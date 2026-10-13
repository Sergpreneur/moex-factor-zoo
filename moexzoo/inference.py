"""Inference ladder (pre-analysis plan §6): NW t-statistics, Holm, BH, BY, Romano–Wolf stepdown
with a stationary bootstrap, Deflated Sharpe Ratio, CSCV/PBO, effective number of tests and an
empirical-Bayes hierarchical model."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import optimize, stats

from . import config as C

EULER = 0.5772156649015329


def nw_lag(T: int) -> int:
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


# ----------------------------------------------------------------------------------
# Newey–West for (possibly gappy) series: missing months contribute zero deviation
# ----------------------------------------------------------------------------------

def _nw_mean_se(X: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X: (..., T, K) with NaN. Returns mean, NW standard error of the mean, n (shape (..., K))."""
    obs = ~np.isnan(X)
    n = obs.sum(-2)
    Xz = np.where(obs, X, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = Xz.sum(-2) / n
    e = np.where(obs, X - mean[..., None, :], 0.0)
    g0 = (e * e).sum(-2)
    s = g0.copy()
    T = X.shape[-2]
    for j in range(1, L + 1):
        w = 1.0 - j / (L + 1.0)
        gj = (e[..., j:, :] * e[..., : T - j, :]).sum(-2)
        s += 2.0 * w * gj
    with np.errstate(invalid="ignore", divide="ignore"):
        var = s / n**2
        se = np.sqrt(np.maximum(var, 0.0))
    return mean, se, n


def tstats(df: pd.DataFrame, L: int | None = None) -> pd.DataFrame:
    X = df.values.astype(float)
    T = int(np.max((~np.isnan(X)).sum(0)))
    L = nw_lag(T) if L is None else L
    mean, se, n = _nw_mean_se(X, L)
    t = mean / se
    p = 2.0 * stats.norm.sf(np.abs(t))
    sd = np.nanstd(X, 0, ddof=1)
    out = pd.DataFrame({"mean": mean, "se": se, "t": t, "p": p, "n": n, "sd": sd, "sr_m": mean / sd}, index=df.columns)
    out["sr_ann"] = out["sr_m"] * np.sqrt(12.0)
    out["skew"] = [stats.skew(df[c].dropna()) for c in df.columns]
    out["kurt"] = [stats.kurtosis(df[c].dropna(), fisher=False) for c in df.columns]
    out["L"] = L
    return out


# ----------------------------------------------------------------------------------
# classical multiplicity corrections (two-sided p-values)
# ----------------------------------------------------------------------------------

def holm(p: np.ndarray, alpha: float = C.ALPHA) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    rej = np.zeros(m, bool)
    for i, k in enumerate(order):
        if p[k] <= alpha / (m - i):
            rej[k] = True
        else:
            break
    return rej


def bh(p: np.ndarray, alpha: float = C.ALPHA, dependence_free: bool = False) -> np.ndarray:
    m = len(p)
    cm = np.sum(1.0 / np.arange(1, m + 1)) if dependence_free else 1.0
    order = np.argsort(p)
    ps = p[order]
    thr = alpha * np.arange(1, m + 1) / (m * cm)
    below = np.where(ps <= thr)[0]
    rej = np.zeros(m, bool)
    if len(below):
        rej[order[: below.max() + 1]] = True
    return rej


def bh_adjusted(p: np.ndarray, dependence_free: bool = False) -> np.ndarray:
    m = len(p)
    cm = np.sum(1.0 / np.arange(1, m + 1)) if dependence_free else 1.0
    order = np.argsort(p)
    ps = p[order] * m * cm / np.arange(1, m + 1)
    ps = np.minimum.accumulate(ps[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ps, 1.0)
    return out


# ----------------------------------------------------------------------------------
# stationary bootstrap and Romano–Wolf stepdown
# ----------------------------------------------------------------------------------

def stationary_indices(T: int, B: int, mean_block: float, rng: np.random.Generator,
                       length: int | None = None) -> np.ndarray:
    """Politis–Romano stationary-bootstrap indices into a series of length T (circular).
    `length` (default T) is the length of each resampled path."""
    n = T if length is None else length
    p = 1.0 / mean_block
    idx = np.empty((B, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, T, B)
    new_block = rng.random((B, n)) < p
    starts = rng.integers(0, T, (B, n))
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % T)
    return idx


def bootstrap_tstats(X: np.ndarray, B: int = C.BOOT_REPS, mean_block: float = C.BOOT_MEAN_BLOCK,
                     seed: int = C.BOOT_SEED, L: int | None = None, chunk: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Studentised, centred bootstrap statistics t*_kb = (mean*_k - mean_k)/se*_k. Returns (t_obs, T*) with
    T* of shape (B, K)."""
    T, K = X.shape
    L = nw_lag(T) if L is None else L
    mean, se, _ = _nw_mean_se(X, L)
    t_obs = mean / se
    rng = np.random.default_rng(seed)
    idx = stationary_indices(T, B, mean_block, rng)
    out = np.empty((B, K))
    for a in range(0, B, chunk):
        Xb = X[idx[a: a + chunk]]  # (b, T, K)
        mb, sb, _ = _nw_mean_se(Xb, L)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a: a + chunk] = (mb - mean) / sb
    return t_obs, out


def romano_wolf(t_obs: np.ndarray, t_boot: np.ndarray, alpha: float = C.ALPHA) -> dict:
    """Two-sided stepdown with adjusted p-values (Romano & Wolf 2005, 2016)."""
    K = len(t_obs)
    a = np.abs(t_obs)
    ab = np.abs(t_boot)
    ab = np.where(np.isnan(ab), 0.0, ab)
    order = np.argsort(-a)
    B = ab.shape[0]
    padj_raw = np.empty(K)
    for i, k in enumerate(order):
        remaining = order[i:]
        mx = ab[:, remaining].max(1)
        padj_raw[i] = (1.0 + np.sum(mx >= a[k])) / (B + 1.0)
    padj_sorted = np.maximum.accumulate(padj_raw)
    padj = np.empty(K)
    padj[order] = padj_sorted
    # stepdown critical values (for reporting)
    crit = []
    active = np.ones(K, bool)
    rejected = np.zeros(K, bool)
    while True:
        mx = ab[:, active].max(1)
        c = np.quantile(mx, 1 - alpha)
        crit.append(float(c))
        new = active & (a > c)
        if not new.any():
            break
        rejected |= new
        active &= ~new
        if not active.any():
            break
    return {"padj": padj, "reject": padj <= alpha, "crit_first": crit[0], "crit_steps": crit,
            "reject_stepdown": rejected}


# ----------------------------------------------------------------------------------
# Deflated Sharpe Ratio and CSCV/PBO
# ----------------------------------------------------------------------------------

def expected_max_sr(var_sr: float, N: float) -> float:
    if N <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - EULER) * stats.norm.ppf(1 - 1.0 / N) + EULER * stats.norm.ppf(1 - 1.0 / (N * math.e)))


def psr(sr: float, sr_star: float, T: int, skew: float, kurt: float) -> float:
    den = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if den <= 0:
        return float("nan")
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(T - 1.0) / math.sqrt(den)))


def deflated_sharpe(ts: pd.DataFrame, N: float, var_sr: float | None = None) -> pd.Series:
    """ts: output of `tstats` (monthly SR, skew, kurt, n)."""
    v = float(np.nanvar(ts["sr_m"].values, ddof=1)) if var_sr is None else var_sr
    sr0 = expected_max_sr(v, N)
    return pd.Series({k: psr(r["sr_m"], sr0, int(r["n"]), r["skew"], r["kurt"]) for k, r in ts.iterrows()}), sr0


def pbo_cscv(X: pd.DataFrame, S: int = C.CSCV_SPLITS) -> dict:
    M = X.dropna().values
    T, K = M.shape
    blocks = np.array_split(np.arange(T), S)
    logits = []
    for comb in itertools.combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        sr_is = M[is_idx].mean(0) / M[is_idx].std(0, ddof=1)
        sr_oos = M[oos_idx].mean(0) / M[oos_idx].std(0, ddof=1)
        best = int(np.argmax(sr_is))
        rank = stats.rankdata(sr_oos)[best]  # 1..K
        w = rank / (K + 1.0)
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "logits": logits, "T": T, "K": K}


# ----------------------------------------------------------------------------------
# effective number of tests
# ----------------------------------------------------------------------------------

def neff(corr: np.ndarray) -> dict:
    lam = np.linalg.eigvalsh(corr)
    lam = np.clip(lam, 0, None)
    M = len(lam)
    nyholt = 1 + (M - 1) * (1 - np.var(lam, ddof=1) / M)
    liji = float(np.sum((lam >= 1).astype(float) + (lam - np.floor(lam))))
    return {"nyholt": float(nyholt), "li_ji": liji, "eigen": lam[::-1]}


# ----------------------------------------------------------------------------------
# empirical Bayes hierarchical model (after Jensen, Kelly & Pedersen 2023)
# ----------------------------------------------------------------------------------

def nearest_psd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    return (V * w) @ V.T


def empirical_bayes(mu_hat: np.ndarray, Sigma: np.ndarray, clusters: list[str]) -> dict:
    """mu_hat ~ N(mu, Sigma); mu = mu0 + Z c + w, c ~ N(0, tc^2 I), w ~ N(0, tw^2 I).
    Hyper-parameters by maximum marginal likelihood; returns posterior summaries."""
    K = len(mu_hat)
    labs = sorted(set(clusters))
    Z = np.array([[1.0 if clusters[i] == l else 0.0 for l in labs] for i in range(K)])
    ZZ = Z @ Z.T
    one = np.ones(K)

    def negll(theta):
        mu0, ltc, ltw = theta
        V = Sigma + np.exp(2 * ltc) * ZZ + np.exp(2 * ltw) * np.eye(K)
        sign, logdet = np.linalg.slogdet(V)
        r = mu_hat - mu0 * one
        return 0.5 * (logdet + r @ np.linalg.solve(V, r))

    best = None
    for s0 in (-6.0, -5.0, -4.0, -3.0):
        for s1 in (-6.0, -5.0, -4.0, -3.0):
            res = optimize.minimize(negll, x0=np.array([float(np.mean(mu_hat)), s0, s1]), method="Nelder-Mead",
                                    options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000})
            if best is None or res.fun < best.fun:
                best = res
    mu0, ltc, ltw = best.x
    tc2, tw2 = math.exp(2 * ltc), math.exp(2 * ltw)
    Omega = tc2 * ZZ + tw2 * np.eye(K)  # prior covariance of mu around mu0
    V = Sigma + Omega
    G = Omega @ np.linalg.inv(V)
    post_mean = mu0 + G @ (mu_hat - mu0)
    post_cov = Omega - G @ Omega
    post_sd = np.sqrt(np.clip(np.diag(post_cov), 0, None))
    p_pos = stats.norm.cdf(post_mean / post_sd)
    return {"mu0": mu0, "tau_c": math.sqrt(tc2), "tau_w": math.sqrt(tw2), "post_mean": post_mean, "post_sd": post_sd,
            "p_pos": p_pos, "lo95": post_mean - 1.96 * post_sd, "hi95": post_mean + 1.96 * post_sd,
            "negll": best.fun}
