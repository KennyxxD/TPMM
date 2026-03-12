import argparse
import os
import json
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp, expit

EPS = 1e-12


def log_binom_pmf(k, n, p):
    p = np.clip(p, 1e-300, 1 - 1e-300)
    k = np.asarray(k)
    n = np.asarray(n)
    return (
        gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        + k * np.log(p) + (n - k) * np.log1p(-p)
    )


def fit_binomial_logit_irls(R, N, x, w, beta_init=None, max_iter=50, tol=1e-8, ridge=1e-6):
    """
    Fit beta for p_i = sigmoid(beta0 + beta1 * x_i) by weighted binomial log-likelihood:
      sum_i w_i * [ R_i log p_i + (N_i-R_i) log(1-p_i) ]  (+ const terms ignored)
    Uses IRLS (Newton) on aggregated binomial.

    R,N: counts
    x: covariate (e.g. log1p(N_total))
    w: nonnegative weights (e.g. responsibility for noise component)
    """
    R = np.asarray(R, dtype=float)
    N = np.asarray(N, dtype=float)
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = (N > 0) & np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        # no info: return something safe
        return np.array([np.log(1e-3) - np.log1p(-1e-3), 0.0], dtype=float)

    Rm = R[m]
    Nm = N[m]
    xm = x[m]
    wm = w[m]

    # Design matrix with intercept
    X = np.column_stack([np.ones_like(xm), xm])

    if beta_init is None:
        # start from overall weighted rate
        rate = np.sum(wm * Rm) / np.maximum(np.sum(wm * Nm), EPS)
        rate = float(np.clip(rate, 1e-6, 1 - 1e-6))
        beta = np.array([np.log(rate) - np.log1p(-rate), 0.0], dtype=float)
    else:
        beta = np.asarray(beta_init, dtype=float).copy()
        if beta.size != 2:
            raise ValueError("beta_init must be length-2 (intercept, slope).")

    for _ in range(max_iter):
        eta = X @ beta
        # guard extreme
        eta = np.clip(eta, -30, 30)
        mu = expit(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        # IRLS for binomial with counts:
        # y = R/N, z = eta + (y-mu)/(mu(1-mu)), W = w*N*mu(1-mu)
        y = Rm / Nm
        denom = mu * (1.0 - mu)
        z = eta + (y - mu) / np.maximum(denom, 1e-12)
        W = wm * Nm * denom

        # Weighted least squares: (X^T W X + ridge I) beta = X^T W z
        XT_W = X.T * W  # (2,m)
        H = XT_W @ X
        g = XT_W @ z

        H = H + ridge * np.eye(2)
        beta_new = np.linalg.solve(H, g)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    return beta


def posterior_mixture3_2ch_logitnoise(R_pe, N_pe, R_span, N_span, pis, p_pe, p_span, noise_beta, x):
    """
    Responsibilities gamma_{ik} = P(z=k | data)
    Components:
      k=0 noise: p_i = sigmoid(beta0 + beta1 * x_i), shared for both channels
      k=1 low:   p_pe[1], p_span[1] constant
      k=2 high:  p_pe[2], p_span[2] constant

    Likelihood uses BOTH channels for ALL components:
      L = Binom(R_pe|N_pe,p_pe_k) * Binom(R_span|N_span,p_span_k)
    For noise component, p_pe_k = p_span_k = p_i (varies by i).
    """
    R_pe = np.asarray(R_pe, dtype=int)
    N_pe = np.asarray(N_pe, dtype=int)
    R_span = np.asarray(R_span, dtype=int)
    N_span = np.asarray(N_span, dtype=int)

    pis = np.asarray(pis, dtype=float)
    p_pe = np.asarray(p_pe, dtype=float)
    p_span = np.asarray(p_span, dtype=float)
    noise_beta = np.asarray(noise_beta, dtype=float)
    x = np.asarray(x, dtype=float)

    pis = np.clip(pis, EPS, 1.0)
    pis = pis / pis.sum()

    # p for k=1,2
    p_pe = np.clip(p_pe, 1e-10, 1 - 1e-10)
    p_span = np.clip(p_span, 1e-10, 1 - 1e-10)

    # p_i for noise
    eta = noise_beta[0] + noise_beta[1] * x
    eta = np.clip(eta, -30, 30)
    p_noise_i = expit(eta)
    p_noise_i = np.clip(p_noise_i, 1e-10, 1 - 1e-10)

    logw = np.zeros((R_pe.size, 3), dtype=float)

    # k=0 noise (shared p_i across channels)
    logw[:, 0] = (
        np.log(pis[0])
        + log_binom_pmf(R_pe, N_pe, p_noise_i)
        + log_binom_pmf(R_span, N_span, p_noise_i)
    )

    # k=1 low
    logw[:, 1] = (
        np.log(pis[1])
        + log_binom_pmf(R_pe, N_pe, p_pe[1])
        + log_binom_pmf(R_span, N_span, p_span[1])
    )

    # k=2 high
    logw[:, 2] = (
        np.log(pis[2])
        + log_binom_pmf(R_pe, N_pe, p_pe[2])
        + log_binom_pmf(R_span, N_span, p_span[2])
    )

    lse = logsumexp(logw, axis=1)
    gamma = np.exp(logw - lse[:, None])
    gamma = np.clip(gamma, 1e-15, 1.0)
    gamma = gamma / gamma.sum(axis=1, keepdims=True)
    return gamma, p_noise_i


def em_fit_params_3_2ch_logitnoise(R_pe, N_pe, R_span, N_span, N_total,
                                  max_iter=500, tol=1e-6, n_restarts=20, seed=0,
                                  pi_init=None, p_init_pe=None, p_init_span=None, noise_beta_init=None):
    """
    EM fit for 3-component mixture with 2 channels (Pe/Span) and read-level logit noise.

    Params:
      - pis (3)
      - p_pe[1], p_pe[2]  (constants); p_pe[0] not used (noise is covariate-based)
      - p_span[1], p_span[2] (constants)
      - noise_beta (2): p_noise_i = sigmoid(beta0 + beta1 * log1p(N_total_i))

    Label control:
      - component 0 fixed as noise
      - components 1 and 2 are ordered so that avg p is increasing
    """
    rng = np.random.default_rng(seed)

    R_pe = np.asarray(R_pe, dtype=int)
    N_pe = np.asarray(N_pe, dtype=int)
    R_span = np.asarray(R_span, dtype=int)
    N_span = np.asarray(N_span, dtype=int)
    N_total = np.asarray(N_total, dtype=int)

    if R_pe.size == 0:
        raise ValueError("Empty fit data.")
    if np.any(N_total <= 0):
        raise ValueError("N_total must be positive in fit data.")
    if np.any(R_pe < 0) or np.any(R_span < 0) or np.any(R_pe > N_pe) or np.any(R_span > N_span):
        raise ValueError("Require 0 <= R <= N in each channel.")

    x = np.log1p(N_total.astype(float))

    # channel-wise observed rates (for initialization of low/high)
    p_hat_pe = R_pe / np.maximum(N_pe, 1)
    p_hat_span = R_span / np.maximum(N_span, 1)
    # overall rate for noise beta init
    R_tot = R_pe + R_span
    N_tot = N_pe + N_span
    p_hat_tot = R_tot / np.maximum(N_tot, 1)

    base_pe = np.quantile(p_hat_pe, [0.30, 0.80])
    base_span = np.quantile(p_hat_span, [0.30, 0.80])
    base_pe = np.clip(base_pe, 1e-6, 1 - 1e-6)
    base_span = np.clip(base_span, 1e-6, 1 - 1e-6)

    # noise intercept init near low quantile of total
    base_noise = float(np.clip(np.quantile(p_hat_tot, 0.05), 1e-6, 1 - 1e-6))
    base_beta0 = float(np.log(base_noise) - np.log1p(-base_noise))
    base_beta = np.array([base_beta0, 0.0], dtype=float)

    def one_run(init_pis, init_pe, init_span, init_beta):
        pis = np.asarray(init_pis, dtype=float)
        pis = np.clip(pis, EPS, 1.0)
        pis = pis / pis.sum()

        # Store p arrays length-3 (index 0 unused for noise; keep placeholder)
        p_pe = np.array([1e-3, float(init_pe[0]), float(init_pe[1])], dtype=float)
        p_span = np.array([1e-3, float(init_span[0]), float(init_span[1])], dtype=float)
        p_pe = np.clip(p_pe, 1e-10, 1 - 1e-10)
        p_span = np.clip(p_span, 1e-10, 1 - 1e-10)

        noise_beta = np.asarray(init_beta, dtype=float).copy()
        if noise_beta.size != 2:
            raise ValueError("noise_beta must be length-2.")

        loglike_prev = -np.inf
        converged = False
        n_iter = 0

        for it in range(1, max_iter + 1):
            n_iter = it

            # E-step
            gamma, p_noise_i = posterior_mixture3_2ch_logitnoise(
                R_pe, N_pe, R_span, N_span, pis, p_pe, p_span, noise_beta, x
            )

            # M-step
            Nk = gamma.sum(axis=0)
            pis_new = Nk / Nk.sum()

            # Update low/high constants (k=1,2) channel-wise
            for k in (1, 2):
                denom_pe = np.sum(gamma[:, k] * N_pe)
                num_pe = np.sum(gamma[:, k] * R_pe)
                denom_span = np.sum(gamma[:, k] * N_span)
                num_span = np.sum(gamma[:, k] * R_span)

                denom_pe = max(denom_pe, EPS)
                denom_span = max(denom_span, EPS)

                p_pe[k] = float(np.clip(num_pe / denom_pe, 1e-10, 1 - 1e-10))
                p_span[k] = float(np.clip(num_span / denom_span, 1e-10, 1 - 1e-10))

            # Order k=1,2 by average p to stabilize labels
            score1 = 0.5 * (p_pe[1] + p_span[1])
            score2 = 0.5 * (p_pe[2] + p_span[2])
            if score2 < score1:
                # swap low/high
                p_pe[1], p_pe[2] = p_pe[2], p_pe[1]
                p_span[1], p_span[2] = p_span[2], p_span[1]
                pis_new[1], pis_new[2] = pis_new[2], pis_new[1]
                gamma[:, [1, 2]] = gamma[:, [2, 1]]  # keep consistency for next step

            # Update noise beta (read-level): use combined counts; weights = gamma[:,0]
            # Using totals is OK for optimizing beta (combinatorial constants don't affect beta optimum)
            noise_beta_new = fit_binomial_logit_irls(
                R=R_tot, N=N_tot, x=x, w=gamma[:, 0], beta_init=noise_beta,
                max_iter=50, tol=1e-8, ridge=1e-6
            )

            # Compute full log-likelihood under updated params (use two channels for all components)
            eta = np.clip(noise_beta_new[0] + noise_beta_new[1] * x, -30, 30)
            p_noise_i2 = np.clip(expit(eta), 1e-10, 1 - 1e-10)

            logw = np.zeros((R_pe.size, 3), dtype=float)
            logw[:, 0] = (
                np.log(np.clip(pis_new[0], EPS, 1.0))
                + log_binom_pmf(R_pe, N_pe, p_noise_i2)
                + log_binom_pmf(R_span, N_span, p_noise_i2)
            )
            logw[:, 1] = (
                np.log(np.clip(pis_new[1], EPS, 1.0))
                + log_binom_pmf(R_pe, N_pe, p_pe[1])
                + log_binom_pmf(R_span, N_span, p_span[1])
            )
            logw[:, 2] = (
                np.log(np.clip(pis_new[2], EPS, 1.0))
                + log_binom_pmf(R_pe, N_pe, p_pe[2])
                + log_binom_pmf(R_span, N_span, p_span[2])
            )
            loglike = float(np.sum(logsumexp(logw, axis=1)))

            if np.isfinite(loglike_prev) and abs(loglike - loglike_prev) < tol:
                pis = pis_new
                noise_beta = noise_beta_new
                loglike_prev = loglike
                converged = True
                break

            pis = pis_new
            noise_beta = noise_beta_new
            loglike_prev = loglike

        return {
            "pis": pis,
            "p_pe": p_pe.copy(),
            "p_span": p_span.copy(),
            "noise_beta": noise_beta.copy(),
            "loglike": loglike_prev,
            "n_iter": n_iter,
            "converged": converged
        }

    best = None

    for r in range(int(n_restarts)):
        # init pis
        if pi_init is not None and r == 0:
            init_pis = np.asarray(pi_init, dtype=float)
            if init_pis.size != 3:
                raise ValueError("pi_init must have 3 values.")
        else:
            init_pis = rng.dirichlet(alpha=np.ones(3))

        # init low/high p (2 values each channel)
        if p_init_pe is not None and r == 0:
            init_pe = np.asarray(p_init_pe, dtype=float)
            if init_pe.size != 2:
                raise ValueError("p_init_pe must have 2 values (low, high).")
            init_pe = np.clip(init_pe, 1e-6, 1 - 1e-6)
        else:
            init_pe = np.clip(base_pe + rng.normal(0, 0.02, size=2), 1e-6, 1 - 1e-6)

        if p_init_span is not None and r == 0:
            init_span = np.asarray(p_init_span, dtype=float)
            if init_span.size != 2:
                raise ValueError("p_init_span must have 2 values (low, high).")
            init_span = np.clip(init_span, 1e-6, 1 - 1e-6)
        else:
            init_span = np.clip(base_span + rng.normal(0, 0.02, size=2), 1e-6, 1 - 1e-6)

        # init noise beta
        if noise_beta_init is not None and r == 0:
            init_beta = np.asarray(noise_beta_init, dtype=float)
            if init_beta.size != 2:
                raise ValueError("noise_beta_init must have 2 values (beta0,beta1).")
        else:
            init_beta = base_beta + rng.normal(0, [0.5, 0.2], size=2)

        res = one_run(init_pis, init_pe, init_span, init_beta)
        if best is None or res["loglike"] > best["loglike"]:
            best = res

    return best


def compute_FRN(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Pe_F", "Span_F", "Pe_R", "Span_R"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df.copy()

    # ensure integer counts for each channel
    for c in ["Pe_F", "Pe_R", "Span_F", "Span_R"]:
        if out[c].isna().any():
            raise ValueError(f"Found NaN in {c}.")
        out[c] = np.rint(out[c].to_numpy()).astype(int)

    # per-channel totals
    out["F_pe"] = out["Pe_F"]
    out["R_pe"] = out["Pe_R"]
    out["N_pe"] = out["F_pe"] + out["R_pe"]

    out["F_span"] = out["Span_F"]
    out["R_span"] = out["Span_R"]
    out["N_span"] = out["F_span"] + out["R_span"]

    # keep legacy aggregated columns for compatibility/output
    out["F"] = out["F_pe"] + out["F_span"]
    out["R"] = out["R_pe"] + out["R_span"]
    out["N"] = out["F"] + out["R"]

    bad = (
        (out["N"] < 0) | (out["R"] < 0) | (out["F"] < 0) |
        (out["N_pe"] < 0) | (out["N_span"] < 0) |
        (out["R_pe"] < 0) | (out["R_span"] < 0) |
        (out["F_pe"] < 0) | (out["F_span"] < 0) |
        (out["R_pe"] > out["N_pe"]) | (out["R_span"] > out["N_span"]) |
        (out["R"] > out["N"])
    )
    if bad.any():
        raise ValueError(f"Invalid counts in rows: {bad.sum()}")

    return out


def fit_highN_score_all(df: pd.DataFrame,
                        min_coverage: int = 8,
                        q_low: float = 0.90,
                        q_high: float = 0.99,
                        max_iter: int = 500,
                        tol: float = 1e-6,
                        restarts: int = 25,
                        seed: int = 0):
    """
    Fit 3-comp EM on high-N subset (quantile range), then score all points.
    Now uses multi-channel (Pe, Span) + read-level logit noise.
    Keeps original index (no reset_index).
    """
    df0 = compute_FRN(df)

    df_all = df0[df0["N"] >= int(min_coverage)].copy()
    if len(df_all) == 0:
        raise ValueError("No rows pass min_coverage.")

    fit_min = float(df_all["N"].quantile(q_low))
    fit_max = float(df_all["N"].quantile(q_high))

    df_fit = df_all[(df_all["N"] >= fit_min) & (df_all["N"] <= fit_max)].copy()
    if len(df_fit) == 0:
        raise ValueError("No rows in fit range; adjust q_low/q_high.")

    fit_res = em_fit_params_3_2ch_logitnoise(
        R_pe=df_fit["R_pe"].to_numpy(),
        N_pe=df_fit["N_pe"].to_numpy(),
        R_span=df_fit["R_span"].to_numpy(),
        N_span=df_fit["N_span"].to_numpy(),
        N_total=df_fit["N"].to_numpy(),
        max_iter=max_iter,
        tol=tol,
        n_restarts=restarts,
        seed=seed
    )

    pis = fit_res["pis"]
    p_pe = fit_res["p_pe"]
    p_span = fit_res["p_span"]
    noise_beta = fit_res["noise_beta"]

    # score all
    gamma, p_noise_i = posterior_mixture3_2ch_logitnoise(
        df_all["R_pe"].to_numpy(),
        df_all["N_pe"].to_numpy(),
        df_all["R_span"].to_numpy(),
        df_all["N_span"].to_numpy(),
        pis, p_pe, p_span,
        noise_beta,
        x=np.log1p(df_all["N"].to_numpy().astype(float))
    )

    # component meanings: 0=noise, 1=low, 2=high
    df_all["posterior_noise"] = gamma[:, 0]
    df_all["posterior_low"] = gamma[:, 1]
    df_all["posterior_high"] = gamma[:, 2]
    df_all["posterior_non_noise"] = 1.0 - df_all["posterior_noise"]

    # keep compatibility:
    df_all["posterior_true"] = df_all["posterior_non_noise"]

    # read-level noise probability per row
    df_all["p_noise_i"] = p_noise_i

    # report reference noise level at median N (just for readability)
    x_med = float(np.log1p(np.median(df_all["N"].to_numpy())))
    p_noise_med = float(np.clip(expit(noise_beta[0] + noise_beta[1] * x_med), 1e-10, 1 - 1e-10))

    params = {
        "pis": [float(x) for x in pis],

        # channel params for low/high
        "p_pe_low": float(p_pe[1]),
        "p_pe_high": float(p_pe[2]),
        "p_span_low": float(p_span[1]),
        "p_span_high": float(p_span[2]),

        # noise is read-level logit
        "noise_beta0": float(noise_beta[0]),
        "noise_beta1": float(noise_beta[1]),
        "p_noise_at_medianN": float(p_noise_med),

        "pi_noise": float(pis[0]),
        "pi_low": float(pis[1]),
        "pi_high": float(pis[2]),

        "q_low": float(q_low),
        "q_high": float(q_high),
        "fit_min_coverage": float(fit_min),
        "fit_max_coverage": float(fit_max),
        "n_all": int(len(df_all)),
        "n_fit": int(len(df_fit)),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "restarts": int(restarts),
        "seed": int(seed),
        "loglike": float(fit_res["loglike"]),
        "converged": bool(fit_res["converged"]),
        "n_iter": int(fit_res["n_iter"]),
    }
    return df_all, params

# --------- robust original-line export (physical line numbers) ---------
def read_table_with_src_line_no(path: str, sep: str = "\t") -> pd.DataFrame:
    """
    Read TSV into a DataFrame and attach '__src_line_no' storing the physical line number
    in the original file for each data row.

    We skip blank lines to match pandas default skip_blank_lines=True.
    """
    src_line_nos = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            raise RuntimeError("Empty input file.")
        for line_no, line in enumerate(f, start=2):  # header is line 1
            if not line.strip():
                continue
            src_line_nos.append(line_no)

    df = pd.read_csv(path, sep=sep, skip_blank_lines=True)

    if len(df) != len(src_line_nos):
        raise RuntimeError(
            f"Row count mismatch: pandas={len(df)} vs scanned_nonblank={len(src_line_nos)}.\n"
            "If the file has malformed rows or embedded newlines inside quoted fields, "
            "fix the input formatting."
        )

    df["__src_line_no"] = src_line_nos
    return df


def write_hits_original_by_line_no(in_path: str, out_path: str, line_nos):
    """
    Copy original lines (preserve raw format) for hit rows, using PHYSICAL line numbers.
    Always writes header.
    """
    keep = set(int(x) for x in line_nos)
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if line_no == 1:
                fout.write(line)  # header
                continue
            if line_no in keep:
                fout.write(line)


# --------- posterior-based (Bayesian) FDR selection ---------
def select_hits_bfdr(df: pd.DataFrame, alpha: float,
                     rank_col: str = "posterior_true",
                     lfdr_col: str = "posterior_noise"):
    """
    Bayesian FDR via cumulative lfdr (posterior_noise):
      - rank by rank_col descending (typically posterior_non_noise)
      - cum_fdr(k) = mean_{top k}(lfdr)
      - take largest k with cum_fdr <= alpha

    Returns:
      hit_mask (aligned to df.index),
      cum_fdr_series (aligned to df.index; NaN for rows not rankable),
      rank_pos_series (1..n in ranking; NaN if not rankable)
    """
    if alpha is None:
        raise ValueError("alpha must not be None for BFDR selection.")
    alpha = float(alpha)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("bfdr_alpha must be in (0,1].")

    # rankable rows: finite rank_col and lfdr_col
    ok = np.isfinite(df[rank_col].to_numpy()) & np.isfinite(df[lfdr_col].to_numpy())
    if not np.any(ok):
        hit_mask = pd.Series(False, index=df.index)
        return hit_mask, pd.Series(np.nan, index=df.index), pd.Series(np.nan, index=df.index)

    d = df.loc[ok, [rank_col, lfdr_col]].copy()
    d = d.sort_values(rank_col, ascending=False, kind="mergesort")
    lfdr = d[lfdr_col].to_numpy(dtype=float)
    cum_fdr = np.cumsum(lfdr) / (np.arange(lfdr.size, dtype=float) + 1.0)

    # find max k satisfying
    good = np.where(cum_fdr <= alpha)[0]
    kmax = int(good[-1] + 1) if good.size else 0

    # build outputs aligned to df.index
    hit_mask = pd.Series(False, index=df.index)
    if kmax > 0:
        hit_mask.loc[d.index[:kmax]] = True

    cum_fdr_series = pd.Series(np.nan, index=df.index)
    cum_fdr_series.loc[d.index] = cum_fdr

    rank_pos_series = pd.Series(np.nan, index=df.index)
    rank_pos_series.loc[d.index] = np.arange(1, len(d) + 1, dtype=float)

    return hit_mask, cum_fdr_series, rank_pos_series


# -----------------------------
# Main
# -----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(
        description="Three-component Binomial EM on high-N subset, score all; report hits, summary."
    )
    ap.add_argument("--input", required=True, help="Input TSV path (e.g., gut1_pos.txt)")
    ap.add_argument("--sep", default="\t", help="Separator (default: tab)")
    ap.add_argument("--min_coverage", type=int, default=8, help="Keep rows with N>=min_coverage (default: 8)")
    ap.add_argument("--q_low", type=float, default=0.90, help="Lower quantile for fit subset (default: 0.90)")
    ap.add_argument("--q_high", type=float, default=0.99, help="Upper quantile for fit subset (default: 0.99)")
    ap.add_argument("--max_iter", type=int, default=500, help="EM max iterations (default: 500)")
    ap.add_argument("--tol", type=float, default=1e-6, help="EM tolerance on loglike (default: 1e-6)")
    ap.add_argument("--restarts", type=int, default=25, help="EM restarts (default: 25)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    # Filtering:
    # posterior_true is now posterior_non_noise
    POSTERIOR_MIN_DEFAULT = 0.90

    ap.add_argument("--posterior_min", type=float, default=POSTERIOR_MIN_DEFAULT,
                    help="Hit filter: posterior_true>= (default: 0.90). Now posterior_true=posterior_non_noise.")

    # Optional: choose which posterior column to use for hits
    ap.add_argument("--posterior_col", default="posterior_true",
                    choices=["posterior_true", "posterior_non_noise", "posterior_high", "posterior_low", "posterior_noise"],
                    help="Which posterior column used in hit filter & grid summaries (default: posterior_true=non_noise)")

    # posterior-based (Bayesian) FDR selection
    ap.add_argument("--use_bfdr", action="store_true",
                    help="Use posterior-based Bayesian FDR selection (cumulative lfdr = posterior_noise).")
    ap.add_argument("--bfdr_alpha", type=float, default=0.05,
                    help="Target Bayesian FDR alpha for --use_bfdr (default: 0.05).")

    # Output control
    ap.add_argument("--outdir", default=None, help="Output directory")
    ap.add_argument("--prefix", default=None, help="Output prefix")

    args = ap.parse_args()

    in_path = args.input
    in_dir = os.path.dirname(os.path.abspath(in_path))
    base = os.path.splitext(os.path.basename(in_path))[0]

    outdir = args.outdir if args.outdir else os.path.join(in_dir, "em_out")
    prefix = args.prefix if args.prefix else base
    ensure_dir(outdir)

    # Load (robustly track physical source line numbers)
    df = read_table_with_src_line_no(in_path, sep=args.sep)

    # Fit + score
    result, params = fit_highN_score_all(
        df,
        min_coverage=args.min_coverage,
        q_low=args.q_low,
        q_high=args.q_high,
        max_iter=args.max_iter,
        tol=args.tol,
        restarts=args.restarts,
        seed=args.seed
    )

    # Hits (preserve original order via index)
    post_col = args.posterior_col

    if args.use_bfdr:
        # Rank by chosen posterior column; lfdr is posterior_noise
        bfdr_mask, cum_fdr_s, rank_pos_s = select_hits_bfdr(
            result, alpha=args.bfdr_alpha, rank_col=post_col, lfdr_col="posterior_noise"
        )
        result["bfdr_rank_pos"] = rank_pos_s
        result["bfdr_cum_fdr"] = cum_fdr_s

        hits = result.loc[bfdr_mask].copy()

        # Keep backward-compatible thresholds as OPTIONAL extra filters:
        # If user left it at default, do NOT apply (so BFDR works "out of the box").
        apply_extra = (float(args.posterior_min) != float(POSTERIOR_MIN_DEFAULT))
        if apply_extra:
            hits = hits[hits[post_col] >= args.posterior_min].copy()

        hits = hits.sort_index()
    else:
        hit_mask = result[post_col] >= args.posterior_min
        hits = result.loc[hit_mask].copy()
        hits = hits.sort_index()

    # Print key info (stdout)
    print(json.dumps(params, indent=2, ensure_ascii=False))
    if args.use_bfdr:
        print(f"\nMain filter: BFDR(alpha={args.bfdr_alpha}) using rank_col={post_col}, lfdr=posterior_noise")
        if ("bfdr_cum_fdr" in result.columns) and len(hits):
            max_cum = float(np.nanmax(hits["bfdr_cum_fdr"].to_numpy()))
            print(f"  (max cum_fdr among hits: {max_cum:.6g})")
    else:
        print(f"\nMain filter: {post_col}>={args.posterior_min}")
    print(f"Hits count: {len(hits)} / {len(result)} (N>= {args.min_coverage})")

    # Write outputs
    out_all = os.path.join(outdir, f"{prefix}.em.tsv")
    out_hits_scored = os.path.join(outdir, f"{prefix}.hits.tsv")
    out_hits_txt = os.path.join(outdir, f"{prefix}.hits.txt")  # original format lines
    out_params = os.path.join(outdir, f"{prefix}.params.json")

    # don't leak internal helper column to scored TSVs (but keep it in-memory for hits.txt export)
    result_out = result.drop(columns=["__src_line_no"], errors="ignore")
    hits_out = hits.drop(columns=["__src_line_no"], errors="ignore")

    result_out.to_csv(out_all, sep="\t", index=False)
    hits_out.to_csv(out_hits_scored, sep="\t", index=False)

    # robust original lines export
    if "__src_line_no" not in hits.columns:
        raise RuntimeError("Internal error: __src_line_no missing; cannot write hits.txt robustly.")
    write_hits_original_by_line_no(in_path, out_hits_txt, hits["__src_line_no"].to_numpy())

    with open(out_params, "w", encoding="utf-8") as f:
        json.dump(
            {
                "params": params,
                "main_filter": (
                    {"method": "bfdr", "alpha": float(args.bfdr_alpha), "rank_col": post_col, "lfdr_col": "posterior_noise"}
                    if args.use_bfdr
                    else {"method": "threshold", "posterior_col": post_col, "posterior_min": args.posterior_min}
                ),
                "counts": {"n_all": int(len(result)), "n_hits": int(len(hits))}
            },
            f,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nWrote:")
    print(f"  {out_all}")
    print(f"  {out_hits_scored}")
    print(f"  {out_hits_txt}  (original-format lines)")
    print(f"  {out_params}")


if __name__ == "__main__":
    main()