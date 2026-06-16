"""Python DIY-loop baselines (naive off-the-shelf + best hand-rolled), keyed by family.

Kernels are lifted verbatim from the original mcpower_benchmark_{ols,glm,lme}.py;
the model constants (beta / targets / baseline / ICC / clusters) are now read from a
Case, and the design matrix is built per-shape (continuous mains, then factor dummies,
then interaction products). Timing comparators only — power is a sanity signal.
"""
from __future__ import annotations
import os
import multiprocessing as mp
import re
import numpy as np
from scipy import stats
from scipy.special import expit
import statsmodels.api as sm
from cases import Case

ALPHA = 0.05


def _rhs(formula: str) -> str:
    # right-hand side after ~ or =, with any "(1|g)" random-effects term removed
    rhs = re.split(r"[~=]", formula, maxsplit=1)[1]
    return re.sub(r"\([^)]*\|[^)]*\)", "", rhs)


def _chol(case: Case, cont: list[str]):
    """Cholesky factor of the continuous-mains correlation matrix, or None."""
    if not case.correlations:
        return None
    C = np.eye(len(cont))
    for a, b, r in re.findall(r"corr\(\s*(\w+)\s*,\s*(\w+)\s*\)\s*=\s*(-?[0-9.]+)",
                              case.correlations):
        i, j = cont.index(a), cont.index(b)
        C[i, j] = C[j, i] = float(r)
    return np.linalg.cholesky(C)


def _design(case: Case):
    """Parse a Case into the fixed column convention.

    Column order: [continuous mains | factor dummy blocks in variable_types
    order | interaction terms in formula order]. Interaction terms expand to
    cc -> 1 col, cf -> one col per dummy of f, ff -> dummy x dummy (f1-major).

    Parser constraints for new cases: continuous predictors must be named
    x<digits> (anything else is silently dropped from the design), and only
    pairwise interactions are parsed (a*b*c silently loses the 3-way term).

    Returns dict with:
      n_cont: int                          # continuous main predictors
      proportions: list[list[float]]       # one list per factor (levels = len)
      n_dummies_of: list[int]              # dummies per factor
      interactions: list[tuple[str,tuple]] # ("cc",(ci,cj)) | ("cf",(ci,fi)) | ("ff",(fi,fj))
      beta: np.ndarray                     # length = total P (non-intercept), col order
      targets: np.ndarray[int]             # column indices of case.targets
      chol: np.ndarray | None              # Cholesky factor for correlated continuous draws
    """
    rhs = _rhs(case.formula)
    fac_names = list(case.variable_types.keys())
    proportions = []
    for spec_str in case.variable_types.values():
        proportions.append([float(x) for x in re.findall(r"[0-9]*\.?[0-9]+", spec_str)])
    n_dummies_of = [len(p) - 1 for p in proportions]

    cont: list[str] = []
    for name in re.findall(r"x\d+", rhs):
        if name not in cont:
            cont.append(name)
    n_cont = len(cont)
    n_dummies = sum(n_dummies_of)

    # interaction terms in formula order: ("cc",(ci,cj)) | ("cf",(ci,fi)) | ("ff",(fi,fj))
    interactions = []
    for a, b in re.findall(r"(\w+)\s*[:*]\s*(\w+)", rhs):
        if a in cont and b in cont:
            interactions.append(("cc", (cont.index(a), cont.index(b))))
        elif a in cont and b in fac_names:
            interactions.append(("cf", (cont.index(a), fac_names.index(b))))
        elif a in fac_names and b in fac_names:
            interactions.append(("ff", (fac_names.index(a), fac_names.index(b))))
        else:
            raise ValueError(f"unparseable interaction {a}:{b} in {case.id}")

    def _width(term):
        kind, idx = term
        if kind == "cc":
            return 1
        if kind == "cf":
            return n_dummies_of[idx[1]]
        return n_dummies_of[idx[0]] * n_dummies_of[idx[1]]

    inter_offset, off = [], n_cont + n_dummies
    for t in interactions:
        inter_offset.append(off)
        off += _width(t)
    P = off

    fac_offset, foff = [], n_cont
    for nd in n_dummies_of:
        fac_offset.append(foff)
        foff += nd

    def _atom(tok):
        """('c', cont_idx, None) or ('f', fac_idx, level)."""
        m = re.match(r"(\w+)\[(\d+)\]$", tok)
        if m:
            return ("f", fac_names.index(m.group(1)), int(m.group(2)))
        if tok in cont:
            return ("c", cont.index(tok), None)
        if tok in fac_names and len(proportions[fac_names.index(tok)]) == 2:
            return ("f", fac_names.index(tok), 2)        # bare 2-level factor -> level 2
        raise ValueError(f"cannot resolve effect token {tok!r} in {case.id}")

    def col_of(name: str) -> int:
        name = name.strip()
        if ":" in name:
            l, r_ = (s.strip() for s in name.split(":"))
            a, b = _atom(l), _atom(r_)
            if (a[0], b[0]) == ("c", "c"):
                return inter_offset[interactions.index(("cc", (a[1], b[1])))]
            if (a[0], b[0]) == ("c", "f"):
                return inter_offset[interactions.index(("cf", (a[1], b[1])))] + (b[2] - 2)
            if (a[0], b[0]) == ("f", "f"):
                ti = interactions.index(("ff", (a[1], b[1])))
                return inter_offset[ti] + (a[2] - 2) * n_dummies_of[b[1]] + (b[2] - 2)
            raise ValueError(f"unsupported interaction order in {name!r} ({case.id})")
        a = _atom(name)
        return a[1] if a[0] == "c" else fac_offset[a[1]] + (a[2] - 2)

    beta = np.zeros(P)
    for part in case.effects.split(","):
        nm, val = part.split("=")
        beta[col_of(nm.strip())] = float(val)

    targets = np.array([col_of(t) for t in case.targets], dtype=int)
    return {"n_cont": n_cont, "proportions": proportions, "n_dummies_of": n_dummies_of,
            "interactions": interactions, "beta": beta, "targets": targets,
            "chol": _chol(case, cont)}


def _draw_X(rng: np.random.Generator, n: int, spec: dict) -> np.ndarray:
    """Build the (n, P) design (no intercept) in the fixed column order."""
    cont = rng.standard_normal((n, spec["n_cont"]))
    if spec["chol"] is not None:
        cont = cont @ spec["chol"].T
    dummy_blocks = []
    for props in spec["proportions"]:
        levels = rng.choice(len(props), size=n, p=np.asarray(props) / np.sum(props))
        dummy_blocks.append(np.stack([(levels == k).astype(np.float64)
                                      for k in range(1, len(props))], axis=1))
    blocks = [cont] + dummy_blocks
    for kind, idx in spec["interactions"]:
        if kind == "cc":
            i, j = idx
            blocks.append((cont[:, i] * cont[:, j])[:, None])
        elif kind == "cf":
            ci, fi = idx
            blocks.append(cont[:, ci:ci + 1] * dummy_blocks[fi])
        else:  # ff, f1-major
            d1, d2 = dummy_blocks[idx[0]], dummy_blocks[idx[1]]
            blocks.append(np.stack([d1[:, k1] * d2[:, k2]
                                    for k1 in range(d1.shape[1])
                                    for k2 in range(d2.shape[1])], axis=1))
    return np.column_stack(blocks)


# ---------------------------------------------------------------------------
# Parallel pool helpers
# ---------------------------------------------------------------------------

def _split_counts(n_sims: int, n_workers: int) -> list[int]:
    base, extra = divmod(n_sims, n_workers)
    return [c for c in (base + (1 if i < extra else 0) for i in range(n_workers)) if c > 0]


def _parallel(chunk_fn, case, n, n_sims, seed):
    """Run a *_best_chunk across a process pool; pool raw counts, then estimate.

    cpu_count() pool workers × multi-threaded BLAS would oversubscribe the machine
    and inflate the loop baseline, flattering MCPower; BLAS is pinned in harness.py.
    np.random.SeedSequence is picklable and accepted by np.random.default_rng().
    """
    counts = _split_counts(n_sims, min(os.cpu_count() or 1, n_sims))
    seeds = np.random.SeedSequence(seed).spawn(len(counts))
    with mp.Pool(len(counts)) as pool:
        parts = pool.starmap(chunk_fn, [(case, n, c, s) for c, s in zip(counts, seeds)])
    t_rej = sum(p[0] for p in parts)
    f_rej = sum(p[1] for p in parts)
    usable = sum(p[2] for p in parts)
    denom = max(usable, 1)
    return t_rej / denom, f_rej / denom, usable


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

def _ols_best_chunk(case, n, n_sims, seed):
    rng = np.random.default_rng(seed)
    spec = _design(case); BETA = spec["beta"]; TARGETS = spec["targets"]; P = BETA.size
    df_resid = n - (P + 1)
    t_crit = stats.t.ppf(1.0 - ALPHA / 2.0, df_resid)
    f_crit = stats.f.ppf(1.0 - ALPHA, TARGETS.size, df_resid)
    t_rejections = np.zeros(P, dtype=np.int64); f_rejections = 0; usable = 0
    ones = np.ones(n)
    for _ in range(n_sims):
        X = _draw_X(rng, n, spec)
        y = X @ BETA + rng.standard_normal(n)
        Xd = np.column_stack([ones, X])
        # A draw can be rank-deficient (e.g. a factor level absent at small n);
        # skip it, mirroring the convergence-skip in the GLM/LME kernels.
        try:
            XtX_inv = np.linalg.inv(Xd.T @ Xd)
        except np.linalg.LinAlgError:
            continue
        usable += 1
        coef = XtX_inv @ (Xd.T @ y)
        resid = y - Xd @ coef
        sigma2 = (resid @ resid) / df_resid
        se = np.sqrt(sigma2 * np.diag(XtX_inv))
        t_stat = coef[1:] / se[1:]
        t_rejections += (np.abs(t_stat) > t_crit).astype(np.int64)
        idx = TARGETS + 1
        b = coef[idx]; V = sigma2 * XtX_inv[np.ix_(idx, idx)]
        f_stat = float(b @ np.linalg.solve(V, b)) / TARGETS.size
        if f_stat > f_crit: f_rejections += 1
    # Return raw counts for parallel aggregation
    return t_rejections, f_rejections, usable


def ols_best(case, n, n_sims, seed):
    return _parallel(_ols_best_chunk, case, n, n_sims, seed)


def ols_naive(case, n, n_sims, seed):
    rng = np.random.default_rng(seed)
    spec = _design(case); BETA = spec["beta"]; TARGETS = spec["targets"]; P = BETA.size
    t_rejections = np.zeros(P, dtype=np.int64); f_rejections = 0; usable = 0
    R = np.zeros((TARGETS.size, P + 1))
    for row, j in enumerate(TARGETS): R[row, j + 1] = 1.0
    for _ in range(n_sims):
        X = _draw_X(rng, n, spec)
        y = X @ BETA + rng.standard_normal(n)
        Xd = sm.add_constant(X)
        # Skip a rank-deficient draw (e.g. a factor level absent at small n).
        try:
            fit = sm.OLS(y, Xd).fit()
            pvals = np.asarray(fit.pvalues)[1:]
            f_reject = float(fit.f_test(R).pvalue) < ALPHA
        except Exception:
            continue
        usable += 1
        t_rejections += (pvals < ALPHA).astype(np.int64)
        if f_reject: f_rejections += 1
    denom = max(usable, 1)
    return t_rejections / denom, f_rejections / denom


# ---------------------------------------------------------------------------
# GLM logit
# ---------------------------------------------------------------------------

def _fit_logistic_irls(
    X: np.ndarray, y: np.ndarray, max_iter: int = 25, tol: float = 1e-6
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Newton/IRLS fit. Returns (beta, cov); (None, None) on failure or
    non-convergence (mirrors the R kernel's glm.fit converged check)."""
    n, k = X.shape
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        eta = X @ beta
        mu = expit(eta)
        w = mu * (1.0 - mu)
        # Guard against W → 0 (near-separation) destabilising the solve
        w = np.clip(w, 1e-8, None)
        XtW = X.T * w
        XtWX = XtW @ X
        grad = X.T @ (y - mu)
        try:
            step = np.linalg.solve(XtWX, grad)
        except np.linalg.LinAlgError:
            return None, None
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    if not converged:
        return None, None
    eta = X @ beta
    mu = expit(eta)
    w = np.clip(mu * (1.0 - mu), 1e-8, None)
    XtWX = (X.T * w) @ X
    try:
        cov = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return beta, None
    return beta, cov


def _glm_best_chunk(case, n, n_sims, seed):
    rng = np.random.default_rng(seed)
    spec = _design(case); BETA = spec["beta"]; TARGETS = spec["targets"]; P = BETA.size
    INTERCEPT = float(np.log(case.baseline_p / (1.0 - case.baseline_p)))
    z_crit = stats.norm.ppf(1.0 - ALPHA / 2.0)
    chi2_crit = stats.chi2.ppf(1.0 - ALPHA, TARGETS.size)
    t_rejections = np.zeros(P, dtype=np.int64)
    f_rejections = 0
    converged = 0
    ones = np.ones(n)
    for _ in range(n_sims):
        X = _draw_X(rng, n, spec)
        eta = INTERCEPT + X @ BETA
        p = expit(eta)
        y = (rng.random(n) < p).astype(np.float64)
        Xd = np.column_stack([ones, X])
        coef, cov = _fit_logistic_irls(Xd, y)
        if coef is None or cov is None:
            continue
        converged += 1
        # Per-coefficient Wald z-tests (skip intercept)
        se = np.sqrt(np.diag(cov))
        z_stat = coef[1:] / se[1:]
        t_rejections += (np.abs(z_stat) > z_crit).astype(np.int64)
        # Joint Wald chi-squared on TARGETS
        idx = TARGETS + 1
        b = coef[idx]
        V = cov[np.ix_(idx, idx)]
        try:
            w_stat = float(b @ np.linalg.solve(V, b))
        except np.linalg.LinAlgError:
            continue
        if w_stat > chi2_crit:
            f_rejections += 1
    # Return raw counts for parallel aggregation
    return t_rejections, f_rejections, converged


def glm_best(case, n, n_sims, seed):
    return _parallel(_glm_best_chunk, case, n, n_sims, seed)


def glm_naive(case, n, n_sims, seed):
    rng = np.random.default_rng(seed)
    spec = _design(case); BETA = spec["beta"]; TARGETS = spec["targets"]; P = BETA.size
    INTERCEPT = float(np.log(case.baseline_p / (1.0 - case.baseline_p)))
    t_rejections = np.zeros(P, dtype=np.int64)
    f_rejections = 0
    converged = 0
    R = np.zeros((TARGETS.size, P + 1))
    for row, j in enumerate(TARGETS): R[row, j + 1] = 1.0
    for _ in range(n_sims):
        X = _draw_X(rng, n, spec)
        eta = INTERCEPT + X @ BETA
        p = expit(eta)
        y = (rng.random(n) < p).astype(np.float64)
        Xd = sm.add_constant(X)
        try:
            fit = sm.GLM(y, Xd, family=sm.families.Binomial()).fit(disp=0)
        except Exception:
            continue
        if not getattr(fit, "converged", True):
            continue
        converged += 1
        pvals = np.asarray(fit.pvalues)[1:]  # drop intercept
        t_rejections += (pvals < ALPHA).astype(np.int64)
        try:
            if float(fit.wald_test(R, scalar=True).pvalue) < ALPHA:
                f_rejections += 1
        except Exception:
            pass
    denom = max(converged, 1)
    return t_rejections / denom, f_rejections / denom, converged


# ---------------------------------------------------------------------------
# LME
# ---------------------------------------------------------------------------

def _assign_clusters(n: int, k: int) -> np.ndarray:
    """Spread n observations across k clusters as evenly as possible."""
    base, extra = divmod(n, k)
    sizes = np.full(k, base, dtype=np.int64)
    sizes[:extra] += 1
    return np.repeat(np.arange(k, dtype=np.int64), sizes)


def _lme_best_chunk(case, n, n_sims, seed):
    rng = np.random.default_rng(seed)
    spec = _design(case); BETA = spec["beta"]; TARGETS = spec["targets"]; P = BETA.size
    N_CLUSTERS = case.cluster["n_clusters"]
    ICC = case.cluster["ICC"]
    TAU = float(np.sqrt(ICC / (1.0 - ICC)))
    z_crit = stats.norm.ppf(1.0 - ALPHA / 2.0)
    chi2_crit = stats.chi2.ppf(1.0 - ALPHA, TARGETS.size)
    t_rejections = np.zeros(P, dtype=np.int64)
    f_rejections = 0
    converged = 0
    cluster_ids = _assign_clusters(n, N_CLUSTERS)
    R = np.zeros((TARGETS.size, P + 1))
    for row, j in enumerate(TARGETS):
        R[row, j + 1] = 1.0
    for _ in range(n_sims):
        X = _draw_X(rng, n, spec)
        b_cluster = rng.standard_normal(N_CLUSTERS) * TAU
        eps = rng.standard_normal(n)
        y = X @ BETA + b_cluster[cluster_ids] + eps
        Xd = sm.add_constant(X)
        try:
            fit = sm.MixedLM(y, Xd, groups=cluster_ids).fit(
                method="lbfgs", reml=True, disp=False
            )
        except Exception:
            continue
        if not getattr(fit, "converged", True):
            continue
        converged += 1
        coef = np.asarray(fit.fe_params)
        cov = np.asarray(fit.cov_params())[: P + 1, : P + 1]
        se = np.sqrt(np.diag(cov))
        z_stat = coef[1:] / se[1:]
        t_rejections += (np.abs(z_stat) > z_crit).astype(np.int64)
        idx = TARGETS + 1
        b = coef[idx]
        V = cov[np.ix_(idx, idx)]
        try:
            w_stat = float(b @ np.linalg.solve(V, b))
        except np.linalg.LinAlgError:
            continue
        if w_stat > chi2_crit:
            f_rejections += 1
    # Return raw counts for parallel aggregation
    return t_rejections, f_rejections, converged


def lme_best(case, n, n_sims, seed):
    return _parallel(_lme_best_chunk, case, n, n_sims, seed)


def lme_naive(case, n, n_sims, seed):
    rng = np.random.default_rng(seed)
    spec = _design(case); BETA = spec["beta"]; TARGETS = spec["targets"]; P = BETA.size
    N_CLUSTERS = case.cluster["n_clusters"]
    ICC = case.cluster["ICC"]
    TAU = float(np.sqrt(ICC / (1.0 - ICC)))
    t_rejections = np.zeros(P, dtype=np.int64)
    converged = 0
    cluster_ids = _assign_clusters(n, N_CLUSTERS)
    for _ in range(n_sims):
        X = _draw_X(rng, n, spec)
        b_cluster = rng.standard_normal(N_CLUSTERS) * TAU
        eps = rng.standard_normal(n)
        y = X @ BETA + b_cluster[cluster_ids] + eps
        Xd = sm.add_constant(X)
        try:
            fit = sm.MixedLM(y, Xd, groups=cluster_ids).fit(disp=False)
        except Exception:
            continue
        if not getattr(fit, "converged", True):
            continue
        converged += 1
        pvals = np.asarray(fit.pvalues)[1 : P + 1]  # drop intercept, keep fixed effects
        t_rejections += (pvals < ALPHA).astype(np.int64)
    # No joint test here — the R naive kernel doesn't compute one, the value is
    # never recorded, and timing dead work would inflate the Python baseline.
    denom = max(converged, 1)
    return t_rejections / denom, 0.0, converged


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LOOPS = {
    "ols":   {"best": ols_best,   "naive": ols_naive},
    "logit": {"best": glm_best,   "naive": glm_naive},
    "lme":   {"best": lme_best,   "naive": lme_naive},
}
