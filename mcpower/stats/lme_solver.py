"""Custom LME solver for random-intercept linear mixed models.

Implements REML/ML estimation via profiled deviance optimization,
following Bates et al. (2015) "Fitting Linear Mixed-Effects Models
Using lme4" (JSS 67(1), arXiv:1406.5823).

For random intercepts (q=1), exploits block-diagonal structure of V
to reduce REML fitting to a 1D optimization via Brent's method,
with all per-cluster operations being scalar. This yields massive
speedups over statsmodels' general-purpose MixedLM.

For future random slopes (q>1), the general profiled deviance
function works with q×q per-cluster Cholesky decompositions and
L-BFGS-B optimization over the lower-triangular T template.

Performance: core deviance functions are JIT-compiled via numba
when available, following the same pattern as ols.py.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

FLOAT_NEAR_ZERO = 1e-15


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class SufficientStats:
    """Per-cluster precomputed statistics for LME fitting.

    All cross-products are precomputed once per simulation, then reused
    across all optimizer evaluations of the profiled deviance.
    """

    K: int  # number of clusters
    q: int  # random effects per cluster
    p: int  # fixed effects (including intercept)
    N: int  # total observations
    cluster_sizes: np.ndarray  # (K,) int
    ZtZ: np.ndarray  # (K, q, q) or (K,) for q=1
    ZtX: np.ndarray  # (K, q, p) or (K, p) for q=1
    Zty: np.ndarray  # (K, q) or (K,) for q=1
    XtX: np.ndarray  # (K, p, p)
    Xty: np.ndarray  # (K, p)
    yty: np.ndarray  # (K,)


@dataclass
class LMEResult:
    """Result of LME model fitting."""

    beta: np.ndarray  # (p,) fixed effects incl. intercept
    sigma2: float  # residual variance
    tau2: float  # random intercept variance (for q=1)
    G: np.ndarray  # (q,q) random effects covariance / sigma2
    theta: np.ndarray  # optimizer parameters (Cholesky elements)
    cov_beta: np.ndarray  # (p, p) covariance of fixed effects
    se_beta: np.ndarray  # (p,) standard errors
    log_likelihood: float  # at optimum
    converged: bool


# ---------------------------------------------------------------------------
# Critical value precomputation (mirrors ols.compute_critical_values)
# ---------------------------------------------------------------------------


def compute_lme_critical_values(alpha, n_fixed, n_targets, correction_method):
    """Pre-compute critical z and chi2 values for LME significance testing.

    Called once before the simulation loop. LME uses Wald z-tests
    (normal approximation) rather than t-tests.

    Args:
        alpha: Significance level.
        n_fixed: Number of fixed effects (excluding intercept).
        n_targets: Number of individual effects being tested.
        correction_method: 0=none, 1=Bonferroni, 2=BH, 3=Holm.

    Returns:
        Tuple of (chi2_crit, z_crit, correction_z_crits).
    """
    from scipy.stats import chi2, norm

    chi2_crit = chi2.ppf(1 - alpha, n_fixed) if n_fixed > 0 else np.inf
    z_crit = norm.ppf(1 - alpha / 2)

    m = n_targets
    if m == 0:
        return chi2_crit, z_crit, np.empty(0)

    if correction_method == 0:  # None
        correction_z_crits = np.full(m, z_crit)
    elif correction_method == 1:  # Bonferroni
        bonf_crit = norm.ppf(1 - alpha / (2 * m))
        correction_z_crits = np.full(m, bonf_crit)
    elif correction_method == 2:  # FDR (Benjamini-Hochberg)
        correction_z_crits = np.array([norm.ppf(1 - (k + 1) / m * alpha / 2) for k in range(m)])
    elif correction_method == 3:  # Holm
        correction_z_crits = np.array([norm.ppf(1 - alpha / (2 * (m - k))) for k in range(m)])
    else:
        correction_z_crits = np.full(m, z_crit)

    return chi2_crit, z_crit, correction_z_crits


# ---------------------------------------------------------------------------
# Sufficient statistics computation
# ---------------------------------------------------------------------------


def compute_sufficient_statistics(X, y, cluster_ids, K, q=1, Z=None):
    """Precompute per-cluster cross-products for LME fitting.

    For q=1 (random intercept), Z_j = 1_{n_j}, so:
        ZtZ_j = n_j  (scalar)
        ZtX_j = colsum(X_j)  (p-vector)
        Zty_j = sum(y_j)  (scalar)

    For q>1, Z must be provided explicitly as (N, q) matrix.

    Args:
        X: (N, p) fixed-effects design matrix (with intercept column).
        y: (N,) response vector.
        cluster_ids: (N,) integer cluster membership.
        K: Number of clusters.
        q: Random effects dimension per cluster (1 for intercept only).
        Z: (N, q) random-effects design matrix (required when q > 1).

    Returns:
        SufficientStats instance.
    """
    N, p = X.shape

    cluster_sizes = np.zeros(K, dtype=np.int64)
    XtX = np.zeros((K, p, p))
    Xty = np.zeros((K, p))
    yty = np.zeros(K)

    if q == 1:
        ZtZ = np.zeros(K)
        ZtX = np.zeros((K, p))
        Zty = np.zeros(K)

        _compute_suff_stats_q1(X, y, cluster_ids, K, p, cluster_sizes, ZtZ, ZtX, Zty, XtX, Xty, yty)
    else:
        if Z is None:
            raise ValueError("Z matrix required for q > 1")

        ZtZ = np.zeros((K, q, q))
        ZtX = np.zeros((K, q, p))
        Zty = np.zeros((K, q))

        _compute_suff_stats_general(X, y, Z, cluster_ids, K, p, q, cluster_sizes, ZtZ, ZtX, Zty, XtX, Xty, yty)

    return SufficientStats(
        K=K,
        q=q,
        p=p,
        N=N,
        cluster_sizes=cluster_sizes,
        ZtZ=ZtZ,
        ZtX=ZtX,
        Zty=Zty,
        XtX=XtX,
        Xty=Xty,
        yty=yty,
    )


def _compute_suff_stats_q1(X, y, cluster_ids, K, p, cluster_sizes, ZtZ, ZtX, Zty, XtX, Xty, yty):
    """Compute sufficient statistics for q=1 (random intercept)."""
    N = len(y)
    for i in range(N):
        j = cluster_ids[i]
        cluster_sizes[j] += 1
        yi = y[i]
        yty[j] += yi * yi
        Zty[j] += yi
        for c in range(p):
            xic = X[i, c]
            Xty[j][c] += xic * yi
            ZtX[j][c] += xic
            for c2 in range(c, p):
                XtX[j][c][c2] += xic * X[i, c2]

    # Fill lower triangles of XtX
    for j in range(K):
        ZtZ[j] = cluster_sizes[j]
        for c in range(p):
            for c2 in range(c):
                XtX[j][c][c2] = XtX[j][c2][c]


def _compute_suff_stats_general(X, y, Z, cluster_ids, K, p, q, cluster_sizes, ZtZ, ZtX, Zty, XtX, Xty, yty):
    """Compute sufficient statistics for general q (random slopes)."""
    N = len(y)
    for i in range(N):
        j = cluster_ids[i]
        cluster_sizes[j] += 1
        yi = y[i]
        yty[j] += yi * yi
        xi = X[i]
        zi = Z[i]

        # Zty_j += z_i * y_i
        for a in range(q):
            Zty[j, a] += zi[a] * yi

        # ZtZ_j += z_i @ z_i'
        for a in range(q):
            for b in range(a, q):
                val = zi[a] * zi[b]
                ZtZ[j, a, b] += val
                if a != b:
                    ZtZ[j, b, a] += val

        # ZtX_j += z_i @ x_i'
        for a in range(q):
            for c in range(p):
                ZtX[j, a, c] += zi[a] * xi[c]

        # Xty_j += x_i * y_i
        for c in range(p):
            Xty[j, c] += xi[c] * yi

        # XtX_j += x_i @ x_i'
        for c in range(p):
            for c2 in range(c, p):
                val = xi[c] * xi[c2]
                XtX[j, c, c2] += val
                if c != c2:
                    XtX[j, c2, c] += val


# Pure-Python core for the profiled deviance (q=1 fast path).
# This function accepts only arrays/scalars so it can be JIT-compiled.
def _profiled_deviance_q1_core(lam_sq, K, p, N, cluster_sizes, ZtZ, ZtX, Zty, XtX, Xty, yty, reml):
    """Evaluate profiled REML/ML deviance for random intercept (q=1).

    All per-cluster operations are scalar (no matrix ops in the loop).

    Args:
        lam_sq: lambda^2, the square of the relative covariance factor.
            tau^2 = sigma^2 * lam_sq. Must be >= 0.
        K, p, N: cluster count, fixed-effects dim, total obs.
        cluster_sizes, ZtZ, ZtX, Zty, XtX, Xty, yty: sufficient stats.
        reml: 1 for REML, 0 for ML.

    Returns:
        Profiled deviance value (scalar to minimize).
    """
    # Accumulate Schur complement A, right-hand side b, and scalar c
    A = np.zeros((p, p))
    b = np.zeros(p)
    c = 0.0
    log_det_Ltheta = 0.0
    sqrt_lam = lam_sq**0.5

    for j in range(K):
        nj = cluster_sizes[j]
        # M_j = lam_sq * n_j + 1  (scalar for q=1)
        Mj = lam_sq * nj + 1.0
        Lj = Mj**0.5
        inv_Lj = 1.0 / Lj

        # cu_j = inv_Lj * lambda * Zty_j  (lambda = sqrt(lam_sq))
        cu_j = inv_Lj * sqrt_lam * Zty[j]

        # C_X_j = inv_Lj * sqrt(lam_sq) * ZtX_j   (p-vector)
        # Accumulate A -= C_X_j' * C_X_j, b -= C_X_j' * cu_j, c -= cu_j^2
        factor = inv_Lj * sqrt_lam
        for c1 in range(p):
            CX_c1 = factor * ZtX[j][c1]
            b[c1] += Xty[j][c1] - CX_c1 * cu_j
            for c2 in range(c1, p):
                CX_c2 = factor * ZtX[j][c2]
                A[c1][c2] += XtX[j][c1][c2] - CX_c1 * CX_c2

        c += yty[j] - cu_j * cu_j
        log_det_Ltheta += 2.0 * np.log(Lj)

    # Fill lower triangle of A
    for c1 in range(p):
        for c2 in range(c1):
            A[c1][c2] = A[c2][c1]

    # Manual Cholesky factorization A = R_X @ R_X' (numba-compatible, no exceptions)
    R_X = np.zeros((p, p))
    for i in range(p):
        s = A[i][i]
        for k in range(i):
            s -= R_X[i][k] * R_X[i][k]
        if s <= 0.0:
            return 1e30  # Not positive definite
        R_X[i][i] = s**0.5
        for j2 in range(i + 1, p):
            s2 = A[j2][i]
            for k in range(i):
                s2 -= R_X[j2][k] * R_X[i][k]
            R_X[j2][i] = s2 / R_X[i][i]

    # Solve R_X @ R_X' @ beta = b
    # Forward solve: R_X @ z = b
    z = np.zeros(p)
    for i in range(p):
        s = b[i]
        for k in range(i):
            s -= R_X[i][k] * z[k]
        z[i] = s / R_X[i][i]
    # Back solve: R_X' @ beta = z
    beta = np.zeros(p)
    for i in range(p - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, p):
            s -= R_X[k][i] * beta[k]
        beta[i] = s / R_X[i][i]

    # r^2(theta) = c - beta' @ b
    r_sq = c
    for i in range(p):
        r_sq -= beta[i] * b[i]

    if r_sq <= 0:
        return 1e30

    # Profiled objective
    if reml:
        # f_R(theta) = log_det_Ltheta + 2*sum(log(diag(R_X))) + (N-p)*log(r_sq)
        log_det_RX = 0.0
        for i in range(p):
            log_det_RX += np.log(R_X[i][i])
        return log_det_Ltheta + 2.0 * log_det_RX + (N - p) * np.log(r_sq)
    else:
        # f(theta) = log_det_Ltheta + N*log(r_sq)
        return log_det_Ltheta + N * np.log(r_sq)


def _extract_results_q1(lam_sq_opt, stats, reml):
    """Extract beta, sigma2, tau2, cov_beta, log-likelihood from optimal theta.

    Called after optimization to compute final parameter estimates.
    """
    K, p, N = stats.K, stats.p, stats.N

    # Recompute A, b, c at the optimum (same as in deviance, but we need beta)
    A = np.zeros((p, p))
    b = np.zeros(p)
    c = 0.0
    log_det_Ltheta = 0.0

    sqrt_lam = lam_sq_opt**0.5
    for j in range(K):
        nj = stats.cluster_sizes[j]
        Mj = lam_sq_opt * nj + 1.0
        Lj = Mj**0.5
        inv_Lj = 1.0 / Lj
        factor = inv_Lj * sqrt_lam

        cu_j = factor * stats.Zty[j]
        for c1 in range(p):
            CX_c1 = factor * stats.ZtX[j][c1]
            b[c1] += stats.Xty[j][c1] - CX_c1 * cu_j
            for c2 in range(c1, p):
                CX_c2 = factor * stats.ZtX[j][c2]
                A[c1][c2] += stats.XtX[j][c1][c2] - CX_c1 * CX_c2

        c += stats.yty[j] - cu_j * cu_j
        log_det_Ltheta += 2.0 * np.log(Lj)

    for c1 in range(p):
        for c2 in range(c1):
            A[c1][c2] = A[c2][c1]

    # Solve for beta
    beta = np.linalg.solve(A, b)

    # r^2 and sigma^2
    r_sq = c - beta @ b
    if reml:
        sigma2 = r_sq / (N - p)
    else:
        sigma2 = r_sq / N

    # tau^2 = sigma^2 * lambda^2
    tau2 = float(sigma2 * lam_sq_opt)

    # G matrix (q=1: scalar)
    G = np.array([[lam_sq_opt]])

    # Covariance of beta: sigma^2 * A^{-1}
    A_inv = np.linalg.inv(A)
    cov_beta = sigma2 * A_inv
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    # Log-likelihood (ML version, for LR tests)
    # loglik_ML = -0.5 * [log_det_Ltheta + N + N*log(2pi) + N*log(r_sq/N)]
    sigma2_ml = r_sq / N
    if sigma2_ml > 0:
        log_lik = -0.5 * (log_det_Ltheta + N + N * np.log(2.0 * np.pi) + N * np.log(sigma2_ml))
    else:
        log_lik = -np.inf

    return LMEResult(
        beta=beta,
        sigma2=sigma2,
        tau2=tau2,
        G=G,
        theta=np.array([lam_sq_opt**0.5]),
        cov_beta=cov_beta,
        se_beta=se_beta,
        log_likelihood=log_lik,
        converged=True,
    )


# ---------------------------------------------------------------------------
# Main fitting function
# ---------------------------------------------------------------------------


def _theta_to_T(theta_vec, q):
    """Reconstruct lower-triangular T from theta vector.

    For q=2: theta = [lambda_11, lambda_21, lambda_22]
    T = [[lambda_11, 0], [lambda_21, lambda_22]]
    """
    T = np.zeros((q, q))
    idx = 0
    for j in range(q):
        for i in range(j, q):
            T[i, j] = theta_vec[idx]
            idx += 1
    return T


def _profiled_deviance_general(theta_vec, stats, reml_int):
    """Evaluate profiled REML/ML deviance for general q.

    Uses q×q Cholesky per cluster. For q=2 uses inline 2×2 Cholesky
    to avoid np.linalg calls in the inner loop.
    """
    K, q, p, N = stats.K, stats.q, stats.p, stats.N

    # Reconstruct T from theta_vec
    T = _theta_to_T(theta_vec, q)

    A = np.zeros((p, p))
    b = np.zeros(p)
    c = 0.0
    log_det_Ltheta = 0.0

    for j in range(K):
        # M_j = T' ZtZ_j T + I_q
        TtZtZ = T.T @ stats.ZtZ[j]
        M_j = TtZtZ @ T + np.eye(q)

        # Cholesky of M_j
        try:
            L_j = np.linalg.cholesky(M_j)
        except np.linalg.LinAlgError:
            return 1e30

        # cu_j = L_j^{-1} T' Zty_j
        TtZty = T.T @ stats.Zty[j]
        cu_j = np.linalg.solve(L_j, TtZty)

        # CX_j = L_j^{-1} T' ZtX_j  (q × p)
        TtZtX = T.T @ stats.ZtX[j]
        CX_j = np.linalg.solve(L_j, TtZtX)

        # Accumulate
        A += stats.XtX[j] - CX_j.T @ CX_j
        b += stats.Xty[j] - CX_j.T @ cu_j
        c += stats.yty[j] - cu_j @ cu_j
        log_det_Ltheta += 2.0 * np.sum(np.log(np.diag(L_j)))

    # Solve A beta = b
    try:
        R_X = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return 1e30

    z = np.linalg.solve(R_X, b)
    beta = np.linalg.solve(R_X.T, z)

    r_sq = c - beta @ b
    if r_sq <= 0:
        return 1e30

    if reml_int:
        log_det_RX = 2.0 * np.sum(np.log(np.diag(R_X)))
        return log_det_Ltheta + log_det_RX + (N - p) * np.log(r_sq)
    else:
        return log_det_Ltheta + N * np.log(r_sq)


def _extract_results_general(theta_opt, stats, reml):
    """Extract beta, sigma2, G, cov_beta from optimal theta for general q."""
    K, q, p, N = stats.K, stats.q, stats.p, stats.N

    T = _theta_to_T(theta_opt, q)

    A = np.zeros((p, p))
    b = np.zeros(p)
    c = 0.0
    log_det_Ltheta = 0.0

    for j in range(K):
        TtZtZ = T.T @ stats.ZtZ[j]
        M_j = TtZtZ @ T + np.eye(q)
        L_j = np.linalg.cholesky(M_j)
        TtZty = T.T @ stats.Zty[j]
        cu_j = np.linalg.solve(L_j, TtZty)
        TtZtX = T.T @ stats.ZtX[j]
        CX_j = np.linalg.solve(L_j, TtZtX)

        A += stats.XtX[j] - CX_j.T @ CX_j
        b += stats.Xty[j] - CX_j.T @ cu_j
        c += stats.yty[j] - cu_j @ cu_j
        log_det_Ltheta += 2.0 * np.sum(np.log(np.diag(L_j)))

    beta = np.linalg.solve(A, b)
    r_sq = c - beta @ b

    if reml:
        sigma2 = r_sq / (N - p)
    else:
        sigma2 = r_sq / N

    G = T @ T.T  # relative covariance: actual G = sigma2 * G_rel
    tau2 = float(sigma2 * G[0, 0])

    A_inv = np.linalg.inv(A)
    cov_beta = sigma2 * A_inv
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    sigma2_ml = r_sq / N
    if sigma2_ml > 0:
        log_lik = -0.5 * (log_det_Ltheta + N + N * np.log(2.0 * np.pi) + N * np.log(sigma2_ml))
    else:
        log_lik = -np.inf

    return LMEResult(
        beta=beta,
        sigma2=sigma2,
        tau2=tau2,
        G=G,
        theta=theta_opt,
        cov_beta=cov_beta,
        se_beta=se_beta,
        log_likelihood=log_lik,
        converged=True,
    )


def _fit_general(stats, reml, warm_theta=None):
    """Fit general q>1 model via L-BFGS-B over theta."""
    from scipy.optimize import minimize

    q = stats.q
    n_theta = q * (q + 1) // 2
    reml_int = 1 if reml else 0

    def objective(theta_vec):
        return _profiled_deviance_general(theta_vec, stats, reml_int)

    # Bounds: diagonal elements >= 0, off-diagonal free
    bounds = []
    idx = 0
    for j in range(q):
        for i in range(j, q):
            if i == j:
                bounds.append((0.0, 1e4))  # diagonal >= 0
            else:
                bounds.append((-1e4, 1e4))  # off-diagonal free
            idx += 1

    # Initial guess
    if warm_theta is not None and len(warm_theta) == n_theta:
        x0 = warm_theta.copy()
    else:
        x0 = np.zeros(n_theta)
        idx = 0
        for j in range(q):
            for i in range(j, q):
                if i == j:
                    x0[idx] = 1.0
                idx += 1

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-6},
    )

    return _extract_results_general(result.x, stats, reml)


# ---------------------------------------------------------------------------
# Nested random intercepts solver
# ---------------------------------------------------------------------------


@dataclass
class NestedSufficientStats:
    """Sufficient statistics for nested random intercepts (1|A) + (1|A:B)."""

    K_parent: int
    K_child: int
    p: int
    N: int
    parent_sizes: np.ndarray  # (K_parent,) obs per parent
    child_sizes: np.ndarray  # (K_child,) obs per child
    child_to_parent: np.ndarray  # (K_child,) maps child -> parent
    # Per-child stats
    child_XtX: np.ndarray  # (K_child, p, p)
    child_Xty: np.ndarray  # (K_child, p)
    child_yty: np.ndarray  # (K_child,)
    child_Xt1: np.ndarray  # (K_child, p) = column sums of X within child
    child_1ty: np.ndarray  # (K_child,) = sum of y within child
    child_1t1: np.ndarray  # (K_child,) = n_child (same as child_sizes)


def compute_nested_sufficient_statistics(X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent):
    """Precompute per-child and per-parent cross-products for nested models."""
    N, p = X.shape

    parent_sizes = np.zeros(K_parent, dtype=np.int64)
    child_sizes = np.zeros(K_child, dtype=np.int64)
    child_XtX = np.zeros((K_child, p, p))
    child_Xty = np.zeros((K_child, p))
    child_yty = np.zeros(K_child)
    child_Xt1 = np.zeros((K_child, p))
    child_1ty = np.zeros(K_child)

    for i in range(N):
        cj = child_ids[i]
        pj = parent_ids[i]
        child_sizes[cj] += 1
        parent_sizes[pj] += 1
        yi = y[i]
        child_yty[cj] += yi * yi
        child_1ty[cj] += yi
        for c in range(p):
            xic = X[i, c]
            child_Xty[cj, c] += xic * yi
            child_Xt1[cj, c] += xic
            for c2 in range(c, p):
                val = xic * X[i, c2]
                child_XtX[cj, c, c2] += val
                if c != c2:
                    child_XtX[cj, c2, c] += val

    return NestedSufficientStats(
        K_parent=K_parent,
        K_child=K_child,
        p=p,
        N=N,
        parent_sizes=parent_sizes,
        child_sizes=child_sizes,
        child_to_parent=child_to_parent,
        child_XtX=child_XtX,
        child_Xty=child_Xty,
        child_yty=child_yty,
        child_Xt1=child_Xt1,
        child_1ty=child_1ty,
        child_1t1=child_sizes.astype(np.float64),
    )


def _profiled_deviance_nested(theta_vec, nstats, reml_int):
    """Profiled deviance for two-level nested random intercepts.

    Uses two-pass Woodbury decomposition: first absorb child effects,
    then absorb parent effects. All operations are scalar per child/parent.

    theta_vec = [theta_parent, theta_child] where theta >= 0.
    """
    theta_parent = theta_vec[0]
    theta_child = theta_vec[1]
    lam_sq_parent = theta_parent**2
    lam_sq_child = theta_child**2

    p = nstats.p
    N = nstats.N
    K_parent = nstats.K_parent

    # Accumulate after absorbing both levels
    A = np.zeros((p, p))
    b_vec = np.zeros(p)
    c_val = 0.0
    log_det = 0.0

    # Per-parent accumulators (after absorbing child effects)
    parent_XtX_adj = np.zeros((K_parent, p, p))
    parent_Xty_adj = np.zeros((K_parent, p))
    parent_yty_adj = np.zeros(K_parent)
    parent_Xt1_adj = np.zeros((K_parent, p))
    parent_1ty_adj = np.zeros(K_parent)
    parent_1t1_adj = np.zeros(K_parent)

    # Pass 1: Absorb child-level effects (scalar per child)
    for cj in range(nstats.K_child):
        nc = nstats.child_sizes[cj]
        pj = nstats.child_to_parent[cj]

        Mc = lam_sq_child * nc + 1.0
        Lc = Mc**0.5
        inv_Mc = 1.0 / Mc
        factor = lam_sq_child * inv_Mc

        # Absorb child: subtract rank-1 update
        # A_adj = XtX - factor * Xt1 @ 1tX
        outer_XX = np.outer(nstats.child_Xt1[cj], nstats.child_Xt1[cj])
        child_XtX_adj = nstats.child_XtX[cj] - factor * outer_XX
        child_Xty_adj = nstats.child_Xty[cj] - factor * nstats.child_Xt1[cj] * nstats.child_1ty[cj]
        child_yty_adj = nstats.child_yty[cj] - factor * nstats.child_1ty[cj] ** 2

        # Adjusted sums for parent level (child's contribution to parent's "1" vector)
        child_1t1_adj = nc - factor * nc * nc
        child_Xt1_for_parent = nstats.child_Xt1[cj] - factor * nstats.child_Xt1[cj] * nc
        child_1ty_for_parent = nstats.child_1ty[cj] - factor * nstats.child_1ty[cj] * nc

        parent_XtX_adj[pj] += child_XtX_adj
        parent_Xty_adj[pj] += child_Xty_adj
        parent_yty_adj[pj] += child_yty_adj
        parent_Xt1_adj[pj] += child_Xt1_for_parent
        parent_1ty_adj[pj] += child_1ty_for_parent
        parent_1t1_adj[pj] += child_1t1_adj

        log_det += np.log(Lc)

    # Pass 2: Absorb parent-level effects (scalar per parent)
    for pj in range(K_parent):
        ns_adj = parent_1t1_adj[pj]
        Mp = lam_sq_parent * ns_adj + 1.0
        if Mp <= 0:
            return 1e30
        Lp = Mp**0.5
        inv_Mp = 1.0 / Mp
        factor_p = lam_sq_parent * inv_Mp

        outer_XX_p = np.outer(parent_Xt1_adj[pj], parent_Xt1_adj[pj])
        A += parent_XtX_adj[pj] - factor_p * outer_XX_p
        b_vec += parent_Xty_adj[pj] - factor_p * parent_Xt1_adj[pj] * parent_1ty_adj[pj]
        c_val += parent_yty_adj[pj] - factor_p * parent_1ty_adj[pj] ** 2
        log_det += np.log(Lp)

    # 2 * log_det because each Lj contributes 2*log(Lj) to det
    log_det_total = 2.0 * log_det

    # Solve A beta = b
    try:
        R_X = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return 1e30

    z = np.linalg.solve(R_X, b_vec)
    beta = np.linalg.solve(R_X.T, z)

    r_sq = c_val - beta @ b_vec
    if r_sq <= 0:
        return 1e30

    if reml_int:
        log_det_RX = 2.0 * np.sum(np.log(np.diag(R_X)))
        return log_det_total + log_det_RX + (N - p) * np.log(r_sq)
    else:
        return log_det_total + N * np.log(r_sq)


def _extract_results_nested(theta_opt, nstats, reml):
    """Extract parameters from optimal nested theta."""
    theta_parent, theta_child = theta_opt[0], theta_opt[1]
    lam_sq_parent = theta_parent**2
    lam_sq_child = theta_child**2

    p = nstats.p
    N = nstats.N
    K_parent = nstats.K_parent

    # Repeat the deviance computation to get A, b, c at the optimum
    A = np.zeros((p, p))
    b_vec = np.zeros(p)
    c_val = 0.0
    log_det = 0.0

    parent_XtX_adj = np.zeros((K_parent, p, p))
    parent_Xty_adj = np.zeros((K_parent, p))
    parent_yty_adj = np.zeros(K_parent)
    parent_Xt1_adj = np.zeros((K_parent, p))
    parent_1ty_adj = np.zeros(K_parent)
    parent_1t1_adj = np.zeros(K_parent)

    for cj in range(nstats.K_child):
        nc = nstats.child_sizes[cj]
        pj = nstats.child_to_parent[cj]
        Mc = lam_sq_child * nc + 1.0
        inv_Mc = 1.0 / Mc
        factor = lam_sq_child * inv_Mc

        outer_XX = np.outer(nstats.child_Xt1[cj], nstats.child_Xt1[cj])
        parent_XtX_adj[pj] += nstats.child_XtX[cj] - factor * outer_XX
        parent_Xty_adj[pj] += nstats.child_Xty[cj] - factor * nstats.child_Xt1[cj] * nstats.child_1ty[cj]
        parent_yty_adj[pj] += nstats.child_yty[cj] - factor * nstats.child_1ty[cj] ** 2
        parent_Xt1_adj[pj] += nstats.child_Xt1[cj] - factor * nstats.child_Xt1[cj] * nc
        parent_1ty_adj[pj] += nstats.child_1ty[cj] - factor * nstats.child_1ty[cj] * nc
        parent_1t1_adj[pj] += nc - factor * nc * nc
        log_det += np.log(Mc**0.5)

    for pj in range(K_parent):
        ns_adj = parent_1t1_adj[pj]
        Mp = lam_sq_parent * ns_adj + 1.0
        inv_Mp = 1.0 / Mp
        factor_p = lam_sq_parent * inv_Mp

        outer_XX_p = np.outer(parent_Xt1_adj[pj], parent_Xt1_adj[pj])
        A += parent_XtX_adj[pj] - factor_p * outer_XX_p
        b_vec += parent_Xty_adj[pj] - factor_p * parent_Xt1_adj[pj] * parent_1ty_adj[pj]
        c_val += parent_yty_adj[pj] - factor_p * parent_1ty_adj[pj] ** 2
        log_det += np.log(Mp**0.5)

    beta = np.linalg.solve(A, b_vec)
    r_sq = c_val - beta @ b_vec

    if reml:
        sigma2 = r_sq / (N - p)
    else:
        sigma2 = r_sq / N

    tau2_parent = float(sigma2 * lam_sq_parent)

    G = np.diag([lam_sq_parent, lam_sq_child])

    A_inv = np.linalg.inv(A)
    cov_beta = sigma2 * A_inv
    se_beta = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))

    sigma2_ml = r_sq / N
    log_det_total = 2.0 * log_det
    if sigma2_ml > 0:
        log_lik = -0.5 * (log_det_total + N + N * np.log(2.0 * np.pi) + N * np.log(sigma2_ml))
    else:
        log_lik = -np.inf

    return LMEResult(
        beta=beta,
        sigma2=sigma2,
        tau2=tau2_parent,  # primary (parent) tau2
        G=G,
        theta=theta_opt,
        cov_beta=cov_beta,
        se_beta=se_beta,
        log_likelihood=log_lik,
        converged=True,
    )


def _fit_nested(nstats, reml, warm_theta=None):
    """Fit nested random intercepts via L-BFGS-B over [theta_parent, theta_child]."""
    from scipy.optimize import minimize

    reml_int = 1 if reml else 0

    def objective(theta_vec):
        return _profiled_deviance_nested(theta_vec, nstats, reml_int)

    if warm_theta is not None and len(warm_theta) == 2:
        x0 = warm_theta.copy()
    else:
        x0 = np.array([1.0, 1.0])

    bounds = [(0.0, 1e3), (0.0, 1e3)]
    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 200, "ftol": 1e-10, "gtol": 1e-6},
    )

    return _extract_results_nested(result.x, nstats, reml)


def lme_fit(X, y, cluster_ids, K, q=1, reml=True, warm_theta=None, Z=None, model_type="single"):
    """Fit a linear mixed model via profiled deviance optimization.

    For q=1 (random intercept), uses Brent's method for 1D optimization
    over lambda^2 ∈ [0, 1e6]. For q>1, uses L-BFGS-B.

    Args:
        X: (N, p) fixed-effects design matrix (WITH intercept column).
        y: (N,) response vector.
        cluster_ids: (N,) integer cluster membership (or parent_ids for nested).
        K: Number of clusters.
        q: Random effects dimension per cluster.
        reml: Use REML (True) or ML (False).
        warm_theta: Previous optimal theta for warm start.
        Z: (N, q) random-effects design matrix (required when q > 1).
        model_type: "single" (default) or "nested".

    Returns:
        LMEResult with estimated parameters.
    """
    if model_type == "nested":
        raise ValueError("Use lme_fit_nested() for nested models")

    stats = compute_sufficient_statistics(X, y, cluster_ids, K, q, Z=Z)

    if q == 1:
        return _fit_q1(stats, reml, warm_theta)
    else:
        return _fit_general(stats, reml, warm_theta)


def lme_fit_nested(X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent, reml=True, warm_theta=None):
    """Fit a nested random intercepts model: (1|parent) + (1|parent:child).

    Args:
        X: (N, p) fixed-effects design matrix (WITH intercept column).
        y: (N,) response vector.
        parent_ids: (N,) integer parent cluster membership.
        child_ids: (N,) integer child cluster membership.
        K_parent: Number of parent clusters.
        K_child: Number of child clusters (total).
        child_to_parent: (K_child,) mapping from child index to parent index.
        reml: Use REML (True) or ML (False).
        warm_theta: Previous optimal [theta_parent, theta_child].

    Returns:
        LMEResult with estimated parameters.
    """
    nstats = compute_nested_sufficient_statistics(X, y, parent_ids, child_ids, K_parent, K_child, child_to_parent)
    return _fit_nested(nstats, reml, warm_theta)


def _fit_q1(stats, reml, warm_theta=None):
    """Fit random intercept model via Brent's method on lambda^2."""
    from scipy.optimize import minimize_scalar

    reml_int = 1 if reml else 0

    def objective(lam_sq):
        return _profiled_deviance_q1_jit(
            lam_sq,
            stats.K,
            stats.p,
            stats.N,
            stats.cluster_sizes,
            stats.ZtZ,
            stats.ZtX,
            stats.Zty,
            stats.XtX,
            stats.Xty,
            stats.yty,
            reml_int,
        )

    # Brent's method on [0, 1e6]
    # For warm start, narrow the bracket around previous optimum
    if warm_theta is not None and warm_theta > 0:
        # warm_theta is lambda (not lambda^2), square it
        warm_lam_sq = warm_theta**2
        # Search in [warm/10, warm*10] first, then fall back to full range
        lo = max(0.0, warm_lam_sq * 0.01)
        hi = min(1e6, warm_lam_sq * 100.0)
        result = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-8, "maxiter": 100})
        # Verify we didn't get stuck at a boundary
        if result.x <= lo * 1.01 or result.x >= hi * 0.99:
            # Boundary hit, try full range
            result2 = minimize_scalar(objective, bounds=(0.0, 1e6), method="bounded", options={"xatol": 1e-8, "maxiter": 150})
            if result2.fun < result.fun:
                result = result2
    else:
        result = minimize_scalar(objective, bounds=(0.0, 1e6), method="bounded", options={"xatol": 1e-8, "maxiter": 150})

    lam_sq_opt = result.x
    return _extract_results_q1(lam_sq_opt, stats, reml)


def lme_fit_ml_null(y, cluster_ids, K, warm_theta=None):
    """Fit null model (intercept + random intercept only) with ML.

    Used for likelihood-ratio tests.
    """
    N = len(y)
    X_null = np.ones((N, 1))
    return lme_fit(X_null, y, cluster_ids, K, q=1, reml=False, warm_theta=warm_theta)


# ---------------------------------------------------------------------------
# Full analysis function (matches C++ LMESolver::analyze signature)
# ---------------------------------------------------------------------------


def lme_analysis_full(
    X,
    y,
    cluster_ids,
    n_clusters,
    target_indices,
    chi2_crit,
    z_crit,
    correction_z_crits,
    correction_method,
    warm_lambda_sq=-1.0,
):
    """Full LME analysis returning results array.

    Matches the C++ LMESolver::analyze interface. Performs REML fit,
    ML fits for LR test, Wald z-tests, and multiple comparison
    corrections.

    Args:
        X: (n, p) design matrix WITHOUT intercept.
        y: (n,) response vector.
        cluster_ids: (n,) integer cluster membership.
        n_clusters: Number of clusters.
        target_indices: Coefficient indices to test.
        chi2_crit: Critical chi-squared for LR test.
        z_crit: Critical z for Wald tests.
        correction_z_crits: Correction critical z values.
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm.
        warm_lambda_sq: Warm start lambda^2 (-1.0 for cold).

    Returns:
        np.ndarray: [f_sig, uncorrected..., corrected..., wald_flag]
        or None on failure.
    """
    n, p = X.shape
    n_targets = len(target_indices)
    K = n_clusters

    # Add intercept column
    X_int = np.column_stack([np.ones(n), X])

    # Convert warm_lambda_sq to warm_theta (sqrt)
    warm_theta = warm_lambda_sq**0.5 if warm_lambda_sq > 0 else None

    # REML fit
    try:
        reml_result = lme_fit(X_int, y, cluster_ids, K, q=1, reml=True, warm_theta=warm_theta)
    except Exception:
        return None

    if not reml_result.converged:
        return None

    # Extract fixed effects (skip intercept)
    beta = reml_result.beta[1:]
    se = reml_result.se_beta[1:]

    # Wald z-tests
    z_abs = np.zeros(p)
    for i in range(p):
        if se[i] > FLOAT_NEAR_ZERO:
            z_abs[i] = abs(beta[i] / se[i])

    # Result array
    results = np.zeros(1 + 2 * n_targets + 1)

    # Uncorrected individual tests
    for idx_pos in range(n_targets):
        coef_idx = target_indices[idx_pos]
        if coef_idx < p:
            results[1 + idx_pos] = 1.0 if z_abs[coef_idx] > z_crit else 0.0

    # Likelihood ratio test
    f_significant = 0.0
    wald_fallback = False

    try:
        if reml_result.tau2 < 1e-10:
            # Boundary: tau2 ≈ 0, OLS equivalent
            Q, R = np.linalg.qr(X_int)
            beta_full = np.linalg.solve(R, Q.T @ y)
            resid_full = y - X_int @ beta_full
            ss_full = resid_full @ resid_full
            sigma2_full = ss_full / n
            llf_full = -0.5 * n * (1.0 + np.log(2 * np.pi) + np.log(sigma2_full))

            y_mean = np.mean(y)
            ss_null = np.sum((y - y_mean) ** 2)
            sigma2_null = ss_null / n
            llf_null = -0.5 * n * (1.0 + np.log(2 * np.pi) + np.log(sigma2_null))

            lr_stat = 2 * (llf_full - llf_null)
        else:
            full_ml = lme_fit(X_int, y, cluster_ids, K, q=1, reml=False, warm_theta=reml_result.theta[0])
            null_ml = lme_fit_ml_null(y, cluster_ids, K, warm_theta=reml_result.theta[0])
            lr_stat = 2 * (full_ml.log_likelihood - null_ml.log_likelihood)

        if np.isnan(lr_stat) or lr_stat < 0 or not np.isfinite(lr_stat):
            wald_fallback = True
        else:
            f_significant = 1.0 if lr_stat > chi2_crit else 0.0
    except Exception:
        wald_fallback = True

    if wald_fallback:
        # Wald chi-squared fallback
        cov_fixed = reml_result.cov_beta[1:, 1:]
        try:
            inv_cov_beta = np.linalg.solve(cov_fixed, beta)
            wald_stat = beta @ inv_cov_beta
            f_significant = 1.0 if wald_stat > chi2_crit else 0.0
        except np.linalg.LinAlgError:
            pass

    results[0] = f_significant

    # Multiple comparison corrections (same logic as C++)
    if correction_method == 0 or correction_method == 1:
        for i in range(n_targets):
            coef_idx = target_indices[i]
            if coef_idx < p:
                results[1 + n_targets + i] = 1.0 if z_abs[coef_idx] > correction_z_crits[i] else 0.0
    elif correction_method == 2:
        # FDR step-up
        target_z = np.array([z_abs[target_indices[i]] if target_indices[i] < p else 0.0 for i in range(n_targets)])
        sorted_idx = np.argsort(-target_z)
        last_sig = -1
        for k in range(n_targets):
            if target_z[sorted_idx[k]] > correction_z_crits[k]:
                last_sig = k
        if last_sig >= 0:
            for k in range(last_sig + 1):
                results[1 + n_targets + sorted_idx[k]] = 1.0
    elif correction_method == 3:
        # Holm step-down
        target_z = np.array([z_abs[target_indices[i]] if target_indices[i] < p else 0.0 for i in range(n_targets)])
        sorted_idx = np.argsort(-target_z)
        for k in range(n_targets):
            if target_z[sorted_idx[k]] > correction_z_crits[k]:
                results[1 + n_targets + sorted_idx[k]] = 1.0
            else:
                break
    else:
        results[1 + n_targets : 1 + 2 * n_targets] = results[1 : 1 + n_targets]

    results[1 + 2 * n_targets] = 1.0 if wald_fallback else 0.0

    return results


def lme_analysis_general(
    X,
    y,
    cluster_ids,
    n_clusters,
    q,
    Z,
    target_indices,
    chi2_crit,
    z_crit,
    correction_z_crits,
    correction_method,
    warm_theta=None,
) -> Optional[np.ndarray]:
    """Full LME analysis for general q>1 (random slopes).

    Similar to lme_analysis_full but uses the general profiled deviance solver.

    Args:
        X: (n, p) design matrix WITHOUT intercept.
        y: (n,) response vector.
        cluster_ids: (n,) integer cluster membership.
        n_clusters: Number of clusters.
        q: Random effects dimension.
        Z: (n, q) random-effects design matrix.
        target_indices: Coefficient indices to test.
        chi2_crit, z_crit, correction_z_crits: Precomputed critical values.
        correction_method: 0=none, 1=Bonferroni, 2=FDR, 3=Holm.
        warm_theta: Previous optimal theta vector.

    Returns:
        np.ndarray or None on failure.
    """
    n, p = X.shape
    n_targets = len(target_indices)
    K = n_clusters

    X_int = np.column_stack([np.ones(n), X])

    try:
        reml_result = lme_fit(X_int, y, cluster_ids, K, q=q, reml=True, warm_theta=warm_theta, Z=Z)
    except Exception:
        return None

    if not reml_result.converged:
        return None

    beta = reml_result.beta[1:]
    se = reml_result.se_beta[1:]

    z_abs = np.zeros(p)
    for i in range(p):
        if se[i] > FLOAT_NEAR_ZERO:
            z_abs[i] = abs(beta[i] / se[i])

    results = np.zeros(1 + 2 * n_targets + 1)

    for idx_pos in range(n_targets):
        coef_idx = target_indices[idx_pos]
        if coef_idx < p:
            results[1 + idx_pos] = 1.0 if z_abs[coef_idx] > z_crit else 0.0

    # LR test for overall significance
    wald_fallback = False
    f_significant = 0.0
    try:
        full_ml = lme_fit(X_int, y, cluster_ids, K, q=q, reml=False, warm_theta=reml_result.theta, Z=Z)
        X_null = np.ones((n, 1))
        null_ml = lme_fit(X_null, y, cluster_ids, K, q=1, reml=False)
        lr_stat = 2 * (full_ml.log_likelihood - null_ml.log_likelihood)
        if np.isnan(lr_stat) or lr_stat < 0 or not np.isfinite(lr_stat):
            wald_fallback = True
        else:
            f_significant = 1.0 if lr_stat > chi2_crit else 0.0
    except Exception:
        wald_fallback = True

    if wald_fallback:
        cov_fixed = reml_result.cov_beta[1:, 1:]
        try:
            inv_cov_beta = np.linalg.solve(cov_fixed, beta)
            wald_stat = beta @ inv_cov_beta
            f_significant = 1.0 if wald_stat > chi2_crit else 0.0
        except np.linalg.LinAlgError:
            pass

    results[0] = f_significant

    # Multiple comparison corrections (same as q=1)
    if correction_method == 0 or correction_method == 1:
        for i in range(n_targets):
            coef_idx = target_indices[i]
            if coef_idx < p:
                results[1 + n_targets + i] = 1.0 if z_abs[coef_idx] > correction_z_crits[i] else 0.0
    elif correction_method == 2:
        target_z = np.array([z_abs[target_indices[i]] if target_indices[i] < p else 0.0 for i in range(n_targets)])
        sorted_idx = np.argsort(-target_z)
        last_sig = -1
        for k in range(n_targets):
            if target_z[sorted_idx[k]] > correction_z_crits[k]:
                last_sig = k
        if last_sig >= 0:
            for k in range(last_sig + 1):
                results[1 + n_targets + sorted_idx[k]] = 1.0
    elif correction_method == 3:
        target_z = np.array([z_abs[target_indices[i]] if target_indices[i] < p else 0.0 for i in range(n_targets)])
        sorted_idx = np.argsort(-target_z)
        for k in range(n_targets):
            if target_z[sorted_idx[k]] > correction_z_crits[k]:
                results[1 + n_targets + sorted_idx[k]] = 1.0
            else:
                break
    else:
        results[1 + n_targets : 1 + 2 * n_targets] = results[1 : 1 + n_targets]

    results[1 + 2 * n_targets] = 1.0 if wald_fallback else 0.0
    return results


def lme_analysis_nested(
    X,
    y,
    parent_ids,
    child_ids,
    K_parent,
    K_child,
    child_to_parent,
    target_indices,
    chi2_crit,
    z_crit,
    correction_z_crits,
    correction_method,
    warm_theta=None,
) -> Optional[np.ndarray]:
    """Full LME analysis for nested random intercepts.

    Args:
        X: (n, p) design matrix WITHOUT intercept.
        y: (n,) response vector.
        parent_ids, child_ids: cluster membership arrays.
        K_parent, K_child: cluster counts.
        child_to_parent: mapping array.
        target_indices, chi2_crit, z_crit, correction_z_crits,
        correction_method: as in lme_analysis_full.
        warm_theta: Previous optimal [theta_parent, theta_child].

    Returns:
        np.ndarray or None on failure.
    """
    n, p = X.shape
    n_targets = len(target_indices)

    X_int = np.column_stack([np.ones(n), X])

    try:
        reml_result = lme_fit_nested(
            X_int,
            y,
            parent_ids,
            child_ids,
            K_parent,
            K_child,
            child_to_parent,
            reml=True,
            warm_theta=warm_theta,
        )
    except Exception:
        return None

    if not reml_result.converged:
        return None

    beta = reml_result.beta[1:]
    se = reml_result.se_beta[1:]

    z_abs = np.zeros(p)
    for i in range(p):
        if se[i] > FLOAT_NEAR_ZERO:
            z_abs[i] = abs(beta[i] / se[i])

    results = np.zeros(1 + 2 * n_targets + 1)

    for idx_pos in range(n_targets):
        coef_idx = target_indices[idx_pos]
        if coef_idx < p:
            results[1 + idx_pos] = 1.0 if z_abs[coef_idx] > z_crit else 0.0

    # LR test
    wald_fallback = False
    f_significant = 0.0
    try:
        full_ml = lme_fit_nested(
            X_int,
            y,
            parent_ids,
            child_ids,
            K_parent,
            K_child,
            child_to_parent,
            reml=False,
            warm_theta=reml_result.theta,
        )
        X_null = np.ones((n, 1))
        null_ml = lme_fit_nested(
            X_null,
            y,
            parent_ids,
            child_ids,
            K_parent,
            K_child,
            child_to_parent,
            reml=False,
        )
        lr_stat = 2 * (full_ml.log_likelihood - null_ml.log_likelihood)
        if np.isnan(lr_stat) or lr_stat < 0 or not np.isfinite(lr_stat):
            wald_fallback = True
        else:
            f_significant = 1.0 if lr_stat > chi2_crit else 0.0
    except Exception:
        wald_fallback = True

    if wald_fallback:
        cov_fixed = reml_result.cov_beta[1:, 1:]
        try:
            inv_cov_beta = np.linalg.solve(cov_fixed, beta)
            wald_stat = beta @ inv_cov_beta
            f_significant = 1.0 if wald_stat > chi2_crit else 0.0
        except np.linalg.LinAlgError:
            pass

    results[0] = f_significant

    if correction_method == 0 or correction_method == 1:
        for i in range(n_targets):
            coef_idx = target_indices[i]
            if coef_idx < p:
                results[1 + n_targets + i] = 1.0 if z_abs[coef_idx] > correction_z_crits[i] else 0.0
    elif correction_method == 2:
        target_z = np.array([z_abs[target_indices[i]] if target_indices[i] < p else 0.0 for i in range(n_targets)])
        sorted_idx = np.argsort(-target_z)
        last_sig = -1
        for k in range(n_targets):
            if target_z[sorted_idx[k]] > correction_z_crits[k]:
                last_sig = k
        if last_sig >= 0:
            for k in range(last_sig + 1):
                results[1 + n_targets + sorted_idx[k]] = 1.0
    elif correction_method == 3:
        target_z = np.array([z_abs[target_indices[i]] if target_indices[i] < p else 0.0 for i in range(n_targets)])
        sorted_idx = np.argsort(-target_z)
        for k in range(n_targets):
            if target_z[sorted_idx[k]] > correction_z_crits[k]:
                results[1 + n_targets + sorted_idx[k]] = 1.0
            else:
                break
    else:
        results[1 + n_targets : 1 + 2 * n_targets] = results[1 : 1 + n_targets]

    results[1 + 2 * n_targets] = 1.0 if wald_fallback else 0.0
    return results


# ---------------------------------------------------------------------------
# JIT compilation (following ols.py pattern)
# ---------------------------------------------------------------------------

# Keep the pure-Python versions as fallbacks
_profiled_deviance_q1_jit = _profiled_deviance_q1_core
_compute_suff_stats_q1_jit = _compute_suff_stats_q1
_USE_LME_JIT = False

try:
    from numba import njit

    _profiled_deviance_q1_jit = njit(cache=True)(_profiled_deviance_q1_core)
    _compute_suff_stats_q1_jit = njit(cache=True)(_compute_suff_stats_q1)
    _USE_LME_JIT = True
except ImportError:
    pass
