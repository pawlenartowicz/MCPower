"""Hand-compute reference for the θ=1 random-intercept LME case.

Computes closed-form NumPy values for the random-intercept LME profiled
deviance at fixed theta=1 (and at theta=2 for a deviance-difference reference),
plus statsmodels MixedLM(reml=True) deviance to cross-check the deviance
*difference* (the additive constant cancels).

The case:
  N=6 rows, 2 clusters of 3 rows, P=2 (intercept + 1 covariate).
  Seed: numpy.random.default_rng(0).

For the LME reparam theta = tau/sigma, each cluster has
  V_c = I_{n_c} + theta^2 * 1 * 1'
  V_c^{-1} = I_{n_c} - (theta^2 / (1 + theta^2 * n_c)) * 1 * 1'

So globally V = block_diag(V_c) and we form
  X' V^{-1} X = X'X - sum_c (theta^2 / (1 + theta^2 * n_c)) * (sum_c X_c) * (sum_c X_c)'
  X' V^{-1} y = X'y - sum_c (theta^2 / (1 + theta^2 * n_c)) * (sum_c X_c) * (sum_c y_c)
  y' V^{-1} y = y'y - sum_c (theta^2 / (1 + theta^2 * n_c)) * (sum_c y_c)^2

REML closed forms at fixed theta:
  beta_hat        = (X' V^{-1} X)^{-1} X' V^{-1} y
  sigma_hat^2     = (y' V^{-1} y - beta_hat' X' V^{-1} y) / (N - P)
  log|V|          = sum_c log(1 + theta^2 * n_c)
  log|X'V^{-1}X|  = 2 * sum_j log L_jj    (where L is the lower Cholesky factor)
  dev_REML(theta) = log|V| + log|X'V^{-1}X| + (N - P) * log(sigma_hat^2)

statsmodels MixedLM minimises over theta to find theta_hat; here we want
beta and sigma^2 *at fixed theta=1*, which are NOT the statsmodels fit. So
we compute them directly from the matrix formulas above. The deviance
DIFFERENCE between theta=1 and theta=2 can be cross-checked against the
statsmodels profiled-deviance evaluator because additive constants cancel.
"""

import warnings

import numpy as np
from numpy.linalg import cholesky
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Suppress noisy convergence warnings — we only call .fit() to populate
# statsmodels' lazy attributes (cov_pen etc.) so that .loglike() can be
# evaluated at arbitrary theta values. The actual numeric results come
# from the closed-form NumPy code above.
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def vinv_action(theta_sq, n_c):
    """Scalar coefficient s_c = theta^2 / (1 + theta^2 * n_c)."""
    return theta_sq / (1.0 + theta_sq * n_c)


def closed_form_at_theta(theta, X, y, cluster_ids):
    """Compute REML closed-form quantities at fixed theta."""
    N, P = X.shape
    unique_clusters = np.unique(cluster_ids)
    theta_sq = theta * theta

    # Per-cluster sums
    XtVinvX = X.T @ X
    XtVinvy = X.T @ y
    ytVinvy = float(y @ y)
    log_det_V = 0.0
    for c in unique_clusters:
        mask = cluster_ids == c
        n_c = int(mask.sum())
        Xc_sum = X[mask].sum(axis=0)            # shape (P,)
        yc_sum = float(y[mask].sum())
        s_c = vinv_action(theta_sq, n_c)
        XtVinvX = XtVinvX - s_c * np.outer(Xc_sum, Xc_sum)
        XtVinvy = XtVinvy - s_c * Xc_sum * yc_sum
        ytVinvy = ytVinvy - s_c * yc_sum * yc_sum
        log_det_V += np.log(1.0 + theta_sq * n_c)

    L = cholesky(XtVinvX)
    # beta_hat: solve L L' beta = X'V^{-1}y
    z = np.linalg.solve(L, XtVinvy)
    beta_hat = np.linalg.solve(L.T, z)
    sigma_sq = (ytVinvy - float(beta_hat @ XtVinvy)) / (N - P)
    log_det_XVinvX = 2.0 * np.log(np.diag(L)).sum()
    dev = log_det_V + log_det_XVinvX + (N - P) * np.log(sigma_sq)
    return {
        "beta_hat": beta_hat,
        "sigma_sq": sigma_sq,
        "log_det_V": log_det_V,
        "log_det_XVinvX": log_det_XVinvX,
        "dev_REML_closed": dev,
        "XtVinvX": XtVinvX,
        "XtVinvy": XtVinvy,
        "ytVinvy": ytVinvy,
    }


def statsmodels_dev_at_theta(theta, X, y, cluster_ids):
    """Evaluate statsmodels' profiled REML log-likelihood at fixed theta.

    statsmodels parametrises the random-effect cov as tau^2 = sigma^2 * theta^2,
    and uses an internal `params` vector containing the (free) Cholesky factor
    of the random-effects covariance, scaled by 1/sigma. For a single random
    intercept this is just theta itself. The profiled REML log-likelihood is
    evaluated at the fixed theta using `MixedLM.loglike(...)`.

    Returns -2 * loglike (= the deviance) so the difference matches the
    closed-form deviance convention.
    """
    # statsmodels expects: endog=y, exog=X, groups=cluster_ids, exog_re=intercept
    md = sm.MixedLM(endog=y, exog=X, groups=cluster_ids,
                    exog_re=np.ones((len(y), 1)))
    # statsmodels lazily initialises some attributes (notably cov_pen) inside
    # .fit(); .loglike() crashes with AttributeError if called on a fresh
    # model object. So run a quick fit first to populate those attributes,
    # then evaluate loglike at the user-supplied theta.
    md.fit(reml=True)
    params = np.array([theta])
    loglike = md.loglike(params, profile_fe=True)  # REML by default
    return -2.0 * loglike


def main():
    rng = np.random.default_rng(0)
    N = 6
    n_clusters = 2
    cluster_size = 3
    P = 2  # intercept + 1 covariate

    # Cluster ids: blocked (cluster 0 → rows 0..2, cluster 1 → rows 3..5).
    cluster_ids = np.array([i // cluster_size for i in range(N)], dtype=np.int64)

    x1 = rng.standard_normal(N)
    X = np.column_stack([np.ones(N), x1])

    # y: linear combination + cluster random intercepts + noise (just any
    # finite reproducible y; the test is closed-form at fixed theta).
    true_beta = np.array([0.5, 1.2])
    u = rng.standard_normal(n_clusters) * np.sqrt(0.4)
    eps = rng.standard_normal(N)
    y = X @ true_beta + u[cluster_ids] + eps

    # Helper for printing native Python floats with full repr precision
    # (numpy 2.x .repr's wrap scalars as `np.float64(...)`, which is not
    # paste-ready Rust syntax).
    def f(v):
        return repr(float(v))

    print("# Input data (numpy default_rng(0)):")
    print(f"# N            = {N}")
    print(f"# P            = {P}")
    print(f"# n_clusters   = {n_clusters}")
    print(f"# cluster_size = {cluster_size}")
    print()
    print(f"cluster_ids = {cluster_ids.tolist()}")
    print("x1 (covariate column 1) = [")
    for v in x1:
        print(f"    {f(v)},")
    print("]")
    print("y = [")
    for v in y:
        print(f"    {f(v)},")
    print("]")
    print()

    res1 = closed_form_at_theta(1.0, X, y, cluster_ids)
    res2 = closed_form_at_theta(2.0, X, y, cluster_ids)

    # Independent statsmodels cross-check for the deviance DIFFERENCE
    # (additive constants cancel, so this verifies the deviance closed form
    # up to a constant offset).
    sm_dev1 = statsmodels_dev_at_theta(1.0, X, y, cluster_ids)
    sm_dev2 = statsmodels_dev_at_theta(2.0, X, y, cluster_ids)

    dev_diff_closed = float(res1["dev_REML_closed"] - res2["dev_REML_closed"])
    dev_diff_sm = float(sm_dev1 - sm_dev2)

    b0, b1 = (float(v) for v in res1["beta_hat"].tolist())
    sigma_sq1 = float(res1["sigma_sq"])

    print("# Closed-form quantities at theta=1:")
    print(f"#   beta_hat(theta=1)      = [{f(b0)}, {f(b1)}]")
    print(f"#   sigma_hat^2(theta=1)   = {f(sigma_sq1)}")
    print(f"#   log|V|(theta=1)        = {f(res1['log_det_V'])}")
    print(f"#   log|X'V^-1 X|(theta=1) = {f(res1['log_det_XVinvX'])}")
    print(f"#   dev_REML_closed(1)     = {f(res1['dev_REML_closed'])}")
    print()
    print("# Closed-form quantities at theta=2:")
    b0_t2, b1_t2 = (float(v) for v in res2["beta_hat"].tolist())
    print(f"#   beta_hat(theta=2)      = [{f(b0_t2)}, {f(b1_t2)}]")
    print(f"#   sigma_hat^2(theta=2)   = {f(res2['sigma_sq'])}")
    print(f"#   dev_REML_closed(2)     = {f(res2['dev_REML_closed'])}")
    print()
    print("# Deviance difference cross-check (additive constants cancel):")
    print(f"#   closed-form dev(1) - dev(2)   = {f(dev_diff_closed)}")
    print(f"#   statsmodels dev(1) - dev(2)   = {f(dev_diff_sm)}")
    print(f"#   absolute mismatch             = {f(abs(dev_diff_closed - dev_diff_sm))}")
    print()

    # --- Final constants to paste into the Rust test (16 sig fig via repr) ---
    print("// ------------------------------------------------------------")
    print("// Paste these into the LME test body")
    print("// Generated by MCPower2/scripts/lme_handcompute_theta1.py")
    print("// ------------------------------------------------------------")
    print(f"let expected_beta_theta1: [f64; 2] = [{f(b0)}, {f(b1)}];")
    print(f"let expected_sigma_sq_theta1: f64 = {f(sigma_sq1)};")
    print(f"let expected_dev_diff_1_minus_2: f64 = {f(dev_diff_closed)};")
    print()


if __name__ == "__main__":
    main()
