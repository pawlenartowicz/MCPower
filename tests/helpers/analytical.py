"""
Analytical power formulas for non-central t and F distributions.

Single source of truth — imported by specs/, backend/, and any future tests.
"""

import numpy as np
from scipy.stats import f as f_dist
from scipy.stats import ncf, nct, ncx2, norm
from scipy.stats import t as t_dist


def analytical_t_power(beta, n, p, sigma_eps, vif_j, alpha=0.05):
    """
    Power of a two-sided t-test for coefficient j, corrected for
    finite-sample variability of X.

    Uses the inverse-Wishart expectation E[(X'X)^{-1}]_jj = Sigma^{-1}_jj / (n - p - 2)
    instead of the plug-in Sigma^{-1}_jj / n.

    Args:
        beta: effect size for predictor j
        n: sample size
        p: total number of predictors
        sigma_eps: residual std dev (1.0 in MCPower's DGP)
        vif_j: [Sigma^{-1}]_jj for predictor j (= VIF for standardised vars)
        alpha: significance level
    """
    df = n - p - 1
    n_eff = max(n - p - 2, p + 2)
    noncentrality = beta * np.sqrt(n_eff) / (sigma_eps * np.sqrt(vif_j))
    t_crit = t_dist.ppf(1 - alpha / 2, df)
    power = 1 - nct.cdf(t_crit, df, noncentrality) + nct.cdf(-t_crit, df, noncentrality)
    return power * 100  # percentage


def analytical_f_power(betas, n, corr_matrix, sigma_eps, alpha=0.05):
    """
    Power of the overall F-test.

    Non-centrality: lambda = n * beta' Sigma beta / sigma_eps^2
    dfn = p, dfd = n - p - 1

    Args:
        betas: array of effect sizes
        n: sample size
        corr_matrix: Sigma (predictor correlation matrix)
        sigma_eps: residual std dev
        alpha: significance level
    """
    betas = np.asarray(betas, dtype=float)
    p = len(betas)
    dfn = p
    dfd = n - p - 1
    lam = n * betas @ corr_matrix @ betas / (sigma_eps**2)
    f_crit = f_dist.ppf(1 - alpha, dfn, dfd)
    power = 1 - ncf.cdf(f_crit, dfn, dfd, lam)
    return power * 100  # percentage


# =========================================================================
# Mixed model (LME) analytical power
# =========================================================================


def _deff_within(n_total, n_clusters, icc):
    """
    Design effect for within-cluster (iid) predictors.

    When X is generated iid (no cluster structure in X),
    Deff = (1 + (m-1)*ICC) / (1 + (m-2)*ICC)  where m = n/K.

    This is much milder than the between-cluster Deff = 1 + (m-1)*ICC.
    """
    m = n_total / n_clusters
    return (1 + (m - 1) * icc) / (1 + (m - 2) * icc)


def analytical_z_power_lme(beta, n_total, n_clusters, icc, sigma_eps=1.0, vif_j=1.0, alpha=0.05):
    """
    Power of a two-sided z-test (Wald) for a single fixed effect in an LME.

    MCPower's DGP: y_ij = Xβ + b_i + ε, X ~ N(0,Σ) iid, b_i ~ N(0,τ²), ε ~ N(0,1).
    statsmodels MixedLM reports z-tests (normal approx) for fixed effects.

    NCP: λ = β * sqrt(n_eff / (VIF_j * Deff_within)) / σ
    Power = 1 - Φ(z_{α/2} - λ) + Φ(-z_{α/2} - λ)

    Args:
        beta: effect size for predictor j
        n_total: total sample size (all observations)
        n_clusters: number of clusters (K)
        icc: intraclass correlation coefficient
        sigma_eps: residual std dev (1.0 in MCPower's DGP)
        vif_j: [Σ^{-1}]_jj for predictor j (= VIF for standardised vars)
        alpha: significance level
    """
    p_fixed = 1  # conservative: at least 1 predictor
    n_eff = max(n_total - p_fixed - 1, p_fixed + 2)
    deff = _deff_within(n_total, n_clusters, icc)
    ncp = abs(beta) * np.sqrt(n_eff / (vif_j * deff)) / sigma_eps
    z_crit = norm.ppf(1 - alpha / 2)
    power = 1 - norm.cdf(z_crit - ncp) + norm.cdf(-z_crit - ncp)
    return power * 100  # percentage


def analytical_lr_power_lme(betas, n_total, n_clusters, icc, corr_matrix, sigma_eps=1.0, alpha=0.05):
    """
    Power of the likelihood-ratio (χ²) test for overall fixed effects in an LME.

    NCP: λ = n * β'Σβ / (σ² * Deff_within)
    Power = 1 - F_{ncχ²}(χ²_{α,p}; p, λ)

    Args:
        betas: array of effect sizes
        n_total: total sample size
        n_clusters: number of clusters (K)
        icc: intraclass correlation coefficient
        corr_matrix: Σ (predictor correlation matrix)
        sigma_eps: residual std dev
        alpha: significance level
    """
    betas = np.asarray(betas, dtype=float)
    p = len(betas)
    deff = _deff_within(n_total, n_clusters, icc)
    lam = n_total * betas @ corr_matrix @ betas / (sigma_eps**2 * deff)
    chi2_crit = ncx2.ppf(1 - alpha, df=p, nc=0)  # central χ² critical value
    power = 1 - ncx2.cdf(chi2_crit, df=p, nc=lam)
    return power * 100  # percentage
