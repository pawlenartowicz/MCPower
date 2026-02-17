#include "lme_solver.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace mcpower {

// -------------------------------------------------------------------------
// Sufficient statistics computation
// -------------------------------------------------------------------------

SufficientStatsQ1 LMESolver::compute_sufficient_stats(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXi>& cluster_ids,
    int K
) {
    const int N = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());

    SufficientStatsQ1 stats;
    stats.K = K;
    stats.p = p;
    stats.N = N;

    stats.cluster_sizes = VectorXi::Zero(K);
    stats.ZtZ = VectorXd::Zero(K);
    stats.ZtX = MatrixXd::Zero(K, p);
    stats.Zty = VectorXd::Zero(K);
    stats.Xty = MatrixXd::Zero(K, p);
    stats.yty = VectorXd::Zero(K);

    stats.XtX.resize(K);
    for (int j = 0; j < K; ++j) {
        stats.XtX[j] = MatrixXd::Zero(p, p);
    }

    // Single pass over data
    for (int i = 0; i < N; ++i) {
        const int j = cluster_ids(i);
        const double yi = y(i);

        stats.cluster_sizes(j) += 1;
        stats.yty(j) += yi * yi;
        stats.Zty(j) += yi;

        for (int c = 0; c < p; ++c) {
            const double xic = X(i, c);
            stats.Xty(j, c) += xic * yi;
            stats.ZtX(j, c) += xic;

            // Upper triangle of XtX (symmetric)
            for (int c2 = c; c2 < p; ++c2) {
                stats.XtX[j](c, c2) += xic * X(i, c2);
            }
        }
    }

    // Fill lower triangles and set ZtZ
    for (int j = 0; j < K; ++j) {
        stats.ZtZ(j) = static_cast<double>(stats.cluster_sizes(j));
        for (int c = 0; c < p; ++c) {
            for (int c2 = 0; c2 < c; ++c2) {
                stats.XtX[j](c, c2) = stats.XtX[j](c2, c);
            }
        }
    }

    return stats;
}

// -------------------------------------------------------------------------
// Profiled deviance evaluation (q=1 fast path)
// -------------------------------------------------------------------------

double LMESolver::profiled_deviance_q1(
    double lam_sq,
    const SufficientStatsQ1& stats,
    bool reml
) {
    const int K = stats.K;
    const int p = stats.p;
    const int N = stats.N;

    // Accumulate Schur complement
    MatrixXd A = MatrixXd::Zero(p, p);
    VectorXd b = VectorXd::Zero(p);
    double c = 0.0;
    double log_det_Ltheta = 0.0;

    const double sqrt_lam = std::sqrt(lam_sq);

    for (int j = 0; j < K; ++j) {
        const double nj = stats.ZtZ(j);
        const double Mj = lam_sq * nj + 1.0;
        const double Lj = std::sqrt(Mj);
        const double inv_Lj = 1.0 / Lj;
        const double factor = inv_Lj * sqrt_lam;

        // cu_j = factor * Zty_j (scalar)
        const double cu_j = factor * stats.Zty(j);

        // Accumulate A, b, c
        for (int c1 = 0; c1 < p; ++c1) {
            const double CX_c1 = factor * stats.ZtX(j, c1);
            b(c1) += stats.Xty(j, c1) - CX_c1 * cu_j;

            for (int c2 = c1; c2 < p; ++c2) {
                const double CX_c2 = factor * stats.ZtX(j, c2);
                A(c1, c2) += stats.XtX[j](c1, c2) - CX_c1 * CX_c2;
            }
        }

        c += stats.yty(j) - cu_j * cu_j;
        log_det_Ltheta += 2.0 * std::log(Lj);
    }

    // Fill lower triangle
    for (int c1 = 0; c1 < p; ++c1) {
        for (int c2 = 0; c2 < c1; ++c2) {
            A(c1, c2) = A(c2, c1);
        }
    }

    // Cholesky decomposition of A
    Eigen::LLT<MatrixXd> llt(A);
    if (llt.info() != Eigen::Success) {
        return LME_DEVIANCE_FAIL;
    }

    // Solve for beta: A * beta = b
    VectorXd beta = llt.solve(b);

    // r^2(theta) = c - beta' * b
    double r_sq = c - beta.dot(b);
    if (r_sq <= 0.0) {
        return LME_DEVIANCE_FAIL;
    }

    // Profiled objective
    if (reml) {
        // f_R = log|L_theta| + 2*sum(log(diag(R_X))) + (N-p)*log(r_sq)
        const MatrixXd& L = llt.matrixL();
        double log_det_RX = 0.0;
        for (int i = 0; i < p; ++i) {
            log_det_RX += std::log(L(i, i));
        }
        return log_det_Ltheta + 2.0 * log_det_RX + static_cast<double>(N - p) * std::log(r_sq);
    } else {
        // f = log|L_theta| + N*log(r_sq)
        return log_det_Ltheta + static_cast<double>(N) * std::log(r_sq);
    }
}

// -------------------------------------------------------------------------
// Brent's method for 1D minimization
// -------------------------------------------------------------------------

double LMESolver::brent_minimize(
    double a, double b, double tol, int max_iter,
    const SufficientStatsQ1& stats, bool reml
) {
    // Brent's method (combination of golden section and parabolic interpolation)
    const double golden = 0.3819660112501051;  // (3 - sqrt(5)) / 2
    const double eps = std::numeric_limits<double>::epsilon();

    double x = a + golden * (b - a);
    double w = x, v = x;
    double fx = profiled_deviance_q1(x, stats, reml);
    double fw = fx, fv = fx;
    double d = 0.0, e = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        double midpoint = 0.5 * (a + b);
        double tol1 = tol * std::abs(x) + eps;
        double tol2 = 2.0 * tol1;

        // Convergence check
        if (std::abs(x - midpoint) <= (tol2 - 0.5 * (b - a))) {
            return x;
        }

        // Try parabolic interpolation
        bool use_golden = true;
        double u;

        if (std::abs(e) > tol1) {
            // Fit parabola
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            else q = -q;
            r = e;
            e = d;

            // Check if parabolic step is acceptable
            if (std::abs(p) < std::abs(0.5 * q * r) &&
                p > q * (a - x) && p < q * (b - x)) {
                d = p / q;
                u = x + d;
                if ((u - a) < tol2 || (b - u) < tol2) {
                    d = (x < midpoint) ? tol1 : -tol1;
                }
                use_golden = false;
            }
        }

        if (use_golden) {
            e = (x < midpoint) ? (b - x) : (a - x);
            d = golden * e;
        }

        // Evaluate function at new point
        u = x + ((std::abs(d) >= tol1) ? d : ((d > 0) ? tol1 : -tol1));
        double fu = profiled_deviance_q1(u, stats, reml);

        // Update bracket
        if (fu <= fx) {
            if (u < x) b = x; else a = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }

    return x;
}

// -------------------------------------------------------------------------
// Extract results at optimal lambda^2
// -------------------------------------------------------------------------

LMEFitResult LMESolver::extract_results_q1(
    double lam_sq_opt,
    const SufficientStatsQ1& stats,
    bool reml
) {
    const int K = stats.K;
    const int p = stats.p;
    const int N = stats.N;

    // Recompute A, b, c at the optimum
    MatrixXd A = MatrixXd::Zero(p, p);
    VectorXd b_vec = VectorXd::Zero(p);
    double c = 0.0;
    double log_det_Ltheta = 0.0;

    const double sqrt_lam = std::sqrt(lam_sq_opt);

    for (int j = 0; j < K; ++j) {
        const double nj = stats.ZtZ(j);
        const double Mj = lam_sq_opt * nj + 1.0;
        const double Lj = std::sqrt(Mj);
        const double inv_Lj = 1.0 / Lj;
        const double factor = inv_Lj * sqrt_lam;

        const double cu_j = factor * stats.Zty(j);

        for (int c1 = 0; c1 < p; ++c1) {
            const double CX_c1 = factor * stats.ZtX(j, c1);
            b_vec(c1) += stats.Xty(j, c1) - CX_c1 * cu_j;
            for (int c2 = c1; c2 < p; ++c2) {
                const double CX_c2 = factor * stats.ZtX(j, c2);
                A(c1, c2) += stats.XtX[j](c1, c2) - CX_c1 * CX_c2;
            }
        }
        c += stats.yty(j) - cu_j * cu_j;
        log_det_Ltheta += 2.0 * std::log(Lj);
    }

    // Fill lower triangle
    for (int c1 = 0; c1 < p; ++c1) {
        for (int c2 = 0; c2 < c1; ++c2) {
            A(c1, c2) = A(c2, c1);
        }
    }

    LMEFitResult result;
    result.lambda_sq = lam_sq_opt;

    // Solve for beta
    result.beta = A.ldlt().solve(b_vec);

    // r^2 and sigma^2
    double r_sq = c - result.beta.dot(b_vec);

    if (reml) {
        result.sigma2 = r_sq / static_cast<double>(N - p);
    } else {
        result.sigma2 = r_sq / static_cast<double>(N);
    }

    // tau^2 = sigma^2 * lambda^2
    result.tau2 = result.sigma2 * lam_sq_opt;

    // Covariance of beta: sigma^2 * A^{-1}
    result.cov_beta = result.sigma2 * A.ldlt().solve(MatrixXd::Identity(p, p));
    result.se_beta = result.cov_beta.diagonal().cwiseMax(0.0).cwiseSqrt();

    // ML log-likelihood (for LR tests)
    double sigma2_ml = r_sq / static_cast<double>(N);
    if (sigma2_ml > 0.0) {
        result.log_likelihood = -0.5 * (log_det_Ltheta +
            static_cast<double>(N) +
            static_cast<double>(N) * std::log(2.0 * M_PI) +
            static_cast<double>(N) * std::log(sigma2_ml));
    } else {
        result.log_likelihood = -std::numeric_limits<double>::infinity();
    }

    result.converged = true;
    return result;
}

// -------------------------------------------------------------------------
// Fit q=1 model via Brent's method
// -------------------------------------------------------------------------

LMEFitResult LMESolver::fit_q1(
    const SufficientStatsQ1& stats,
    bool reml,
    double warm_lambda_sq
) {
    double lam_sq_opt;

    if (warm_lambda_sq > 0.0) {
        // Warm start: search around previous optimum
        double lo = std::max(0.0, warm_lambda_sq * 0.01);
        double hi = std::min(1e6, warm_lambda_sq * 100.0);
        lam_sq_opt = brent_minimize(lo, hi, 1e-8, 100, stats, reml);

        // Verify we didn't get stuck at a boundary
        double f_warm = profiled_deviance_q1(lam_sq_opt, stats, reml);
        if (lam_sq_opt <= lo * 1.01 || lam_sq_opt >= hi * 0.99) {
            double lam_sq_full = brent_minimize(0.0, 1e6, 1e-8, 150, stats, reml);
            double f_full = profiled_deviance_q1(lam_sq_full, stats, reml);
            if (f_full < f_warm) {
                lam_sq_opt = lam_sq_full;
            }
        }
    } else {
        lam_sq_opt = brent_minimize(0.0, 1e6, 1e-8, 150, stats, reml);
    }

    return extract_results_q1(lam_sq_opt, stats, reml);
}

// -------------------------------------------------------------------------
// Full analysis
// -------------------------------------------------------------------------

Eigen::VectorXd LMESolver::analyze(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXi>& cluster_ids,
    int n_clusters,
    const Eigen::Ref<const VectorXi>& target_indices,
    double chi2_crit,
    double z_crit,
    const Eigen::Ref<const VectorXd>& correction_z_crits,
    int correction_method,
    double warm_lambda_sq
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    const int n_targets = static_cast<int>(target_indices.size());
    const int K = n_clusters;

    // Result: [f_sig, uncorrected_1..n, corrected_1..n, wald_flag]
    VectorXd results = VectorXd::Zero(1 + 2 * n_targets + 1);

    // Add intercept column
    MatrixXd X_int(n, p + 1);
    X_int.col(0) = VectorXd::Ones(n);
    X_int.rightCols(p) = X;

    // Compute sufficient statistics (once for all fits)
    SufficientStatsQ1 stats = compute_sufficient_stats(X_int, y, cluster_ids, K);

    // REML fit
    LMEFitResult reml_result = fit_q1(stats, true, warm_lambda_sq);
    if (!reml_result.converged) {
        return VectorXd();  // empty = failure
    }

    // Extract fixed effects (skip intercept)
    VectorXd beta = reml_result.beta.tail(p);
    VectorXd se = reml_result.se_beta.tail(p);

    // Wald z-tests for individual effects
    VectorXd z_abs = VectorXd::Zero(p);
    for (int i = 0; i < p; ++i) {
        if (se(i) > LME_FLOAT_NEAR_ZERO) {
            z_abs(i) = std::abs(beta(i) / se(i));
        }
    }

    // Uncorrected individual tests
    for (int idx_pos = 0; idx_pos < n_targets; ++idx_pos) {
        int coef_idx = target_indices(idx_pos);
        if (coef_idx < p) {
            results(1 + idx_pos) = (z_abs(coef_idx) > z_crit) ? 1.0 : 0.0;
        }
    }

    // Likelihood ratio test for overall significance
    double f_significant = 0.0;
    bool wald_fallback = false;

    if (reml_result.tau2 < 1e-10) {
        // Boundary case: tau2 ≈ 0, effectively OLS
        // Simple OLS log-likelihood comparison
        Eigen::HouseholderQR<MatrixXd> qr(X_int);
        VectorXd beta_full = qr.solve(y);
        VectorXd resid_full = y - X_int * beta_full;
        double ss_full = resid_full.squaredNorm();
        double sigma2_full = ss_full / n;
        double llf_full = -0.5 * n * (1.0 + std::log(2.0 * M_PI) + std::log(sigma2_full));

        double y_mean = y.mean();
        double ss_null = (y.array() - y_mean).square().sum();
        double sigma2_null = ss_null / n;
        double llf_null = -0.5 * n * (1.0 + std::log(2.0 * M_PI) + std::log(sigma2_null));

        double lr_stat = 2.0 * (llf_full - llf_null);
        if (std::isnan(lr_stat) || lr_stat < 0.0 || !std::isfinite(lr_stat)) {
            wald_fallback = true;
        } else {
            f_significant = (lr_stat > chi2_crit) ? 1.0 : 0.0;
        }
    } else {
        // Full ML fit + null ML fit for LR test
        // Recompute stats for ML (same data, just different objective)
        LMEFitResult full_ml = fit_q1(stats, false, reml_result.lambda_sq);

        // Null model: intercept + random intercept only
        MatrixXd X_null = MatrixXd::Ones(n, 1);
        SufficientStatsQ1 null_stats = compute_sufficient_stats(X_null, y, cluster_ids, K);
        LMEFitResult null_ml = fit_q1(null_stats, false, reml_result.lambda_sq);

        double lr_stat = 2.0 * (full_ml.log_likelihood - null_ml.log_likelihood);
        if (std::isnan(lr_stat) || lr_stat < 0.0 || !std::isfinite(lr_stat)) {
            wald_fallback = true;
        } else {
            f_significant = (lr_stat > chi2_crit) ? 1.0 : 0.0;
        }
    }

    // Wald fallback for overall significance
    if (wald_fallback) {
        // Wald chi-squared: beta' * inv(cov_beta) * beta
        MatrixXd cov_fixed = reml_result.cov_beta.bottomRightCorner(p, p);
        Eigen::LDLT<MatrixXd> ldlt(cov_fixed);
        if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
            VectorXd inv_cov_beta = ldlt.solve(beta);
            double wald_stat = beta.dot(inv_cov_beta);
            f_significant = (wald_stat > chi2_crit) ? 1.0 : 0.0;
        }
    }

    results(0) = f_significant;

    // Multiple comparison corrections
    if (correction_method == 0 || correction_method == 1) {
        // No correction or Bonferroni: direct comparison
        for (int i = 0; i < n_targets; ++i) {
            int coef_idx = target_indices(i);
            if (coef_idx < p) {
                results(1 + n_targets + i) =
                    (z_abs(coef_idx) > correction_z_crits(i)) ? 1.0 : 0.0;
            }
        }
    } else if (correction_method == 2) {
        // FDR (Benjamini-Hochberg): step-up procedure
        VectorXd target_z = VectorXd::Zero(n_targets);
        for (int i = 0; i < n_targets; ++i) {
            int coef_idx = target_indices(i);
            if (coef_idx < p) target_z(i) = z_abs(coef_idx);
        }

        // Sort indices by |z| descending
        std::vector<int> sorted_idx(n_targets);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&target_z](int a, int b) { return target_z(a) > target_z(b); });

        int last_sig = -1;
        for (int k = 0; k < n_targets; ++k) {
            if (target_z(sorted_idx[k]) > correction_z_crits(k)) {
                last_sig = k;
            }
        }
        if (last_sig >= 0) {
            for (int k = 0; k <= last_sig; ++k) {
                results(1 + n_targets + sorted_idx[k]) = 1.0;
            }
        }
    } else if (correction_method == 3) {
        // Holm: step-down procedure
        VectorXd target_z = VectorXd::Zero(n_targets);
        for (int i = 0; i < n_targets; ++i) {
            int coef_idx = target_indices(i);
            if (coef_idx < p) target_z(i) = z_abs(coef_idx);
        }

        std::vector<int> sorted_idx(n_targets);
        std::iota(sorted_idx.begin(), sorted_idx.end(), 0);
        std::sort(sorted_idx.begin(), sorted_idx.end(),
                  [&target_z](int a, int b) { return target_z(a) > target_z(b); });

        for (int k = 0; k < n_targets; ++k) {
            if (target_z(sorted_idx[k]) > correction_z_crits(k)) {
                results(1 + n_targets + sorted_idx[k]) = 1.0;
            } else {
                break;
            }
        }
    } else {
        // Unknown: same as uncorrected
        results.segment(1 + n_targets, n_targets) = results.segment(1, n_targets);
    }

    // Wald flag (last element)
    results(1 + 2 * n_targets) = wald_fallback ? 1.0 : 0.0;

    return results;
}

}  // namespace mcpower
