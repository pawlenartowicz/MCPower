#include "lme_solver.hpp"
#include "optimizers.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>

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

    // Validate cluster_ids are in [0, K)
    for (int i = 0; i < N; ++i) {
        if (cluster_ids(i) < 0 || cluster_ids(i) >= K) {
            throw std::invalid_argument("cluster_ids contains value outside [0, K)");
        }
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

    // Solve for beta via LDLT (computed once, reused for cov_beta)
    Eigen::LDLT<MatrixXd> ldlt_A(A);
    if (ldlt_A.info() != Eigen::Success) {
        result.converged = false;
        return result;
    }
    result.beta = ldlt_A.solve(b_vec);

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
    result.cov_beta = result.sigma2 * ldlt_A.solve(MatrixXd::Identity(p, p));
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
// Fit q=1 model via L-BFGS-B (LBFGSPP)
// -------------------------------------------------------------------------

LMEFitResult LMESolver::fit_q1_lbfgsb(
    const SufficientStatsQ1& stats,
    bool reml,
    double warm_lambda_sq
) {
    // Wrap the q=1 profiled deviance as a 1D function for LBFGSB
    // theta = [lambda] where lambda_sq = lambda^2, lambda >= 0
    // We optimize over lambda (not lambda_sq) to match the
    // parameterization used in the general solver (theta = lower-triangular
    // Cholesky factor elements). For q=1, theta has just one element.

    auto objective = [this, &stats, reml](const Eigen::VectorXd& theta) -> double {
        double lambda = theta(0);
        double lam_sq = lambda * lambda;
        return this->profiled_deviance_q1(lam_sq, stats, reml);
    };

    // Bounds: lambda >= 0 (diagonal element of Cholesky factor)
    Eigen::VectorXd x0(1), lb(1), ub(1);
    lb(0) = 0.0;
    ub(0) = 1e3;  // upper bound for lambda (1e6 for lambda_sq)

    if (warm_lambda_sq > 0.0) {
        x0(0) = std::sqrt(warm_lambda_sq);
    } else {
        x0(0) = 1.0;  // default: lambda = 1 (ICC ~= 0.5)
    }

    LBFGSBResult opt_result = lbfgsb_minimize_fd(
        objective, x0, lb, ub,
        /* maxiter */ 200,
        /* ftol */ 1e-10,
        /* epsilon */ 1e-6
    );

    double lam_sq_opt = opt_result.x(0) * opt_result.x(0);
    LMEFitResult result = extract_results_q1(lam_sq_opt, stats, reml);
    result.converged = opt_result.converged;
    return result;
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

        double y_mean = y.mean();
        double ss_null = (y.array() - y_mean).square().sum();
        double sigma2_null = ss_null / n;

        if (sigma2_full <= 0.0 || sigma2_null <= 0.0) {
            wald_fallback = true;
        } else {
            double llf_full = -0.5 * n * (1.0 + std::log(2.0 * M_PI) + std::log(sigma2_full));
            double llf_null = -0.5 * n * (1.0 + std::log(2.0 * M_PI) + std::log(sigma2_null));

            double lr_stat = 2.0 * (llf_full - llf_null);
            if (std::isnan(lr_stat) || lr_stat < 0.0 || !std::isfinite(lr_stat)) {
                wald_fallback = true;
            } else {
                f_significant = (lr_stat > chi2_crit) ? 1.0 : 0.0;
            }
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

    // Multiple comparison corrections (shared helper)
    apply_corrections(results, z_abs, p, target_indices, n_targets,
                      correction_z_crits, correction_method);

    // Wald flag (last element)
    results(1 + 2 * n_targets) = wald_fallback ? 1.0 : 0.0;

    return results;
}

// =========================================================================
// Shared utilities
// =========================================================================

Eigen::MatrixXd LMESolver::theta_to_T(const VectorXd& theta_vec, int q) {
    MatrixXd T = MatrixXd::Zero(q, q);
    int idx = 0;
    for (int j = 0; j < q; ++j) {
        for (int i = j; i < q; ++i) {
            T(i, j) = theta_vec(idx);
            ++idx;
        }
    }
    return T;
}

void LMESolver::apply_corrections(
    VectorXd& results,
    const VectorXd& z_abs,
    int p,
    const Eigen::Ref<const VectorXi>& target_indices,
    int n_targets,
    const Eigen::Ref<const VectorXd>& correction_z_crits,
    int correction_method
) {
    if (correction_method == 0 || correction_method == 1) {
        for (int i = 0; i < n_targets; ++i) {
            int coef_idx = target_indices(i);
            if (coef_idx < p) {
                results(1 + n_targets + i) =
                    (z_abs(coef_idx) > correction_z_crits(i)) ? 1.0 : 0.0;
            }
        }
    } else if (correction_method == 2) {
        // FDR (Benjamini-Hochberg): step-up
        VectorXd target_z = VectorXd::Zero(n_targets);
        for (int i = 0; i < n_targets; ++i) {
            int coef_idx = target_indices(i);
            if (coef_idx < p) target_z(i) = z_abs(coef_idx);
        }
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
        // Holm: step-down
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
        results.segment(1 + n_targets, n_targets) = results.segment(1, n_targets);
    }
}

// =========================================================================
// General solver (q>1, random slopes)
// =========================================================================

SufficientStatsGeneral LMESolver::compute_sufficient_stats_general(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const MatrixXd>& Z,
    const Eigen::Ref<const VectorXi>& cluster_ids,
    int K, int q
) {
    const int N = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());

    SufficientStatsGeneral stats;
    stats.K = K;
    stats.q = q;
    stats.p = p;
    stats.N = N;

    stats.cluster_sizes = VectorXi::Zero(K);
    stats.Zty = MatrixXd::Zero(K, q);
    stats.Xty = MatrixXd::Zero(K, p);
    stats.yty = VectorXd::Zero(K);

    stats.ZtZ.resize(K);
    stats.ZtX.resize(K);
    stats.XtX.resize(K);
    for (int j = 0; j < K; ++j) {
        stats.ZtZ[j] = MatrixXd::Zero(q, q);
        stats.ZtX[j] = MatrixXd::Zero(q, p);
        stats.XtX[j] = MatrixXd::Zero(p, p);
    }

    // Validate cluster_ids are in [0, K)
    for (int i = 0; i < N; ++i) {
        if (cluster_ids(i) < 0 || cluster_ids(i) >= K) {
            throw std::invalid_argument("cluster_ids contains value outside [0, K)");
        }
    }

    // Single pass over data
    for (int i = 0; i < N; ++i) {
        const int j = cluster_ids(i);
        const double yi = y(i);

        stats.cluster_sizes(j) += 1;
        stats.yty(j) += yi * yi;

        // Zty_j += z_i * y_i
        for (int a = 0; a < q; ++a) {
            stats.Zty(j, a) += Z(i, a) * yi;
        }

        // ZtZ_j += z_i @ z_i' (symmetric)
        for (int a = 0; a < q; ++a) {
            for (int b = a; b < q; ++b) {
                double val = Z(i, a) * Z(i, b);
                stats.ZtZ[j](a, b) += val;
                if (a != b) stats.ZtZ[j](b, a) += val;
            }
        }

        // ZtX_j += z_i @ x_i'
        for (int a = 0; a < q; ++a) {
            for (int c = 0; c < p; ++c) {
                stats.ZtX[j](a, c) += Z(i, a) * X(i, c);
            }
        }

        // Xty_j += x_i * y_i
        for (int c = 0; c < p; ++c) {
            stats.Xty(j, c) += X(i, c) * yi;
        }

        // XtX_j += x_i @ x_i' (symmetric)
        for (int c = 0; c < p; ++c) {
            for (int c2 = c; c2 < p; ++c2) {
                double val = X(i, c) * X(i, c2);
                stats.XtX[j](c, c2) += val;
                if (c != c2) stats.XtX[j](c2, c) += val;
            }
        }
    }

    return stats;
}

double LMESolver::profiled_deviance_general(
    const VectorXd& theta_vec,
    const SufficientStatsGeneral& stats,
    bool reml
) {
    const int K = stats.K;
    const int q = stats.q;
    const int p = stats.p;
    const int N = stats.N;

    // Reconstruct T from theta_vec
    MatrixXd T = theta_to_T(theta_vec, q);

    MatrixXd A = MatrixXd::Zero(p, p);
    VectorXd b = VectorXd::Zero(p);
    double c = 0.0;
    double log_det_Ltheta = 0.0;

    for (int j = 0; j < K; ++j) {
        // M_j = T' ZtZ_j T + I_q
        MatrixXd TtZtZ = T.transpose() * stats.ZtZ[j];
        MatrixXd M_j = TtZtZ * T + MatrixXd::Identity(q, q);

        // Cholesky of M_j
        Eigen::LLT<MatrixXd> llt_M(M_j);
        if (llt_M.info() != Eigen::Success) {
            return LME_DEVIANCE_FAIL;
        }

        // cu_j = L_j^{-1} T' Zty_j
        VectorXd TtZty = T.transpose() * stats.Zty.row(j).transpose();
        VectorXd cu_j = llt_M.matrixL().solve(TtZty);

        // CX_j = L_j^{-1} T' ZtX_j  (q × p)
        MatrixXd TtZtX = T.transpose() * stats.ZtX[j];
        MatrixXd CX_j = llt_M.matrixL().solve(TtZtX);

        // Accumulate
        A.noalias() += stats.XtX[j] - CX_j.transpose() * CX_j;
        b.noalias() += stats.Xty.row(j).transpose() - CX_j.transpose() * cu_j;
        c += stats.yty(j) - cu_j.dot(cu_j);

        // log det contribution: 2 * sum(log(diag(L_j)))
        const MatrixXd& L_j = llt_M.matrixL();
        for (int i = 0; i < q; ++i) {
            log_det_Ltheta += 2.0 * std::log(L_j(i, i));
        }
    }

    // Solve A beta = b via Cholesky
    Eigen::LLT<MatrixXd> llt_A(A);
    if (llt_A.info() != Eigen::Success) {
        return LME_DEVIANCE_FAIL;
    }

    VectorXd beta = llt_A.solve(b);
    double r_sq = c - beta.dot(b);
    if (r_sq <= 0.0) {
        return LME_DEVIANCE_FAIL;
    }

    if (reml) {
        const MatrixXd& R_X = llt_A.matrixL();
        double log_det_RX = 0.0;
        for (int i = 0; i < p; ++i) {
            log_det_RX += std::log(R_X(i, i));
        }
        return log_det_Ltheta + 2.0 * log_det_RX + static_cast<double>(N - p) * std::log(r_sq);
    } else {
        return log_det_Ltheta + static_cast<double>(N) * std::log(r_sq);
    }
}

LMEFitResultGeneral LMESolver::extract_results_general(
    const VectorXd& theta_opt,
    const SufficientStatsGeneral& stats,
    bool reml
) {
    const int K = stats.K;
    const int q = stats.q;
    const int p = stats.p;
    const int N = stats.N;

    MatrixXd T = theta_to_T(theta_opt, q);

    MatrixXd A = MatrixXd::Zero(p, p);
    VectorXd b = VectorXd::Zero(p);
    double c = 0.0;
    double log_det_Ltheta = 0.0;

    for (int j = 0; j < K; ++j) {
        MatrixXd TtZtZ = T.transpose() * stats.ZtZ[j];
        MatrixXd M_j = TtZtZ * T + MatrixXd::Identity(q, q);
        Eigen::LLT<MatrixXd> llt_M(M_j);

        if (llt_M.info() != Eigen::Success) {
            LMEFitResultGeneral fail_result;
            fail_result.converged = false;
            return fail_result;
        }

        VectorXd TtZty = T.transpose() * stats.Zty.row(j).transpose();
        VectorXd cu_j = llt_M.matrixL().solve(TtZty);
        MatrixXd TtZtX = T.transpose() * stats.ZtX[j];
        MatrixXd CX_j = llt_M.matrixL().solve(TtZtX);

        A.noalias() += stats.XtX[j] - CX_j.transpose() * CX_j;
        b.noalias() += stats.Xty.row(j).transpose() - CX_j.transpose() * cu_j;
        c += stats.yty(j) - cu_j.dot(cu_j);

        const MatrixXd& L_j = llt_M.matrixL();
        for (int i = 0; i < q; ++i) {
            log_det_Ltheta += 2.0 * std::log(L_j(i, i));
        }
    }

    LMEFitResultGeneral result;
    result.theta = theta_opt;

    // Solve for beta via LDLT (computed once, reused for cov_beta)
    Eigen::LDLT<MatrixXd> ldlt_A(A);
    if (ldlt_A.info() != Eigen::Success) {
        result.converged = false;
        return result;
    }
    result.beta = ldlt_A.solve(b);
    double r_sq = c - result.beta.dot(b);

    if (reml) {
        result.sigma2 = r_sq / static_cast<double>(N - p);
    } else {
        result.sigma2 = r_sq / static_cast<double>(N);
    }

    // G = T * T' (relative covariance)
    result.G = T * T.transpose();
    result.tau2 = result.sigma2 * result.G(0, 0);

    // Covariance of beta
    result.cov_beta = result.sigma2 * ldlt_A.solve(MatrixXd::Identity(p, p));
    result.se_beta = result.cov_beta.diagonal().cwiseMax(0.0).cwiseSqrt();

    // ML log-likelihood
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

LMEFitResultGeneral LMESolver::fit_general(
    const SufficientStatsGeneral& stats,
    bool reml,
    const VectorXd& warm_theta
) {
    const int q = stats.q;
    const int n_theta = q * (q + 1) / 2;

    auto objective = [this, &stats, reml](const Eigen::VectorXd& theta) -> double {
        return this->profiled_deviance_general(theta, stats, reml);
    };

    // Bounds: diagonal elements >= 0, off-diagonal free
    VectorXd lb(n_theta), ub(n_theta);
    int idx = 0;
    for (int j = 0; j < q; ++j) {
        for (int i = j; i < q; ++i) {
            if (i == j) {
                lb(idx) = 0.0;
                ub(idx) = 1e4;
            } else {
                lb(idx) = -1e4;
                ub(idx) = 1e4;
            }
            ++idx;
        }
    }

    // Initial guess
    VectorXd x0(n_theta);
    if (warm_theta.size() == n_theta) {
        x0 = warm_theta;
    } else {
        x0.setZero();
        idx = 0;
        for (int j = 0; j < q; ++j) {
            for (int i = j; i < q; ++i) {
                if (i == j) x0(idx) = 1.0;
                ++idx;
            }
        }
    }

    LBFGSBResult opt_result = lbfgsb_minimize_fd(
        objective, x0, lb, ub,
        /* maxiter */ 200,
        /* ftol */ 1e-10,
        /* epsilon */ 1e-6
    );

    LMEFitResultGeneral result = extract_results_general(opt_result.x, stats, reml);
    result.converged = opt_result.converged;
    return result;
}

Eigen::VectorXd LMESolver::analyze_general(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const MatrixXd>& Z,
    const Eigen::Ref<const VectorXi>& cluster_ids,
    int n_clusters, int q,
    const Eigen::Ref<const VectorXi>& target_indices,
    double chi2_crit,
    double z_crit,
    const Eigen::Ref<const VectorXd>& correction_z_crits,
    int correction_method,
    const Eigen::Ref<const VectorXd>& warm_theta
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

    // Compute sufficient stats
    SufficientStatsGeneral stats = compute_sufficient_stats_general(X_int, y, Z, cluster_ids, K, q);

    // REML fit
    VectorXd warm_theta_copy = warm_theta;
    LMEFitResultGeneral reml_result = fit_general(stats, true, warm_theta_copy);
    if (!reml_result.converged) {
        return VectorXd();
    }

    // Extract fixed effects (skip intercept)
    VectorXd beta = reml_result.beta.tail(p);
    VectorXd se = reml_result.se_beta.tail(p);

    // Wald z-tests
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

    // LR test for overall significance
    double f_significant = 0.0;
    bool wald_fallback = false;

    // Full ML fit (general)
    LMEFitResultGeneral full_ml = fit_general(stats, false, reml_result.theta);
    if (!full_ml.converged) {
        wald_fallback = true;
    } else {
        // Null ML fit (q=1 intercept only, using existing q=1 solver)
        MatrixXd X_null = MatrixXd::Ones(n, 1);
        SufficientStatsQ1 null_stats = compute_sufficient_stats(X_null, y, cluster_ids, K);
        LMEFitResult null_ml = fit_q1(null_stats, false);

        double lr_stat = 2.0 * (full_ml.log_likelihood - null_ml.log_likelihood);
        if (std::isnan(lr_stat) || lr_stat < 0.0 || !std::isfinite(lr_stat)) {
            wald_fallback = true;
        } else {
            f_significant = (lr_stat > chi2_crit) ? 1.0 : 0.0;
        }
    }

    // Wald fallback
    if (wald_fallback) {
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
    apply_corrections(results, z_abs, p, target_indices, n_targets,
                      correction_z_crits, correction_method);

    // Wald flag
    results(1 + 2 * n_targets) = wald_fallback ? 1.0 : 0.0;

    return results;
}

// =========================================================================
// Nested solver (two-level random intercepts)
// =========================================================================

NestedSufficientStats LMESolver::compute_sufficient_stats_nested(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXi>& parent_ids,
    const Eigen::Ref<const VectorXi>& child_ids,
    int K_parent, int K_child,
    const Eigen::Ref<const VectorXi>& child_to_parent
) {
    const int N = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());

    NestedSufficientStats nstats;
    nstats.K_parent = K_parent;
    nstats.K_child = K_child;
    nstats.p = p;
    nstats.N = N;

    nstats.parent_sizes = VectorXi::Zero(K_parent);
    nstats.child_sizes = VectorXi::Zero(K_child);
    nstats.child_to_parent = child_to_parent;

    nstats.child_Xty = MatrixXd::Zero(K_child, p);
    nstats.child_yty = VectorXd::Zero(K_child);
    nstats.child_Xt1 = MatrixXd::Zero(K_child, p);
    nstats.child_1ty = VectorXd::Zero(K_child);
    nstats.child_1t1 = VectorXd::Zero(K_child);

    nstats.child_XtX.resize(K_child);
    for (int cj = 0; cj < K_child; ++cj) {
        nstats.child_XtX[cj] = MatrixXd::Zero(p, p);
    }

    // Validate parent_ids in [0, K_parent) and child_ids in [0, K_child)
    for (int i = 0; i < N; ++i) {
        if (child_ids(i) < 0 || child_ids(i) >= K_child) {
            throw std::invalid_argument("child_ids contains value outside [0, K_child)");
        }
        if (parent_ids(i) < 0 || parent_ids(i) >= K_parent) {
            throw std::invalid_argument("parent_ids contains value outside [0, K_parent)");
        }
    }

    for (int i = 0; i < N; ++i) {
        const int cj = child_ids(i);
        const int pj = parent_ids(i);
        const double yi = y(i);

        nstats.child_sizes(cj) += 1;
        nstats.parent_sizes(pj) += 1;
        nstats.child_yty(cj) += yi * yi;
        nstats.child_1ty(cj) += yi;

        for (int c = 0; c < p; ++c) {
            const double xic = X(i, c);
            nstats.child_Xty(cj, c) += xic * yi;
            nstats.child_Xt1(cj, c) += xic;

            for (int c2 = c; c2 < p; ++c2) {
                double val = xic * X(i, c2);
                nstats.child_XtX[cj](c, c2) += val;
                if (c != c2) nstats.child_XtX[cj](c2, c) += val;
            }
        }
    }

    // child_1t1 = child_sizes as double
    for (int cj = 0; cj < K_child; ++cj) {
        nstats.child_1t1(cj) = static_cast<double>(nstats.child_sizes(cj));
    }

    return nstats;
}

double LMESolver::profiled_deviance_nested(
    const VectorXd& theta_vec,
    const NestedSufficientStats& nstats,
    bool reml
) {
    const double theta_parent = theta_vec(0);
    const double theta_child = theta_vec(1);
    const double lam_sq_parent = theta_parent * theta_parent;
    const double lam_sq_child = theta_child * theta_child;

    const int p = nstats.p;
    const int N = nstats.N;
    const int K_parent = nstats.K_parent;
    const int K_child = nstats.K_child;

    MatrixXd A = MatrixXd::Zero(p, p);
    VectorXd b_vec = VectorXd::Zero(p);
    double c_val = 0.0;
    double log_det = 0.0;

    // Per-parent accumulators (after absorbing child effects)
    std::vector<MatrixXd> parent_XtX_adj(K_parent, MatrixXd::Zero(p, p));
    MatrixXd parent_Xty_adj = MatrixXd::Zero(K_parent, p);
    VectorXd parent_yty_adj = VectorXd::Zero(K_parent);
    MatrixXd parent_Xt1_adj = MatrixXd::Zero(K_parent, p);
    VectorXd parent_1ty_adj = VectorXd::Zero(K_parent);
    VectorXd parent_1t1_adj = VectorXd::Zero(K_parent);

    // Pass 1: Absorb child-level effects (scalar per child)
    for (int cj = 0; cj < K_child; ++cj) {
        const double nc = nstats.child_1t1(cj);
        const int pj = nstats.child_to_parent(cj);

        const double Mc = lam_sq_child * nc + 1.0;
        const double Lc = std::sqrt(Mc);
        const double inv_Mc = 1.0 / Mc;
        const double factor = lam_sq_child * inv_Mc;

        // Absorb child: subtract rank-1 update
        for (int c1 = 0; c1 < p; ++c1) {
            for (int c2 = 0; c2 < p; ++c2) {
                parent_XtX_adj[pj](c1, c2) +=
                    nstats.child_XtX[cj](c1, c2) -
                    factor * nstats.child_Xt1(cj, c1) * nstats.child_Xt1(cj, c2);
            }
            parent_Xty_adj(pj, c1) +=
                nstats.child_Xty(cj, c1) -
                factor * nstats.child_Xt1(cj, c1) * nstats.child_1ty(cj);

            // Child's contribution to parent's "1" vector
            parent_Xt1_adj(pj, c1) +=
                nstats.child_Xt1(cj, c1) - factor * nstats.child_Xt1(cj, c1) * nc;
        }
        parent_yty_adj(pj) +=
            nstats.child_yty(cj) - factor * nstats.child_1ty(cj) * nstats.child_1ty(cj);
        parent_1ty_adj(pj) +=
            nstats.child_1ty(cj) - factor * nstats.child_1ty(cj) * nc;
        parent_1t1_adj(pj) += nc - factor * nc * nc;

        log_det += std::log(Lc);
    }

    // Pass 2: Absorb parent-level effects (scalar per parent)
    for (int pj = 0; pj < K_parent; ++pj) {
        const double ns_adj = parent_1t1_adj(pj);
        const double Mp = lam_sq_parent * ns_adj + 1.0;
        if (Mp <= 0.0) return LME_DEVIANCE_FAIL;
        const double Lp = std::sqrt(Mp);
        const double inv_Mp = 1.0 / Mp;
        const double factor_p = lam_sq_parent * inv_Mp;

        for (int c1 = 0; c1 < p; ++c1) {
            for (int c2 = 0; c2 < p; ++c2) {
                A(c1, c2) += parent_XtX_adj[pj](c1, c2) -
                    factor_p * parent_Xt1_adj(pj, c1) * parent_Xt1_adj(pj, c2);
            }
            b_vec(c1) += parent_Xty_adj(pj, c1) -
                factor_p * parent_Xt1_adj(pj, c1) * parent_1ty_adj(pj);
        }
        c_val += parent_yty_adj(pj) - factor_p * parent_1ty_adj(pj) * parent_1ty_adj(pj);
        log_det += std::log(Lp);
    }

    const double log_det_total = 2.0 * log_det;

    // Solve A beta = b
    Eigen::LLT<MatrixXd> llt_A(A);
    if (llt_A.info() != Eigen::Success) {
        return LME_DEVIANCE_FAIL;
    }

    VectorXd z_vec = llt_A.matrixL().solve(b_vec);
    VectorXd beta = llt_A.matrixL().transpose().solve(z_vec);

    double r_sq = c_val - beta.dot(b_vec);
    if (r_sq <= 0.0) return LME_DEVIANCE_FAIL;

    if (reml) {
        const MatrixXd& R_X = llt_A.matrixL();
        double log_det_RX = 0.0;
        for (int i = 0; i < p; ++i) {
            log_det_RX += std::log(R_X(i, i));
        }
        return log_det_total + 2.0 * log_det_RX + static_cast<double>(N - p) * std::log(r_sq);
    } else {
        return log_det_total + static_cast<double>(N) * std::log(r_sq);
    }
}

LMEFitResultGeneral LMESolver::extract_results_nested(
    const VectorXd& theta_opt,
    const NestedSufficientStats& nstats,
    bool reml
) {
    const double theta_parent = theta_opt(0);
    const double theta_child = theta_opt(1);
    const double lam_sq_parent = theta_parent * theta_parent;
    const double lam_sq_child = theta_child * theta_child;

    const int p = nstats.p;
    const int N = nstats.N;
    const int K_parent = nstats.K_parent;
    const int K_child = nstats.K_child;

    MatrixXd A = MatrixXd::Zero(p, p);
    VectorXd b_vec = VectorXd::Zero(p);
    double c_val = 0.0;
    double log_det = 0.0;

    std::vector<MatrixXd> parent_XtX_adj(K_parent, MatrixXd::Zero(p, p));
    MatrixXd parent_Xty_adj = MatrixXd::Zero(K_parent, p);
    VectorXd parent_yty_adj = VectorXd::Zero(K_parent);
    MatrixXd parent_Xt1_adj = MatrixXd::Zero(K_parent, p);
    VectorXd parent_1ty_adj = VectorXd::Zero(K_parent);
    VectorXd parent_1t1_adj = VectorXd::Zero(K_parent);

    for (int cj = 0; cj < K_child; ++cj) {
        const double nc = nstats.child_1t1(cj);
        const int pj = nstats.child_to_parent(cj);
        const double Mc = lam_sq_child * nc + 1.0;
        const double inv_Mc = 1.0 / Mc;
        const double factor = lam_sq_child * inv_Mc;

        for (int c1 = 0; c1 < p; ++c1) {
            for (int c2 = 0; c2 < p; ++c2) {
                parent_XtX_adj[pj](c1, c2) +=
                    nstats.child_XtX[cj](c1, c2) -
                    factor * nstats.child_Xt1(cj, c1) * nstats.child_Xt1(cj, c2);
            }
            parent_Xty_adj(pj, c1) +=
                nstats.child_Xty(cj, c1) -
                factor * nstats.child_Xt1(cj, c1) * nstats.child_1ty(cj);
            parent_Xt1_adj(pj, c1) +=
                nstats.child_Xt1(cj, c1) - factor * nstats.child_Xt1(cj, c1) * nc;
        }
        parent_yty_adj(pj) += nstats.child_yty(cj) - factor * nstats.child_1ty(cj) * nstats.child_1ty(cj);
        parent_1ty_adj(pj) += nstats.child_1ty(cj) - factor * nstats.child_1ty(cj) * nc;
        parent_1t1_adj(pj) += nc - factor * nc * nc;
        log_det += std::log(std::sqrt(Mc));
    }

    for (int pj = 0; pj < K_parent; ++pj) {
        const double ns_adj = parent_1t1_adj(pj);
        const double Mp = lam_sq_parent * ns_adj + 1.0;
        const double inv_Mp = 1.0 / Mp;
        const double factor_p = lam_sq_parent * inv_Mp;

        for (int c1 = 0; c1 < p; ++c1) {
            for (int c2 = 0; c2 < p; ++c2) {
                A(c1, c2) += parent_XtX_adj[pj](c1, c2) -
                    factor_p * parent_Xt1_adj(pj, c1) * parent_Xt1_adj(pj, c2);
            }
            b_vec(c1) += parent_Xty_adj(pj, c1) -
                factor_p * parent_Xt1_adj(pj, c1) * parent_1ty_adj(pj);
        }
        c_val += parent_yty_adj(pj) - factor_p * parent_1ty_adj(pj) * parent_1ty_adj(pj);
        log_det += std::log(std::sqrt(Mp));
    }

    LMEFitResultGeneral result;
    result.theta = theta_opt;

    // Solve for beta via LDLT (computed once, reused for cov_beta)
    Eigen::LDLT<MatrixXd> ldlt_A(A);
    if (ldlt_A.info() != Eigen::Success) {
        result.converged = false;
        return result;
    }
    result.beta = ldlt_A.solve(b_vec);
    double r_sq = c_val - result.beta.dot(b_vec);

    if (reml) {
        result.sigma2 = r_sq / static_cast<double>(N - p);
    } else {
        result.sigma2 = r_sq / static_cast<double>(N);
    }

    result.tau2 = result.sigma2 * lam_sq_parent;

    // G = diag(lam_sq_parent, lam_sq_child) for nested
    result.G = MatrixXd::Zero(2, 2);
    result.G(0, 0) = lam_sq_parent;
    result.G(1, 1) = lam_sq_child;

    result.cov_beta = result.sigma2 * ldlt_A.solve(MatrixXd::Identity(p, p));
    result.se_beta = result.cov_beta.diagonal().cwiseMax(0.0).cwiseSqrt();

    double sigma2_ml = r_sq / static_cast<double>(N);
    double log_det_total = 2.0 * log_det;
    if (sigma2_ml > 0.0) {
        result.log_likelihood = -0.5 * (log_det_total +
            static_cast<double>(N) +
            static_cast<double>(N) * std::log(2.0 * M_PI) +
            static_cast<double>(N) * std::log(sigma2_ml));
    } else {
        result.log_likelihood = -std::numeric_limits<double>::infinity();
    }

    result.converged = true;
    return result;
}

LMEFitResultGeneral LMESolver::fit_nested(
    const NestedSufficientStats& nstats,
    bool reml,
    const VectorXd& warm_theta
) {
    auto objective = [this, &nstats, reml](const Eigen::VectorXd& theta) -> double {
        return this->profiled_deviance_nested(theta, nstats, reml);
    };

    VectorXd lb(2), ub(2);
    lb(0) = 0.0; lb(1) = 0.0;
    ub(0) = 1e3; ub(1) = 1e3;

    VectorXd x0(2);
    if (warm_theta.size() == 2) {
        x0 = warm_theta;
    } else {
        x0(0) = 1.0;
        x0(1) = 1.0;
    }

    LBFGSBResult opt_result = lbfgsb_minimize_fd(
        objective, x0, lb, ub,
        /* maxiter */ 200,
        /* ftol */ 1e-10,
        /* epsilon */ 1e-6
    );

    LMEFitResultGeneral result = extract_results_nested(opt_result.x, nstats, reml);
    result.converged = opt_result.converged;
    return result;
}

Eigen::VectorXd LMESolver::analyze_nested(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXi>& parent_ids,
    const Eigen::Ref<const VectorXi>& child_ids,
    int K_parent, int K_child,
    const Eigen::Ref<const VectorXi>& child_to_parent,
    const Eigen::Ref<const VectorXi>& target_indices,
    double chi2_crit,
    double z_crit,
    const Eigen::Ref<const VectorXd>& correction_z_crits,
    int correction_method,
    const Eigen::Ref<const VectorXd>& warm_theta
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    const int n_targets = static_cast<int>(target_indices.size());

    VectorXd results = VectorXd::Zero(1 + 2 * n_targets + 1);

    // Add intercept column
    MatrixXd X_int(n, p + 1);
    X_int.col(0) = VectorXd::Ones(n);
    X_int.rightCols(p) = X;

    // Compute sufficient stats
    NestedSufficientStats nstats = compute_sufficient_stats_nested(
        X_int, y, parent_ids, child_ids, K_parent, K_child, child_to_parent);

    // REML fit
    VectorXd warm_theta_copy = warm_theta;
    LMEFitResultGeneral reml_result = fit_nested(nstats, true, warm_theta_copy);
    if (!reml_result.converged) {
        return VectorXd();
    }

    // Extract fixed effects (skip intercept)
    VectorXd beta = reml_result.beta.tail(p);
    VectorXd se = reml_result.se_beta.tail(p);

    // Wald z-tests
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

    // LR test
    double f_significant = 0.0;
    bool wald_fallback = false;

    // Full ML (nested)
    LMEFitResultGeneral full_ml = fit_nested(nstats, false, reml_result.theta);
    if (!full_ml.converged) {
        wald_fallback = true;
    } else {
        // Null ML: intercept only, still nested
        MatrixXd X_null = MatrixXd::Ones(n, 1);
        NestedSufficientStats null_nstats = compute_sufficient_stats_nested(
            X_null, y, parent_ids, child_ids, K_parent, K_child, child_to_parent);
        LMEFitResultGeneral null_ml = fit_nested(null_nstats, false);

        double lr_stat = 2.0 * (full_ml.log_likelihood - null_ml.log_likelihood);
        if (std::isnan(lr_stat) || lr_stat < 0.0 || !std::isfinite(lr_stat)) {
            wald_fallback = true;
        } else {
            f_significant = (lr_stat > chi2_crit) ? 1.0 : 0.0;
        }
    }

    if (wald_fallback) {
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
    apply_corrections(results, z_abs, p, target_indices, n_targets,
                      correction_z_crits, correction_method);

    // Wald flag
    results(1 + 2 * n_targets) = wald_fallback ? 1.0 : 0.0;

    return results;
}

}  // namespace mcpower
