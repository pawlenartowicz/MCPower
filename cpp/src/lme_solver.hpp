#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mcpower {

constexpr double LME_FLOAT_NEAR_ZERO = 1e-15;
constexpr double LME_DEVIANCE_FAIL = 1e30;

/**
 * Per-cluster sufficient statistics for random-intercept LME (q=1).
 *
 * For random intercept, Z_j = 1_{n_j}, so:
 *   ZtZ_j = n_j  (scalar)
 *   ZtX_j = colsum(X_j)  (p-vector)
 *   Zty_j = sum(y_j)  (scalar)
 */
struct SufficientStatsQ1 {
    int K;    // number of clusters
    int p;    // fixed effects (including intercept)
    int N;    // total observations

    Eigen::VectorXi cluster_sizes;  // (K,)
    Eigen::VectorXd ZtZ;            // (K,) = cluster_sizes as double
    Eigen::MatrixXd ZtX;            // (K, p)
    Eigen::VectorXd Zty;            // (K,)
    // XtX stored as vector of p×p matrices (one per cluster)
    std::vector<Eigen::MatrixXd> XtX;  // K × (p, p)
    Eigen::MatrixXd Xty;            // (K, p)
    Eigen::VectorXd yty;            // (K,)
};

/**
 * Result of LME model fitting.
 */
struct LMEFitResult {
    Eigen::VectorXd beta;     // (p,) fixed effects including intercept
    double sigma2;             // residual variance
    double tau2;               // random intercept variance
    double lambda_sq;          // optimal lambda^2 (theta^2)
    Eigen::MatrixXd cov_beta; // (p, p) covariance of fixed effects
    Eigen::VectorXd se_beta;  // (p,) standard errors
    double log_likelihood;     // ML log-likelihood at optimum
    bool converged;
};

/**
 * Custom LME solver for random-intercept linear mixed models.
 *
 * Implements REML/ML estimation via profiled deviance optimization
 * following Bates et al. (2015). For q=1 (random intercept), all
 * per-cluster operations are scalar, and optimization is 1D via
 * Brent's method.
 */
class LMESolver {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXi = Eigen::VectorXi;

    LMESolver() = default;

    /**
     * Compute per-cluster sufficient statistics for q=1.
     */
    SufficientStatsQ1 compute_sufficient_stats(
        const Eigen::Ref<const MatrixXd>& X,
        const Eigen::Ref<const VectorXd>& y,
        const Eigen::Ref<const VectorXi>& cluster_ids,
        int K
    );

    /**
     * Evaluate profiled REML/ML deviance at given lambda^2 (q=1).
     *
     * @param lam_sq  lambda^2 >= 0, the relative covariance parameter
     * @param stats   precomputed sufficient statistics
     * @param reml    true for REML, false for ML
     * @return profiled deviance value (to minimize)
     */
    double profiled_deviance_q1(
        double lam_sq,
        const SufficientStatsQ1& stats,
        bool reml
    );

    /**
     * Fit model and extract parameters at optimal lambda^2.
     */
    LMEFitResult fit_q1(
        const SufficientStatsQ1& stats,
        bool reml,
        double warm_lambda_sq = -1.0
    );

    /**
     * Full LME analysis returning results array.
     *
     * Performs REML fit, ML fits for LR test, Wald z-tests, and
     * multiple comparison corrections. Returns same format as OLS:
     * [f_significant, uncorrected_1..n, corrected_1..n, wald_flag]
     *
     * @param X Design matrix (n, p), no intercept column
     * @param y Response vector (n,)
     * @param cluster_ids Cluster membership (n,)
     * @param n_clusters Number of clusters
     * @param target_indices Coefficient indices to test
     * @param chi2_crit Critical chi-squared value for LR test
     * @param z_crit Critical z value for individual Wald tests
     * @param correction_z_crits Correction critical values array
     * @param correction_method 0=none, 1=Bonferroni, 2=FDR, 3=Holm
     * @param warm_lambda_sq Warm start (-1.0 for cold start)
     * @return Results vector, or empty vector on failure
     */
    VectorXd analyze(
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
    );

private:
    /**
     * Brent's method for 1D minimization on [a, b].
     *
     * @return optimal x minimizing f(x)
     */
    double brent_minimize(
        double a, double b, double tol, int max_iter,
        const SufficientStatsQ1& stats, bool reml
    );

    /**
     * Extract beta, sigma2, cov_beta etc. at optimal lambda^2.
     */
    LMEFitResult extract_results_q1(
        double lam_sq_opt,
        const SufficientStatsQ1& stats,
        bool reml
    );
};

}  // namespace mcpower
