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
    int K = 0;    // number of clusters
    int p = 0;    // fixed effects (including intercept)
    int N = 0;    // total observations

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
    double sigma2 = 0.0;      // residual variance
    double tau2 = 0.0;        // random intercept variance
    double lambda_sq = 0.0;   // optimal lambda^2 (theta^2)
    Eigen::MatrixXd cov_beta; // (p, p) covariance of fixed effects
    Eigen::VectorXd se_beta;  // (p,) standard errors
    double log_likelihood = -std::numeric_limits<double>::infinity();
    bool converged = false;
};

/**
 * Per-cluster sufficient statistics for general LME (q>1, random slopes).
 *
 * For general q, Z_j is (n_j, q), so:
 *   ZtZ_j = Z_j' Z_j  (q×q matrix)
 *   ZtX_j = Z_j' X_j  (q×p matrix)
 *   Zty_j = Z_j' y_j  (q-vector)
 */
struct SufficientStatsGeneral {
    int K = 0;    // number of clusters
    int q = 0;    // random effects per cluster
    int p = 0;    // fixed effects (including intercept)
    int N = 0;    // total observations

    Eigen::VectorXi cluster_sizes;           // (K,)
    std::vector<Eigen::MatrixXd> ZtZ;        // K × (q,q)
    std::vector<Eigen::MatrixXd> ZtX;        // K × (q,p)
    Eigen::MatrixXd Zty;                     // (K,q)
    std::vector<Eigen::MatrixXd> XtX;        // K × (p,p)
    Eigen::MatrixXd Xty;                     // (K,p)
    Eigen::VectorXd yty;                     // (K,)
};

/**
 * Result of general LME model fitting (q>1).
 */
struct LMEFitResultGeneral {
    Eigen::VectorXd beta;      // (p,)
    double sigma2 = 0.0;       // residual variance
    double tau2 = 0.0;         // random intercept variance (G[0,0] * sigma2)
    Eigen::MatrixXd G;         // (q,q) relative covariance (T*T')
    Eigen::VectorXd theta;     // Cholesky entries, q*(q+1)/2
    Eigen::MatrixXd cov_beta;  // (p,p)
    Eigen::VectorXd se_beta;   // (p,)
    double log_likelihood = -std::numeric_limits<double>::infinity();
    bool converged = false;
};

/**
 * Sufficient statistics for nested random intercepts.
 *
 * Two-level nesting: (1|parent) + (1|parent:child).
 * All per-child and per-parent stats are precomputed.
 */
struct NestedSufficientStats {
    int K_parent = 0;
    int K_child = 0;
    int p = 0;
    int N = 0;

    Eigen::VectorXi parent_sizes;     // (K_parent,)
    Eigen::VectorXi child_sizes;      // (K_child,)
    Eigen::VectorXi child_to_parent;  // (K_child,) maps child -> parent

    // Per-child statistics
    std::vector<Eigen::MatrixXd> child_XtX;  // K_child × (p,p)
    Eigen::MatrixXd child_Xty;               // (K_child, p)
    Eigen::VectorXd child_yty;               // (K_child,)
    Eigen::MatrixXd child_Xt1;               // (K_child, p)
    Eigen::VectorXd child_1ty;               // (K_child,)
    Eigen::VectorXd child_1t1;               // (K_child,) = child_sizes as double
};

/**
 * Custom LME solver for linear mixed models.
 *
 * Implements REML/ML estimation via profiled deviance optimization
 * following Bates et al. (2015). Three solver paths:
 * - q=1 (random intercept): 1D Brent optimization
 * - q>1 (random slopes): L-BFGS-B over theta (Cholesky entries)
 * - Nested intercepts: L-BFGS-B over [theta_parent, theta_child]
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
     * Fit q=1 model via L-BFGS-B (LBFGSPP) instead of Brent's method.
     *
     * This wraps the 1D profiled deviance in a 1D L-BFGS-B problem.
     * Useful as a validation path and for consistency with the
     * general/nested solvers that also use L-BFGS-B.
     *
     * Uses the same sufficient statistics and deviance function as fit_q1,
     * but optimizes with LBFGSPP + finite differences instead of Brent.
     */
    LMEFitResult fit_q1_lbfgsb(
        const SufficientStatsQ1& stats,
        bool reml,
        double warm_lambda_sq = -1.0
    );

    /**
     * Full LME analysis returning results array (q=1).
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

    // ===== General solver (q>1, random slopes) =====

    /**
     * Compute per-cluster sufficient statistics for general q.
     */
    SufficientStatsGeneral compute_sufficient_stats_general(
        const Eigen::Ref<const MatrixXd>& X,
        const Eigen::Ref<const VectorXd>& y,
        const Eigen::Ref<const MatrixXd>& Z,
        const Eigen::Ref<const VectorXi>& cluster_ids,
        int K, int q
    );

    /**
     * Evaluate profiled REML/ML deviance for general q.
     *
     * @param theta_vec Lower-triangular Cholesky entries, q*(q+1)/2
     * @param stats Precomputed sufficient statistics
     * @param reml true for REML, false for ML
     * @return Profiled deviance value
     */
    double profiled_deviance_general(
        const VectorXd& theta_vec,
        const SufficientStatsGeneral& stats,
        bool reml
    );

    /**
     * Fit general q>1 model via L-BFGS-B over theta.
     */
    LMEFitResultGeneral fit_general(
        const SufficientStatsGeneral& stats,
        bool reml,
        const VectorXd& warm_theta = VectorXd()
    );

    /**
     * Full LME analysis for general q>1 (random slopes).
     *
     * @param X Design matrix (n, p), no intercept column
     * @param y Response vector (n,)
     * @param Z Random effects design matrix (n, q)
     * @param cluster_ids Cluster membership (n,)
     * @param n_clusters Number of clusters
     * @param q Random effects dimension
     * @param target_indices Coefficient indices to test
     * @param chi2_crit, z_crit, correction_z_crits, correction_method As in analyze()
     * @param warm_theta Previous optimal theta vector (empty for cold start)
     * @return Results vector, or empty vector on failure
     */
    VectorXd analyze_general(
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
    );

    // ===== Nested solver (two-level random intercepts) =====

    /**
     * Compute sufficient statistics for nested random intercepts.
     */
    NestedSufficientStats compute_sufficient_stats_nested(
        const Eigen::Ref<const MatrixXd>& X,
        const Eigen::Ref<const VectorXd>& y,
        const Eigen::Ref<const VectorXi>& parent_ids,
        const Eigen::Ref<const VectorXi>& child_ids,
        int K_parent, int K_child,
        const Eigen::Ref<const VectorXi>& child_to_parent
    );

    /**
     * Evaluate profiled deviance for nested random intercepts.
     *
     * @param theta_vec [theta_parent, theta_child], both >= 0
     * @param nstats Precomputed nested sufficient statistics
     * @param reml true for REML, false for ML
     * @return Profiled deviance value
     */
    double profiled_deviance_nested(
        const VectorXd& theta_vec,
        const NestedSufficientStats& nstats,
        bool reml
    );

    /**
     * Fit nested random intercepts via L-BFGS-B over [theta_parent, theta_child].
     */
    LMEFitResultGeneral fit_nested(
        const NestedSufficientStats& nstats,
        bool reml,
        const VectorXd& warm_theta = VectorXd()
    );

    /**
     * Full LME analysis for nested random intercepts.
     *
     * @param X Design matrix (n, p), no intercept column
     * @param y Response vector (n,)
     * @param parent_ids Parent cluster membership (n,)
     * @param child_ids Child cluster membership (n,)
     * @param K_parent, K_child Number of parent/child clusters
     * @param child_to_parent Mapping from child to parent (K_child,)
     * @param target_indices, chi2_crit, z_crit, correction_z_crits, correction_method As in analyze()
     * @param warm_theta Previous optimal [theta_parent, theta_child] (empty for cold start)
     * @return Results vector, or empty vector on failure
     */
    VectorXd analyze_nested(
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

    /**
     * Extract results at optimal theta for general q>1.
     */
    LMEFitResultGeneral extract_results_general(
        const VectorXd& theta_opt,
        const SufficientStatsGeneral& stats,
        bool reml
    );

    /**
     * Extract results at optimal theta for nested random intercepts.
     */
    LMEFitResultGeneral extract_results_nested(
        const VectorXd& theta_opt,
        const NestedSufficientStats& nstats,
        bool reml
    );

    /**
     * Reconstruct lower-triangular T from theta vector.
     */
    static MatrixXd theta_to_T(const VectorXd& theta_vec, int q);

    /**
     * Apply multiple comparison corrections to results array.
     * Shared by all analyze methods.
     */
    void apply_corrections(
        VectorXd& results,
        const VectorXd& z_abs,
        int p,
        const Eigen::Ref<const VectorXi>& target_indices,
        int n_targets,
        const Eigen::Ref<const VectorXd>& correction_z_crits,
        int correction_method
    );
};

}  // namespace mcpower
