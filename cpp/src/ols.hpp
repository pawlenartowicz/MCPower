#pragma once

#include <Eigen/Dense>
#include <Eigen/QR>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace mcpower {

constexpr double FLOAT_NEAR_ZERO = 1e-15;

/**
 * OLS regression analysis with F/t-tests and multiple comparison corrections.
 *
 * Uses precomputed critical values (computed once in Python via scipy)
 * instead of per-simulation p-value lookup tables.
 *
 * Implements:
 * - QR decomposition for numerical stability
 * - F-test for overall model significance
 * - t-tests for individual coefficients
 * - Multiple comparison corrections (Bonferroni, FDR, Holm)
 */
class OLSAnalyzer {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXi = Eigen::VectorXi;

    /**
     * Default constructor — no lookup tables needed.
     */
    OLSAnalyzer() = default;

    /**
     * Run OLS analysis on data.
     *
     * @param X Design matrix (n_samples, n_features), no intercept
     * @param y Response vector (n_samples,)
     * @param target_indices Coefficient indices to test
     * @param f_crit Precomputed F critical value
     * @param t_crit Precomputed t critical value (two-tailed, uncorrected)
     * @param correction_t_crits Precomputed correction critical values array
     * @param correction_method 0=none, 1=Bonferroni, 2=FDR, 3=Holm
     * @return Vector: [f_sig, uncorrected_sigs..., corrected_sigs...]
     */
    VectorXd analyze(
        const Eigen::Ref<const MatrixXd>& X,
        const Eigen::Ref<const VectorXd>& y,
        const Eigen::Ref<const VectorXi>& target_indices,
        double f_crit,
        double t_crit,
        const Eigen::Ref<const VectorXd>& correction_t_crits,
        int correction_method
    );
};

/**
 * Generate dependent variable with heterogeneity and heteroskedasticity.
 *
 * @param X Design matrix (n_samples, n_features)
 * @param effects Effect sizes (n_features,)
 * @param heterogeneity SD of effect size variation
 * @param heteroskedasticity Correlation between predictor and error variance
 * @param seed Random seed (-1 for random)
 * @param residual_dist Error distribution: 0=normal, 1=heavy_tailed (t), 2=skewed (chi2)
 * @param residual_df Degrees of freedom for non-normal residuals (min clamped to 3)
 * @return Response vector (n_samples,)
 */
Eigen::VectorXd generate_y(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& effects,
    double heterogeneity,
    double heteroskedasticity,
    int seed,
    int residual_dist = 0,
    double residual_df = 10.0
);

}  // namespace mcpower
