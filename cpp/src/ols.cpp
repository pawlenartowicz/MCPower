#include "ols.hpp"
#include <random>
#include <algorithm>
#include <numeric>

namespace mcpower {

Eigen::VectorXd OLSAnalyzer::analyze(
    const Eigen::Ref<const MatrixXd>& X,
    const Eigen::Ref<const VectorXd>& y,
    const Eigen::Ref<const VectorXi>& target_indices,
    double f_crit,
    double t_crit,
    const Eigen::Ref<const VectorXd>& correction_t_crits,
    int correction_method
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    const int n_targets = static_cast<int>(target_indices.size());

    // Result vector: [f_sig, uncorrected..., corrected...]
    VectorXd results = VectorXd::Zero(1 + 2 * n_targets);

    // Add intercept column
    MatrixXd X_int(n, p + 1);
    X_int.col(0) = VectorXd::Ones(n);
    X_int.rightCols(p) = X;

    // QR decomposition for numerical stability
    Eigen::HouseholderQR<MatrixXd> qr(X_int);
    MatrixXd Q = qr.householderQ() * MatrixXd::Identity(n, p + 1);
    MatrixXd R = qr.matrixQR().topRows(p + 1).triangularView<Eigen::Upper>();

    // Solve for beta
    VectorXd QTy = Q.transpose() * y;
    VectorXd beta_all = R.triangularView<Eigen::Upper>().solve(QTy);
    VectorXd beta = beta_all.tail(p);

    // Residuals and MSE
    VectorXd y_pred = X_int * beta_all;
    VectorXd residuals = y - y_pred;
    double ss_res = residuals.squaredNorm();
    int dof = n - (p + 1);

    if (dof <= 0) {
        return results;
    }

    double mse = ss_res / dof;

    // F-test for overall model significance
    double f_significant = 0.0;
    if (p > 0) {
        double y_mean = y.mean();
        double ss_tot = (y.array() - y_mean).square().sum();

        if (ss_tot > 1e-10) {
            double f_stat = ((ss_tot - ss_res) / p) / mse;
            f_significant = (f_stat > f_crit) ? 1.0 : 0.0;
        }
    }

    results(0) = f_significant;

    // Individual coefficient tests
    if (n_targets > 0 && mse > FLOAT_NEAR_ZERO) {
        // Compute |t| values for all targets
        VectorXd t_abs_values = VectorXd::Zero(n_targets);

        // Compute (R'R)^-1 for standard errors
        MatrixXd R_inv = R.triangularView<Eigen::Upper>().solve(
            MatrixXd::Identity(p + 1, p + 1)
        );
        MatrixXd var_coef = mse * (R_inv * R_inv.transpose());

        for (int idx_pos = 0; idx_pos < n_targets; ++idx_pos) {
            int coef_idx = target_indices(idx_pos);

            if (coef_idx < p) {
                int param_idx = coef_idx + 1;  // Account for intercept
                double std_err = std::sqrt(var_coef(param_idx, param_idx));

                if (std_err > FLOAT_NEAR_ZERO) {
                    t_abs_values(idx_pos) = std::abs(beta(coef_idx) / std_err);
                }
            }
        }

        // Uncorrected significances: compare |t| against t_crit
        for (int i = 0; i < n_targets; ++i) {
            results(1 + i) = (t_abs_values(i) > t_crit) ? 1.0 : 0.0;
        }

        // Apply correction
        if (correction_method == 0 || correction_method == 1) {
            // No correction or Bonferroni: direct comparison per target
            for (int i = 0; i < n_targets; ++i) {
                results(1 + n_targets + i) =
                    (t_abs_values(i) > correction_t_crits(i)) ? 1.0 : 0.0;
            }
        } else if (correction_method == 2) {
            // FDR (Benjamini-Hochberg): step-up procedure
            // Sort indices by |t| descending
            std::vector<int> sorted_indices(n_targets);
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            std::sort(sorted_indices.begin(), sorted_indices.end(),
                      [&t_abs_values](int a, int b) {
                          return t_abs_values(a) > t_abs_values(b);
                      });

            // Find last k where |t|_(k) > correction_t_crits[k]
            int last_sig = -1;
            for (int k = 0; k < n_targets; ++k) {
                if (t_abs_values(sorted_indices[k]) > correction_t_crits(k)) {
                    last_sig = k;
                }
            }
            if (last_sig >= 0) {
                for (int k = 0; k <= last_sig; ++k) {
                    results(1 + n_targets + sorted_indices[k]) = 1.0;
                }
            }
        } else if (correction_method == 3) {
            // Holm: step-down procedure
            std::vector<int> sorted_indices(n_targets);
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            std::sort(sorted_indices.begin(), sorted_indices.end(),
                      [&t_abs_values](int a, int b) {
                          return t_abs_values(a) > t_abs_values(b);
                      });

            for (int k = 0; k < n_targets; ++k) {
                if (t_abs_values(sorted_indices[k]) > correction_t_crits(k)) {
                    results(1 + n_targets + sorted_indices[k]) = 1.0;
                } else {
                    break;
                }
            }
        } else {
            // Unknown correction: same as uncorrected
            results.segment(1 + n_targets, n_targets) = results.segment(1, n_targets);
        }
    }

    return results;
}

// Y generation function
Eigen::VectorXd generate_y(
    const Eigen::Ref<const Eigen::MatrixXd>& X,
    const Eigen::Ref<const Eigen::VectorXd>& effects,
    double heterogeneity,
    double heteroskedasticity,
    int seed
) {
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());

    // Set up random generator
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(static_cast<unsigned int>(seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::normal_distribution<double> normal(0.0, 1.0);

    // Linear predictor with heterogeneity
    Eigen::VectorXd linear_pred(n);

    if (std::abs(heterogeneity) < FLOAT_NEAR_ZERO) {
        // No heterogeneity: simple matrix multiplication
        linear_pred = X * effects;
    } else {
        // Heterogeneity: vary effect sizes per observation
        linear_pred.setZero();

        // Change seed for heterogeneity noise
        if (seed >= 0) {
            gen.seed(static_cast<unsigned int>(seed + 1));
        }

        for (int j = 0; j < p; ++j) {
            double base_effect = effects(j);
            double noise_scale = heterogeneity * std::abs(base_effect);

            for (int i = 0; i < n; ++i) {
                double het_effect = base_effect + normal(gen) * noise_scale;
                linear_pred(i) += X(i, j) * het_effect;
            }
        }
    }

    // Generate errors
    if (seed >= 0) {
        gen.seed(static_cast<unsigned int>(seed + 2));
    }

    Eigen::VectorXd error(n);
    for (int i = 0; i < n; ++i) {
        error(i) = normal(gen);
    }

    // Apply heteroskedasticity
    if (std::abs(heteroskedasticity) >= FLOAT_NEAR_ZERO) {
        double lp_mean = linear_pred.mean();
        double lp_std = std::sqrt((linear_pred.array() - lp_mean).square().mean());

        if (lp_std > FLOAT_NEAR_ZERO) {
            Eigen::VectorXd standardized = (linear_pred.array() - lp_mean) / lp_std;

            // Error variance depends on predictor
            Eigen::VectorXd error_variance =
                (1.0 - heteroskedasticity) +
                heteroskedasticity * standardized.array().abs();

            // Clamp to minimum variance
            error_variance = error_variance.array().max(0.1);

            // Scale errors
            error = error.array() * error_variance.array().sqrt();

            // Re-standardize to maintain unit variance
            double error_std = std::sqrt(error.array().square().mean());
            if (error_std > FLOAT_NEAR_ZERO) {
                error /= error_std;
            }
        }
    }

    return linear_pred + error;
}

}  // namespace mcpower
