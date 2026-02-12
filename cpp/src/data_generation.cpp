#include "data_generation.hpp"
#include <algorithm>
#include <stdexcept>

namespace mcpower {

DataGenerator::DataGenerator()
    : tables_loaded_(false)
{}

void DataGenerator::set_tables(
    const Eigen::Ref<const VectorXd>& norm_cdf,
    const Eigen::Ref<const VectorXd>& t3_ppf
) {
    norm_cdf_table_.resize(norm_cdf.size());
    for (int i = 0; i < norm_cdf.size(); ++i) {
        norm_cdf_table_[i] = norm_cdf(i);
    }

    t3_ppf_table_.resize(t3_ppf.size());
    for (int i = 0; i < t3_ppf.size(); ++i) {
        t3_ppf_table_[i] = t3_ppf(i);
    }

    tables_loaded_ = true;
}

Eigen::MatrixXd DataGenerator::robust_cholesky(const MatrixXd& corr_matrix) {
    // Try standard Cholesky first
    Eigen::LLT<MatrixXd> llt(corr_matrix);

    if (llt.info() == Eigen::Success) {
        return llt.matrixL();
    }

    // Fallback: eigenvalue decomposition with regularization
    Eigen::SelfAdjointEigenSolver<MatrixXd> es(corr_matrix);
    VectorXd eigenvals = es.eigenvalues();
    MatrixXd eigenvecs = es.eigenvectors();

    // Ensure positive eigenvalues
    for (int i = 0; i < eigenvals.size(); ++i) {
        if (eigenvals(i) < FLOAT_NEAR_ZERO) {
            eigenvals(i) = FLOAT_NEAR_ZERO;
        }
    }

    // Reconstruct: V * sqrt(D)
    return eigenvecs * eigenvals.array().sqrt().matrix().asDiagonal();
}

double DataGenerator::norm_cdf(double x) const {
    if (!tables_loaded_) {
        // Fallback: error function approximation
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }

    if (x <= NORM_RANGE_MIN) return 0.0;
    if (x >= NORM_RANGE_MAX) return 1.0;

    const double scale = (DIST_RESOLUTION - 1) / (NORM_RANGE_MAX - NORM_RANGE_MIN);
    const double idx = (x - NORM_RANGE_MIN) * scale;
    const int idx_int = static_cast<int>(idx);

    if (idx_int >= DIST_RESOLUTION - 1) {
        return norm_cdf_table_.back();
    }

    const double frac = idx - idx_int;
    return norm_cdf_table_[idx_int] * (1.0 - frac) +
           norm_cdf_table_[idx_int + 1] * frac;
}

double DataGenerator::t3_ppf(double percentile) const {
    if (!tables_loaded_) {
        // Fallback: return input (no transform)
        return percentile;
    }

    if (percentile <= PERC_RANGE_MIN) return t3_ppf_table_.front();
    if (percentile >= PERC_RANGE_MAX) return t3_ppf_table_.back();

    const double scale = (DIST_RESOLUTION - 1) / (PERC_RANGE_MAX - PERC_RANGE_MIN);
    const double idx = (percentile - PERC_RANGE_MIN) * scale;
    const int idx_int = static_cast<int>(idx);

    if (idx_int >= DIST_RESOLUTION - 1) {
        return t3_ppf_table_.back();
    }

    const double frac = idx - idx_int;
    return t3_ppf_table_[idx_int] * (1.0 - frac) +
           t3_ppf_table_[idx_int + 1] * frac;
}

double DataGenerator::uploaded_lookup(
    double normal_val,
    const VectorXd& normal_vals,
    const VectorXd& uploaded_vals
) {
    const int n = static_cast<int>(normal_vals.size());

    if (normal_val <= normal_vals(0)) return uploaded_vals(0);
    if (normal_val >= normal_vals(n - 1)) return uploaded_vals(n - 1);

    // Binary search
    int left = 0, right = n - 1;
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (normal_vals(mid) <= normal_val) {
            left = mid;
        } else {
            right = mid;
        }
    }

    // Linear interpolation
    double frac = (normal_val - normal_vals(left)) /
                  (normal_vals(right) - normal_vals(left));
    return uploaded_vals(left) * (1.0 - frac) + uploaded_vals(right) * frac;
}

void DataGenerator::transform_distribution(
    VectorXd& data,
    int dist_type,
    double param,
    int var_idx,
    const MatrixXd& upload_normal,
    const MatrixXd& upload_data
) {
    const int n = static_cast<int>(data.size());

    switch (dist_type) {
        case 0:  // Normal - no transform needed
            break;

        case 1: {  // Binary
            for (int i = 0; i < n; ++i) {
                double percentile = norm_cdf(data(i));
                data(i) = (percentile < param) ? 1.0 : 0.0;
            }
            // Center at 0
            double mean = data.mean();
            data.array() -= mean;
            break;
        }

        case 2: {  // Right skewed (exponential-like via lognormal)
            for (int i = 0; i < n; ++i) {
                double percentile = norm_cdf(data(i));
                percentile = std::max(PERC_RANGE_MIN, std::min(PERC_RANGE_MAX, percentile));
                data(i) = (-std::log(percentile) - SKEW_MEAN) / SKEW_STD;
            }
            break;
        }

        case 3: {  // Left skewed
            for (int i = 0; i < n; ++i) {
                double percentile = norm_cdf(data(i));
                percentile = std::max(PERC_RANGE_MIN, std::min(PERC_RANGE_MAX, percentile));
                data(i) = (std::log(1.0 - percentile) + SKEW_MEAN) / SKEW_STD;
            }
            break;
        }

        case 4: {  // High kurtosis (t-distribution)
            for (int i = 0; i < n; ++i) {
                double percentile = norm_cdf(data(i));
                data(i) = t3_ppf(percentile);
            }
            break;
        }

        case 5: {  // Uniform
            for (int i = 0; i < n; ++i) {
                double percentile = norm_cdf(data(i));
                data(i) = SQRT3 * (2.0 * percentile - 1.0);
            }
            break;
        }

        case 97:  // Uploaded factor (handled via bootstrap, no-op here)
            break;

        case 98:  // Uploaded binary (handled via bootstrap, no-op here)
            break;

        case 99: {  // Uploaded data
            if (var_idx < upload_normal.rows()) {
                VectorXd normal_vals = upload_normal.row(var_idx);
                VectorXd uploaded_vals = upload_data.row(var_idx);

                for (int i = 0; i < n; ++i) {
                    data(i) = uploaded_lookup(data(i), normal_vals, uploaded_vals);
                }
            }
            break;
        }

        default:
            // No transform
            break;
    }
}

Eigen::MatrixXd DataGenerator::generate_X(
    int n_samples,
    int n_vars,
    const Eigen::Ref<const MatrixXd>& correlation_matrix,
    const Eigen::Ref<const VectorXi>& var_types,
    const Eigen::Ref<const VectorXd>& var_params,
    const Eigen::Ref<const MatrixXd>& upload_normal,
    const Eigen::Ref<const MatrixXd>& upload_data,
    int seed
) {
    // Set up random generator
    std::mt19937 gen;
    if (seed >= 0) {
        gen.seed(static_cast<unsigned int>(seed));
    } else {
        std::random_device rd;
        gen.seed(rd());
    }
    std::normal_distribution<double> normal(0.0, 1.0);

    // Generate base normal data
    MatrixXd base_normal(n_samples, n_vars);
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_vars; ++j) {
            base_normal(i, j) = normal(gen);
        }
    }

    // Apply correlation structure
    MatrixXd cholesky = robust_cholesky(correlation_matrix);
    MatrixXd correlated = base_normal * cholesky.transpose();

    // Transform each variable to target distribution
    MatrixXd X(n_samples, n_vars);
    for (int j = 0; j < n_vars; ++j) {
        VectorXd col = correlated.col(j);
        int dist_type = (j < var_types.size()) ? var_types(j) : 0;
        double param = (j < var_params.size()) ? var_params(j) : 0.5;

        transform_distribution(col, dist_type, param, j, upload_normal, upload_data);
        X.col(j) = col;
    }

    return X;
}

}  // namespace mcpower
