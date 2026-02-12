#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <vector>
#include <random>
#include <cmath>

namespace mcpower {

/**
 * Distribution type codes.
 */
enum class DistributionType : int {
    Normal = 0,
    Binary = 1,
    RightSkewed = 2,
    LeftSkewed = 3,
    HighKurtosis = 4,
    Uniform = 5,
    UploadedFactor = 97,
    UploadedBinary = 98,
    UploadedData = 99
};

/**
 * Data generator for Monte Carlo simulations.
 *
 * Generates correlated data with various distributions using:
 * - Cholesky decomposition for correlation structure
 * - Lookup tables for distribution transforms
 */
class DataGenerator {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXi = Eigen::VectorXi;

    /**
     * Initialize data generator with lookup tables.
     */
    DataGenerator();

    /**
     * Set lookup tables for distribution transforms.
     */
    void set_tables(
        const Eigen::Ref<const VectorXd>& norm_cdf,
        const Eigen::Ref<const VectorXd>& t3_ppf
    );

    /**
     * Generate design matrix with specified distributions and correlations.
     *
     * @param n_samples Number of observations
     * @param n_vars Number of variables
     * @param correlation_matrix Correlation structure
     * @param var_types Distribution types per variable
     * @param var_params Distribution parameters (e.g., binary proportions)
     * @param upload_normal Normal quantiles for uploaded data
     * @param upload_data Uploaded data values
     * @param seed Random seed (-1 for random)
     * @return Design matrix (n_samples, n_vars)
     */
    MatrixXd generate_X(
        int n_samples,
        int n_vars,
        const Eigen::Ref<const MatrixXd>& correlation_matrix,
        const Eigen::Ref<const VectorXi>& var_types,
        const Eigen::Ref<const VectorXd>& var_params,
        const Eigen::Ref<const MatrixXd>& upload_normal,
        const Eigen::Ref<const MatrixXd>& upload_data,
        int seed
    );

private:
    std::vector<double> norm_cdf_table_;
    std::vector<double> t3_ppf_table_;
    bool tables_loaded_;

    // Table parameters
    static constexpr int DIST_RESOLUTION = 2048;
    static constexpr double NORM_RANGE_MIN = -6.0;
    static constexpr double NORM_RANGE_MAX = 6.0;
    static constexpr double PERC_RANGE_MIN = 0.001;
    static constexpr double PERC_RANGE_MAX = 0.999;
    static constexpr double SQRT3 = 1.7320508075688772;

    static constexpr double FLOAT_NEAR_ZERO = 1e-15;

    // Lognormal standardization constants
    static constexpr double SKEW_MEAN = 1.6487212707001282;  // exp(0.5)
    static constexpr double SKEW_STD = 2.1611974158950877;   // sqrt(exp(2) - exp(1))

    /**
     * Robust Cholesky decomposition with eigenvalue correction.
     */
    MatrixXd robust_cholesky(const MatrixXd& corr_matrix);

    /**
     * Normal CDF lookup.
     */
    double norm_cdf(double x) const;

    /**
     * t(3) PPF lookup.
     */
    double t3_ppf(double percentile) const;

    /**
     * Transform normal data to target distribution.
     */
    void transform_distribution(
        VectorXd& data,
        int dist_type,
        double param,
        int var_idx,
        const MatrixXd& upload_normal,
        const MatrixXd& upload_data
    );

    /**
     * Binary search and interpolation for uploaded data.
     */
    double uploaded_lookup(
        double normal_val,
        const VectorXd& normal_vals,
        const VectorXd& uploaded_vals
    );
};

}  // namespace mcpower
