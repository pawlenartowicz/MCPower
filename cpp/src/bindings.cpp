#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <stdexcept>

#include "ols.hpp"
#include "data_generation.hpp"
#include "lme_solver.hpp"
#include "distributions.hpp"
#include "optimizers.hpp"

namespace py = pybind11;

// Row-major matrix type for mapping numpy C-contiguous arrays
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace mcpower {

// Global instances (shared across function calls for efficiency)
static DataGenerator g_data_generator;
static bool g_tables_initialized = false;

/**
 * Initialize data generation tables from Python arrays.
 * Only loads norm_cdf and t3_ppf (OLS no longer needs lookup tables).
 */
void init_tables(
    py::array_t<double> norm_cdf,
    py::array_t<double> t3_ppf
) {
    auto norm_cdf_buf = norm_cdf.request();
    auto t3_ppf_buf = t3_ppf.request();

    // Map distribution tables
    Eigen::Map<const Eigen::VectorXd> norm_cdf_map(
        static_cast<double*>(norm_cdf_buf.ptr),
        norm_cdf_buf.size
    );

    Eigen::Map<const Eigen::VectorXd> t3_ppf_map(
        static_cast<double*>(t3_ppf_buf.ptr),
        t3_ppf_buf.size
    );

    // Initialize data generation tables
    g_data_generator.set_tables(norm_cdf_map, t3_ppf_map);
    g_tables_initialized = true;
}

/**
 * OLS analysis wrapper for Python.
 */
py::array_t<double> ols_analysis(
    py::array_t<double> X,
    py::array_t<double> y,
    py::array_t<int32_t> target_indices,
    double f_crit,
    double t_crit,
    py::array_t<double> correction_t_crits,
    int correction_method
) {
    // Convert inputs to Eigen
    auto X_buf = X.request();
    auto y_buf = y.request();
    auto idx_buf = target_indices.request();
    auto crits_buf = correction_t_crits.request();

    const int n = static_cast<int>(X_buf.shape[0]);
    const int p = static_cast<int>(X_buf.shape[1]);
    const int n_targets = static_cast<int>(idx_buf.size);
    const int n_crits = static_cast<int>(crits_buf.size);

    // Numpy arrays are row-major; convert X to column-major for Eigen
    Eigen::Map<const RowMajorMatrixXd> X_row(
        static_cast<double*>(X_buf.ptr), n, p
    );
    Eigen::MatrixXd X_map = X_row;

    Eigen::Map<const Eigen::VectorXd> y_map(
        static_cast<double*>(y_buf.ptr), n
    );
    Eigen::Map<const Eigen::VectorXi> idx_map(
        static_cast<int*>(idx_buf.ptr), n_targets
    );
    Eigen::Map<const Eigen::VectorXd> crits_map(
        static_cast<double*>(crits_buf.ptr), n_crits
    );

    // Run analysis
    OLSAnalyzer analyzer;
    Eigen::VectorXd results = analyzer.analyze(
        X_map, y_map, idx_map, f_crit, t_crit, crits_map, correction_method
    );

    // Convert result to numpy
    py::array_t<double> result(results.size());
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, results.data(), results.size() * sizeof(double));

    return result;
}

/**
 * Y generation wrapper for Python.
 */
py::array_t<double> generate_y_wrapper(
    py::array_t<double> X,
    py::array_t<double> effects,
    double heterogeneity,
    double heteroskedasticity,
    int seed
) {
    auto X_buf = X.request();
    auto effects_buf = effects.request();

    const int n = static_cast<int>(X_buf.shape[0]);
    const int p = static_cast<int>(X_buf.shape[1]);

    // Numpy arrays are row-major; convert X to column-major for Eigen
    Eigen::Map<const RowMajorMatrixXd> X_row(
        static_cast<double*>(X_buf.ptr), n, p
    );
    Eigen::MatrixXd X_map = X_row;

    Eigen::Map<const Eigen::VectorXd> effects_map(
        static_cast<double*>(effects_buf.ptr), p
    );

    Eigen::VectorXd y = generate_y(
        X_map, effects_map, heterogeneity, heteroskedasticity, seed
    );

    py::array_t<double> result(n);
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, y.data(), n * sizeof(double));

    return result;
}

/**
 * X generation wrapper for Python.
 */
py::array_t<double> generate_X_wrapper(
    int n_samples,
    int n_vars,
    py::array_t<double> correlation_matrix,
    py::array_t<int32_t> var_types,
    py::array_t<double> var_params,
    py::array_t<double> upload_normal,
    py::array_t<double> upload_data,
    int seed
) {
    auto corr_buf = correlation_matrix.request();
    auto types_buf = var_types.request();
    auto params_buf = var_params.request();
    auto un_buf = upload_normal.request();
    auto ud_buf = upload_data.request();

    // Numpy arrays are row-major; convert to column-major for Eigen
    Eigen::Map<const RowMajorMatrixXd> corr_row(
        static_cast<double*>(corr_buf.ptr), n_vars, n_vars
    );
    Eigen::MatrixXd corr_map = corr_row;

    Eigen::Map<const Eigen::VectorXi> types_map(
        static_cast<int*>(types_buf.ptr), n_vars
    );
    Eigen::Map<const Eigen::VectorXd> params_map(
        static_cast<double*>(params_buf.ptr), n_vars
    );

    // Handle upload tables (may have different dimensions)
    int un_rows = (un_buf.ndim >= 2) ? static_cast<int>(un_buf.shape[0]) : 0;
    int un_cols = (un_buf.ndim >= 2) ? static_cast<int>(un_buf.shape[1]) : 0;
    int ud_rows = (ud_buf.ndim >= 2) ? static_cast<int>(ud_buf.shape[0]) : 0;
    int ud_cols = (ud_buf.ndim >= 2) ? static_cast<int>(ud_buf.shape[1]) : 0;

    Eigen::MatrixXd un_mat = Eigen::MatrixXd::Zero(std::max(1, un_rows), std::max(1, un_cols));
    Eigen::MatrixXd ud_mat = Eigen::MatrixXd::Zero(std::max(1, ud_rows), std::max(1, ud_cols));

    if (un_rows > 0 && un_cols > 0) {
        Eigen::Map<const RowMajorMatrixXd> un_row(
            static_cast<double*>(un_buf.ptr), un_rows, un_cols
        );
        un_mat = un_row;
    }
    if (ud_rows > 0 && ud_cols > 0) {
        Eigen::Map<const RowMajorMatrixXd> ud_row(
            static_cast<double*>(ud_buf.ptr), ud_rows, ud_cols
        );
        ud_mat = ud_row;
    }

    Eigen::MatrixXd X = g_data_generator.generate_X(
        n_samples, n_vars, corr_map, types_map, params_map,
        un_mat, ud_mat, seed
    );

    // Convert to numpy (row-major for Python)
    py::array_t<double> result({n_samples, n_vars});
    auto result_buf = result.request();
    double* result_ptr = static_cast<double*>(result_buf.ptr);

    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_vars; ++j) {
            result_ptr[i * n_vars + j] = X(i, j);
        }
    }

    return result;
}

/**
 * LME analysis wrapper for Python.
 */
py::array_t<double> lme_analysis(
    py::array_t<double> X,
    py::array_t<double> y,
    py::array_t<int32_t> cluster_ids,
    int n_clusters,
    py::array_t<int32_t> target_indices,
    double chi2_crit,
    double z_crit,
    py::array_t<double> correction_z_crits,
    int correction_method,
    double warm_lambda_sq
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    auto cid_buf = cluster_ids.request();
    auto idx_buf = target_indices.request();
    auto crits_buf = correction_z_crits.request();

    const int n = static_cast<int>(X_buf.shape[0]);
    const int p = static_cast<int>(X_buf.shape[1]);
    const int n_obs = static_cast<int>(y_buf.size);
    const int n_targets = static_cast<int>(idx_buf.size);
    const int n_crits = static_cast<int>(crits_buf.size);

    if (n != n_obs) {
        throw std::invalid_argument("X.shape[0] and y.size must match");
    }

    // Numpy arrays are row-major; convert X to column-major for Eigen
    Eigen::Map<const RowMajorMatrixXd> X_row(
        static_cast<double*>(X_buf.ptr), n, p
    );
    Eigen::MatrixXd X_map = X_row;

    Eigen::Map<const Eigen::VectorXd> y_map(
        static_cast<double*>(y_buf.ptr), n
    );
    Eigen::Map<const Eigen::VectorXi> cid_map(
        static_cast<int*>(cid_buf.ptr), n
    );
    Eigen::Map<const Eigen::VectorXi> idx_map(
        static_cast<int*>(idx_buf.ptr), n_targets
    );
    Eigen::Map<const Eigen::VectorXd> crits_map(
        static_cast<double*>(crits_buf.ptr), n_crits
    );

    // Run LME analysis
    LMESolver solver;
    Eigen::VectorXd results = solver.analyze(
        X_map, y_map, cid_map, n_clusters,
        idx_map, chi2_crit, z_crit, crits_map,
        correction_method, warm_lambda_sq
    );

    if (results.size() == 0) {
        // Return empty array to signal failure
        return py::array_t<double>(0);
    }

    // Convert result to numpy
    py::array_t<double> result(results.size());
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, results.data(), results.size() * sizeof(double));

    return result;
}

/**
 * LME analysis wrapper for general q>1 (random slopes).
 */
py::array_t<double> lme_analysis_general(
    py::array_t<double> X,
    py::array_t<double> y,
    py::array_t<double> Z,
    py::array_t<int32_t> cluster_ids,
    int n_clusters,
    int q,
    py::array_t<int32_t> target_indices,
    double chi2_crit,
    double z_crit,
    py::array_t<double> correction_z_crits,
    int correction_method,
    py::array_t<double> warm_theta
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    auto Z_buf = Z.request();
    auto cid_buf = cluster_ids.request();
    auto idx_buf = target_indices.request();
    auto crits_buf = correction_z_crits.request();
    auto wt_buf = warm_theta.request();

    const int n = static_cast<int>(X_buf.shape[0]);
    const int p = static_cast<int>(X_buf.shape[1]);
    const int n_targets = static_cast<int>(idx_buf.size);
    const int n_crits = static_cast<int>(crits_buf.size);
    const int n_wt = static_cast<int>(wt_buf.size);

    // Row-major to col-major conversions
    Eigen::Map<const RowMajorMatrixXd> X_row(static_cast<double*>(X_buf.ptr), n, p);
    Eigen::MatrixXd X_map = X_row;

    Eigen::Map<const RowMajorMatrixXd> Z_row(static_cast<double*>(Z_buf.ptr), n, q);
    Eigen::MatrixXd Z_map = Z_row;

    Eigen::Map<const Eigen::VectorXd> y_map(static_cast<double*>(y_buf.ptr), n);
    Eigen::Map<const Eigen::VectorXi> cid_map(static_cast<int*>(cid_buf.ptr), n);
    Eigen::Map<const Eigen::VectorXi> idx_map(static_cast<int*>(idx_buf.ptr), n_targets);
    Eigen::Map<const Eigen::VectorXd> crits_map(static_cast<double*>(crits_buf.ptr), n_crits);
    Eigen::Map<const Eigen::VectorXd> wt_map(static_cast<double*>(wt_buf.ptr), n_wt);

    LMESolver solver;
    Eigen::VectorXd results = solver.analyze_general(
        X_map, y_map, Z_map, cid_map, n_clusters, q,
        idx_map, chi2_crit, z_crit, crits_map,
        correction_method, wt_map
    );

    if (results.size() == 0) {
        return py::array_t<double>(0);
    }

    py::array_t<double> result(results.size());
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, results.data(), results.size() * sizeof(double));
    return result;
}

/**
 * LME analysis wrapper for nested random intercepts.
 */
py::array_t<double> lme_analysis_nested(
    py::array_t<double> X,
    py::array_t<double> y,
    py::array_t<int32_t> parent_ids,
    py::array_t<int32_t> child_ids,
    int K_parent,
    int K_child,
    py::array_t<int32_t> child_to_parent,
    py::array_t<int32_t> target_indices,
    double chi2_crit,
    double z_crit,
    py::array_t<double> correction_z_crits,
    int correction_method,
    py::array_t<double> warm_theta
) {
    auto X_buf = X.request();
    auto y_buf = y.request();
    auto pid_buf = parent_ids.request();
    auto cid_buf = child_ids.request();
    auto c2p_buf = child_to_parent.request();
    auto idx_buf = target_indices.request();
    auto crits_buf = correction_z_crits.request();
    auto wt_buf = warm_theta.request();

    const int n = static_cast<int>(X_buf.shape[0]);
    const int p = static_cast<int>(X_buf.shape[1]);
    const int n_targets = static_cast<int>(idx_buf.size);
    const int n_crits = static_cast<int>(crits_buf.size);
    const int n_wt = static_cast<int>(wt_buf.size);

    Eigen::Map<const RowMajorMatrixXd> X_row(static_cast<double*>(X_buf.ptr), n, p);
    Eigen::MatrixXd X_map = X_row;

    Eigen::Map<const Eigen::VectorXd> y_map(static_cast<double*>(y_buf.ptr), n);
    Eigen::Map<const Eigen::VectorXi> pid_map(static_cast<int*>(pid_buf.ptr), n);
    Eigen::Map<const Eigen::VectorXi> cid_map(static_cast<int*>(cid_buf.ptr), n);
    Eigen::Map<const Eigen::VectorXi> c2p_map(static_cast<int*>(c2p_buf.ptr), K_child);
    Eigen::Map<const Eigen::VectorXi> idx_map(static_cast<int*>(idx_buf.ptr), n_targets);
    Eigen::Map<const Eigen::VectorXd> crits_map(static_cast<double*>(crits_buf.ptr), n_crits);
    Eigen::Map<const Eigen::VectorXd> wt_map(static_cast<double*>(wt_buf.ptr), n_wt);

    LMESolver solver;
    Eigen::VectorXd results = solver.analyze_nested(
        X_map, y_map, pid_map, cid_map, K_parent, K_child, c2p_map,
        idx_map, chi2_crit, z_crit, crits_map,
        correction_method, wt_map
    );

    if (results.size() == 0) {
        return py::array_t<double>(0);
    }

    py::array_t<double> result(results.size());
    auto result_buf = result.request();
    std::memcpy(result_buf.ptr, results.data(), results.size() * sizeof(double));
    return result;
}

/**
 * Check if tables are initialized.
 */
bool tables_initialized() {
    return g_tables_initialized;
}

}  // namespace mcpower

PYBIND11_MODULE(mcpower_native, m) {
    m.doc() = "MCPower native C++ backend for high-performance Monte Carlo simulations";

    // Table initialization (data generation tables only)
    m.def("init_tables", &mcpower::init_tables,
        py::arg("norm_cdf"),
        py::arg("t3_ppf"),
        "Initialize data generation tables from numpy arrays"
    );

    m.def("tables_initialized", &mcpower::tables_initialized,
        "Check if data generation tables have been initialized"
    );

    // OLS analysis
    m.def("ols_analysis", &mcpower::ols_analysis,
        py::arg("X"),
        py::arg("y"),
        py::arg("target_indices"),
        py::arg("f_crit"),
        py::arg("t_crit"),
        py::arg("correction_t_crits"),
        py::arg("correction_method") = 0,
        "Run OLS analysis with precomputed critical values"
    );

    // Y generation
    m.def("generate_y", &mcpower::generate_y_wrapper,
        py::arg("X"),
        py::arg("effects"),
        py::arg("heterogeneity") = 0.0,
        py::arg("heteroskedasticity") = 0.0,
        py::arg("seed") = -1,
        "Generate dependent variable with heterogeneity and heteroskedasticity"
    );

    // LME analysis (q=1 random intercept)
    m.def("lme_analysis", &mcpower::lme_analysis,
        py::arg("X"),
        py::arg("y"),
        py::arg("cluster_ids"),
        py::arg("n_clusters"),
        py::arg("target_indices"),
        py::arg("chi2_crit"),
        py::arg("z_crit"),
        py::arg("correction_z_crits"),
        py::arg("correction_method") = 0,
        py::arg("warm_lambda_sq") = -1.0,
        "Run LME analysis with precomputed critical values (random intercept)"
    );

    // LME analysis (general q>1 random slopes)
    m.def("lme_analysis_general", &mcpower::lme_analysis_general,
        py::arg("X"),
        py::arg("y"),
        py::arg("Z"),
        py::arg("cluster_ids"),
        py::arg("n_clusters"),
        py::arg("q"),
        py::arg("target_indices"),
        py::arg("chi2_crit"),
        py::arg("z_crit"),
        py::arg("correction_z_crits"),
        py::arg("correction_method") = 0,
        py::arg("warm_theta") = py::array_t<double>(0),
        "Run LME analysis for random slopes (q>1)"
    );

    // LME analysis (nested random intercepts)
    m.def("lme_analysis_nested", &mcpower::lme_analysis_nested,
        py::arg("X"),
        py::arg("y"),
        py::arg("parent_ids"),
        py::arg("child_ids"),
        py::arg("K_parent"),
        py::arg("K_child"),
        py::arg("child_to_parent"),
        py::arg("target_indices"),
        py::arg("chi2_crit"),
        py::arg("z_crit"),
        py::arg("correction_z_crits"),
        py::arg("correction_method") = 0,
        py::arg("warm_theta") = py::array_t<double>(0),
        "Run LME analysis for nested random intercepts"
    );

    // X generation
    m.def("generate_X", &mcpower::generate_X_wrapper,
        py::arg("n_samples"),
        py::arg("n_vars"),
        py::arg("correlation_matrix"),
        py::arg("var_types"),
        py::arg("var_params"),
        py::arg("upload_normal"),
        py::arg("upload_data"),
        py::arg("seed") = -1,
        "Generate design matrix with specified distributions and correlations"
    );

    // ---- Distribution functions ----

    m.def("norm_ppf", &mcpower::dist::norm_ppf,
          py::arg("p"),
          "Standard normal quantile function (inverse CDF)");

    m.def("norm_cdf", &mcpower::dist::norm_cdf,
          py::arg("x"),
          "Standard normal CDF");

    m.def("t_ppf", &mcpower::dist::t_ppf,
          py::arg("p"), py::arg("df"),
          "Student's t quantile function");

    m.def("f_ppf", &mcpower::dist::f_ppf,
          py::arg("p"), py::arg("dfn"), py::arg("dfd"),
          "Fisher F quantile function");

    m.def("chi2_ppf", &mcpower::dist::chi2_ppf,
          py::arg("p"), py::arg("df"),
          "Chi-squared quantile function");

    m.def("chi2_cdf", &mcpower::dist::chi2_cdf,
          py::arg("x"), py::arg("df"),
          "Chi-squared CDF");

    m.def("studentized_range_ppf", &mcpower::dist::studentized_range_ppf,
          py::arg("p"), py::arg("k"), py::arg("df"),
          "Studentized range quantile (Tukey). k=groups, df=denom df.");

    // ---- Batch critical value computation ----

    m.def("compute_critical_values_ols",
        [](double alpha, int dfn, int dfd, int n_targets, int correction_method) {
            auto cv = mcpower::dist::compute_critical_values_ols(alpha, dfn, dfd, n_targets, correction_method);
            return py::make_tuple(cv.f_crit, cv.t_crit, cv.correction_t_crits);
        },
        py::arg("alpha"), py::arg("dfn"), py::arg("dfd"),
        py::arg("n_targets"), py::arg("correction_method"),
        "Compute OLS critical values. Returns (f_crit, t_crit, correction_t_crits).");

    m.def("compute_tukey_critical_value", &mcpower::dist::compute_tukey_critical_value,
          py::arg("alpha"), py::arg("n_levels"), py::arg("dfd"),
          "Compute Tukey HSD critical value (q / sqrt(2)).");

    m.def("compute_critical_values_lme",
        [](double alpha, int n_fixed, int n_targets, int correction_method) {
            auto cv = mcpower::dist::compute_critical_values_lme(alpha, n_fixed, n_targets, correction_method);
            return py::make_tuple(cv.chi2_crit, cv.z_crit, cv.correction_z_crits);
        },
        py::arg("alpha"), py::arg("n_fixed"),
        py::arg("n_targets"), py::arg("correction_method"),
        "Compute LME critical values. Returns (chi2_crit, z_crit, correction_z_crits).");

    // ---- Table generation ----

    m.def("generate_norm_cdf_table", &mcpower::dist::generate_norm_cdf_table,
          py::arg("x_min"), py::arg("x_max"), py::arg("resolution"),
          "Generate normal CDF lookup table.");

    m.def("generate_t3_ppf_table", &mcpower::dist::generate_t3_ppf_table,
          py::arg("perc_min"), py::arg("perc_max"), py::arg("resolution"),
          "Generate t(3) PPF lookup table (divided by sqrt(3)).");

    m.def("norm_ppf_array", &mcpower::dist::norm_ppf_array,
          py::arg("percentiles"),
          "Vectorized normal PPF for percentile array.");

    // ---- Optimizer result types ----

    py::class_<mcpower::BrentResult>(m, "BrentResult")
        .def_readonly("x", &mcpower::BrentResult::x)
        .def_readonly("fun", &mcpower::BrentResult::fun)
        .def_readonly("converged", &mcpower::BrentResult::converged);

    py::class_<mcpower::LBFGSBResult>(m, "LBFGSBResult")
        .def_readonly("x", &mcpower::LBFGSBResult::x)
        .def_readonly("fun", &mcpower::LBFGSBResult::fun)
        .def_readonly("converged", &mcpower::LBFGSBResult::converged)
        .def_readonly("n_iterations", &mcpower::LBFGSBResult::n_iterations);

    // ---- Standalone optimizers ----

    m.def("brent_minimize_scalar", &mcpower::brent_minimize_scalar,
          py::arg("objective"),
          py::arg("a"),
          py::arg("b"),
          py::arg("tol") = 1e-8,
          py::arg("maxiter") = 150,
          "Brent's method for 1D minimization on [a, b]");

    m.def("lbfgsb_minimize_fd", &mcpower::lbfgsb_minimize_fd,
          py::arg("objective"),
          py::arg("x0"),
          py::arg("lb"),
          py::arg("ub"),
          py::arg("maxiter") = 200,
          py::arg("ftol") = 1e-10,
          py::arg("epsilon") = 1e-6,
          "L-BFGS-B minimization with finite-difference gradients");
}
