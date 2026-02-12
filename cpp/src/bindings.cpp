#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "ols.hpp"
#include "data_generation.hpp"

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
}
