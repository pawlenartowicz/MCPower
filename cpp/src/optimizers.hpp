#pragma once

#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <limits>
#include <algorithm>

namespace mcpower {

// =========================================================================
// L-BFGS-B result (box-constrained multivariate optimization)
// =========================================================================

struct LBFGSBResult {
    Eigen::VectorXd x;   // optimal point
    double fun;           // objective value at optimum
    bool converged;       // true if optimization converged without error
    int n_iterations;     // number of iterations used
};

/**
 * Minimize a scalar function using L-BFGS-B (box-constrained L-BFGS).
 *
 * Uses finite-difference gradients (central differences) since we do not
 * have analytic gradients for the profiled deviance in the general case.
 * This is acceptable because:
 *   - n_theta is small (typically 1-6 parameters)
 *   - each objective evaluation is cheap (profiled deviance)
 *   - 2*n+1 evaluations per iteration is fine for small n
 *
 * Parameters match scipy.optimize.minimize(method='L-BFGS-B'):
 *   - objective: f(x) -> double
 *   - x0: initial guess
 *   - lb, ub: lower and upper bounds (element-wise)
 *   - maxiter: maximum iterations (0 = unlimited, default 200)
 *   - ftol: function value convergence tolerance (maps to LBFGSBParam::delta)
 *   - epsilon: gradient norm convergence tolerance (maps to LBFGSBParam::epsilon)
 *
 * The finite difference step h is adaptive: h = sqrt(eps) * max(1, |x_i|).
 */
LBFGSBResult lbfgsb_minimize_fd(
    const std::function<double(const Eigen::VectorXd&)>& objective,
    const Eigen::VectorXd& x0,
    const Eigen::VectorXd& lb,
    const Eigen::VectorXd& ub,
    int maxiter = 200,
    double ftol = 1e-10,
    double epsilon = 1e-6
);

// =========================================================================
// Brent's method result (1D scalar optimization)
// =========================================================================

struct BrentResult {
    double x;       // optimal point
    double fun;     // objective value at optimum
    bool converged; // true if converged within tolerance
};

/**
 * Generic Brent's method for 1D minimization on [a, b].
 *
 * This is a standalone version of Brent's method that accepts any
 * callable objective. The LMESolver class has its own Brent that's
 * tied to SufficientStatsQ1; this one can be used from Python via
 * pybind11 to replace scipy.optimize.minimize_scalar.
 *
 * @param objective  Callable f(x) -> double to minimize
 * @param a          Left bound of search interval
 * @param b          Right bound of search interval
 * @param tol        Convergence tolerance (default 1e-8)
 * @param maxiter    Maximum iterations (default 150)
 * @return BrentResult with optimal x, f(x), and convergence flag
 */
BrentResult brent_minimize_scalar(
    const std::function<double(double)>& objective,
    double a,
    double b,
    double tol = 1e-8,
    int maxiter = 150
);

}  // namespace mcpower
