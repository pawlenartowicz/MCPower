#include "optimizers.hpp"
#include <LBFGSB.h>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace mcpower {

// =========================================================================
// L-BFGS-B minimizer with finite-difference gradients
// =========================================================================

LBFGSBResult lbfgsb_minimize_fd(
    const std::function<double(const Eigen::VectorXd&)>& objective,
    const Eigen::VectorXd& x0,
    const Eigen::VectorXd& lb,
    const Eigen::VectorXd& ub,
    int maxiter,
    double ftol,
    double epsilon
) {
    const int n = static_cast<int>(x0.size());

    // Adaptive finite difference step: h_i = sqrt(eps) * max(1, |x_i|)
    // Use a base step of ~1.5e-8 (sqrt of machine epsilon)
    constexpr double h_base = 1.4901161193847656e-8;  // sqrt(2.2e-16)

    // Functor for LBFGSPP: computes value and gradient via central differences
    auto func = [&](const Eigen::VectorXd& x, Eigen::VectorXd& grad) -> double {
        double fx = objective(x);

        // Check for non-finite objective — return large value + zero gradient
        // to let the optimizer recover or fail gracefully
        if (!std::isfinite(fx)) {
            grad.setZero();
            return 1e30;
        }

        // Central finite differences for gradient
        Eigen::VectorXd x_fwd = x;
        Eigen::VectorXd x_bwd = x;

        for (int i = 0; i < n; ++i) {
            const double hi = h_base * std::max(1.0, std::abs(x(i)));

            x_fwd(i) = x(i) + hi;
            x_bwd(i) = x(i) - hi;

            // Clamp to bounds to avoid out-of-domain evaluations
            x_fwd(i) = std::min(x_fwd(i), ub(i));
            x_bwd(i) = std::max(x_bwd(i), lb(i));

            const double actual_hi = x_fwd(i) - x_bwd(i);
            if (actual_hi > 0.0) {
                const double f_fwd = objective(x_fwd);
                const double f_bwd = objective(x_bwd);
                grad(i) = (f_fwd - f_bwd) / actual_hi;
            } else {
                // At a boundary corner; gradient = 0 (the projection will handle it)
                grad(i) = 0.0;
            }

            // Restore
            x_fwd(i) = x(i);
            x_bwd(i) = x(i);
        }

        return fx;
    };

    // Configure L-BFGS-B parameters
    LBFGSpp::LBFGSBParam<double> param;
    param.m = 6;                     // number of corrections (default)
    param.max_iterations = maxiter;
    param.epsilon = epsilon;         // gradient convergence tolerance
    param.epsilon_rel = 0.0;        // disable relative tolerance
    param.past = 1;                  // check function decrease over 1 step
    param.delta = ftol;              // function value convergence tolerance
    param.max_linesearch = 20;

    LBFGSpp::LBFGSBSolver<double> solver(param);

    Eigen::VectorXd x = x0;
    double fx = 0.0;

    LBFGSBResult result;
    try {
        int niter = solver.minimize(func, x, fx, lb, ub);
        result.x = x;
        result.fun = fx;
        result.converged = true;
        result.n_iterations = niter;
    } catch (const std::exception&) {
        // Optimization failed (e.g., line search failure, numerical issues)
        // Return the best point found so far
        result.x = x;
        result.fun = objective(x);
        result.converged = false;
        result.n_iterations = -1;
    } catch (...) {
        result.x = x;
        result.fun = objective(x);
        result.converged = false;
        result.n_iterations = -1;
    }

    return result;
}

// =========================================================================
// Generic Brent's method for 1D scalar minimization
// =========================================================================

BrentResult brent_minimize_scalar(
    const std::function<double(double)>& objective,
    double a,
    double b,
    double tol,
    int maxiter
) {
    // Brent's method: combination of golden section and parabolic interpolation
    // This is a standard implementation following Brent (1973) "Algorithms for
    // Minimization without Derivatives", also used in scipy.optimize.minimize_scalar.

    constexpr double golden = 0.3819660112501051;  // (3 - sqrt(5)) / 2
    const double eps = std::numeric_limits<double>::epsilon();

    double x = a + golden * (b - a);
    double w = x, v = x;
    double fx = objective(x);
    double fw = fx, fv = fx;
    double d = 0.0, e = 0.0;

    BrentResult result;
    result.converged = false;

    for (int iter = 0; iter < maxiter; ++iter) {
        double midpoint = 0.5 * (a + b);
        double tol1 = tol * std::abs(x) + eps;
        double tol2 = 2.0 * tol1;

        // Convergence check
        if (std::abs(x - midpoint) <= (tol2 - 0.5 * (b - a))) {
            result.x = x;
            result.fun = fx;
            result.converged = true;
            return result;
        }

        // Try parabolic interpolation
        bool use_golden = true;
        double u;

        if (std::abs(e) > tol1) {
            // Fit parabola through (v, fv), (w, fw), (x, fx)
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            else q = -q;
            r = e;
            e = d;

            // Accept parabolic step if it falls within bounds and is smaller
            // than half the previous step
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
        double fu = objective(u);

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

    // Did not converge within maxiter, but return best found
    result.x = x;
    result.fun = fx;
    result.converged = false;
    return result;
}

}  // namespace mcpower
