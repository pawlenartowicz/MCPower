#include "distributions.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/fisher_f.hpp>
#include <boost/math/distributions/chi_squared.hpp>

// Ensure M_LN2 is available (POSIX, but not guaranteed on all platforms)
#ifndef M_LN2
#define M_LN2 0.693147180559945309417232121458176568
#endif

namespace mcpower { namespace dist {

// ============================================================================
// Individual distribution functions (Boost.Math)
// ============================================================================

double norm_ppf(double p) {
    boost::math::normal_distribution<> d(0.0, 1.0);
    return boost::math::quantile(d, p);
}

double norm_cdf(double x) {
    boost::math::normal_distribution<> d(0.0, 1.0);
    return boost::math::cdf(d, x);
}

double t_ppf(double p, double df) {
    boost::math::students_t_distribution<> d(df);
    return boost::math::quantile(d, p);
}

double f_ppf(double p, double dfn, double dfd) {
    boost::math::fisher_f_distribution<> d(dfn, dfd);
    return boost::math::quantile(d, p);
}

double chi2_ppf(double p, double df) {
    boost::math::chi_squared_distribution<> d(df);
    return boost::math::quantile(d, p);
}

double chi2_cdf(double x, double df) {
    boost::math::chi_squared_distribution<> d(df);
    return boost::math::cdf(d, x);
}

// ============================================================================
// Studentized range distribution (Tukey HSD)
// Ported from R's src/nmath/ptukey.c and qtukey.c
// Original: Copenhaver & Holland (1988), AS 190 by Lund & Lund (1983)
// R source: GPL-2+, compatible with this project's GPL-3 license
// Accuracy: ptukey ~1e-14 (16-pt Legendre), qtukey ~4 sig. digits (secant)
//
// Copyright (C) 1998       Ross Ihaka
// Copyright (C) 2000--2007 The R Core Team
// ============================================================================

namespace detail {

// Standard normal distribution object (reused across calls)
static const boost::math::normal_distribution<> std_normal(0.0, 1.0);

// Helper: standard normal CDF
static inline double pnorm_std(double x) {
    return boost::math::cdf(std_normal, x);
}

// Helper: standard normal PDF
static inline double dnorm_std(double x) {
    return boost::math::pdf(std_normal, x);
}

// Helper: standard normal quantile
static inline double qnorm_std(double p) {
    return boost::math::quantile(std_normal, p);
}

// --------------------------------------------------------------------------
// wprob(): probability integral of Hartley's form of the range
//
// w     = value of range
// rr    = no. of rows or groups
// cc    = no. of columns or treatments
// returns probability integral from (0, w)
// --------------------------------------------------------------------------
static double wprob(double w, double rr, double cc)
{
    constexpr int nleg = 12;
    constexpr int ihalf = 6;

    static const double C1 = -30.0;
    static const double C2 = -50.0;
    static const double C3 = 60.0;
    static const double bb = 8.0;
    static const double wlar = 3.0;
    static const double wincr1 = 2.0;
    static const double wincr2 = 3.0;
    static const double xleg[ihalf] = {
        0.981560634246719250690549090149,
        0.904117256370474856678465866119,
        0.769902674194304687036893833213,
        0.587317954286617447296702418941,
        0.367831498998180193752691536644,
        0.125233408511468915472441369464
    };
    static const double aleg[ihalf] = {
        0.047175336386511827194615961485,
        0.106939325995318430960254718194,
        0.160078328543346226334652529543,
        0.203167426723065921749064455810,
        0.233492536538354808760849898925,
        0.249147045813402785000562436043
    };

    double a, ac, pr_w, b, binc, c, cc1,
           pminus, pplus, qexpo, qsqz, rinsum, wi, wincr, xx;
    long double blb, bub, einsum, elsum;
    int j, jj;

    qsqz = w * 0.5;

    // if w >= 16 then the integral lower bound (occurs for c=20)
    // is 0.99999999999995 so return a value of 1.
    if (qsqz >= bb)
        return 1.0;

    // find (f(w/2) - 1) ^ cc
    // (first term in integral of hartley's form).
    pr_w = 2.0 * pnorm_std(qsqz) - 1.0;  // erf(qsqz / M_SQRT2)

    // if pr_w ^ cc < 2e-22 then set pr_w = 0
    if (pr_w >= std::exp(C2 / cc))
        pr_w = std::pow(pr_w, cc);
    else
        pr_w = 0.0;

    // if w is large then the second component of the
    // integral is small, so fewer intervals are needed.
    if (w > wlar)
        wincr = wincr1;
    else
        wincr = wincr2;

    // find the integral of second term of hartley's form
    // for the integral of the range for equal-length
    // intervals using legendre quadrature.  limits of
    // integration are from (w/2, 8).  two or three
    // equal-length intervals are used.

    blb = qsqz;
    binc = (bb - qsqz) / wincr;
    bub = blb + binc;
    einsum = 0.0;

    // integrate over each interval
    cc1 = cc - 1.0;
    for (wi = 1; wi <= wincr; wi++) {
        elsum = 0.0;
        a = static_cast<double>(0.5 * (bub + blb));

        // legendre quadrature with order = nleg
        b = static_cast<double>(0.5 * (bub - blb));

        for (jj = 1; jj <= nleg; jj++) {
            if (ihalf < jj) {
                j = (nleg - jj) + 1;
                xx = xleg[j - 1];
            } else {
                j = jj;
                xx = -xleg[j - 1];
            }
            c = b * xx;
            ac = a + c;

            // if exp(-qexpo/2) < 9e-14,
            // then doesn't contribute to integral
            qexpo = ac * ac;
            if (qexpo > C3)
                break;

            pplus = 2.0 * pnorm_std(ac);
            pminus = 2.0 * pnorm_std(ac - w);

            // if rinsum ^ (cc-1) < 9e-14,
            // then doesn't contribute to integral
            rinsum = (pplus * 0.5) - (pminus * 0.5);
            if (rinsum >= std::exp(C1 / cc1)) {
                rinsum = (aleg[j - 1] * std::exp(-(0.5 * qexpo))) * std::pow(rinsum, cc1);
                elsum += rinsum;
            }
        }
        // M_1_SQRT_2PI = 1 / sqrt(2 * pi)
        static const double M_1_SQRT_2PI = 0.398942280401432677939946059934;
        elsum *= (((2.0 * b) * cc) * M_1_SQRT_2PI);
        einsum += elsum;
        blb = bub;
        bub += binc;
    }

    // if pr_w ^ rr < 9e-14, then return 0
    pr_w += static_cast<double>(einsum);
    if (pr_w <= std::exp(C1 / rr))
        return 0.0;

    pr_w = std::pow(pr_w, rr);
    if (pr_w >= 1.0)
        return 1.0;
    return pr_w;
}

// --------------------------------------------------------------------------
// ptukey_impl(): CDF of the studentized range distribution
//
// q          = value of studentized range
// rr         = no. of rows or groups (nranges)
// cc         = no. of columns or treatments (nmeans)
// df         = degrees of freedom of error term
// lower_tail = if true, returns P(X <= q); if false, P(X > q)
// log_p      = if true, returns log probability
// --------------------------------------------------------------------------
static double ptukey_impl(double q, double rr, double cc, double df,
                           bool lower_tail, bool log_p)
{
    constexpr int nlegq = 16;
    constexpr int ihalfq = 8;

    static const double eps1 = -30.0;
    static const double eps2 = 1.0e-14;
    static const double dhaf  = 100.0;
    static const double dquar = 800.0;
    static const double deigh = 5000.0;
    static const double dlarg = 25000.0;
    static const double ulen1 = 1.0;
    static const double ulen2 = 0.5;
    static const double ulen3 = 0.25;
    static const double ulen4 = 0.125;
    static const double xlegq[ihalfq] = {
        0.989400934991649932596154173450,
        0.944575023073232576077988415535,
        0.865631202387831743880467897712,
        0.755404408355003033895101194847,
        0.617876244402643748446671764049,
        0.458016777657227386342419442984,
        0.281603550779258913230460501460,
        0.950125098376374401853193354250e-1
    };
    static const double alegq[ihalfq] = {
        0.271524594117540948517805724560e-1,
        0.622535239386478928628438369944e-1,
        0.951585116824927848099251076022e-1,
        0.124628971255533872052476282192,
        0.149595988816576732081501730547,
        0.169156519395002538189312079030,
        0.182603415044923588866763667969,
        0.189450610455068496285396723208
    };

    double ans, f2, f21, f2lf, ff4, otsum, qsqz, rotsum, t1, twa1, ulen, wprb;
    int i, j, jj;

    if (std::isnan(q) || std::isnan(rr) || std::isnan(cc) || std::isnan(df))
        return std::numeric_limits<double>::quiet_NaN();

    if (q <= 0) {
        // R_DT_0: lower_tail ? (log_p ? -inf : 0) : (log_p ? 0 : 1)
        if (lower_tail)
            return log_p ? -std::numeric_limits<double>::infinity() : 0.0;
        else
            return log_p ? 0.0 : 1.0;
    }

    // df must be > 1, there must be at least two values
    if (df < 2 || rr < 1 || cc < 2)
        return std::numeric_limits<double>::quiet_NaN();

    if (!std::isfinite(q)) {
        // R_DT_1: lower_tail ? (log_p ? 0 : 1) : (log_p ? -inf : 0)
        if (lower_tail)
            return log_p ? 0.0 : 1.0;
        else
            return log_p ? -std::numeric_limits<double>::infinity() : 0.0;
    }

    if (df > dlarg) {
        // Use range distribution directly for large df
        double val = wprob(q, rr, cc);
        // R_DT_val(val):
        if (!lower_tail) val = 1.0 - val;
        if (log_p) val = std::log(val);
        return val;
    }

    // Calculate leading constant
    f2 = df * 0.5;
    f2lf = ((f2 * std::log(df)) - (df * M_LN2)) - std::lgamma(f2);
    f21 = f2 - 1.0;

    // integral is divided into unit, half-unit, quarter-unit, or
    // eighth-unit length intervals depending on the value of the
    // degrees of freedom.
    ff4 = df * 0.25;
    if      (df <= dhaf)  ulen = ulen1;
    else if (df <= dquar) ulen = ulen2;
    else if (df <= deigh) ulen = ulen3;
    else                  ulen = ulen4;

    f2lf += std::log(ulen);

    // integrate over each subinterval
    ans = 0.0;

    for (i = 1; i <= 50; i++) {
        otsum = 0.0;

        // legendre quadrature with order = nlegq
        // nodes (stored in xlegq) are symmetric around zero.
        twa1 = (2 * i - 1) * ulen;

        for (jj = 1; jj <= nlegq; jj++) {
            if (ihalfq < jj) {
                j = jj - ihalfq - 1;
                t1 = (f2lf + (f21 * std::log(twa1 + (xlegq[j] * ulen))))
                    - (((xlegq[j] * ulen) + twa1) * ff4);
            } else {
                j = jj - 1;
                t1 = (f2lf + (f21 * std::log(twa1 - (xlegq[j] * ulen))))
                    + (((xlegq[j] * ulen) - twa1) * ff4);
            }

            // if exp(t1) < 9e-14, then doesn't contribute to integral
            if (t1 >= eps1) {
                if (ihalfq < jj) {
                    qsqz = q * std::sqrt(((xlegq[j] * ulen) + twa1) * 0.5);
                } else {
                    qsqz = q * std::sqrt(((-(xlegq[j] * ulen)) + twa1) * 0.5);
                }

                // call wprob to find integral of range portion
                wprb = wprob(qsqz, rr, cc);
                rotsum = (wprb * alegq[j]) * std::exp(t1);
                otsum += rotsum;
            }
        }

        // if integral for interval i < 1e-14, then stop.
        // However, in order to avoid small area under left tail,
        // at least  1 / ulen  intervals are calculated.
        if (i * ulen >= 1.0 && otsum <= eps2)
            break;

        ans += otsum;
    }

    if (ans > 1.0)
        ans = 1.0;

    // R_DT_val(ans):
    if (!lower_tail) ans = 1.0 - ans;
    if (log_p) ans = std::log(ans);
    return ans;
}

// --------------------------------------------------------------------------
// qinv_impl(): initial estimate for the secant method in qtukey
//
// Adapted from AS 70 (Applied Statistics, 1974, Vol. 23, No. 1)
// by Odeh, R. E. and Evans, J. O.
//
// p = percentage point
// c = no. of columns or treatments
// v = degrees of freedom
// --------------------------------------------------------------------------
static double qinv_impl(double p, double c, double v)
{
    static const double p0 = 0.322232421088;
    static const double q0 = 0.993484626060e-01;
    static const double p1 = -1.0;
    static const double q1 = 0.588581570495;
    static const double p2 = -0.342242088547;
    static const double q2 = 0.531103462366;
    static const double p3 = -0.204231210125;
    static const double q3 = 0.103537752850;
    static const double p4 = -0.453642210148e-04;
    static const double q4 = 0.38560700634e-02;
    static const double c1 = 0.8832;
    static const double c2 = 0.2368;
    static const double c3 = 1.214;
    static const double c4 = 1.208;
    static const double c5 = 1.4142;
    static const double vmax = 120.0;

    double ps, q, t, yi;

    ps = 0.5 - 0.5 * p;
    yi = std::sqrt(std::log(1.0 / (ps * ps)));
    t = yi + ((((yi * p4 + p3) * yi + p2) * yi + p1) * yi + p0)
           / ((((yi * q4 + q3) * yi + q2) * yi + q1) * yi + q0);
    if (v < vmax) t += (t * t * t + t) / v / 4.0;
    q = c1 - c2 * t;
    if (v < vmax) q += -c3 / v + c4 * t / v;
    return t * (q * std::log(c - 1.0) + c5);
}

// --------------------------------------------------------------------------
// qtukey_impl(): quantile function (inverse CDF) of the studentized range
//
// Uses the secant method with eps=0.0001, max 50 iterations.
//
// p          = probability (confidence level)
// rr         = no. of rows or groups (nranges)
// cc         = no. of columns or treatments (nmeans)
// df         = degrees of freedom of error term
// lower_tail = if true, p is P(X <= q); if false, P(X > q)
// log_p      = if true, p is given as log(probability)
// --------------------------------------------------------------------------
static double qtukey_impl(double p, double rr, double cc, double df,
                           bool lower_tail, bool log_p)
{
    static const double eps = 0.0001;
    const int maxiter = 50;

    double ans = 0.0, valx0, valx1, x0, x1, xabs;
    int iter;

    if (std::isnan(p) || std::isnan(rr) || std::isnan(cc) || std::isnan(df))
        return std::numeric_limits<double>::quiet_NaN();

    // df must be > 1 ; there must be at least two values
    if (df < 2 || rr < 1 || cc < 2)
        return std::numeric_limits<double>::quiet_NaN();

    // R_Q_P01_boundaries(p, 0, ML_POSINF):
    // Handle boundary cases after converting to lower_tail non-log form
    {
        double p_internal = p;
        if (log_p) {
            if (p_internal > 0.0)
                return std::numeric_limits<double>::quiet_NaN();
            if (p_internal == 0.0)  // log(1)
                return lower_tail ? std::numeric_limits<double>::infinity() : 0.0;
            if (p_internal == -std::numeric_limits<double>::infinity())
                return lower_tail ? 0.0 : std::numeric_limits<double>::infinity();
        } else {
            if (p_internal < 0.0 || p_internal > 1.0)
                return std::numeric_limits<double>::quiet_NaN();
            if (p_internal == 0.0)
                return lower_tail ? 0.0 : std::numeric_limits<double>::infinity();
            if (p_internal == 1.0)
                return lower_tail ? std::numeric_limits<double>::infinity() : 0.0;
        }
    }

    // R_DT_qIv(p): convert to lower_tail, non-log "p"
    {
        double p_work = p;
        if (log_p)
            p_work = std::exp(p_work);
        if (!lower_tail)
            p_work = 1.0 - p_work;
        p = p_work;
    }

    // Initial value
    x0 = qinv_impl(p, cc, df);

    // Find prob(value < x0)
    valx0 = ptukey_impl(x0, rr, cc, df, true, false) - p;

    // Find the second iterate and prob(value < x1).
    // If the first iterate has probability value
    // exceeding p then second iterate is 1 less than
    // first iterate; otherwise it is 1 greater.
    if (valx0 > 0.0)
        x1 = std::fmax(0.0, x0 - 1.0);
    else
        x1 = x0 + 1.0;
    valx1 = ptukey_impl(x1, rr, cc, df, true, false) - p;

    // Find new iterate using secant method
    for (iter = 1; iter < maxiter; iter++) {
        ans = x1 - ((valx1 * (x1 - x0)) / (valx1 - valx0));
        valx0 = valx1;

        // New iterate must be >= 0
        x0 = x1;
        if (ans < 0.0) {
            ans = 0.0;
            valx1 = -p;
        }
        // Find prob(value < new iterate)
        valx1 = ptukey_impl(ans, rr, cc, df, true, false) - p;
        x1 = ans;

        // If the difference between two successive
        // iterates is less than eps, stop
        xabs = std::fabs(x1 - x0);
        if (xabs < eps)
            return ans;
    }

    // The process did not converge in 'maxiter' iterations
    // Return best estimate anyway (matches R behavior)
    return ans;
}

}  // namespace detail

double studentized_range_ppf(double p, int k, double df) {
    if (df < 2 || k < 2 || k > 200 || p <= 0.0 || p >= 1.0) {
        return std::numeric_limits<double>::infinity();
    }
    // qtukey(p, nranges=1, nmeans=k, df, lower_tail=true, log_p=false)
    return detail::qtukey_impl(p, 1.0, static_cast<double>(k), df, true, false);
}

// ============================================================================
// Batch critical value helpers
// ============================================================================

OLSCriticalValues compute_critical_values_ols(
    double alpha, int dfn, int dfd, int n_targets, int correction_method
) {
    OLSCriticalValues cv;

    if (dfd <= 0) {
        cv.f_crit = std::numeric_limits<double>::infinity();
        cv.t_crit = std::numeric_limits<double>::infinity();
        cv.correction_t_crits = Eigen::VectorXd::Constant(
            std::max(n_targets, 1), std::numeric_limits<double>::infinity());
        return cv;
    }

    cv.f_crit = (dfn > 0) ? f_ppf(1.0 - alpha, static_cast<double>(dfn), static_cast<double>(dfd))
                           : std::numeric_limits<double>::infinity();
    cv.t_crit = t_ppf(1.0 - alpha / 2.0, static_cast<double>(dfd));

    int m = n_targets;
    if (m == 0) {
        cv.correction_t_crits = Eigen::VectorXd();
        return cv;
    }

    cv.correction_t_crits.resize(m);
    double dfd_d = static_cast<double>(dfd);

    if (correction_method == 0) {         // None
        cv.correction_t_crits.setConstant(cv.t_crit);
    } else if (correction_method == 1) {  // Bonferroni
        double bonf = t_ppf(1.0 - alpha / (2.0 * m), dfd_d);
        cv.correction_t_crits.setConstant(bonf);
    } else if (correction_method == 2) {  // FDR (Benjamini-Hochberg)
        for (int k = 0; k < m; ++k) {
            double eff_alpha = static_cast<double>(k + 1) / m * alpha / 2.0;
            cv.correction_t_crits(k) = (eff_alpha < 1e-12)
                ? std::numeric_limits<double>::infinity()
                : t_ppf(1.0 - eff_alpha, dfd_d);
        }
    } else if (correction_method == 3) {  // Holm
        for (int k = 0; k < m; ++k) {
            double eff_alpha = alpha / (2.0 * (m - k));
            cv.correction_t_crits(k) = (eff_alpha < 1e-12)
                ? std::numeric_limits<double>::infinity()
                : t_ppf(1.0 - eff_alpha, dfd_d);
        }
    } else {
        cv.correction_t_crits.setConstant(cv.t_crit);
    }

    return cv;
}

double compute_tukey_critical_value(double alpha, int n_levels, int dfd) {
    if (dfd <= 0) return std::numeric_limits<double>::infinity();
    double q_crit = studentized_range_ppf(1.0 - alpha, n_levels, static_cast<double>(dfd));
    return q_crit / std::sqrt(2.0);
}

LMECriticalValues compute_critical_values_lme(
    double alpha, int n_fixed, int n_targets, int correction_method
) {
    LMECriticalValues cv;
    cv.chi2_crit = (n_fixed > 0) ? chi2_ppf(1.0 - alpha, static_cast<double>(n_fixed))
                                  : std::numeric_limits<double>::infinity();
    cv.z_crit = norm_ppf(1.0 - alpha / 2.0);

    int m = n_targets;
    if (m == 0) {
        cv.correction_z_crits = Eigen::VectorXd();
        return cv;
    }

    cv.correction_z_crits.resize(m);

    if (correction_method == 0) {         // None
        cv.correction_z_crits.setConstant(cv.z_crit);
    } else if (correction_method == 1) {  // Bonferroni
        double bonf = norm_ppf(1.0 - alpha / (2.0 * m));
        cv.correction_z_crits.setConstant(bonf);
    } else if (correction_method == 2) {  // FDR (Benjamini-Hochberg)
        for (int k = 0; k < m; ++k) {
            double eff_alpha = static_cast<double>(k + 1) / m * alpha / 2.0;
            cv.correction_z_crits(k) = (eff_alpha < 1e-12)
                ? std::numeric_limits<double>::infinity()
                : norm_ppf(1.0 - eff_alpha);
        }
    } else if (correction_method == 3) {  // Holm
        for (int k = 0; k < m; ++k) {
            double eff_alpha = alpha / (2.0 * (m - k));
            cv.correction_z_crits(k) = (eff_alpha < 1e-12)
                ? std::numeric_limits<double>::infinity()
                : norm_ppf(1.0 - eff_alpha);
        }
    } else {
        cv.correction_z_crits.setConstant(cv.z_crit);
    }

    return cv;
}

// ============================================================================
// Table generation
// ============================================================================

Eigen::VectorXd generate_norm_cdf_table(double x_min, double x_max, int resolution) {
    Eigen::VectorXd table(resolution);
    for (int i = 0; i < resolution; ++i) {
        double x = x_min + (x_max - x_min) * i / (resolution - 1);
        table(i) = norm_cdf(x);
    }
    return table;
}

Eigen::VectorXd generate_t3_ppf_table(double perc_min, double perc_max, int resolution) {
    double sqrt3 = std::sqrt(3.0);
    Eigen::VectorXd table(resolution);
    for (int i = 0; i < resolution; ++i) {
        double p = perc_min + (perc_max - perc_min) * i / (resolution - 1);
        table(i) = t_ppf(p, 3.0) / sqrt3;
    }
    return table;
}

Eigen::VectorXd norm_ppf_array(const Eigen::VectorXd& percentiles) {
    Eigen::VectorXd result(percentiles.size());
    for (Eigen::Index i = 0; i < percentiles.size(); ++i) {
        result(i) = norm_ppf(percentiles(i));
    }
    return result;
}

}}  // namespace mcpower::dist
