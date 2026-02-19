#pragma once

#include <cmath>
#include <limits>
#include <Eigen/Dense>

namespace mcpower { namespace dist {

// --- Individual distribution functions ---

// Normal distribution
double norm_ppf(double p);
double norm_cdf(double x);

// Student's t distribution
double t_ppf(double p, double df);

// F distribution
double f_ppf(double p, double dfn, double dfd);

// Chi-squared distribution
double chi2_ppf(double p, double df);
double chi2_cdf(double x, double df);

// Studentized range (Tukey) — ported from R's qtukey/ptukey
// Implemented in Task 3
double studentized_range_ppf(double p, int k, double df);

// --- Batch critical value helpers (Task 4) ---

struct OLSCriticalValues {
    double f_crit;
    double t_crit;
    Eigen::VectorXd correction_t_crits;
};

OLSCriticalValues compute_critical_values_ols(
    double alpha, int dfn, int dfd, int n_targets, int correction_method
);

double compute_tukey_critical_value(double alpha, int n_levels, int dfd);

struct LMECriticalValues {
    double chi2_crit;
    double z_crit;
    Eigen::VectorXd correction_z_crits;
};

LMECriticalValues compute_critical_values_lme(
    double alpha, int n_fixed, int n_targets, int correction_method
);

// --- Table generation ---

Eigen::VectorXd generate_norm_cdf_table(double x_min, double x_max, int resolution);
Eigen::VectorXd generate_t3_ppf_table(double perc_min, double perc_max, int resolution);
Eigen::VectorXd norm_ppf_array(const Eigen::VectorXd& percentiles);

}}  // namespace mcpower::dist
