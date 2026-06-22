//! t(3) PPF lookup table — 2048-knot uniform percentile grid over
//! [PERC_MIN, PERC_MAX], linearly interpolated at lookup time. The band is
//! retightened from the original [0.001, 0.999] so every synthetic marginal
//! stays within ±6 SD (constants: PERC_RANGE_MIN = 0.001, PERC_RANGE_MAX =
//! 0.999, DIST_RESOLUTION = 2048; our band narrows to [0.001_21, 0.998_79]).
//!
//! The table is `t_ppf(p, 3)` standardized by the SD of the marginal it
//! actually implements — the percentile-censored, linearly-interpolated
//! t(3) — so the transform has exactly unit variance (excess kurtosis ≈ 6.39,
//! support ≈ ±6.0 SD). The original implementation divided by √3, the SD of
//! the *uncensored* t(3); censoring removes ~14% of t(3)'s tail-dominated
//! variance, so that left the marginal at var ≈ 0.858 — a silent effect shrink
//! for every high-kurtosis predictor. Deliberate departure from that behaviour
//! (the original had the same fault). Shared across all sims and all columns;
//! build once per run.

use std::sync::Arc;

use crate::critvals::t_ppf;

// Band retightened from v1's [0.001, 0.999] (support 6.4 SD, exkurt ≈ 7.0)
// so every synthetic marginal stays within ±6 SD: support 6.0 SD, exkurt 6.39.
const PERC_MIN: f64 = 0.001_21;
const PERC_MAX: f64 = 0.998_79;
const RESOLUTION: usize = 2048;

/// Precomputed t(3) inverse-CDF table on `[perc_min, perc_max]`, sampled at
/// `RESOLUTION` knots.
#[derive(Debug, Clone, PartialEq)]
pub struct T3PpfTable {
    values: Vec<f64>,
    perc_min: f64,
    perc_max: f64,
}

impl T3PpfTable {
    /// Build the table at the default band/resolution above. Construction cost
    /// is `RESOLUTION` calls to `t_ppf(_, 3.0)`. Build once per run and clone
    /// the `Arc`.
    pub fn build_default() -> Arc<Self> {
        Self::build(PERC_MIN, PERC_MAX, RESOLUTION)
    }

    fn build(perc_min: f64, perc_max: f64, resolution: usize) -> Arc<Self> {
        assert!(resolution >= 2, "T3PpfTable: resolution must be >= 2");
        assert!(perc_min > 0.0 && perc_max < 1.0 && perc_min < perc_max);
        let denom = (resolution - 1) as f64;
        let mut values = Vec::with_capacity(resolution);
        for i in 0..resolution {
            let p = perc_min + (perc_max - perc_min) * (i as f64) / denom;
            values.push(t_ppf(p, 3.0));
        }
        // Standardize by the table's own SD so the marginal has exactly unit
        // variance regardless of censor range or resolution. Moments are
        // closed-form: under U = Φ(Z) ~ Unif(0,1) the lookup output is
        // piecewise linear in U, with mass perc_min / (1 − perc_max) parked
        // at the end knots; per interval ∫lin du = du(a+b)/2 and
        // ∫lin² du = du(a²+ab+b²)/3.
        // Mirrors t3_table_moments() in validation/common.R — change together.
        let du = (perc_max - perc_min) / denom;
        let mut m1 = perc_min * values[0] + (1.0 - perc_max) * values[resolution - 1];
        let mut m2 = perc_min * values[0] * values[0]
            + (1.0 - perc_max) * values[resolution - 1] * values[resolution - 1];
        for w in values.windows(2) {
            let (a, b) = (w[0], w[1]);
            m1 += du * (a + b) / 2.0;
            m2 += du * (a * a + a * b + b * b) / 3.0;
        }
        let sd = (m2 - m1 * m1).sqrt();
        for v in &mut values {
            *v /= sd;
        }
        Arc::new(Self {
            values,
            perc_min,
            perc_max,
        })
    }

    /// Linear-interp lookup. `percentile` is clamped to `[perc_min, perc_max]`.
    #[inline]
    pub fn lookup(&self, percentile: f64) -> f64 {
        let p = percentile.clamp(self.perc_min, self.perc_max);
        let scale = (self.values.len() - 1) as f64 / (self.perc_max - self.perc_min);
        let idx = (p - self.perc_min) * scale;
        let idx_int = idx.floor() as usize;
        let frac = idx - idx_int as f64;
        if idx_int + 1 >= self.values.len() {
            return *self.values.last().unwrap();
        }
        self.values[idx_int] * (1.0 - frac) + self.values[idx_int + 1] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_median_is_zero() {
        let t = T3PpfTable::build_default();
        assert!(
            t.lookup(0.5).abs() < 1e-6,
            "median should map to 0, got {}",
            t.lookup(0.5)
        );
    }

    /// The PPF lookup is monotonically non-decreasing across the
    /// table's percentile range — the quantile function of a unimodal
    /// symmetric distribution is monotone. A broken table (unsorted entries,
    /// wrong index math) would produce an inversion.
    #[test]
    fn lookup_is_monotone() {
        let t = T3PpfTable::build_default();
        let mut prev = f64::NEG_INFINITY;
        let mut p = 0.001_f64;
        while p <= 0.999 {
            let v = t.lookup(p);
            assert!(
                v >= prev - 1e-12,
                "lookup non-monotone at p={p}: {v} < prev {prev}"
            );
            prev = v;
            p += 0.001;
        }
    }

    #[test]
    fn clamp_below_min_returns_first_entry() {
        let t = T3PpfTable::build_default();
        assert_eq!(t.lookup(-0.5), t.values[0]);
    }

    #[test]
    fn clamp_above_max_returns_last_entry() {
        let t = T3PpfTable::build_default();
        let last = *t.values.last().unwrap();
        assert_eq!(t.lookup(1.5), last);
    }
}
