//! When a non-factor column uses DIST_HIGH_KURTOSIS (var_type=4),
//! the resulting column is drawn from the censored-t(3) marginal via the
//! lookup table, standardized at build time to the censored distribution's
//! own SD — so the empirical variance is ~1.0 and the excess kurtosis ~6.4
//! (the censored table's law; full t(3) kurtosis is infinite).
//! This test verifies that the DIST_HIGH_KURTOSIS path uses the table (not a
//! passthrough z): variance alone no longer discriminates (both give ~1.0),
//! so the passthrough detector is the excess kurtosis — a normal z would
//! give ~0, the table ~6.4.

mod common;

use engine_core::data_gen::generate_sim_data;
use engine_core::marginals::T3PpfTable;
use engine_core::spec::{Distribution, SimulationSpec};
use engine_core::workspace::SimWorkspace;

#[test]
fn high_kurtosis_marginal_has_unit_variance() {
    let spec = SimulationSpec {
        var_types: vec![Distribution::HighKurtosis],
        effect_sizes: vec![0.0, 0.0],
        t3_table: Some(T3PpfTable::build_default()),
        ..common::base_spec()
    };

    let n = 100_000;
    let mut ws = SimWorkspace::new(n, 2, 1, 0, None);
    generate_sim_data(&spec, 0, 42, &mut ws).unwrap();

    let mut sum = 0.0;
    let mut sq = 0.0;
    let mut quart = 0.0;
    for i in 0..n {
        let xi = ws.x_full[(i, 1)];
        sum += xi;
        sq += xi * xi;
        quart += xi * xi * xi * xi;
    }
    let mean = sum as f64 / n as f64;
    let var = sq as f64 / n as f64 - mean * mean;
    // Table is standardized at build to the censored distribution's own SD.
    // MC band: SE(var̂) = √((exkurt+2)/n) ≈ 0.0095 at n=1e5 → ±0.05 is >5σ.
    assert!(
        (0.95..=1.05).contains(&var),
        "var={var}, expected ~1.0 (standardized censored t(3))."
    );
    // Passthrough detector: a normal z gives excess kurtosis ≈ 0; the
    // censored t(3) table gives ≈ 6.4. Variance no longer discriminates.
    let exkurt = quart as f64 / n as f64 / (var * var) - 3.0;
    assert!(
        exkurt > 3.0,
        "exkurt={exkurt}, expected ~6.4 (censored t(3)). \
         If exkurt≈0, the DIST_HIGH_KURTOSIS arm is passing z through."
    );
}
