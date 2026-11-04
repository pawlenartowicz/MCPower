//! Scenario perturbations — per-sim design draws.
//!
//! Scenarios are the headline v2 feature. The Rust side owns the perturbation
//! math because it executes inside the per-sim hot path; Python only assembles
//! the `ScenarioPerturbations` payload from defaults + user overrides.
//!
//! Every sim draws its own perturbed correlation matrix and var_types from the
//! scenario stream, independent of the *data* stream that supplies X rows and
//! y values. Sims are therefore iid Bernoulli trials and the naive Wilson
//! interval on `successes / n_sims` is exact.

use crate::rng::SimRng;
use crate::spec::{Distribution, ScenarioPerturbations};
use faer::{MatMut, MatRef};

/// Stream tag for the scenario-perturbation RNG. Independent of the data-gen
/// stream so prefix stability (`X_full[:N]` equality across `max_n`) is
/// preserved per scenario.
pub const STREAM_TAG_SCENARIO: u64 = 0x5C5C_5C5C_5C5C_5C5C;

/// Stream tag for the per-study heterogeneity (β-jitter) RNG. The per-study
/// effect draw δ is keyed `SimRng::new(base_seed, sim_id ^ STREAM_TAG_HET)` —
/// the same mixing site and key as `scenario_rng`, domain-separated from both
/// the data-gen stream (tag 0) and the scenario stream. Because the stream is separate, δ never perturbs X or
/// residuals: heterogeneity = 0 stays byte-identical, and δ is N-stable (drawn
/// once per sim, independent of `max_n` → prefix-stable `find_sample_size`).
/// See the per-study draw in `data_gen::generate_sim_data`.
pub const STREAM_TAG_HET: u64 = 0x4848_4848_4848_4848;

// Continuous-normal var-type variant. The scenario perturbation only swaps out
// variables tagged as continuous-normal.

// ---------------------------------------------------------------------------
// Scenario RNG
// ---------------------------------------------------------------------------

/// Returns a fresh `SimRng` keyed on `(base_seed, sim_id ^ STREAM_TAG_SCENARIO)`.
///
/// Independent of the data-gen RNG, so prefix stability of the data path is
/// preserved regardless of scenario contents (the two `SimRng::new` keyings
/// differ by `STREAM_TAG_SCENARIO`).
///
/// Paired-comparison note: the scenario RNG is keyed only on
/// `(base_seed, sim_id)` — no scenario name or index is folded in — and the
/// orchestrator hands every scenario in one call the same call-level seed.
/// Two scenarios at the same `(base_seed, sim_id)` therefore draw the *same
/// raw noise stream*, scaled by their respective knobs, and the data stream
/// (`SimRng`) is equally shared: cross-scenario power deltas are attributable
/// to perturbation magnitude rather than RNG re-seeding.
pub fn scenario_rng(base_seed: u64, sim_id: u64) -> SimRng {
    SimRng::new(base_seed, sim_id ^ STREAM_TAG_SCENARIO)
}

// ---------------------------------------------------------------------------
// Perturbations
// ---------------------------------------------------------------------------

/// Writes the perturbed correlation into `out` (n×n, column-major).
/// `base` is `spec.correlation` — never mutated. `noise_buf` is caller-owned
/// scratch of length ≥ n² (used only on the noise path; fully overwritten
/// before read, so stale contents are fine).
///
/// Fast path: when `scenario.correlation_noise_sd == 0.0`, copies `base` into
/// `out`. Otherwise: adds symmetric Gaussian noise with SD
/// `correlation_noise_sd` to every off-diagonal entry, clips to ±0.8, enforces
/// unit diagonal. PSD repair is **not** performed here — the caller invokes
/// `correlation::psd_repair_and_factor` on `out` immediately after. Mirrored by
/// `perturb_correlation` in `validation/agreement/example_knobs.R` — change
/// together.
pub fn perturb_correlation(
    scenario: &ScenarioPerturbations,
    base: MatRef<'_, f64>,
    rng: &mut SimRng,
    mut out: MatMut<'_, f64>,
    noise_buf: &mut [f64],
) {
    let n = base.nrows();
    assert_eq!(n, base.ncols(), "perturb_correlation: base must be square");
    assert_eq!(n, out.nrows(), "perturb_correlation: out must match base");
    assert_eq!(n, out.ncols(), "perturb_correlation: out must match base");
    assert!(
        noise_buf.len() >= n * n,
        "perturb_correlation: noise_buf too small"
    );

    // Fast path: copy base, return.
    if scenario.correlation_noise_sd == 0.0 {
        for j in 0..n {
            for i in 0..n {
                out[(i, j)] = base[(i, j)];
            }
        }
        return;
    }

    // Each off-diagonal is the mean (Mᵢⱼ + Mⱼᵢ)/2 of two iid draws, whose SD
    // is the draw SD / √2; drawing at s·√2 gives noise SD s.
    let sd = scenario.correlation_noise_sd * std::f64::consts::SQRT_2;
    // Draw n×n iid normals into out as scratch.
    for j in 0..n {
        for i in 0..n {
            out[(i, j)] = rng.next_normal() as f64 * sd;
        }
    }
    // Symmetrize via (M + Mᵀ) / 2 then add to base; clip; diagonal=1.
    // Two-pass to avoid aliasing: snapshot the noise into the caller-owned
    // scratch (no per-draw heap allocation).
    let noise = &mut noise_buf[..n * n];
    for j in 0..n {
        for i in 0..n {
            noise[j * n + i] = out[(i, j)];
        }
    }
    for j in 0..n {
        for i in 0..n {
            let sym = 0.5 * (noise[j * n + i] + noise[i * n + j]);
            let v = base[(i, j)] + sym;
            let clipped = v.clamp(-0.8, 0.8);
            out[(i, j)] = clipped;
        }
    }
    for i in 0..n {
        out[(i, i)] = 1.0;
    }
}

/// Copies `base` into `out`, then with probability `distribution_change_prob`
/// replaces each *unpinned* continuous-normal entry with a uniform draw from
/// `scenario.new_distributions`, plus the pin rule (default = scenarios decide;
/// explicitly set = pinned). `pinned`
/// is aligned with `base`; a short/empty slice reads as all-unpinned.
pub fn perturb_var_types(
    scenario: &ScenarioPerturbations,
    base: &[Distribution],
    pinned: &[bool],
    rng: &mut SimRng,
    out: &mut [Distribution],
) {
    assert_eq!(base.len(), out.len(), "perturb_var_types: length mismatch");
    out.copy_from_slice(base);

    let prob = scenario.distribution_change_prob;
    if prob <= 0.0 || scenario.new_distributions.is_empty() {
        return;
    }
    for (j, item) in out.iter_mut().enumerate() {
        // v1 draws rng.random() unconditionally per variable; non-normal
        // (and now pinned) entries simply don't get swapped. Preserve that
        // draw count so the RNG state matches v1's per-design-draw
        // expectations — unpinned-default payloads stay byte-identical.
        let u = rng.next_uniform() as f64;
        let is_pinned = pinned.get(j).copied().unwrap_or(false);
        if matches!(*item, Distribution::Normal) && !is_pinned && u < prob {
            // Uniform pick from new_distributions (clamp guards the rare u→1 edge).
            let k = ((rng.next_uniform() as f64 * scenario.new_distributions.len() as f64)
                as usize)
                .min(scenario.new_distributions.len() - 1);
            *item = scenario.new_distributions[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::pcg_mix64;
    use faer::Mat;

    fn mat_from_rows(n: usize, rows: &[&[f64]]) -> Mat<f64> {
        let mut m = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                m[(i, j)] = rows[i][j];
            }
        }
        m
    }

    #[test]
    fn optimistic_fast_path_correlation_is_copy() {
        let scenario = ScenarioPerturbations::default();
        assert!(scenario.is_optimistic());
        let base = mat_from_rows(3, &[&[1.0, 0.3, 0.2], &[0.3, 1.0, 0.4], &[0.2, 0.4, 1.0]]);
        let mut out = Mat::<f64>::zeros(3, 3);
        let mut noise_buf = vec![0.0; 9];
        let mut rng = scenario_rng(42, 0);
        perturb_correlation(
            &scenario,
            base.as_ref(),
            &mut rng,
            out.as_mut(),
            &mut noise_buf,
        );
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(out[(i, j)], base[(i, j)], "({i},{j})");
            }
        }
    }

    #[test]
    fn optimistic_fast_path_var_types_is_copy() {
        let scenario = ScenarioPerturbations::default();
        let base = vec![
            Distribution::Normal,
            Distribution::Normal,
            Distribution::Binary,
            Distribution::RightSkewed,
        ];
        let mut out = vec![Distribution::Normal; 4];
        let mut rng = scenario_rng(42, 0);
        perturb_var_types(&scenario, &base, &[], &mut rng, &mut out);
        assert_eq!(out, base);
    }

    #[test]
    fn rng_stream_separation_data_vs_scenario() {
        // SCEN-12 / DGEN-06: the data stream and the scenario stream are seeded from
        // disjoint derivations for the same (base_seed, sim_id). The deterministic
        // mechanic is that the two seeds differ; the (low) sample correlation of the
        // realized streams is a statistical property (L3 seed), not asserted here.
        let base_seed = 42_u64;
        let sim_id = 17_u64;

        let data_seed = pcg_mix64(base_seed, sim_id);
        let scenario_seed = pcg_mix64(base_seed, sim_id ^ STREAM_TAG_SCENARIO);
        assert_ne!(data_seed, scenario_seed, "stream collision");
    }

    #[test]
    fn perturb_correlation_keeps_symmetry_and_unit_diag() {
        // Build a "realistic" scenario with mild correlation noise.
        let scenario = ScenarioPerturbations {
            name: "realistic".into(),
            correlation_noise_sd: 0.15,
            ..Default::default()
        };
        // (0,1) and (2,3) sit beyond the ±0.8 clip so the clamp fires.
        let base = mat_from_rows(
            4,
            &[
                &[1.0, 0.99, 0.2, 0.1],
                &[0.99, 1.0, 0.4, 0.2],
                &[0.2, 0.4, 1.0, -0.99],
                &[0.1, 0.2, -0.99, 1.0],
            ],
        );
        let mut rng = scenario_rng(42, 0);
        let mut out = Mat::<f64>::zeros(4, 4);
        let mut noise_buf = vec![0.0; 16];
        perturb_correlation(
            &scenario,
            base.as_ref(),
            &mut rng,
            out.as_mut(),
            &mut noise_buf,
        );

        // Symmetry.
        for i in 0..4 {
            for j in 0..4 {
                assert!((out[(i, j)] - out[(j, i)]).abs() < 1e-12, "asymm ({i},{j})");
            }
            // Unit diagonal.
            assert_eq!(out[(i, i)], 1.0);
        }
        // Off-diagonals clipped to ±0.8; the out-of-range base entries land
        // exactly on the clip.
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    assert!(
                        out[(i, j)] >= -0.8 && out[(i, j)] <= 0.8,
                        "clip ({i},{j}) = {}",
                        out[(i, j)]
                    );
                }
            }
        }
        assert_eq!(out[(0, 1)], 0.8, "upper clamp must fire");
        assert_eq!(out[(2, 3)], -0.8, "lower clamp must fire");
        // The noise is live: in-range entries move off the base.
        assert_ne!(out[(0, 2)], base[(0, 2)], "noise must perturb (0,2)");
        assert_ne!(out[(1, 2)], base[(1, 2)], "noise must perturb (1,2)");
    }

    #[test]
    fn perturb_correlation_noise_has_the_set_sd() {
        // ρ = 0 with s = 0.15 puts the ±0.8 clip 5.3σ out, so the realised SD
        // is s. 20,000 draws give SE(SD) ≈ s/√(2·20000) ≈ 0.00075; the ±0.005
        // band is ~6.7 SE wide, and the s/√2 ≈ 0.106 slip sits ~59 SE out.
        let s = 0.15;
        let scenario = ScenarioPerturbations {
            name: "realistic".into(),
            correlation_noise_sd: s,
            ..Default::default()
        };
        let base = mat_from_rows(2, &[&[1.0, 0.0], &[0.0, 1.0]]);
        let mut rng = scenario_rng(7, 0);
        let mut out = Mat::<f64>::zeros(2, 2);
        let mut noise_buf = vec![0.0; 4];
        let k = 20_000;
        let mut sum_sq = 0.0;
        for _ in 0..k {
            perturb_correlation(
                &scenario,
                base.as_ref(),
                &mut rng,
                out.as_mut(),
                &mut noise_buf,
            );
            sum_sq += out[(0, 1)] * out[(0, 1)];
        }
        let sd = (sum_sq / k as f64).sqrt(); // mean is 0 by construction
        assert!((sd - s).abs() < 0.005, "realised SD {sd}, set {s}");
    }

    #[test]
    fn perturb_var_types_swaps_only_continuous_normal() {
        let scenario = ScenarioPerturbations {
            distribution_change_prob: 1.0, // always swap when normal
            new_distributions: vec![
                Distribution::RightSkewed,
                Distribution::LeftSkewed,
                Distribution::Uniform,
            ],
            ..Default::default()
        };
        let base = vec![
            Distribution::Normal,
            Distribution::Binary,
            Distribution::Normal,
            Distribution::RightSkewed,
            Distribution::Normal,
        ];
        let mut out = vec![Distribution::HighKurtosis; 5];
        let mut rng = scenario_rng(42, 0);
        perturb_var_types(&scenario, &base, &[], &mut rng, &mut out);
        // Non-normal entries must survive unchanged.
        assert_eq!(out[1], Distribution::Binary);
        assert_eq!(out[3], Distribution::RightSkewed);
        // Normal entries must be swapped to one of the new distributions.
        for &i in &[0, 2, 4] {
            assert!(
                scenario.new_distributions.contains(&out[i]),
                "out[{i}] = {:?} not in {:?}",
                out[i],
                scenario.new_distributions
            );
        }
    }

    #[test]
    fn perturb_var_types_skips_pinned_normal() {
        // Pin rule: an explicitly-set normal is never swapped, while the
        // unpinned normals around it still swap. The pinned entry consumes its
        // unconditional uniform but skips the pool-pick draw, so entries after
        // it do NOT replay an all-unpinned run's draws; the one-entry pool makes
        // the swapped value draw-independent.
        let scenario = ScenarioPerturbations {
            distribution_change_prob: 1.0,
            new_distributions: vec![Distribution::RightSkewed],
            ..Default::default()
        };
        let base = vec![
            Distribution::Normal,
            Distribution::Normal,
            Distribution::Normal,
        ];
        let mut out_pinned = vec![Distribution::Normal; 3];
        let mut rng = scenario_rng(42, 7);
        perturb_var_types(
            &scenario,
            &base,
            &[false, true, false],
            &mut rng,
            &mut out_pinned,
        );
        assert_eq!(out_pinned[0], Distribution::RightSkewed);
        assert_eq!(
            out_pinned[1],
            Distribution::Normal,
            "pinned normal must survive"
        );
        assert_eq!(out_pinned[2], Distribution::RightSkewed);
    }
}
