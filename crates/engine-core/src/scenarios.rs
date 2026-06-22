//! Scenario perturbations — per-sim design draws.
//!
//! Scenarios are the headline v2 feature. The Rust side owns the perturbation
//! math because it executes inside the per-sim hot path; Python only assembles
//! the `ScenarioPerturbations` payload from defaults + user overrides.
//!
//! Every sim draws its own perturbed correlation matrix / var_types / residual
//! choice from the scenario stream, independent of the *data* stream that
//! supplies X rows and y values. Sims are therefore iid Bernoulli trials and
//! the naive Wilson interval on `successes / n_sims` is exact.

use crate::rng::SimRng;
use crate::spec::{Distribution, ResidualDist, ScenarioPerturbations};
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
/// `out`. Otherwise: adds symmetric Gaussian noise, clips to ±0.8, enforces
/// unit diagonal. PSD repair is **not** performed here — the caller invokes
/// `correlation::psd_repair_and_factor` on `out` immediately after.
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

    let sd = scenario.correlation_noise_sd;
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

/// With probability `scenario.residual_change_prob`, returns a residual
/// distribution drawn from `scenario.residual_dists`; otherwise returns
/// `spec_residual_dist`. The df for any t-kernel realization comes from
/// `scenario.residual_df` regardless (no model-level df). The uniform is
/// drawn unconditionally before the eligibility check, plus the pin rule.
pub fn pick_residual(
    scenario: &ScenarioPerturbations,
    spec_residual_dist: ResidualDist,
    spec_residual_pinned: bool,
    rng: &mut SimRng,
) -> ResidualDist {
    let prob = scenario.residual_change_prob;
    if prob <= 0.0 || scenario.residual_dists.is_empty() {
        return spec_residual_dist;
    }
    // Deliberate v1 deviation: only an unpinned default-normal residual is
    // swap-eligible (v1 swapped whatever the spec carried). The uniform is
    // drawn before the eligibility check so default-normal payloads keep the
    // exact v1 stream; ineligible specs discard the draw.
    let u = rng.next_uniform() as f64;
    if u < prob && matches!(spec_residual_dist, ResidualDist::Normal) && !spec_residual_pinned {
        // Clamp guards the rare u→1 edge.
        let k = ((rng.next_uniform() as f64 * scenario.residual_dists.len() as f64) as usize)
            .min(scenario.residual_dists.len() - 1);
        scenario.residual_dists[k]
    } else {
        spec_residual_dist
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
    fn optimistic_fast_path_pick_residual_returns_spec() {
        let scenario = ScenarioPerturbations::default();
        let mut rng = scenario_rng(42, 0);
        let dist = pick_residual(&scenario, ResidualDist::HighKurtosis, false, &mut rng);
        assert_eq!(dist, ResidualDist::HighKurtosis);
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
    fn scenario_rng_is_pure_function() {
        let mut a = scenario_rng(42, 0);
        let mut b = scenario_rng(42, 0);
        for _ in 0..100 {
            let xa = a.next_uniform();
            let xb = b.next_uniform();
            assert_eq!(
                xa, xb,
                "scenario_rng must be a pure function of (seed, sim_id)"
            );
        }
    }

    #[test]
    fn sim_id_changes_stream() {
        let mut a = scenario_rng(42, 0);
        let mut b = scenario_rng(42, 1);
        let xa = a.next_uniform();
        let xb = b.next_uniform();
        assert_ne!(xa, xb, "different sim_ids must produce different streams");
    }

    #[test]
    fn perturb_correlation_keeps_symmetry_and_unit_diag() {
        // Build a "realistic" scenario with mild correlation noise.
        let scenario = ScenarioPerturbations {
            name: "realistic".into(),
            correlation_noise_sd: 0.15,
            ..Default::default()
        };
        let base = mat_from_rows(
            4,
            &[
                &[1.0, 0.3, 0.2, 0.1],
                &[0.3, 1.0, 0.4, 0.2],
                &[0.2, 0.4, 1.0, 0.3],
                &[0.1, 0.2, 0.3, 1.0],
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
        // Off-diagonals clipped to ±0.8.
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
        // Pin rule: an explicitly-set normal is never swapped, and the
        // unconditional per-entry uniform draw is preserved, so the unpinned
        // entries realize exactly what an all-unpinned run realizes.
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

    #[test]
    fn pick_residual_swap_returns_pool_dist() {
        // SCEN-10: when `pick_residual` swaps, it returns a dist drawn from the
        // scenario's `residual_dists` pool; when it does not swap, it returns
        // the spec dist unchanged (df always comes from scenario.residual_df,
        // outside this function). These per-draw invariants are deterministic
        // and hold on every draw. The swap *rate* (≈ change_prob) is a
        // statistical property (L3 seed) and is not asserted.
        let scenario = ScenarioPerturbations {
            residual_change_prob: 0.5,
            residual_dists: vec![ResidualDist::HighKurtosis, ResidualDist::RightSkewed],
            residual_df: 8.0,
            ..Default::default()
        };
        let mut rng = scenario_rng(42, 0);
        let mut saw_swap = false;
        let mut saw_keep = false;
        for _ in 0..5000 {
            let dist = pick_residual(&scenario, ResidualDist::Normal, false, &mut rng);
            if !matches!(dist, ResidualDist::Normal) {
                saw_swap = true;
                assert!(
                    matches!(dist, ResidualDist::HighKurtosis | ResidualDist::RightSkewed),
                    "swapped dist must come from the pool, got {dist:?}"
                );
            } else {
                saw_keep = true;
            }
        }
        // With change_prob=0.5 over 5000 draws both branches are reached (this is a
        // reachability check on the deterministic branches, not a rate assertion).
        assert!(
            saw_swap && saw_keep,
            "both swap and keep branches must be exercised"
        );
    }

    #[test]
    fn pick_residual_never_swaps_pinned_or_non_normal() {
        // Pin rule (and the deliberate v1 deviation): a pinned normal and any
        // non-normal spec residual are never swapped, even at change_prob 1.
        let scenario = ScenarioPerturbations {
            residual_change_prob: 1.0,
            residual_dists: vec![ResidualDist::RightSkewed],
            residual_df: 8.0,
            ..Default::default()
        };
        let mut rng = scenario_rng(42, 0);
        for _ in 0..100 {
            let pinned = pick_residual(&scenario, ResidualDist::Normal, true, &mut rng);
            assert_eq!(pinned, ResidualDist::Normal, "pinned normal must survive");
            let non_normal = pick_residual(&scenario, ResidualDist::HighKurtosis, false, &mut rng);
            assert_eq!(
                non_normal,
                ResidualDist::HighKurtosis,
                "non-normal spec residual must survive"
            );
        }
    }
}
