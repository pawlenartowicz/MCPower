//! Aggregation primitives: Wilson 95% score CI; per-batch fold to PowerResult.
//!
//! Internal-stability surface: `pub` for the crate's integration tests, not
//! re-exported at the crate root and not part of the port contract — free to
//! change at any minor version.
//!
//! **WS27** = Wilson, E. B. (1927), *Probable inference, the law of succession,
//! and statistical inference*, JASA 22(158), 209–212.

use crate::result::Ci;

/// Wilson score 95% CI for a binomial proportion (WS27): with p̂ = k/n,
/// `center = (p̂ + z²/2n) / (1 + z²/n)` and
/// `margin = z·√(p̂(1−p̂)/n + z²/4n²) / (1 + z²/n)`. Wilson score CI with
/// `z = 1.96` hardcoded, computed at f64 precision.
///
/// Returns `(0.0, 1.0)` when `n == 0` (Python's edge case).
pub fn wilson_ci(k: u64, n: u64) -> Ci {
    const Z: f64 = 1.96;
    if n == 0 {
        return Ci { lo: 0.0, hi: 1.0 };
    }
    let n_f = n as f64;
    let p = k as f64 / n_f;
    let z2 = Z * Z;
    let denom = 1.0 + z2 / n_f;
    let center = (p + z2 / (2.0 * n_f)) / denom;
    let margin = Z * (p * (1.0 - p) / n_f + z2 / (4.0 * n_f * n_f)).sqrt() / denom;
    Ci {
        lo: (center - margin).max(0.0),
        hi: (center + margin).min(1.0),
    }
}

/// Significance rate `k / n`, guarded so `n == 0 → 0.0` (matches the Python
/// `_rate` edge case). NOTE: `convergence_rate` deliberately defaults to `1.0`
/// on an empty batch, so it does *not* use this helper.
#[inline]
fn rate(k: u64, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        k as f64 / n as f64
    }
}

use crate::result::{EstimatorExtras, PosthocPower, PowerResult};
use engine_core::BatchResult;
use engine_core::EstimatorSpec;

/// Fold the raw `BatchResult` (uncorrected/corrected uint8 grids of shape
/// `[n_sims, n_ss, n_targets]`, converged of shape `[n_sims, n_ss]`) into one
/// `PowerResult` per sample size.
///
/// Caller fills in `n` (sample size) after this returns. `estimator` drives which
/// `EstimatorExtras` variant is constructed.
///
/// Counter-pool mirror: `merge_one_power_result` (merge.rs) re-pools every raw
/// counter this fold emits — a new counter field added here must be pooled
/// there too.
pub fn aggregate_batch(
    batch: &BatchResult,
    target_indices: &[u32],
    contrast_pairs: &[(u32, u32)],
    estimator: &EstimatorSpec,
) -> Vec<PowerResult> {
    let n_sims = batch.shape.n_sims as usize;
    let n_ss = batch.shape.n_sample_sizes as usize;
    let n_targets = batch.shape.n_targets as usize;
    // Total post-hoc contrasts across all blocks. Hoisted so both the post-hoc
    // fold and the joint histogram (below) share one definition. 0 when no
    // post-hoc was requested.
    let total_contrasts: usize = batch
        .shape
        .posthoc_blocks
        .iter()
        .map(|b| b.n_contrasts as usize)
        .sum();
    let sim_posthoc_stride = n_ss * total_contrasts;
    let target_indices_vec: Vec<usize> = target_indices.iter().map(|&i| i as usize).collect();

    let mut out = Vec::with_capacity(n_ss);
    for ss_idx in 0..n_ss {
        let mut power_unc = Vec::with_capacity(n_targets);
        let mut power_cor = Vec::with_capacity(n_targets);
        let mut ci_unc = Vec::with_capacity(n_targets);
        let mut ci_cor = Vec::with_capacity(n_targets);
        let mut k_unc_vec = Vec::with_capacity(n_targets);
        let mut k_cor_vec = Vec::with_capacity(n_targets);

        for t in 0..n_targets {
            let (mut k_unc, mut k_cor) = (0u64, 0u64);
            for sim in 0..n_sims {
                let idx = sim * n_ss * n_targets + ss_idx * n_targets + t;
                k_unc += batch.uncorrected[idx] as u64;
                k_cor += batch.corrected[idx] as u64;
            }
            let p_unc = rate(k_unc, n_sims);
            let p_cor = rate(k_cor, n_sims);
            power_unc.push(p_unc);
            power_cor.push(p_cor);
            ci_unc.push(wilson_ci(k_unc, n_sims as u64));
            ci_cor.push(wilson_ci(k_cor, n_sims as u64));
            k_unc_vec.push(k_unc);
            k_cor_vec.push(k_cor);
        }

        // Per-estimator meaning differs (Ols: closed-form rank ok; Glm: IRLS
        // converged; Mle: optimiser didn't hit singularity); the count is just
        // the sum of nonzero u8s over this ss slot's sims.
        let conv_count: u64 = if n_sims == 0 {
            0
        } else {
            (0..n_sims)
                .map(|sim| batch.converged[sim * n_ss + ss_idx] as u64)
                .sum()
        };
        let conv_rate = if n_sims == 0 {
            1.0
        } else {
            conv_count as f64 / n_sims as f64
        };

        // Collect per-sim boundary_hit values for this sample-size slot.
        let bh_slice: Vec<u8> = (0..n_sims)
            .map(|sim| batch.boundary_hit[sim * n_ss + ss_idx])
            .collect();

        let estimator_extras = EstimatorExtras::from_batch(estimator, batch, ss_idx);

        // Posthoc fold: for each block in batch.shape.posthoc_blocks, sum the
        // per-sim contrast flags for this ss_idx and compute power + Wilson CIs.
        // Layout: posthoc_{unc,cor} are (n_sims, n_ss, total_contrasts) sim-major.
        // sim_posthoc_stride = n_ss * total_contrasts; per (sim, ss_idx) the
        // contrasts for block b start at offset O_b in a window of width W_b.
        let posthoc: Vec<PosthocPower> = if batch.shape.posthoc_blocks.is_empty() {
            Vec::new()
        } else {
            let mut blocks_out = Vec::with_capacity(batch.shape.posthoc_blocks.len());
            let mut offset = 0usize;
            for block in &batch.shape.posthoc_blocks {
                let w = block.n_contrasts as usize;
                let mut k_unc_vec = vec![0u64; w];
                let mut k_cor_vec = vec![0u64; w];
                for sim in 0..n_sims {
                    let base = sim * sim_posthoc_stride + ss_idx * total_contrasts + offset;
                    for c in 0..w {
                        k_unc_vec[c] += batch.posthoc_unc[base + c] as u64;
                        k_cor_vec[c] += batch.posthoc_cor[base + c] as u64;
                    }
                }
                let power_unc: Vec<f64> = k_unc_vec.iter().map(|&k| rate(k, n_sims)).collect();
                let power_cor: Vec<f64> = k_cor_vec.iter().map(|&k| rate(k, n_sims)).collect();
                let ci_unc: Vec<Ci> = k_unc_vec
                    .iter()
                    .map(|&k| wilson_ci(k, n_sims as u64))
                    .collect();
                let ci_cor: Vec<Ci> = k_cor_vec
                    .iter()
                    .map(|&k| wilson_ci(k, n_sims as u64))
                    .collect();
                blocks_out.push(PosthocPower {
                    n_levels: block.n_levels as usize,
                    power_uncorrected: power_unc,
                    power_corrected: power_cor,
                    ci_uncorrected: ci_unc,
                    ci_corrected: ci_cor,
                    success_counts_uncorrected: k_unc_vec,
                    success_counts_corrected: k_cor_vec,
                });
                offset += w;
            }
            blocks_out
        };

        let (overall_significant_rate, overall_count) = if batch.overall.is_empty() {
            (None, 0u64)
        } else {
            let k: u64 = (0..n_sims)
                .map(|sim| batch.overall[sim * n_ss + ss_idx] as u64)
                .sum();
            let r = rate(k, n_sims);
            (Some(r), k)
        };

        // Joint significance histogram over ALL marginal tests: the main targets
        // (marginals + contrast_pairs) AND every post-hoc pairwise contrast.
        // Excludes only the overall omnibus test. Per sim, count significant
        // tests across both buffers, then bump bucket k. Each test contributes
        // its own decision (main correction for targets, the family-wise/Tukey
        // correction for post-hoc), so the "corrected" histogram pools each
        // test at its own corrected threshold. Length n_targets + total_contrasts + 1.
        let n_joint = n_targets + total_contrasts;
        let mut hist_unc = vec![0u64; n_joint + 1];
        let mut hist_cor = vec![0u64; n_joint + 1];
        for sim in 0..n_sims {
            let mut k_unc = 0usize;
            let mut k_cor = 0usize;
            for t in 0..n_targets {
                let idx = sim * n_ss * n_targets + ss_idx * n_targets + t;
                if batch.uncorrected[idx] != 0 {
                    k_unc += 1;
                }
                if batch.corrected[idx] != 0 {
                    k_cor += 1;
                }
            }
            let base = sim * sim_posthoc_stride + ss_idx * total_contrasts;
            for c in 0..total_contrasts {
                if batch.posthoc_unc[base + c] != 0 {
                    k_unc += 1;
                }
                if batch.posthoc_cor[base + c] != 0 {
                    k_cor += 1;
                }
            }
            hist_unc[k_unc] += 1;
            hist_cor[k_cor] += 1;
        }

        // Wilson CI for the overall test, paired with overall_significant_rate.
        let overall_significant_ci =
            overall_significant_rate.map(|_| wilson_ci(overall_count, n_sims as u64));

        // Factor exclusion counters: per-factor counts of sparse-excluded (code 1)
        // and separation-fallback-dropped (code 2) sims for this ss slot.
        // Buffer layout: factor_excluded is (n_sims × n_ss × n_factors) sim-major.
        let n_factors = batch.shape.n_factors as usize;
        let mut excl_sparse = vec![0u64; n_factors];
        let mut excl_sep = vec![0u64; n_factors];
        if n_factors > 0 {
            for sim in 0..n_sims {
                let base = sim * n_ss * n_factors + ss_idx * n_factors;
                for f in 0..n_factors {
                    match batch.factor_excluded[base + f] {
                        1 => excl_sparse[f] += 1,
                        2 => excl_sep[f] += 1,
                        _ => {}
                    }
                }
            }
        }

        out.push(PowerResult {
            n: 0, // filled by caller in find_power.rs
            n_sims: n_sims as u64,
            target_indices: target_indices_vec.clone(),
            contrast_pairs: contrast_pairs.to_vec(),
            power_uncorrected: power_unc,
            power_corrected: power_cor,
            ci_uncorrected: ci_unc,
            ci_corrected: ci_cor,
            convergence_rate: conv_rate,
            boundary_hit: bh_slice,
            estimator_extras,
            overall_significant_rate,
            success_counts_uncorrected: k_unc_vec,
            success_counts_corrected: k_cor_vec,
            convergence_count: conv_count,
            overall_significant_count: overall_count,
            overall_significant_ci,
            success_count_histogram_uncorrected: hist_unc,
            success_count_histogram_corrected: hist_cor,
            grid_warnings: Vec::new(),
            posthoc,
            factor_exclusion_counts: excl_sparse,
            factor_separation_counts: excl_sep,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_core::{BatchResult, EstimatorSpec, ResultShape};

    fn make_batch_posthoc(
        posthoc_unc: Vec<u8>,
        posthoc_cor: Vec<u8>,
        shape: ResultShape,
    ) -> BatchResult {
        let n_sims = shape.n_sims as usize;
        let n_ss = shape.n_sample_sizes as usize;
        let n_targets = shape.n_targets as usize;
        let total = n_sims * n_ss * n_targets;
        let n_sims_ss = n_sims * n_ss;
        BatchResult {
            uncorrected: vec![1u8; total],
            corrected: vec![1u8; total],
            posthoc_unc,
            posthoc_cor,
            converged: vec![1u8; n_sims_ss],
            boundary_hit: vec![0u8; n_sims_ss],
            pinned_components: vec![0u32; n_sims_ss],
            joint_unc: vec![0u8; n_sims_ss],
            joint_cor: vec![0u8; n_sims_ss],
            overall: vec![],
            factor_excluded: vec![],
            tau_squared_hat: vec![],
            shape,
        }
    }

    /// Two sample sizes × two posthoc blocks of different widths (3-contrast +
    /// 1-contrast, total_contrasts=4) over 4 sims.
    ///
    /// Buffer layout: posthoc_unc[sim * (n_ss * total_contrasts) + ss_idx * total_contrasts + global_contrast]
    ///   sim_stride = 2 * 4 = 8; ss0 window = [0..3], ss1 window = [4..7]
    ///   block-0 offset = 0 (contrasts 0,1,2); block-1 offset = 3 (contrast 3)
    ///
    ///   Designed so that:
    ///   - ss0/block0-c0: fires sims 0,1,2 → power=3/4
    ///   - ss0/block1:    fires sims 0,1   → power=2/4=0.5   (different pattern than ss1)
    ///   - ss1/block0-c0: fires sim 0 only → power=1/4
    ///   - ss1/block1:    fires sim 3 only → power=1/4       (different sim than ss0/block1)
    ///
    /// Any bug in offset accumulation (block-1 still reading at offset 0) or in the
    /// ss_idx×total_contrasts stride (ss1 reading ss0's window) produces wrong counts.
    #[test]
    fn posthoc_multi_ss_multi_block_striding() {
        use engine_core::PosthocBlockShape;
        let shape = ResultShape {
            n_sims: 4,
            n_sample_sizes: 2,
            n_targets: 1,
            posthoc_blocks: vec![
                PosthocBlockShape {
                    n_levels: 3,
                    n_contrasts: 3,
                },
                PosthocBlockShape {
                    n_levels: 2,
                    n_contrasts: 1,
                },
            ],
            n_factors: 0,
            n_variance_components: 0,
        };
        // sim_stride = n_ss * total_contrasts = 2 * 4 = 8
        // Per sim: [ss0_b0c0, ss0_b0c1, ss0_b0c2, ss0_b1c0, ss1_b0c0, ss1_b0c1, ss1_b0c2, ss1_b1c0]
        #[rustfmt::skip]
        let posthoc_unc: Vec<u8> = vec![
            1, 0, 0, 1,  1, 0, 0, 0,  // sim 0
            1, 0, 0, 1,  0, 0, 0, 0,  // sim 1
            1, 0, 0, 0,  0, 0, 0, 0,  // sim 2
            0, 0, 0, 0,  0, 0, 0, 1,  // sim 3
        ];
        let posthoc_cor = posthoc_unc.clone();
        let batch = make_batch_posthoc(posthoc_unc, posthoc_cor, shape);
        let aggs = aggregate_batch(&batch, &[0], &[], &EstimatorSpec::Ols);

        assert_eq!(aggs.len(), 2, "two ss slots");

        // --- ss_idx = 0 ---
        let ph0 = &aggs[0].posthoc;
        assert_eq!(ph0.len(), 2, "ss0: two posthoc blocks");
        assert_eq!(ph0[0].n_levels, 3);
        assert_eq!(ph0[1].n_levels, 2);

        // block 0: c0 fires in sims 0,1,2 → 3/4; c1 and c2 never fire
        assert!(
            (ph0[0].power_uncorrected[0] - 0.75).abs() < 1e-12,
            "ss0 block0 c0: 3/4"
        );
        assert_eq!(ph0[0].power_uncorrected[1], 0.0, "ss0 block0 c1: 0");
        assert_eq!(ph0[0].power_uncorrected[2], 0.0, "ss0 block0 c2: 0");
        assert_eq!(ph0[0].success_counts_uncorrected, vec![3, 0, 0]);

        // block 1: fires in sims 0 and 1 → 2/4 = 0.5
        assert!(
            (ph0[1].power_uncorrected[0] - 0.5).abs() < 1e-12,
            "ss0 block1: 2/4"
        );
        assert_eq!(ph0[1].success_counts_uncorrected, vec![2]);

        // --- ss_idx = 1 ---
        let ph1 = &aggs[1].posthoc;
        assert_eq!(ph1.len(), 2, "ss1: two posthoc blocks");
        assert_eq!(ph1[0].n_levels, 3);
        assert_eq!(ph1[1].n_levels, 2);

        // block 0: c0 fires in sim 0 only → 1/4; c1 and c2 never fire
        assert!(
            (ph1[0].power_uncorrected[0] - 0.25).abs() < 1e-12,
            "ss1 block0 c0: 1/4"
        );
        assert_eq!(ph1[0].power_uncorrected[1], 0.0, "ss1 block0 c1: 0");
        assert_eq!(ph1[0].power_uncorrected[2], 0.0, "ss1 block0 c2: 0");
        assert_eq!(ph1[0].success_counts_uncorrected, vec![1, 0, 0]);

        // block 1: fires in sim 3 only → 1/4 (different sim from ss0/block1)
        assert!(
            (ph1[1].power_uncorrected[0] - 0.25).abs() < 1e-12,
            "ss1 block1: 1/4"
        );
        assert_eq!(ph1[1].success_counts_uncorrected, vec![1]);
    }

    #[test]
    fn joint_histogram_pools_posthoc_significance() {
        use engine_core::PosthocBlockShape;
        // 4 sims, 1 ss, 1 main target, one 3-contrast post-hoc block.
        let shape = ResultShape {
            n_sims: 4,
            n_sample_sizes: 1,
            n_targets: 1,
            posthoc_blocks: vec![PosthocBlockShape {
                n_levels: 3,
                n_contrasts: 3,
            }],
            n_factors: 0,
            n_variance_components: 0,
        };
        // main UNCORRECTED (sim, ss=0, target): significant in sims 0 and 1.
        let uncorrected = vec![1u8, 1, 0, 0];
        // main CORRECTED: significant in sim 0 only (independent of uncorrected).
        let corrected = vec![1u8, 0, 0, 0];
        // post-hoc UNCORRECTED, 3 contrasts/sim, sim-major:
        //   sim0 [1,0,0]=1 sig; sim1 [1,1,0]=2 sig; sim2 [1,0,0]=1 sig; sim3 [0,0,0]=0
        let posthoc_unc = vec![1u8, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0];
        // post-hoc CORRECTED: only sim0 contrast0 fires.
        let posthoc_cor = vec![1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let batch = BatchResult {
            uncorrected,
            corrected,
            posthoc_unc,
            posthoc_cor,
            converged: vec![1u8; 4],
            boundary_hit: vec![0u8; 4],
            pinned_components: vec![0u32; 4],
            joint_unc: vec![0u8; 4],
            joint_cor: vec![0u8; 4],
            overall: vec![],
            factor_excluded: vec![],
            tau_squared_hat: vec![],
            shape,
        };
        let aggs = aggregate_batch(&batch, &[1], &[], &EstimatorSpec::Ols);
        assert_eq!(aggs.len(), 1);
        let pr = &aggs[0];

        // Histogram length = n_targets(1) + total_contrasts(3) + 1 = 5 (k = 0..=4).
        assert_eq!(pr.success_count_histogram_uncorrected.len(), 5);
        assert_eq!(pr.success_count_histogram_corrected.len(), 5);

        // Per-sim UNCORRECTED totals = main + post-hoc:
        //   sim0 = 1+1 = 2; sim1 = 1+2 = 3; sim2 = 0+1 = 1; sim3 = 0+0 = 0
        //   → bucket counts k0:1, k1:1, k2:1, k3:1, k4:0
        assert_eq!(pr.success_count_histogram_uncorrected, vec![1, 1, 1, 1, 0]);

        // Per-sim CORRECTED totals:
        //   sim0 = 1+1 = 2; sim1 = 0; sim2 = 0; sim3 = 0
        //   → k0:3, k1:0, k2:1, k3:0, k4:0
        assert_eq!(pr.success_count_histogram_corrected, vec![3, 0, 1, 0, 0]);

        // Bucket-sum invariant: every sim lands in exactly one bucket.
        assert_eq!(
            pr.success_count_histogram_uncorrected.iter().sum::<u64>(),
            4
        );
        assert_eq!(pr.success_count_histogram_corrected.iter().sum::<u64>(), 4);
    }

    #[test]
    fn posthoc_counts_are_surfaced_in_power_result() {
        // 1 sample size, 2 main targets, one 3-contrast post-hoc block (from a
        // 3-level factor → n_levels 3, n_contrasts 3), 4 sims.
        //
        // posthoc_unc layout: (sim, ss_idx=0, contrast) sim-major, stride=3.
        //   sim0: [1, 0, 0]
        //   sim1: [1, 0, 0]
        //   sim2: [1, 0, 0]
        //   sim3: [0, 0, 0]
        // → contrast 0: 3/4 sims (power=0.75), contrast 1: 0/4, contrast 2: 0/4.
        //
        // posthoc_cor layout: same stride.
        //   sim0: [0, 1, 0]
        //   sim1: [0, 0, 0]
        //   sim2: [0, 0, 0]
        //   sim3: [0, 0, 0]
        // → contrast 0: 0/4, contrast 1: 1/4 (power=0.25), contrast 2: 0/4.
        use engine_core::PosthocBlockShape;
        let shape = ResultShape {
            n_sims: 4,
            n_sample_sizes: 1,
            n_targets: 2,
            posthoc_blocks: vec![PosthocBlockShape {
                n_levels: 3,
                n_contrasts: 3,
            }],
            n_factors: 0,
            n_variance_components: 0,
        };
        // posthoc_unc: 4 sims × 1 ss × 3 contrasts = 12 bytes, sim-major.
        let posthoc_unc = vec![
            1u8, 0, 0, // sim 0
            1, 0, 0, // sim 1
            1, 0, 0, // sim 2
            0, 0, 0, // sim 3
        ];
        // posthoc_cor: contrast 1 fires in sim 0 only.
        let posthoc_cor = vec![
            0u8, 1, 0, // sim 0
            0, 0, 0, // sim 1
            0, 0, 0, // sim 2
            0, 0, 0, // sim 3
        ];
        let batch = make_batch_posthoc(posthoc_unc, posthoc_cor, shape);
        let aggs = aggregate_batch(&batch, &[0, 1], &[], &EstimatorSpec::Ols);

        assert_eq!(aggs.len(), 1, "one ss slot");
        assert_eq!(aggs[0].posthoc.len(), 1, "one block for the one factor");
        let ph = &aggs[0].posthoc[0];
        assert_eq!(ph.n_levels, 3);
        assert_eq!(ph.power_uncorrected.len(), 3);
        assert!(
            (ph.power_uncorrected[0] - 0.75).abs() < 1e-12,
            "contrast 0 unc: 3/4"
        );
        assert_eq!(ph.power_uncorrected[1], 0.0, "contrast 1 unc: 0/4");
        assert_eq!(ph.power_uncorrected[2], 0.0, "contrast 2 unc: 0/4");
        assert_eq!(ph.success_counts_uncorrected, vec![3, 0, 0]);
        assert_eq!(ph.power_corrected.len(), 3);
        assert_eq!(ph.power_corrected[0], 0.0, "contrast 0 cor: 0/4");
        assert!(
            (ph.power_corrected[1] - 0.25).abs() < 1e-12,
            "contrast 1 cor: 1/4"
        );
        assert_eq!(ph.power_corrected[2], 0.0, "contrast 2 cor: 0/4");
        assert_eq!(ph.success_counts_corrected, vec![0, 1, 0]);
    }

    /// 2 factors, 2 ss-slots, 3 sims. Buffer layout:
    /// factor_excluded: (n_sims=3 × n_ss=2 × n_factors=2) sim-major.
    /// Per sim: [ss0_f0, ss0_f1, ss1_f0, ss1_f1]
    ///   sim0: [1, 0, 2, 1]  → ss0: f0=sparse, f1=none; ss1: f0=sep, f1=sparse
    ///   sim1: [0, 1, 0, 2]  → ss0: f0=none,   f1=sparse; ss1: f0=none, f1=sep
    ///   sim2: [1, 2, 0, 0]  → ss0: f0=sparse, f1=sep;    ss1: nothing
    ///
    /// Expected ss0 counts: excl_sparse=[2,1], excl_sep=[0,1]
    /// Expected ss1 counts: excl_sparse=[0,1], excl_sep=[1,1]
    #[test]
    fn factor_exclusion_counts_correct_per_ss_slot() {
        let shape = ResultShape {
            n_sims: 3,
            n_sample_sizes: 2,
            n_targets: 1,
            posthoc_blocks: vec![],
            n_factors: 2,
            n_variance_components: 0,
        };
        let n_sims = 3usize;
        let n_ss = 2usize;
        let n_targets = 1usize;
        let total_main = n_sims * n_ss * n_targets;
        let total_sims_ss = n_sims * n_ss;
        // factor_excluded: sim-major (n_sims × n_ss × n_factors)
        // sim0: ss0=[1,0], ss1=[2,1]; sim1: ss0=[0,1], ss1=[0,2]; sim2: ss0=[1,2], ss1=[0,0]
        let factor_excluded = vec![
            1u8, 0, 2, 1, // sim0: [ss0_f0,ss0_f1, ss1_f0,ss1_f1]
            0, 1, 0, 2, // sim1
            1, 2, 0, 0, // sim2
        ];
        let batch = BatchResult {
            uncorrected: vec![1u8; total_main],
            corrected: vec![1u8; total_main],
            posthoc_unc: vec![],
            posthoc_cor: vec![],
            converged: vec![1u8; total_sims_ss],
            boundary_hit: vec![0u8; total_sims_ss],
            pinned_components: vec![0u32; total_sims_ss],
            joint_unc: vec![0u8; total_sims_ss],
            joint_cor: vec![0u8; total_sims_ss],
            overall: vec![],
            factor_excluded,
            tau_squared_hat: vec![],
            shape,
        };
        let aggs = aggregate_batch(&batch, &[0], &[], &EstimatorSpec::Ols);
        assert_eq!(aggs.len(), 2);

        // ss_idx = 0: f0 sparse in sims 0,2 → 2; f1 sparse in sim 1 → 1; sep: f1 in sim 2 → 1
        assert_eq!(aggs[0].factor_exclusion_counts, vec![2, 1]);
        assert_eq!(aggs[0].factor_separation_counts, vec![0, 1]);

        // ss_idx = 1: f0 sep in sim 0 → 1; f1 sparse in sim 0 → 1; f1 sep in sim 1 → 1
        assert_eq!(aggs[1].factor_exclusion_counts, vec![0, 1]);
        assert_eq!(aggs[1].factor_separation_counts, vec![1, 1]);
    }
}
