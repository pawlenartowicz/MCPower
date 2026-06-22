//! Multiple-testing correction methods on t² test statistics.
//!
//! All branches compare in `t²` space — never sqrt the SE, never abs(β).
//! For Holm / BH the targets are sorted by `t²` descending (NaN sinks to the
//! end, ascending target index breaks ties).

use crate::spec::CorrectionMethod;

/// Apply the configured multiple-testing correction.
///
/// - `correction_method`: typed enum (None / Bonferroni / Holm / BenjaminiHochberg).
/// - `t_sq`: per-target squared t-statistics (length `T`); NaN → fail.
/// - `t_crit_sq_uncorrected`: scalar squared critical value (used by `None`).
/// - `crit_sq`: per-rank squared critical values (Bonferroni/Holm/BH; length `T`).
/// - `out`: per-target pass/fail (0/1), length `T`.
pub fn apply_correction(
    correction_method: CorrectionMethod,
    t_sq: &[f64],
    t_crit_sq_uncorrected: f64,
    crit_sq: &[f64],
    out: &mut [u8],
) {
    let t = t_sq.len();
    debug_assert_eq!(out.len(), t, "out length must match t_sq");

    // Zero output first so NaN entries / unwalked positions land as fail.
    out.fill(0);

    if t == 0 {
        return;
    }

    match correction_method {
        CorrectionMethod::None => {
            for i in 0..t {
                if !t_sq[i].is_nan() && t_sq[i] > t_crit_sq_uncorrected {
                    out[i] = 1;
                }
            }
        }
        CorrectionMethod::Bonferroni | CorrectionMethod::TukeyHsd => {
            // Bonferroni — per-target, same threshold per rank (per critvals build).
            // Tukey HSD — per-target single-step comparison, identical in shape:
            // the studentized-range q-math (with each target's factor `k`) already
            // happened in `CritValueTable::build`, so both arms only compare `t²`
            // against the precomputed per-target threshold.
            for i in 0..t {
                if !t_sq[i].is_nan() && t_sq[i] > crit_sq[i] {
                    out[i] = 1;
                }
            }
        }
        CorrectionMethod::Holm => {
            // Holm step-down.
            // Sort indices by t² descending; ties broken by ascending target index.
            // Use a stack buffer up to 64 entries (target counts are small).
            let order = sort_indices_desc(t_sq);
            for k in 0..t {
                let i = order[k] as usize;
                let v = t_sq[i];
                if !v.is_nan() && v > crit_sq[k] {
                    out[i] = 1;
                } else {
                    // Holm: stop at first failure; remaining stay 0.
                    break;
                }
            }
        }
        CorrectionMethod::BenjaminiHochberg => {
            // Benjamini-Hochberg step-up.
            let order = sort_indices_desc(t_sq);
            // Walk descending, track last k where the ordered t² exceeds threshold[k];
            // all positions up to that k pass.
            let mut last_sig: isize = -1;
            for k in 0..t {
                let i = order[k] as usize;
                let v = t_sq[i];
                if !v.is_nan() && v > crit_sq[k] {
                    last_sig = k as isize;
                }
            }
            if last_sig >= 0 {
                for k in 0..=(last_sig as usize) {
                    out[order[k] as usize] = 1;
                }
            }
        }
    }
}

/// Sort `0..t_sq.len()` indices by `t_sq[i]` descending using `f64::total_cmp`.
/// Ties on equal `t²` break on ascending index (deterministic).
/// NaN sorts to the end under descending order via `total_cmp` (NaN is total-cmp
/// greater than all real values; we want failures to land at the end, so we put
/// NaNs last by special-casing in the comparator).
pub(crate) fn sort_indices_desc(t_sq: &[f64]) -> Vec<u32> {
    let n = t_sq.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_by(|&a, &b| {
        let va = t_sq[a as usize];
        let vb = t_sq[b as usize];
        let a_nan = va.is_nan();
        let b_nan = vb.is_nan();
        match (a_nan, b_nan) {
            (true, true) => a.cmp(&b),                    // both NaN: ascending index
            (true, false) => std::cmp::Ordering::Greater, // NaN sinks to end
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => {
                // descending by value (total_cmp), tie-break ascending by index
                vb.total_cmp(&va).then(a.cmp(&b))
            }
        }
    });
    idx
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_method_compares_to_uncorrected() {
        let t_sq = vec![1.0_f64, 5.0, 3.0];
        let mut out = vec![0u8; 3];
        apply_correction(
            CorrectionMethod::None,
            &t_sq,
            4.0,
            &[10.0, 10.0, 10.0],
            &mut out,
        );
        assert_eq!(out, vec![0, 1, 0]);
    }

    #[test]
    fn bonferroni_per_target() {
        let t_sq = vec![3.0_f64, 9.0, 4.5];
        let crit_sq = [4.0, 4.0, 4.0];
        let mut out = vec![0u8; 3];
        apply_correction(CorrectionMethod::Bonferroni, &t_sq, 4.0, &crit_sq, &mut out);
        assert_eq!(out, vec![0, 1, 1]);
    }

    #[test]
    fn holm_step_down_known_outcome() {
        // Three targets, t² = [9, 4, 1], crit_sq sequence by rank = [10, 5, 3].
        // Sorted desc: [0 → 9, 1 → 4, 2 → 1].
        // Rank 0: 9 > 10? no → stop, all fail.
        let t_sq = vec![9.0_f64, 4.0, 1.0];
        let crit_sq = [10.0, 5.0, 3.0];
        let mut out = vec![0u8; 3];
        apply_correction(CorrectionMethod::Holm, &t_sq, 3.84, &crit_sq, &mut out);
        assert_eq!(out, vec![0, 0, 0]);

        // Now make rank 0 pass; rank 1 fails → only target with rank 0 passes.
        let crit_sq2 = [5.0, 5.0, 3.0];
        let mut out = vec![0u8; 3];
        apply_correction(CorrectionMethod::Holm, &t_sq, 3.84, &crit_sq2, &mut out);
        assert_eq!(out, vec![1, 0, 0]);

        // All thresholds easy → all pass.
        let crit_sq3 = [0.5, 0.5, 0.5];
        let mut out = vec![0u8; 3];
        apply_correction(CorrectionMethod::Holm, &t_sq, 0.5, &crit_sq3, &mut out);
        assert_eq!(out, vec![1, 1, 1]);
    }

    #[test]
    fn bh_step_up_known_outcome() {
        // BH walks all and applies step-up.
        // t² = [9, 1, 4] → sorted desc by value: [0:9, 2:4, 1:1]
        // crit_sq by rank = [10, 3, 0.5].
        // Walk: k=0 → 9 > 10? no.  k=1 → 4 > 3? yes → last_sig = 1.
        // k=2 → 1 > 0.5? yes → last_sig = 2.
        // → ranks 0..=2 pass → all targets pass.
        let t_sq = vec![9.0_f64, 1.0, 4.0];
        let crit_sq = [10.0, 3.0, 0.5];
        let mut out = vec![0u8; 3];
        apply_correction(
            CorrectionMethod::BenjaminiHochberg,
            &t_sq,
            3.84,
            &crit_sq,
            &mut out,
        );
        assert_eq!(out, vec![1, 1, 1]);
    }

    #[test]
    fn nan_t_sq_fails_under_every_method() {
        let t_sq = vec![100.0_f64, f64::NAN, 100.0];
        let crit_sq = [1.0, 1.0, 1.0];
        for m in [
            CorrectionMethod::None,
            CorrectionMethod::Bonferroni,
            CorrectionMethod::Holm,
            CorrectionMethod::BenjaminiHochberg,
            CorrectionMethod::TukeyHsd,
        ] {
            let mut out = vec![0u8; 3];
            apply_correction(m, &t_sq, 1.0, &crit_sq, &mut out);
            assert_eq!(out[1], 0, "method {m:?}: NaN must fail");
            // The non-NaN entries should pass.
            assert_eq!(out[0], 1, "method {m:?}: t² > crit must pass");
            assert_eq!(out[2], 1, "method {m:?}: t² > crit must pass");
        }
    }

    /// Hand-computed reference: three targets, both Holm and BH verified in t² space.
    /// Booleans must match the step-down/step-up procedure exactly.
    #[test]
    fn d3_step2_vs_v1_reference_holm_and_bh() {
        // Fixture 1: |t| = [3.5, 0.7, 1.2, 2.3] → all distinct, no ties.
        let abs_t = [3.5_f64, 0.7, 1.2, 2.3];
        let t_sq: Vec<f64> = abs_t.iter().map(|x| x * x).collect();
        // v1-style hand reference (descending order [0, 3, 2, 1]).
        // Use a fixed threshold sequence (squared on v2 side).
        let crit_abs = [2.5_f64, 2.0, 1.5, 1.0]; // for Holm
        let crit_sq: Vec<f64> = crit_abs.iter().map(|x| x * x).collect();

        // V1 reference (run by hand): sort desc by |t|: 3.5, 2.3, 1.2, 0.7
        // Holm step-down with crit sequence [2.5, 2.0, 1.5, 1.0]:
        //   k=0: 3.5 > 2.5 → pass (target 0)
        //   k=1: 2.3 > 2.0 → pass (target 3)
        //   k=2: 1.2 > 1.5? no → stop. Targets 2 & 1 fail.
        // Expected: out = [1, 0, 0, 1]
        let mut out = vec![0u8; 4];
        apply_correction(CorrectionMethod::Holm, &t_sq, 1.0, &crit_sq, &mut out);
        assert_eq!(out, vec![1, 0, 0, 1], "Holm fixture 1");

        // BH on same |t| sequence with thresholds [3.0, 2.0, 1.5, 1.0].
        // Sorted desc: 3.5, 2.3, 1.2, 0.7
        //   k=0: 3.5 > 3.0 → last_sig = 0.
        //   k=1: 2.3 > 2.0 → last_sig = 1.
        //   k=2: 1.2 > 1.5? no.
        //   k=3: 0.7 > 1.0? no.
        // last_sig = 1 → ranks 0..=1 pass → targets 0 and 3.
        let crit_abs_bh = [3.0_f64, 2.0, 1.5, 1.0];
        let crit_sq_bh: Vec<f64> = crit_abs_bh.iter().map(|x| x * x).collect();
        let mut out = vec![0u8; 4];
        apply_correction(
            CorrectionMethod::BenjaminiHochberg,
            &t_sq,
            1.0,
            &crit_sq_bh,
            &mut out,
        );
        assert_eq!(out, vec![1, 0, 0, 1], "BH fixture 1");

        // Fixture 2: tie on |t|, BH walks past it.
        let abs_t = [2.0_f64, 2.0, 0.5];
        let t_sq: Vec<f64> = abs_t.iter().map(|x| x * x).collect();
        let crit_abs_bh = [3.0_f64, 1.8, 0.4];
        let crit_sq_bh: Vec<f64> = crit_abs_bh.iter().map(|x| x * x).collect();
        // Sort desc: ties on first two, asc-index tie-break: 0, 1, 2.
        //   k=0: 2 > 3? no.
        //   k=1: 2 > 1.8? yes → last_sig = 1.
        //   k=2: 0.5 > 0.4? yes → last_sig = 2.
        // → all pass.
        let mut out = vec![0u8; 3];
        apply_correction(
            CorrectionMethod::BenjaminiHochberg,
            &t_sq,
            1.0,
            &crit_sq_bh,
            &mut out,
        );
        assert_eq!(out, vec![1, 1, 1], "BH tied fixture");
    }

    // -----------------------------------------------------------------------
    // Tukey HSD threshold construction (CritValueTable::build_with_tukey_k)
    // -----------------------------------------------------------------------

    use crate::critvals::CritValueTable;
    use crate::spec::{CritValues, EstimatorSpec};

    /// Tukey single-step threshold for one target whose factor has `l` levels,
    /// at sample size `n` and `n_predictors` (df = n − n_predictors). Returns the
    /// squared-t critical value emitted by the Tukey arm of the build.
    fn tukey_crit_sq(alpha: f64, n: u32, n_predictors: u32, l: f64) -> f64 {
        let crit = CritValues {
            alpha,
            posthoc_alpha: None,
        };
        let table = CritValueTable::build_with_tukey_k(
            &crit,
            &[n],
            n_predictors,
            1,
            CorrectionMethod::TukeyHsd,
            EstimatorSpec::Ols,
            &[l],
        )
        .unwrap();
        table.correction_t_crit_sq[0][0]
    }

    /// Step 1 — Tukey sits between uncorrected and Bonferroni. The family Tukey
    /// controls is the `C(L,2)` pairwise comparisons of an `L`-level factor, so
    /// the fair Bonferroni reference uses `m = C(L,2)`. Thresholds are squared-t
    /// (higher = more conservative). Tukey must be ≥ uncorrected and ≤ that
    /// Bonferroni for every L ≥ 2 (equality at L = 2, the single-comparison case).
    #[test]
    fn tukey_between_uncorrected_and_bonferroni() {
        let alpha = 0.05;
        let n = 100u32;
        let n_predictors = 3u32;
        let df = (n - n_predictors) as f64;

        let crit = CritValues {
            alpha,
            posthoc_alpha: None,
        };
        for l in [2.0_f64, 3.0, 4.0, 5.0] {
            let tukey = tukey_crit_sq(alpha, n, n_predictors, l);

            // Uncorrected squared-t threshold (t_ppf(1-α/2, df))².
            let uncorrected = {
                let table = CritValueTable::build(
                    &crit,
                    &[n],
                    n_predictors,
                    1,
                    CorrectionMethod::None,
                    EstimatorSpec::Ols,
                )
                .unwrap();
                table.t_crit_sq_uncorrected[0]
            };

            // Bonferroni with m = C(L,2) comparisons.
            let m = (l * (l - 1.0) / 2.0).round() as u32;
            let bonferroni = {
                let table = CritValueTable::build(
                    &crit,
                    &[n],
                    n_predictors,
                    m,
                    CorrectionMethod::Bonferroni,
                    EstimatorSpec::Ols,
                )
                .unwrap();
                // Bonferroni emits the same threshold for every rank.
                table.correction_t_crit_sq[0][0]
            };

            // Slack absorbs the ~1e-6 numerical gap between the studentized-range
            // quadrature/Newton path and the direct t-quantile path; it is orders
            // of magnitude below any genuine correction gap (the L≥3 spread is
            // ~0.3+ in t² space).
            let eps = 1e-5;
            assert!(
                tukey >= uncorrected - eps,
                "L={l}: tukey {tukey} must be ≥ uncorrected {uncorrected} (df={df})"
            );
            assert!(
                tukey <= bonferroni + eps,
                "L={l}: tukey {tukey} must be ≤ bonferroni(m={m}) {bonferroni} (df={df})"
            );
        }
    }

    /// Step 1b — multi-factor: two targets belonging to factors with DIFFERENT
    /// level counts must get DIFFERENT Tukey thresholds within the SAME build
    /// call. This fails if `k` is treated as a single scalar.
    #[test]
    fn tukey_different_k_per_factor_in_one_build() {
        let alpha = 0.05;
        let n = 100u32;
        let n_predictors = 3u32;
        let crit = CritValues {
            alpha,
            posthoc_alpha: None,
        };
        // Target 0 → factor with L=3; target 1 → factor with L=2.
        let table = CritValueTable::build_with_tukey_k(
            &crit,
            &[n],
            n_predictors,
            2,
            CorrectionMethod::TukeyHsd,
            EstimatorSpec::Ols,
            &[3.0, 2.0],
        )
        .unwrap();
        let row = &table.correction_t_crit_sq[0];
        assert_eq!(row.len(), 2);
        assert!(
            row[0] > row[1],
            "k=3 threshold {} must exceed k=2 threshold {} (more means → more conservative)",
            row[0],
            row[1]
        );
        // Both thresholds must be finite, positive critical values.
        assert!(
            row[0].is_finite() && row[0] > 0.0,
            "k=3 tukey t² must be finite>0"
        );
        assert!(
            row[1].is_finite() && row[1] > 0.0,
            "k=2 tukey t² must be finite>0"
        );
    }

    /// Step 6.3 — NaN entries must never be marked significant under any
    /// correction method. Verifies that sort_indices_desc already has the
    /// explicit NaN-last comparator and that the !v.is_nan() write guard keeps
    /// NaN slots at 0 even when the finite entries exceed the threshold.
    #[test]
    fn correction_arms_treat_nan_as_never_significant() {
        // t_sq = [25.0, NaN, 9.0] with crits that make 25.0 and 9.0 pass.
        // The NaN slot (index 1) must never appear as 1 in output.
        let t_sq = vec![25.0_f64, f64::NAN, 9.0];
        let crit_unc = 4.0;
        let crit_sq = vec![4.0_f64, 4.0, 4.0];
        for m in [
            CorrectionMethod::None,
            CorrectionMethod::Bonferroni,
            CorrectionMethod::TukeyHsd,
            CorrectionMethod::Holm,
            CorrectionMethod::BenjaminiHochberg,
        ] {
            let mut out = vec![0u8; 3];
            apply_correction(m, &t_sq, crit_unc, &crit_sq, &mut out);
            assert_eq!(
                out[1], 0,
                "method {m:?}: NaN entry must never be significant"
            );
            // Finite entries that exceed the threshold must still pass.
            assert_eq!(out[0], 1, "method {m:?}: t²=25 > crit=4 must pass");
            assert_eq!(out[2], 1, "method {m:?}: t²=9 > crit=4 must pass");
        }
        // Holm step-down: verify NaN does not cause early termination for the
        // finite entries. NaN sorts last (total_cmp), so all finite entries are
        // evaluated before Holm reaches the NaN entry.
        // t_sq = [25.0, NaN, 9.0] sorted desc → [0:25, 2:9, 1:NaN].
        // Rank 0: 25 > 4 → pass. Rank 1: 9 > 4 → pass. Rank 2: NaN → break
        // (Holm does break here, but both finite entries already passed — harmless).
        let mut holm_out = vec![0u8; 3];
        apply_correction(
            CorrectionMethod::Holm,
            &t_sq,
            crit_unc,
            &crit_sq,
            &mut holm_out,
        );
        assert_eq!(
            holm_out,
            vec![1, 0, 1],
            "Holm: NaN must not cause early stop before finite passing entries"
        );
    }

    #[test]
    fn sort_tie_breaker_is_ascending_index() {
        let t_sq = vec![5.0_f64, 5.0, 5.0];
        let order = sort_indices_desc(&t_sq);
        assert_eq!(order, vec![0, 1, 2], "ties must break on ascending index");
    }

    #[test]
    fn sort_nans_go_last() {
        let t_sq = vec![1.0_f64, f64::NAN, 3.0, f64::NAN, 2.0];
        let order = sort_indices_desc(&t_sq);
        // Descending: 3, 2, 1, then NaNs in ascending index order.
        assert_eq!(order, vec![2, 4, 0, 1, 3]);
    }

    // -----------------------------------------------------------------
    // C5 — BH step-up pull-through: target with rank-0 t² < rank-0 crit
    // passes because later ranks set last_sig. Bonferroni would reject it.
    // -----------------------------------------------------------------

    #[test]
    fn bh_step_up_pull_through_property() {
        // t² = [4, 25, 1]: sorted desc → rank0=target1(25), rank1=target0(4), rank2=target2(1)
        // crit_sq by rank = [30, 3, 0.5]
        // Walk: rank0 25>30? no.  rank1 4>3? yes (last_sig=1).  rank2 1>0.5? yes (last_sig=2).
        // → all pass. target1 (rank0, t²=25) passes ONLY via pull-through (25 < 30).
        // A broken BH that applies rank-0 threshold to all would return [1, 0, 1]
        // (target1 fails) — the assert_eq below catches that.
        let t_sq = vec![4.0_f64, 25.0, 1.0];
        let crit_sq = [30.0_f64, 3.0, 0.5];
        let mut out = vec![0u8; 3];
        apply_correction(
            CorrectionMethod::BenjaminiHochberg,
            &t_sq,
            0.5,
            &crit_sq,
            &mut out,
        );
        assert_eq!(out, vec![1, 1, 1], "BH: all pass via step-up pull-through");

        // Discriminator: Bonferroni with rank-0 threshold (crit=30) applied to all
        // targets → target1 (t²=25 < 30) fails. This proves the step-up pull-through
        // is doing real work — without it, target1 would not pass.
        let bonf_crit = [30.0_f64, 30.0, 30.0];
        let mut out_bonf = vec![0u8; 3];
        apply_correction(
            CorrectionMethod::Bonferroni,
            &t_sq,
            30.0,
            &bonf_crit,
            &mut out_bonf,
        );
        assert_eq!(
            out_bonf[1], 0,
            "Bonferroni: target1 (t²=25 < crit=30) must fail without pull-through"
        );
    }
}
