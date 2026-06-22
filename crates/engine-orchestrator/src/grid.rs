//! Sample-size grid construction + per-target first-N lookup.
//!
//! `build_grid` builds a cluster-aware sample-size grid: it snaps endpoints
//! to multiples of `atom` (the cluster atom), enforces `hard_min`, pins the
//! `to` endpoint even when the regular step skips it, auto-counts points from
//! the feasible span, and emits advisory warnings when bounds were adjusted or
//! the grid is coarse.
//!
//! Internal-stability surface: `pub` for the crate's integration tests, not
//! re-exported at the crate root and not part of the port contract — free to
//! change at any minor version.

use crate::fit::fit_crossing;
use crate::result::{ByValue, CrossingFit, GridMode, OrchestratorError, PowerResult};

#[inline]
fn ceil_to(x: usize, a: usize) -> usize {
    let a = a.max(1);
    x.div_ceil(a) * a
}
#[inline]
fn floor_to(x: usize, a: usize) -> usize {
    let a = a.max(1);
    (x / a) * a
}
#[inline]
fn round_to(x: usize, a: usize) -> usize {
    let a = a.max(1);
    ((x + a / 2) / a) * a
}

/// Build a cluster-aware sample-size grid.
///
/// `atom` is the smallest legal N increment (`ClusterSizing::atom()`, or 1 when
/// unclustered). `hard_min` is the regime floor (e.g. `n_clusters *
/// MIN_ROWS_PER_CLUSTER`, or 1). Returns the grid plus advisory warnings.
pub fn build_grid(
    from_req: usize,
    to_req: usize,
    by: ByValue,
    mode: GridMode,
    atom: usize,
    hard_min: usize,
) -> Result<(Vec<usize>, Vec<String>), OrchestratorError> {
    if atom == 0 {
        return Err(OrchestratorError::InvalidClusterAtom);
    }
    // Guard 2: raw bounds.
    if from_req < 1 || to_req < from_req {
        return Err(OrchestratorError::InvalidGridBounds {
            from: from_req,
            to: to_req,
            mode,
        });
    }

    let mut warnings: Vec<String> = Vec::new();

    // Bounds (min/max behaviour, cluster-folded).
    let snapped_from = ceil_to(from_req, atom).max(ceil_to(hard_min, atom));
    let snapped_to = floor_to(to_req, atom);

    // Guard 3 / 4: report bound adjustments.
    if snapped_from > from_req {
        warnings.push(format!(
            "raised `from` from {from_req} to {snapped_from} so each cluster keeps enough rows (atom={atom})"
        ));
    }
    if snapped_to < to_req {
        warnings.push(format!(
            "lowered `to` from {to_req} to {snapped_to} (nearest multiple of the cluster atom {atom})"
        ));
    }

    // Guard 5: range collapse.
    if snapped_to < snapped_from {
        return Err(OrchestratorError::ClusterGridEmpty {
            from: snapped_from,
            to: snapped_to,
            atom,
        });
    }

    let max_feasible = (snapped_to - snapped_from) / atom + 1;

    // Guard 6: single point.
    if max_feasible < 2 {
        return Err(OrchestratorError::ClusterGridSinglePoint {
            from: snapped_from,
            to: snapped_to,
            atom,
        });
    }
    // Guard 7: coarse crossing.
    if max_feasible < 4 {
        warnings.push(format!(
            "only {max_feasible} grid points after cluster snapping; required-N (and model-based crossing) will be coarse"
        ));
    }

    let grid = match mode {
        GridMode::Linear => {
            let step = match by {
                ByValue::Fixed(s) => ceil_to(s, atom).max(atom),
                ByValue::Auto { count } => {
                    let c = count.clamp(2, max_feasible);
                    let span = snapped_to - snapped_from;
                    let denom = atom * (c - 1);
                    atom * ((span + denom / 2) / denom).max(1)
                }
            };
            let mut out = Vec::new();
            let mut n = snapped_from;
            while n <= snapped_to {
                out.push(n);
                n += step;
            }
            if *out.last().unwrap() != snapped_to {
                out.push(snapped_to);
            }
            out
        }
        GridMode::Log => {
            let c = match by {
                ByValue::Fixed(points) => points.clamp(2, max_feasible),
                ByValue::Auto { count } => count.clamp(2, max_feasible),
            };
            let lo = (snapped_from as f64).log10();
            let hi = (snapped_to as f64).log10();
            let mut raw: Vec<usize> = (0..c)
                .map(|i| {
                    let t = i as f64 / (c - 1) as f64;
                    let v = 10f64.powf(lo + t * (hi - lo)).round() as usize;
                    round_to(v, atom).clamp(snapped_from, snapped_to)
                })
                .collect();
            raw[0] = snapped_from;
            *raw.last_mut().unwrap() = snapped_to;
            raw.sort_unstable();
            raw.dedup();
            raw
        }
    };

    Ok((grid, warnings))
}

/// Normalize a target-power value to a proportion in `[0, 1]`: values `> 1.0`
/// are interpreted as percentages (Python parity), e.g. `80.0 → 0.8`.
#[inline]
pub(crate) fn as_proportion(p: f64) -> f64 {
    if p > 1.0 {
        p / 100.0
    } else {
        p
    }
}

/// Smallest N at which `powers_corrected[i][target_index] >= p_target`,
/// or `None` if the target is never reached on the grid. Accepts
/// `target_power > 1.0` as a percentage (Python parity).
pub fn first_n_at_target(
    powers_corrected: &[Vec<f64>],
    sample_sizes: &[usize],
    target_power: f64,
    target_index: usize,
) -> Option<usize> {
    let p_target = as_proportion(target_power);
    for (i, &n) in sample_sizes.iter().enumerate() {
        if powers_corrected[i][target_index] >= p_target {
            return Some(n);
        }
    }
    None
}

/// Per-target required-N from a per-N `PowerResult` slice: the smallest grid N at
/// which each target independently meets `target_power`, read off the **corrected**
/// power. Shared by the multi-core, single-core, and merge sample-size paths (each
/// holds a `Vec<PowerResult>` — fresh `aggregate_batch` output or pooled results).
/// The joint counterpart (`first_joint_achieved`) is computed per-path from the
/// per-N histograms.
pub fn derive_first_achieved(
    per_n: &[PowerResult],
    sample_sizes: &[usize],
    target_power: f64,
) -> Vec<Option<usize>> {
    let powers_corrected: Vec<Vec<f64>> =
        per_n.iter().map(|pr| pr.power_corrected.clone()).collect();
    // Entry count = marginals + contrasts (the power vectors' own length), so
    // contrast targets get a required-N record too.
    let n_targets = per_n
        .first()
        .map(|pr| pr.power_corrected.len())
        .unwrap_or(0);
    (0..n_targets)
        .map(|t| first_n_at_target(&powers_corrected, sample_sizes, target_power, t))
        .collect()
}

/// All four sample-size derivations from one per-N `PowerResult` slice: the
/// grid-empirical `first_achieved` / `first_joint_achieved` plus the
/// model-based `fitted` / `fitted_joint` crossing fits.
pub(crate) struct SampleSizeDerivations {
    pub first_achieved: Vec<Option<usize>>,
    pub first_joint_achieved: Vec<Option<usize>>,
    pub fitted: Vec<CrossingFit>,
    pub fitted_joint: Vec<CrossingFit>,
    /// Grid-empirical / model-based crossing for the overall (omnibus) test.
    /// Both `None` when the grid carries no overall test (OLS F / unclustered
    /// GLM LRT only); singular, since the omnibus is one test.
    pub first_overall_achieved: Option<usize>,
    pub fitted_overall: Option<CrossingFit>,
}

/// Derive the four sample-size outputs from the per-N results. One helper
/// shared by the multi-core, single-core, and merge paths so they cannot
/// drift: the dispatch twins must stay value-identical, and merge recomputes
/// everything from pooled counts (the fit is deterministic over counts, so
/// per-worker fitted values are simply discarded and recomputed there).
///
/// `n_sims` for the joint family comes from the histogram row sum, not the
/// `n_sims` field — under merge's tolerant skip-on-empty histogram pooling
/// the two can differ, and the row sum is the denominator the pooled buckets
/// were accumulated against. Identical for fresh `aggregate_batch` output
/// (rows sum to `n_sims` by invariant).
pub(crate) fn derive_sample_size_outputs(
    per_n: &[PowerResult],
    sample_sizes: &[usize],
    target_power: f64,
    atom: usize,
) -> SampleSizeDerivations {
    let first_achieved = derive_first_achieved(per_n, sample_sizes, target_power);

    let n_sims = per_n.first().map(|pr| pr.n_sims).unwrap_or(0);
    // Marginals + contrasts — mirrors derive_first_achieved above.
    let n_targets = per_n
        .first()
        .map(|pr| pr.success_counts_corrected.len())
        .unwrap_or(0);
    let fitted: Vec<CrossingFit> = (0..n_targets)
        .map(|t| {
            let counts: Vec<u64> = per_n
                .iter()
                .map(|pr| pr.success_counts_corrected[t])
                .collect();
            fit_crossing(sample_sizes, &counts, n_sims, target_power, atom)
        })
        .collect();

    // Joint family from the per-N CORRECTED histograms (matching
    // `first_achieved`, which reads corrected power). Histograms absent ⇒
    // both joint vecs empty. Index j is k = j+1; the bound comes from the
    // histogram length so post-hoc contrasts are included.
    let histograms_corrected: Vec<Vec<u64>> = per_n
        .iter()
        .map(|pr| pr.success_count_histogram_corrected.clone())
        .collect();
    let joint_n_sims: u64 = histograms_corrected
        .first()
        .map(|h| h.iter().sum())
        .unwrap_or(0);
    let n_joint = histograms_corrected
        .first()
        .map(|h| h.len().saturating_sub(1))
        .unwrap_or(0);
    let first_joint_achieved: Vec<Option<usize>> = (1..=n_joint)
        .map(|k| {
            first_n_joint_at_target(
                &histograms_corrected,
                joint_n_sims,
                sample_sizes,
                target_power,
                k,
            )
        })
        .collect();
    let fitted_joint: Vec<CrossingFit> = (1..=n_joint)
        .map(|k| {
            // P(>=k) successes per grid point = tail sum of buckets j >= k.
            let counts: Vec<u64> = histograms_corrected
                .iter()
                .map(|h| h.iter().skip(k).sum())
                .collect();
            fit_crossing(sample_sizes, &counts, joint_n_sims, target_power, atom)
        })
        .collect();

    // Overall (omnibus) crossing: a single test, gated on the grid actually
    // carrying it (OLS F / unclustered GLM LRT). The grid-empirical scan
    // mirrors `first_n_at_target` (overall has no corrected variant, so the
    // single rate is used) and the model-based fit reuses `fit_crossing` over
    // the per-N overall success counts — identical machinery to every marginal.
    let has_overall = per_n
        .first()
        .map(|pr| pr.overall_significant_rate.is_some())
        .unwrap_or(false);
    let (first_overall_achieved, fitted_overall) = if has_overall {
        let p_target = as_proportion(target_power);
        let first = sample_sizes.iter().zip(per_n.iter()).find_map(|(&n, pr)| {
            (pr.overall_significant_rate.unwrap_or(0.0) >= p_target).then_some(n)
        });
        let overall_counts: Vec<u64> = per_n
            .iter()
            .map(|pr| pr.overall_significant_count)
            .collect();
        let fitted_overall =
            fit_crossing(sample_sizes, &overall_counts, n_sims, target_power, atom);
        (first, Some(fitted_overall))
    } else {
        (None, None)
    };

    SampleSizeDerivations {
        first_achieved,
        first_joint_achieved,
        fitted,
        fitted_joint,
        first_overall_achieved,
        fitted_overall,
    }
}

/// Smallest N at which P(at least `k` tests jointly significant) >= `target_power`.
///
/// `histograms_corrected[i]` is the per-N **corrected** success-count histogram at
/// `sample_sizes[i]`: bucket `j` counts sims with exactly `j` tests significant
/// (main targets + post-hoc contrasts), `len() == n_joint + 1`. P(>=k) at a grid
/// point = (sum_{j>=k} hist[j]) / n_sims.
///
/// INVARIANT: each `histograms_corrected[i]` must sum to `n_sims`, otherwise
/// P(>=k) can exceed 1. This function divides by `n_sims` itself (cf.
/// `first_n_at_target`, which receives already-divided powers).
pub fn first_n_joint_at_target(
    histograms_corrected: &[Vec<u64>],
    n_sims: u64,
    sample_sizes: &[usize],
    target_power: f64,
    k: usize,
) -> Option<usize> {
    if n_sims == 0 {
        return None;
    }
    let p_target = as_proportion(target_power);
    for (i, &n) in sample_sizes.iter().enumerate() {
        let hist = &histograms_corrected[i];
        // P(>=k) = sum of buckets j = k..len  (bucket j = "exactly j significant").
        let ge_k: u64 = hist.iter().skip(k).sum();
        let p = ge_k as f64 / n_sims as f64;
        if p >= p_target {
            return Some(n);
        }
    }
    None
}

#[cfg(test)]
mod grid_builder_tests {
    use super::*;
    use crate::result::{ByValue, GridMode};

    // No clusters: atom=1, hard_min=1.
    #[test]
    fn auto_count_unclustered_linear_pins_endpoints() {
        let (g, w) =
            build_grid(30, 200, ByValue::Auto { count: 12 }, GridMode::Linear, 1, 1).unwrap();
        assert_eq!(*g.first().unwrap(), 30, "starts at from");
        assert_eq!(*g.last().unwrap(), 200, "ends at to (endpoint pinned)");
        assert!(
            g.len() >= 11 && g.len() <= 13,
            "~12 points, got {}",
            g.len()
        );
        assert!(w.is_empty(), "no cluster warnings unclustered");
        assert!(g.windows(2).all(|p| p[0] < p[1]));
    }

    // Note: the basic linear walk [30,35,40,45,50] is pinned by the integration
    // test `linear_grid_matches_python_range` (tests/test_grid.rs), which also
    // documents the Python range() parity — not duplicated here.

    #[test]
    fn fixed_step_unclustered_pins_dropped_endpoint() {
        let (g, _) = build_grid(30, 52, ByValue::Fixed(5), GridMode::Linear, 1, 1).unwrap();
        assert_eq!(*g.last().unwrap(), 52, "max pinned even when step skips it");
        assert_eq!(g[0], 30);
    }

    // Regime A: atom = n_clusters = 20, hard_min = 20*2 = 40.
    #[test]
    fn auto_count_regime_a_every_point_multiple_of_atom() {
        let (g, w) = build_grid(
            30,
            200,
            ByValue::Auto { count: 12 },
            GridMode::Linear,
            20,
            40,
        )
        .unwrap();
        assert!(
            g.iter().all(|&n| n % 20 == 0),
            "every N a multiple of atom 20: {g:?}"
        );
        assert_eq!(
            *g.first().unwrap(),
            40,
            "from raised to hard_min (40), snapped to atom"
        );
        assert_eq!(
            *g.last().unwrap(),
            200,
            "to snapped down to a multiple of 20"
        );
        assert!(
            w.iter().any(|m| m.contains("raised")),
            "warns it raised from"
        );
    }

    #[test]
    fn regime_a_to_lowered_to_atom_multiple_warns() {
        let (g, w) = build_grid(
            40,
            205,
            ByValue::Auto { count: 12 },
            GridMode::Linear,
            20,
            40,
        )
        .unwrap();
        assert_eq!(*g.last().unwrap(), 200);
        assert!(w.iter().any(|m| m.contains("lowered")));
    }

    #[test]
    fn range_collapse_after_snapping_errors() {
        let r = build_grid(
            190,
            200,
            ByValue::Auto { count: 12 },
            GridMode::Linear,
            60,
            60,
        );
        assert!(matches!(r, Err(OrchestratorError::ClusterGridEmpty { .. })));
    }

    #[test]
    fn single_feasible_point_errors() {
        let r = build_grid(
            40,
            59,
            ByValue::Auto { count: 12 },
            GridMode::Linear,
            20,
            40,
        );
        assert!(matches!(
            r,
            Err(OrchestratorError::ClusterGridSinglePoint { .. })
        ));
    }

    #[test]
    fn coarse_crossing_warns_when_few_points() {
        let (g, w) = build_grid(
            40,
            80,
            ByValue::Auto { count: 12 },
            GridMode::Linear,
            20,
            40,
        )
        .unwrap();
        assert!(g.len() < 4);
        assert!(w
            .iter()
            .any(|m| m.contains("coarse") || m.contains("grid points")));
    }

    #[test]
    fn auto_count_caps_at_feasible() {
        let (g, _) = build_grid(
            40,
            120,
            ByValue::Auto { count: 12 },
            GridMode::Linear,
            20,
            40,
        )
        .unwrap();
        assert_eq!(g, vec![40, 60, 80, 100, 120]);
    }

    #[test]
    fn log_mode_points_are_atom_multiples_sorted_unique() {
        let (g, _) =
            build_grid(40, 400, ByValue::Auto { count: 8 }, GridMode::Log, 20, 40).unwrap();
        assert!(g.iter().all(|&n| n % 20 == 0));
        assert_eq!(*g.first().unwrap(), 40);
        assert_eq!(*g.last().unwrap(), 400);
        assert!(g.windows(2).all(|p| p[0] < p[1]), "sorted + deduped");
    }
}

#[cfg(test)]
mod overall_derivation_tests {
    use super::*;
    use crate::result::{Ci, EstimatorExtras, PowerResult};

    /// One grid point carrying only the fields `derive_sample_size_outputs`
    /// reads for the overall crossing (plus the minimum to keep the marginal
    /// path well-formed). `overall` is `Some((rate, count))` or `None`.
    fn pr(n: usize, n_sims: u64, overall: Option<(f64, u64)>) -> PowerResult {
        let (rate, count) = match overall {
            Some((r, c)) => (Some(r), c),
            None => (None, 0),
        };
        PowerResult {
            n,
            n_sims,
            target_indices: vec![0],
            contrast_pairs: vec![],
            power_uncorrected: vec![0.0],
            power_corrected: vec![0.0],
            ci_uncorrected: vec![Ci { lo: 0.0, hi: 0.0 }],
            ci_corrected: vec![Ci { lo: 0.0, hi: 0.0 }],
            convergence_rate: 1.0,
            boundary_hit: vec![],
            estimator_extras: EstimatorExtras::Ols {},
            overall_significant_rate: rate,
            success_counts_uncorrected: vec![0],
            success_counts_corrected: vec![0],
            convergence_count: n_sims,
            overall_significant_count: count,
            overall_significant_ci: None,
            success_count_histogram_uncorrected: vec![],
            success_count_histogram_corrected: vec![],
            grid_warnings: vec![],
            posthoc: vec![],
            factor_exclusion_counts: vec![],
            factor_separation_counts: vec![],
        }
    }

    #[test]
    fn overall_crossing_matches_direct_fit_and_hand_scan() {
        // 3-point grid, n_sims = 100. overall rates 0.45 / 0.70 / 0.90.
        let n_sims = 100u64;
        let grid = vec![50usize, 100, 150];
        let counts = vec![45u64, 70, 90];
        let per_n: Vec<PowerResult> = grid
            .iter()
            .zip(counts.iter())
            .map(|(&n, &c)| pr(n, n_sims, Some((c as f64 / n_sims as f64, c))))
            .collect();

        let d = derive_sample_size_outputs(&per_n, &grid, 80.0, 1);
        // First N where rate >= 0.80 is the third point (0.90).
        assert_eq!(d.first_overall_achieved, Some(150));
        // The model-based fit is exactly fit_crossing over the overall counts.
        let direct = fit_crossing(&grid, &counts, n_sims, 80.0, 1);
        assert_eq!(d.fitted_overall, Some(direct));
    }

    #[test]
    fn overall_crossing_none_when_grid_has_no_overall() {
        let grid = vec![50usize, 100, 150];
        let per_n: Vec<PowerResult> = grid.iter().map(|&n| pr(n, 100, None)).collect();
        let d = derive_sample_size_outputs(&per_n, &grid, 80.0, 1);
        assert_eq!(d.first_overall_achieved, None);
        assert_eq!(d.fitted_overall, None);
    }
}

#[cfg(test)]
mod joint_tests {
    use super::*;

    // Hand-built: 3 grid points, n_targets = 2 => histogram length 3 (k=0,1,2).
    // n_sims = 100 at every grid point. Bucket-sum invariant: each row sums to 100.
    fn fixture() -> (Vec<Vec<u64>>, Vec<usize>) {
        let histograms = vec![
            vec![80, 15, 5],  // N=50:  P(>=1)=0.20, P(>=2)=0.05
            vec![40, 40, 20], // N=100: P(>=1)=0.60, P(>=2)=0.20
            vec![5, 35, 60],  // N=200: P(>=1)=0.95, P(>=2)=0.60
        ];
        let sample_sizes = vec![50usize, 100, 200];
        (histograms, sample_sizes)
    }

    #[test]
    fn bucket_sum_equals_n_sims_invariant() {
        let (h, _) = fixture();
        for row in &h {
            assert_eq!(
                row.iter().sum::<u64>(),
                100,
                "each histogram row must sum to n_sims"
            );
        }
    }

    #[test]
    fn first_n_for_k1_and_k2_with_50pct_target() {
        let (h, ss) = fixture();
        // P(>=1): 0.20, 0.60, 0.95 -> first >= 0.50 at N=100.
        assert_eq!(first_n_joint_at_target(&h, 100, &ss, 0.50, 1), Some(100));
        // P(>=2): 0.05, 0.20, 0.60 -> first >= 0.50 at N=200.
        assert_eq!(first_n_joint_at_target(&h, 100, &ss, 0.50, 2), Some(200));
    }

    #[test]
    fn required_n_is_non_decreasing_in_k() {
        let (h, ss) = fixture();
        let n1 = first_n_joint_at_target(&h, 100, &ss, 0.50, 1).unwrap();
        let n2 = first_n_joint_at_target(&h, 100, &ss, 0.50, 2).unwrap();
        assert!(n2 >= n1, "required N for >=k must be non-decreasing in k");
    }

    #[test]
    fn unreached_target_returns_none() {
        let (h, ss) = fixture();
        // P(>=2) tops out at 0.60; a 0.90 target is never reached.
        assert_eq!(first_n_joint_at_target(&h, 100, &ss, 0.90, 2), None);
    }

    #[test]
    fn percentage_target_is_divided_by_100() {
        let (h, ss) = fixture();
        // 50.0 (percent) must behave identically to 0.50 (proportion).
        assert_eq!(
            first_n_joint_at_target(&h, 100, &ss, 50.0, 1),
            first_n_joint_at_target(&h, 100, &ss, 0.50, 1),
        );
    }
}
