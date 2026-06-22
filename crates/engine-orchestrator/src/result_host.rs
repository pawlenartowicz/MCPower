//! Host-neutral projection of the orchestrator result types.
//!
//! `PowerResult` / `SampleSizeResult` are projected into a single ordered
//! `HostValue` tree here, once. Each host port then walks that tree with a
//! single generic converter (`host_value_to_py`, `host_value_to_robj`, …)
//! instead of hand-mirroring field names, field order, the boundary_hit
//! reshape, the index-map keying, and the estimator-extras encoding. This is
//! the single source of truth for the **per-scenario** result shape across all
//! ports; the trivial `{scenarios, comparison}` envelope (whose keys are the
//! dynamic scenario labels) stays in each port's idiom.
//!
//! The tree owns field names, field order, and the leaf *values*. Each host's
//! walker owns only how a leaf *kind* is realized in that host (a `Usize` is a
//! Python int but an R integer vector; `BoundaryHit` is a 2-D nested list in
//! Python but a flat R vector; an `IndexMap` is keyed by int in Python but by
//! string in R).
//! Those are genuine, consumer-invisible per-port idioms, not drift.
//!
//! `#[derive(Serialize)]` lets the future app-spec / wasm ports get a JSON
//! encoding of the same tree for free.

use crate::result::{
    Ci, CrossingFit, EstimatorExtras, PosthocPower, PowerResult, SampleSizeResult,
};
use serde::Serialize;

/// Host-neutral value tree. Built once per scenario by the `*_to_host`
/// functions below and walked by each port's generic converter.
#[derive(Debug, Clone, Serialize)]
pub enum HostValue {
    F64(f64),
    OptF64(Option<f64>),
    Usize(usize),
    OptUsize(Option<usize>),
    VecF64(Vec<f64>),
    VecU64(Vec<u64>),
    VecStr(Vec<String>),
    VecCi(Vec<Ci>),
    OptCi(Option<Ci>),
    Str(String),
    /// Struct-like map with static keys (a result's fields, estimator extras,
    /// a post-hoc block). Order is significant and host-uniform.
    Map(Vec<(&'static str, HostValue)>),
    /// Ordered list (e.g. the per-N power vectors, the post-hoc blocks).
    Seq(Vec<HostValue>),
    /// A sequence exposed to the host as a mapping keyed by 0-based index —
    /// Python renders int keys, R renders string keys. Used by
    /// `first_achieved` / `first_joint_achieved`.
    IndexMap(Vec<HostValue>),
    /// Per-simulation boundary_hit codes in row-major `(rows=n_sims,
    /// cols=n_sample_sizes)` order. Python realizes a 2-D nested list (list of
    /// per-sim rows); R keeps the flat vector.
    BoundaryHit {
        flat: Vec<u8>,
        rows: usize,
        cols: usize,
    },
}

/// Project one `PowerResult` (a `find_power` scenario) into the neutral tree.
/// `label` is the scenario name (the `scenario` field). Field order is the
/// canonical host order every port reproduces.
pub fn power_result_to_host(pr: &PowerResult, label: &str) -> HostValue {
    let n_sims_bh = pr.boundary_hit.len();
    HostValue::Map(vec![
        ("n_sims", HostValue::Usize(pr.n_sims as usize)),
        ("n_sample_sizes", HostValue::Usize(1)),
        ("n_targets", HostValue::Usize(pr.target_indices.len())),
        ("target_indices", vec_usize(&pr.target_indices)),
        ("contrast_pairs", contrast_pairs_to_host(&pr.contrast_pairs)),
        (
            "power_uncorrected",
            HostValue::Seq(vec![HostValue::VecF64(pr.power_uncorrected.clone())]),
        ),
        (
            "power_corrected",
            HostValue::Seq(vec![HostValue::VecF64(pr.power_corrected.clone())]),
        ),
        (
            "ci_uncorrected",
            HostValue::Seq(vec![HostValue::VecCi(pr.ci_uncorrected.clone())]),
        ),
        (
            "ci_corrected",
            HostValue::Seq(vec![HostValue::VecCi(pr.ci_corrected.clone())]),
        ),
        (
            "convergence_rate",
            HostValue::VecF64(vec![pr.convergence_rate]),
        ),
        ("scenario", HostValue::Str(label.to_string())),
        // `pr.n` (the snapped N actually run), never the caller-supplied N —
        // they differ when cluster-atom snapping fires.
        ("sample_sizes", HostValue::VecU64(vec![pr.n as u64])),
        (
            "estimator_extras",
            estimator_extras_to_host(&pr.estimator_extras),
        ),
        (
            "boundary_hit",
            HostValue::BoundaryHit {
                flat: pr.boundary_hit.clone(),
                rows: n_sims_bh,
                cols: 1,
            },
        ),
        (
            "overall_significant_rate",
            HostValue::OptF64(pr.overall_significant_rate),
        ),
        (
            "overall_significant_ci",
            HostValue::OptCi(pr.overall_significant_ci),
        ),
        (
            "success_count_histogram_uncorrected",
            HostValue::VecU64(pr.success_count_histogram_uncorrected.clone()),
        ),
        (
            "success_count_histogram_corrected",
            HostValue::VecU64(pr.success_count_histogram_corrected.clone()),
        ),
        ("grid_warnings", HostValue::VecStr(pr.grid_warnings.clone())),
        (
            "posthoc",
            HostValue::Seq(pr.posthoc.iter().map(posthoc_to_host).collect()),
        ),
        (
            "factor_exclusion_counts",
            HostValue::VecU64(pr.factor_exclusion_counts.clone()),
        ),
        (
            "factor_separation_counts",
            HostValue::VecU64(pr.factor_separation_counts.clone()),
        ),
    ])
}

/// Project one `SampleSizeResult` (a `find_sample_size` scenario) into the
/// neutral tree. The per-N fields become `Seq`s over `grid_or_trace`.
pub fn sample_size_result_to_host(ssr: &SampleSizeResult, label: &str) -> HostValue {
    let grid = &ssr.grid_or_trace;
    let first = grid.first();
    let n_sims = first.map(|pr| pr.n_sims as usize).unwrap_or(0);
    let n_targets = first.map(|pr| pr.target_indices.len()).unwrap_or(0);
    let target_indices: Vec<u64> = first
        .map(|pr| pr.target_indices.iter().map(|&i| i as u64).collect())
        .unwrap_or_default();
    let (bh_flat, bh_rows, bh_cols) = interleave_boundary_hit(grid);

    let contrast_pairs = first.map(|pr| pr.contrast_pairs.as_slice()).unwrap_or(&[]);

    HostValue::Map(vec![
        ("n_sims", HostValue::Usize(n_sims)),
        ("n_sample_sizes", HostValue::Usize(grid.len())),
        ("n_targets", HostValue::Usize(n_targets)),
        ("target_indices", HostValue::VecU64(target_indices)),
        ("contrast_pairs", contrast_pairs_to_host(contrast_pairs)),
        (
            "power_uncorrected",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecF64(pr.power_uncorrected.clone()))
                    .collect(),
            ),
        ),
        (
            "power_corrected",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecF64(pr.power_corrected.clone()))
                    .collect(),
            ),
        ),
        (
            "ci_uncorrected",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecCi(pr.ci_uncorrected.clone()))
                    .collect(),
            ),
        ),
        (
            "ci_corrected",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecCi(pr.ci_corrected.clone()))
                    .collect(),
            ),
        ),
        (
            "convergence_rate",
            HostValue::VecF64(grid.iter().map(|pr| pr.convergence_rate).collect()),
        ),
        (
            "sample_sizes",
            HostValue::VecU64(grid.iter().map(|pr| pr.n as u64).collect()),
        ),
        (
            "first_achieved",
            HostValue::IndexMap(
                ssr.first_achieved
                    .iter()
                    .map(|v| HostValue::OptUsize(*v))
                    .collect(),
            ),
        ),
        (
            "success_count_histogram_uncorrected",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecU64(pr.success_count_histogram_uncorrected.clone()))
                    .collect(),
            ),
        ),
        (
            "success_count_histogram_corrected",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecU64(pr.success_count_histogram_corrected.clone()))
                    .collect(),
            ),
        ),
        (
            "grid_warnings",
            HostValue::VecStr(ssr.grid_warnings.clone()),
        ),
        (
            "factor_exclusion_counts",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecU64(pr.factor_exclusion_counts.clone()))
                    .collect(),
            ),
        ),
        (
            "factor_separation_counts",
            HostValue::Seq(
                grid.iter()
                    .map(|pr| HostValue::VecU64(pr.factor_separation_counts.clone()))
                    .collect(),
            ),
        ),
        (
            "first_joint_achieved",
            HostValue::IndexMap(
                ssr.first_joint_achieved
                    .iter()
                    .map(|v| HostValue::OptUsize(*v))
                    .collect(),
            ),
        ),
        (
            "first_overall_achieved",
            HostValue::OptUsize(ssr.first_overall_achieved),
        ),
        ("target_power", HostValue::F64(ssr.target_power)),
        ("scenario", HostValue::Str(label.to_string())),
        (
            "boundary_hit",
            HostValue::BoundaryHit {
                flat: bh_flat,
                rows: bh_rows,
                cols: bh_cols,
            },
        ),
        (
            "fitted",
            HostValue::IndexMap(ssr.fitted.iter().map(crossing_fit_to_host).collect()),
        ),
        (
            "fitted_joint",
            HostValue::IndexMap(ssr.fitted_joint.iter().map(crossing_fit_to_host).collect()),
        ),
        // Singular overall crossing surfaced as a 0-or-1-element IndexMap (no
        // null/optional-Map HostValue variant): empty `[]` when no overall test
        // was requested, `[fit]` when present. Hosts read it as
        // `fitted_overall.get(0)`, the same idiom `fitted` uses per position.
        (
            "fitted_overall",
            HostValue::IndexMap(
                ssr.fitted_overall
                    .iter()
                    .map(crossing_fit_to_host)
                    .collect(),
            ),
        ),
        ("cluster_atom", HostValue::Usize(ssr.cluster_atom)),
    ])
}

/// Project one `CrossingFit` as a status-tagged map, mirroring the serde
/// `status` tag so hosts read the same discriminator on both encodings.
fn crossing_fit_to_host(cf: &CrossingFit) -> HostValue {
    match cf {
        CrossingFit::Fitted {
            n_star,
            n_achievable,
            ci_lo,
            ci_hi,
        } => HostValue::Map(vec![
            ("status", HostValue::Str("fitted".into())),
            ("n_star", HostValue::F64(*n_star)),
            ("n_achievable", HostValue::Usize(*n_achievable)),
            ("ci_lo", HostValue::OptF64(*ci_lo)),
            ("ci_hi", HostValue::OptF64(*ci_hi)),
        ]),
        CrossingFit::AtOrBelowMin { n_min } => HostValue::Map(vec![
            ("status", HostValue::Str("at_or_below_min".into())),
            ("n_min", HostValue::Usize(*n_min)),
        ]),
        CrossingFit::NotReached { n_approx } => HostValue::Map(vec![
            ("status", HostValue::Str("not_reached".into())),
            ("n_approx", HostValue::OptUsize(*n_approx)),
        ]),
        CrossingFit::NonMonotone { max_violation } => HostValue::Map(vec![
            ("status", HostValue::Str("non_monotone".into())),
            ("max_violation", HostValue::F64(*max_violation)),
        ]),
    }
}

fn vec_usize(v: &[usize]) -> HostValue {
    HostValue::VecU64(v.iter().map(|&i| i as u64).collect())
}

/// `contrast_pairs` as a Seq of `[positive, negative]` two-element vectors —
/// Python renders a list of lists, R a list of integer vectors. Empty Seq when
/// no contrasts were requested.
fn contrast_pairs_to_host(pairs: &[(u32, u32)]) -> HostValue {
    HostValue::Seq(
        pairs
            .iter()
            .map(|&(p, n)| HostValue::VecU64(vec![p as u64, n as u64]))
            .collect(),
    )
}

fn posthoc_to_host(ph: &PosthocPower) -> HostValue {
    HostValue::Map(vec![
        ("n_levels", HostValue::Usize(ph.n_levels)),
        (
            "power_uncorrected",
            HostValue::VecF64(ph.power_uncorrected.clone()),
        ),
        (
            "power_corrected",
            HostValue::VecF64(ph.power_corrected.clone()),
        ),
        (
            "ci_uncorrected",
            HostValue::VecCi(ph.ci_uncorrected.clone()),
        ),
        ("ci_corrected", HostValue::VecCi(ph.ci_corrected.clone())),
        (
            "success_counts_uncorrected",
            HostValue::VecU64(ph.success_counts_uncorrected.clone()),
        ),
        (
            "success_counts_corrected",
            HostValue::VecU64(ph.success_counts_corrected.clone()),
        ),
    ])
}

fn estimator_extras_to_host(fx: &EstimatorExtras) -> HostValue {
    match fx {
        EstimatorExtras::Ols {} => {
            HostValue::Map(vec![("estimator", HostValue::Str("ols".into()))])
        }
        EstimatorExtras::Glm {
            baseline_prob_realized,
            singular_fit_rate,
            tau_squared_hat_mean,
            ..
        } => HostValue::Map(vec![
            ("estimator", HostValue::Str("glm".into())),
            (
                "baseline_prob_realized",
                HostValue::F64(*baseline_prob_realized),
            ),
            // GLMM (Glm + cluster) numerics — zero for plain GLM, populated by the
            // Laplace kernel for clustered logistic. tau_squared_hat_mean is the
            // host-side Laplace-bias warning signal; singular_fit_rate mirrors the
            // Mle arm. The shared Glm variant covers both plain-GLM and GLMM
            // (clustered-logistic) results under one estimator tag.
            ("singular_fit_rate", HostValue::F64(*singular_fit_rate)),
            (
                "tau_squared_hat_mean",
                HostValue::F64(*tau_squared_hat_mean),
            ),
        ]),
        EstimatorExtras::Mle {
            tau_estimate,
            boundary_hits,
            joint_uncorrected_rate,
            joint_corrected_rate,
            singular_fit_rate,
            boundary_rate_per_component,
            ..
        } => HostValue::Map(vec![
            ("estimator", HostValue::Str("mle".into())),
            ("tau_estimate", HostValue::F64(*tau_estimate)),
            ("boundary_hits", HostValue::Usize(*boundary_hits as usize)),
            (
                "joint_uncorrected_rate",
                HostValue::F64(*joint_uncorrected_rate),
            ),
            (
                "joint_corrected_rate",
                HostValue::F64(*joint_corrected_rate),
            ),
            ("singular_fit_rate", HostValue::F64(*singular_fit_rate)),
            // Per-diagonal-component pin rate: fraction of converged fits that
            // pinned component k (ordered intercept, slope_0, …). Empty for
            // non-cluster or intercept-only Mle. Derived from the merge-support
            // `boundary_component_counts` / `singular_n` pairing in merge.rs.
            (
                "boundary_rate_per_component",
                HostValue::VecF64(boundary_rate_per_component.clone()),
            ),
        ]),
    }
}

/// Flatten the per-N `boundary_hit` columns into one row-major
/// `(rows=n_sims, cols=n_sample_sizes)` buffer. For `find_power` (one column)
/// this is just the single column; for `find_sample_size` it interleaves the
/// per-N columns. Returns `(flat, rows, cols)`.
fn interleave_boundary_hit(grid: &[PowerResult]) -> (Vec<u8>, usize, usize) {
    let cols = grid.len();
    if cols == 0 {
        return (Vec::new(), 0, 0);
    }
    let rows = grid[0].boundary_hit.len();
    let mut flat = vec![0u8; rows * cols];
    for (c, pr) in grid.iter().enumerate() {
        for (sim, &v) in pr.boundary_hit.iter().enumerate() {
            flat[sim * cols + c] = v;
        }
    }
    (flat, rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::{ByValue, EstimatorExtras, GridMode, SampleSizeMethod};

    fn sample_pr(n: usize) -> PowerResult {
        PowerResult {
            n,
            n_sims: 4,
            target_indices: vec![0, 1],
            contrast_pairs: vec![],
            power_uncorrected: vec![0.8, 0.6],
            power_corrected: vec![0.75, 0.55],
            ci_uncorrected: vec![Ci { lo: 0.7, hi: 0.9 }, Ci { lo: 0.5, hi: 0.7 }],
            ci_corrected: vec![Ci { lo: 0.65, hi: 0.85 }, Ci { lo: 0.45, hi: 0.65 }],
            convergence_rate: 1.0,
            boundary_hit: vec![0, 1, 0, 2],
            estimator_extras: EstimatorExtras::Ols {},
            overall_significant_rate: Some(0.9),
            success_counts_uncorrected: vec![],
            success_counts_corrected: vec![],
            convergence_count: 4,
            overall_significant_count: 0,
            overall_significant_ci: Some(Ci { lo: 0.8, hi: 0.95 }),
            success_count_histogram_uncorrected: vec![1, 2, 1],
            success_count_histogram_corrected: vec![2, 1, 1],
            grid_warnings: vec!["snapped".to_string()],
            posthoc: vec![],
            factor_exclusion_counts: vec![],
            factor_separation_counts: vec![],
        }
    }

    /// Pull a field out of a `HostValue::Map` by key, asserting it exists.
    fn field<'a>(hv: &'a HostValue, key: &str) -> &'a HostValue {
        match hv {
            HostValue::Map(pairs) => pairs
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v)
                .unwrap_or_else(|| panic!("missing field {key}")),
            _ => panic!("not a Map"),
        }
    }

    /// The ordered list of field keys in a `HostValue::Map`.
    fn keys(hv: &HostValue) -> Vec<&'static str> {
        match hv {
            HostValue::Map(pairs) => pairs.iter().map(|(k, _)| *k).collect(),
            _ => panic!("not a Map"),
        }
    }

    #[test]
    fn power_host_field_order_is_canonical() {
        let hv = power_result_to_host(&sample_pr(100), "opt");
        assert_eq!(
            keys(&hv),
            vec![
                "n_sims",
                "n_sample_sizes",
                "n_targets",
                "target_indices",
                "contrast_pairs",
                "power_uncorrected",
                "power_corrected",
                "ci_uncorrected",
                "ci_corrected",
                "convergence_rate",
                "scenario",
                "sample_sizes",
                "estimator_extras",
                "boundary_hit",
                "overall_significant_rate",
                "overall_significant_ci",
                "success_count_histogram_uncorrected",
                "success_count_histogram_corrected",
                "grid_warnings",
                "posthoc",
                "factor_exclusion_counts",
                "factor_separation_counts",
            ]
        );
    }

    #[test]
    fn power_host_sample_sizes_uses_snapped_n() {
        // pr.n == 90 even though a caller might have asked for 100; the host
        // shape must report the snapped N.
        let hv = power_result_to_host(&sample_pr(90), "opt");
        match field(&hv, "sample_sizes") {
            HostValue::VecU64(v) => assert_eq!(v, &vec![90u64]),
            other => panic!("sample_sizes not VecU64: {other:?}"),
        }
    }

    #[test]
    fn power_host_boundary_hit_single_column() {
        let hv = power_result_to_host(&sample_pr(100), "opt");
        match field(&hv, "boundary_hit") {
            HostValue::BoundaryHit { flat, rows, cols } => {
                assert_eq!(flat, &vec![0u8, 1, 0, 2]);
                assert_eq!(*rows, 4);
                assert_eq!(*cols, 1);
            }
            other => panic!("not BoundaryHit: {other:?}"),
        }
    }

    #[test]
    fn power_host_scalar_wrappers() {
        // power_uncorrected is wrapped one level (a 1-element Seq of VecF64),
        // convergence_rate is a flat 1-element VecF64.
        let hv = power_result_to_host(&sample_pr(100), "opt");
        match field(&hv, "power_uncorrected") {
            HostValue::Seq(items) => {
                assert_eq!(items.len(), 1);
                assert!(matches!(items[0], HostValue::VecF64(_)));
            }
            other => panic!("power_uncorrected not Seq: {other:?}"),
        }
        match field(&hv, "convergence_rate") {
            HostValue::VecF64(v) => assert_eq!(v, &vec![1.0]),
            other => panic!("convergence_rate not VecF64: {other:?}"),
        }
    }

    fn sample_ssr() -> SampleSizeResult {
        SampleSizeResult {
            grid_or_trace: vec![sample_pr(50), sample_pr(100)],
            first_achieved: vec![Some(100), None],
            first_joint_achieved: vec![Some(50), None],
            fitted: vec![
                CrossingFit::Fitted {
                    n_star: 84.3,
                    n_achievable: 90,
                    ci_lo: Some(76.2),
                    ci_hi: None,
                },
                CrossingFit::NotReached {
                    n_approx: Some(330),
                },
            ],
            fitted_joint: vec![
                CrossingFit::AtOrBelowMin { n_min: 50 },
                CrossingFit::NonMonotone {
                    max_violation: 0.054,
                },
            ],
            first_overall_achieved: Some(100),
            fitted_overall: Some(CrossingFit::Fitted {
                n_star: 92.5,
                n_achievable: 100,
                ci_lo: Some(80.0),
                ci_hi: None,
            }),
            cluster_atom: 10,
            target_power: 0.8,
            method: SampleSizeMethod::Grid {
                by: ByValue::Fixed(50),
                mode: GridMode::Linear,
            },
            grid_warnings: vec![],
        }
    }

    #[test]
    fn sample_size_host_interleaves_boundary_hit_and_keys_index_maps() {
        let ssr = sample_ssr();
        let hv = sample_size_result_to_host(&ssr, "opt");
        assert_eq!(
            keys(&hv),
            vec![
                "n_sims",
                "n_sample_sizes",
                "n_targets",
                "target_indices",
                "contrast_pairs",
                "power_uncorrected",
                "power_corrected",
                "ci_uncorrected",
                "ci_corrected",
                "convergence_rate",
                "sample_sizes",
                "first_achieved",
                "success_count_histogram_uncorrected",
                "success_count_histogram_corrected",
                "grid_warnings",
                "factor_exclusion_counts",
                "factor_separation_counts",
                "first_joint_achieved",
                "first_overall_achieved",
                "target_power",
                "scenario",
                "boundary_hit",
                "fitted",
                "fitted_joint",
                "fitted_overall",
                "cluster_atom",
            ]
        );
        // 4 sims × 2 sample sizes, row-major interleave.
        match field(&hv, "boundary_hit") {
            HostValue::BoundaryHit { flat, rows, cols } => {
                assert_eq!(*rows, 4);
                assert_eq!(*cols, 2);
                // sim 0: [col0=0, col1=0]; sim 1: [1,1]; sim 2: [0,0]; sim 3: [2,2]
                assert_eq!(flat, &vec![0u8, 0, 1, 1, 0, 0, 2, 2]);
            }
            other => panic!("not BoundaryHit: {other:?}"),
        }
        match field(&hv, "first_achieved") {
            HostValue::IndexMap(items) => {
                assert_eq!(items.len(), 2);
                assert!(matches!(items[0], HostValue::OptUsize(Some(100))));
                assert!(matches!(items[1], HostValue::OptUsize(None)));
            }
            other => panic!("first_achieved not IndexMap: {other:?}"),
        }
    }

    #[test]
    fn sample_size_host_projects_crossing_fits_as_status_tagged_maps() {
        let hv = sample_size_result_to_host(&sample_ssr(), "opt");
        match field(&hv, "fitted") {
            HostValue::IndexMap(items) => {
                assert_eq!(items.len(), 2);
                // Variant tag + per-variant fields, in declaration order.
                assert_eq!(
                    keys(&items[0]),
                    vec!["status", "n_star", "n_achievable", "ci_lo", "ci_hi"]
                );
                match field(&items[0], "status") {
                    HostValue::Str(s) => assert_eq!(s, "fitted"),
                    other => panic!("status not Str: {other:?}"),
                }
                assert!(matches!(
                    field(&items[0], "n_achievable"),
                    HostValue::Usize(90)
                ));
                assert!(matches!(field(&items[0], "ci_hi"), HostValue::OptF64(None)));
                assert_eq!(keys(&items[1]), vec!["status", "n_approx"]);
                assert!(matches!(
                    field(&items[1], "n_approx"),
                    HostValue::OptUsize(Some(330))
                ));
            }
            other => panic!("fitted not IndexMap: {other:?}"),
        }
        match field(&hv, "fitted_joint") {
            HostValue::IndexMap(items) => {
                assert_eq!(keys(&items[0]), vec!["status", "n_min"]);
                assert_eq!(keys(&items[1]), vec!["status", "max_violation"]);
            }
            other => panic!("fitted_joint not IndexMap: {other:?}"),
        }
        // Overall crossing: a 1-element IndexMap (the singular fit) plus a
        // scalar grid-empirical first-N.
        assert!(matches!(
            field(&hv, "first_overall_achieved"),
            HostValue::OptUsize(Some(100))
        ));
        match field(&hv, "fitted_overall") {
            HostValue::IndexMap(items) => {
                assert_eq!(items.len(), 1);
                assert_eq!(
                    keys(&items[0]),
                    vec!["status", "n_star", "n_achievable", "ci_lo", "ci_hi"]
                );
            }
            other => panic!("fitted_overall not IndexMap: {other:?}"),
        }
        assert!(matches!(field(&hv, "cluster_atom"), HostValue::Usize(10)));
    }

    /// `estimator_extras_to_host` for the Mle arm must include
    /// `boundary_rate_per_component` as a `VecF64`. Non-empty for mixed-effects
    /// fits (one entry per variance component); empty for intercept-only / OLS.
    #[test]
    fn mle_estimator_extras_host_includes_boundary_rate_per_component() {
        let mle_extras = EstimatorExtras::Mle {
            tau_estimate: 0.3,
            boundary_hits: 2,
            joint_uncorrected_rate: 0.6,
            joint_corrected_rate: 0.55,
            tau_sum: 0.0,
            tau_n: 0,
            joint_uncorrected_count: 6,
            joint_corrected_count: 5,
            singular_fit_rate: 0.1,
            singular_count: 1,
            singular_n: 10,
            boundary_rate_per_component: vec![0.1, 0.05],
            boundary_component_counts: vec![1, 0],
        };
        // Call through power_result_to_host so we exercise the full projection
        // path (including the estimator_extras_to_host dispatch), not just the
        // private helper directly.
        let mut pr = sample_pr(100);
        pr.estimator_extras = mle_extras;
        let hv = power_result_to_host(&pr, "test");
        let ee = field(&hv, "estimator_extras");
        // Field order in the Mle arm.
        assert_eq!(
            keys(ee),
            vec![
                "estimator",
                "tau_estimate",
                "boundary_hits",
                "joint_uncorrected_rate",
                "joint_corrected_rate",
                "singular_fit_rate",
                "boundary_rate_per_component",
            ]
        );
        match field(ee, "boundary_rate_per_component") {
            HostValue::VecF64(v) => assert_eq!(v, &vec![0.1, 0.05]),
            other => panic!("boundary_rate_per_component not VecF64: {other:?}"),
        }
        // Empty for intercept-only Mle (no variance components).
        let mut pr2 = sample_pr(100);
        pr2.estimator_extras = EstimatorExtras::Mle {
            tau_estimate: f64::NAN,
            boundary_hits: 0,
            joint_uncorrected_rate: 0.0,
            joint_corrected_rate: 0.0,
            tau_sum: 0.0,
            tau_n: 0,
            joint_uncorrected_count: 0,
            joint_corrected_count: 0,
            singular_fit_rate: 0.0,
            singular_count: 0,
            singular_n: 0,
            boundary_rate_per_component: vec![],
            boundary_component_counts: vec![],
        };
        let hv2 = power_result_to_host(&pr2, "test");
        let ee2 = field(&hv2, "estimator_extras");
        match field(ee2, "boundary_rate_per_component") {
            HostValue::VecF64(v) => assert!(v.is_empty(), "expected empty vec, got {v:?}"),
            other => panic!("boundary_rate_per_component not VecF64: {other:?}"),
        }
    }
}
