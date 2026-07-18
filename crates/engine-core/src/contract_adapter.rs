//! Mechanical translation from `engine_contract::SimulationContract` to the
//! kernel-side `engine_core::spec::SimulationSpec` POD.
//!
//! The contract is the host-facing surface (label-free, structured). The
//! POD is the kernel's hot-loop layout (flat buffers, integer codes). This
//! module is the one-way bridge between them; the kernel never sees a
//! `SimulationContract` and the contract never sees the POD.
//!
//! ## Joint test routing
//!
//! `TestTarget::Joint { terms }` is routed in one of two ways:
//!   * `terms == <all non-intercept positions of design_test>` (v1-parity
//!     omnibus shape) → `SimulationSpec.report_overall = true`. The OLS
//!     F-test / Logit LRT "overall" channel handles the rest.
//!   * Any other shape (arbitrary subset, single term, or extra terms not in
//!     the test design) → `AdapterError::JointNotSupported`. Arbitrary joint
//!     subsets need a separate contrast surface that hasn't been designed
//!     yet, and the LME asymptotic joint Wald-χ² (`joint_t_crit_sq` +
//!     `BatchResult.{joint_unc,joint_cor}`) is reached via the spec-builder's
//!     per-marginal routing, not through `TestTarget::Joint`.

use engine_contract::{
    ColumnSpec, Correlations, DesignSpec, DesignTerm, SimulationContract, SyntheticKind, TestTarget,
};

use crate::spec::{
    CorrectionMethod, CritValues, Distribution, HeteroskedasticityCoeffs,
    PosthocSpec as KernelPosthoc, ResidualDist, ScenarioPerturbations as KernelScenario,
    SimulationSpec,
};

/// Errors from lowering a `SimulationContract` into a `SimulationSpec`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AdapterError {
    /// `TestTarget::Joint` is supported only for the v1-parity omnibus shape
    /// — `terms == <all non-intercept positions of design_test>` — which
    /// routes to `SimulationSpec.report_overall = true`. Arbitrary subsets
    /// would need a separate per-joint contrast surface and are not in scope.
    #[error(
        "TestTarget::Joint only supports the v1-parity omnibus shape (all non-intercept \
         terms); got {got:?}"
    )]
    JointNotSupported { got: Vec<u32> },
    #[error("contract failed validate(): {0}")]
    ContractInvalid(String),
}

/// True iff `terms` is the set of all non-intercept positions in `design` —
/// i.e. the v1-parity "overall" omnibus shape (F-test / LRT vs intercept-only).
fn joint_is_omnibus(terms: &[u32], design: &DesignSpec) -> bool {
    let mut expected: Vec<u32> = design
        .terms
        .iter()
        .enumerate()
        .filter_map(|(i, t)| match t {
            DesignTerm::Const => None,
            _ => Some(i as u32),
        })
        .collect();
    expected.sort_unstable();
    let mut got = terms.to_vec();
    got.sort_unstable();
    got.dedup();
    got == expected
}

/// Lower a `SimulationContract` into the kernel's `SimulationSpec`: validates
/// the contract, resolves `design_test` defaulting, and routes v1-parity Joint
/// targets to `report_overall`.
///
/// # Errors
/// `ContractInvalid` if `c.validate()` fails; `JointNotSupported` for any
/// Joint target that is not the omnibus shape.
pub fn contract_to_simulation_spec(c: &SimulationContract) -> Result<SimulationSpec, AdapterError> {
    c.validate()
        .map_err(|e| AdapterError::ContractInvalid(e.to_string()))?;

    // Resolve the fitted design once: design_test defaults to design_generation.
    let design_test = c.design_test.as_ref().unwrap_or(&c.design_generation);

    // Route Joint targets: omnibus shape → report_overall; anything else is
    // rejected. See module-level comment for the rationale.
    let mut report_overall = false;
    for target in &c.test.targets {
        if let TestTarget::Joint { terms } = target {
            if joint_is_omnibus(terms, design_test) {
                report_overall = true;
            } else {
                let mut got: Vec<u32> = terms.clone();
                got.sort_unstable();
                return Err(AdapterError::JointNotSupported { got });
            }
        }
    }

    let (
        var_types,
        var_pinned,
        var_params,
        factor_n_levels,
        factor_proportions,
        factor_sampled,
        n_non_factor,
        n_factor_dummies,
    ) = translate_generation(c);
    let correlation = translate_correlation(c, n_non_factor);
    let (upload_normal, upload_normal_shape, upload_data, upload_data_shape, bootstrap_frame_map) =
        translate_uploaded_frame(c);

    let interactions = translate_interactions(c, n_non_factor);
    let effect_sizes = translate_effect_sizes(c, n_non_factor, n_factor_dummies);
    let between_var_indices = expand_cluster_level_columns(c, n_non_factor);
    // Spec carries the generation-kernel space the kernel reads; the host-report
    // (design_test term) space is recovered by `report_targets_and_contrasts`.
    let (target_indices, _, contrast_pairs, _) =
        translate_targets_and_contrasts(c, n_non_factor, n_factor_dummies, design_test);
    let correction_method: CorrectionMethod = c.test.correction;
    let (posthoc, posthoc_alpha) = translate_posthoc(c, design_test);
    // OutcomeKind, EstimatorSpec, and ClusterSpec are re-exported verbatim from
    // engine_contract (see crate::spec), so they pass straight through with no
    // translation.
    let outcome_kind = c.outcome.kind;
    let estimator = c.estimator;
    let heteroskedasticity_driver = translate_heteroskedasticity_driver(c);
    let residual_dist: ResidualDist = c.outcome.residual.distribution;
    let cluster = c.generation.cluster.clone();
    let scenario = translate_scenario(&c.scenario);
    // Slope covariates → x_full columns, mirroring translate_heteroskedasticity:
    // column_position_for_continuous returns the 1-based x_full index (cursor
    // starts at 1, skipping the intercept), so no -1. validate() guarantees each
    // column is continuous and Direct, so every position is real.
    let cluster_slope_design_cols: Vec<u32> = c
        .generation
        .cluster
        .as_ref()
        .map(|cl| {
            cl.slopes
                .iter()
                .map(|s| column_position_for_continuous(c, s.column.0))
                .collect()
        })
        .unwrap_or_default();
    // Crossed/nested analogue: each extra grouping's slope covariates → x_full
    // columns, declaration order. Same `column_position_for_continuous` resolution
    // as the primary; validate() (invariant_19) guarantees each is continuous and
    // Direct. Empty inner vecs for intercept-only extras.
    let extra_slope_cols: Vec<Vec<u32>> = c
        .generation
        .cluster
        .as_ref()
        .map(|cl| {
            cl.extra_groupings
                .iter()
                .map(|g| {
                    g.slopes
                        .iter()
                        .map(|s| column_position_for_continuous(c, s.column.0))
                        .collect()
                })
                .collect()
        })
        .unwrap_or_default();
    // Fitted (test) design's kept kernel columns, for `test_formula` reduced
    // fits. Fires only when an explicit design_test drops at least one column:
    // map each test-design term to its generation kernel column (ascending,
    // deduped via BTreeSet). An absent design_test, or one that still covers the
    // whole generation design, leaves fit_columns empty (fit == full generation
    // design; no reduction). The reduced-fit path in run_batch reads this.
    let fit_columns: Vec<u32> = if c.design_test.is_some() {
        let n_predictors_total =
            1 + n_non_factor as usize + n_factor_dummies as usize + interactions.len();
        let kept: std::collections::BTreeSet<u32> = (0..design_test.terms.len() as u32)
            .map(|t| term_to_kernel_col(c, t, n_non_factor, n_factor_dummies, design_test))
            .collect();
        if kept.len() == n_predictors_total {
            Vec::new()
        } else {
            kept.into_iter().collect()
        }
    } else {
        Vec::new()
    };

    Ok(SimulationSpec {
        n_non_factor,
        n_factor_dummies,
        correlation,
        var_types,
        var_pinned,
        var_params,
        upload_normal,
        upload_normal_shape,
        upload_data,
        upload_data_shape,
        bootstrap_frame_map,
        between_var_indices,
        factor_n_levels,
        factor_proportions,
        factor_sampled,
        effect_sizes,
        target_indices,
        fit_columns,
        contrast_pairs,
        interactions,
        correction_method,
        crit_values: CritValues {
            alpha: c.test.alpha,
            posthoc_alpha,
        },
        heteroskedasticity_driver,
        cluster_slope_design_cols,
        extra_slope_cols,
        residual_dist,
        residual_pinned: c.outcome.residual.pinned,
        outcome_kind,
        link: c.outcome.link,
        estimator,
        intercept: c.outcome.intercept,
        posthoc,
        max_failed_fraction: c.max_failed_fraction,
        cluster,
        scenario,
        t3_table: None,
        het_coeffs: HeteroskedasticityCoeffs::default(),
        report_overall,
        factor_min_level_count: 0,
        wald_se: c.wald_se,
        nagq: c.nagq,
    })
}

fn synthetic_to_distribution(k: &SyntheticKind) -> (Distribution, f64) {
    match k {
        SyntheticKind::Normal => (Distribution::Normal, 0.0),
        // The kernel reads `p` from `var_params`, not from the scenario menu —
        // we still surface the param here so the parallel `var_params` slot is
        // populated correctly when the column is consumed directly.
        SyntheticKind::Binary { p } => (Distribution::Binary, *p),
        SyntheticKind::RightSkewed => (Distribution::RightSkewed, 0.0),
        SyntheticKind::LeftSkewed => (Distribution::LeftSkewed, 0.0),
        SyntheticKind::HighKurtosis => (Distribution::HighKurtosis, 0.0),
        SyntheticKind::Uniform => (Distribution::Uniform, 0.0),
    }
}

#[allow(clippy::type_complexity)]
fn translate_generation(
    c: &SimulationContract,
) -> (
    Vec<Distribution>,
    Vec<bool>,
    Vec<f64>,
    Vec<i32>,
    Vec<f64>,
    Vec<Option<bool>>,
    u32,
    u32,
) {
    let mut var_types = Vec::new();
    let mut var_pinned = Vec::new();
    let mut var_params = Vec::new();
    let mut factor_n_levels = Vec::new();
    let mut factor_proportions = Vec::new();
    let mut factor_sampled = Vec::new();
    let mut n_non_factor: u32 = 0;
    let mut n_factor_dummies: u32 = 0;

    for col in &c.generation.columns {
        match col {
            ColumnSpec::Synthetic { kind, pinned } => {
                let (dist, param) = synthetic_to_distribution(kind);
                var_types.push(dist);
                var_pinned.push(*pinned);
                var_params.push(param);
                n_non_factor += 1;
            }
            ColumnSpec::Resampled { .. } => {
                // Uploaded columns are never Distribution::Normal, so the swap
                // rule can't fire regardless — the pin value is inert.
                var_types.push(Distribution::UploadedData);
                var_pinned.push(false);
                var_params.push(0.0);
                n_non_factor += 1;
            }
            ColumnSpec::ResampledBinary { proportion, .. } => {
                var_types.push(Distribution::UploadedBinary);
                var_pinned.push(false);
                var_params.push(*proportion);
                n_non_factor += 1;
            }
            ColumnSpec::FactorSynthetic {
                n_levels,
                proportions,
                sampled_proportions,
            }
            | ColumnSpec::FactorFromFrame {
                n_levels,
                proportions,
                sampled_proportions,
                ..
            } => {
                factor_n_levels.push(*n_levels as i32);
                factor_proportions.extend_from_slice(proportions);
                factor_sampled.push(*sampled_proportions);
                n_factor_dummies += n_levels.saturating_sub(1);
            }
        }
    }
    (
        var_types,
        var_pinned,
        var_params,
        factor_n_levels,
        factor_proportions,
        factor_sampled,
        n_non_factor,
        n_factor_dummies,
    )
}

fn translate_correlation(c: &SimulationContract, n_non_factor: u32) -> Vec<f64> {
    let n = n_non_factor as usize;
    match &c.generation.correlations {
        Correlations::Identity => {
            let mut m = vec![0.0; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        }
        Correlations::Matrix { values, .. } => values.clone(),
    }
}

/// Return shape of `translate_uploaded_frame`: `(upload_normal,
/// upload_normal_shape, upload_data, upload_data_shape, bootstrap_frame_map)`.
type UploadedFrameParts = (Vec<f64>, (u32, u32), Vec<f64>, (u32, u32), Vec<Option<u32>>);

/// Translate the contract's optional `uploaded_frame` into the kernel's upload
/// slots, dispatching on the frame's `bootstrap` flag.
///
/// Returns `(upload_normal, upload_normal_shape, upload_data, upload_data_shape,
/// bootstrap_frame_map)`.
///
/// NORTA path (`bootstrap == false`, i.e. `none`/`partial`): builds
/// `upload_normal`, a row-major (U, n_non_factor) lookup table for the marginal
/// path — column j holds the sorted-ascending standardised values from the
/// uploaded frame column referenced by the j-th non-factor column spec
/// (Resampled | ResampledBinary); Synthetic columns leave their region as zeros.
/// `upload_data` carries the raw frame and `bootstrap_frame_map` is empty.
///
/// Bootstrap path (`bootstrap == true`, i.e. strict): skips `upload_normal`
/// entirely (returned empty with shape (0,0)). `upload_data` carries the raw
/// frame and `bootstrap_frame_map` maps every kernel column to its frame column
/// — non-factor columns first (`Some(frame_col)` for Resampled/ResampledBinary,
/// `None` for Synthetic), then factor columns (`Some(frame_col)` for
/// FactorFromFrame, `None` for FactorSynthetic). The kernel uses the map to
/// row-sample whole frame rows.
fn translate_uploaded_frame(c: &SimulationContract) -> UploadedFrameParts {
    let frame = match &c.generation.uploaded_frame {
        None => return (Vec::new(), (0, 0), Vec::new(), (0, 0), Vec::new()),
        Some(f) => f,
    };

    if frame.bootstrap {
        // Bootstrap (strict): no NORTA lookup. Build the per-column frame map
        // instead — non-factor entries first (column order), then factor entries.
        let mut bootstrap_frame_map: Vec<Option<u32>> = Vec::new();
        for col in &c.generation.columns {
            match col {
                ColumnSpec::Resampled { frame_column }
                | ColumnSpec::ResampledBinary { frame_column, .. } => {
                    bootstrap_frame_map.push(Some(*frame_column));
                }
                ColumnSpec::Synthetic { .. } => bootstrap_frame_map.push(None),
                ColumnSpec::FactorSynthetic { .. } | ColumnSpec::FactorFromFrame { .. } => {
                    // Factors are appended in a second pass below.
                }
            }
        }
        for col in &c.generation.columns {
            match col {
                ColumnSpec::FactorFromFrame { frame_column, .. } => {
                    bootstrap_frame_map.push(Some(*frame_column));
                }
                ColumnSpec::FactorSynthetic { .. } => bootstrap_frame_map.push(None),
                _ => {}
            }
        }
        return (
            Vec::new(),
            (0, 0),
            frame.data.clone(),
            (frame.n_rows, frame.n_cols),
            bootstrap_frame_map,
        );
    }

    // NORTA path. `upload_normal[r * n_nf + j]` is the r-th quantile for
    // non-factor column j (0-indexed, 0..U-1 ascending).
    // Count non-factor columns (Synthetic | Resampled | ResampledBinary).
    let n_nf = c
        .generation
        .columns
        .iter()
        .filter(|col| {
            matches!(
                col,
                ColumnSpec::Synthetic { .. }
                    | ColumnSpec::Resampled { .. }
                    | ColumnSpec::ResampledBinary { .. }
            )
        })
        .count();

    let u_rows = frame.n_rows as usize;
    let n_cols_frame = frame.n_cols as usize;

    // Allocate: row-major (U, n_nf), all zeros. Synthetic columns stay zero.
    let mut upload_normal = vec![0.0f64; u_rows * n_nf];

    // Walk generation.columns, tracking non-factor index j.
    let mut j = 0usize;
    for col in &c.generation.columns {
        match col {
            ColumnSpec::Resampled { frame_column }
            | ColumnSpec::ResampledBinary { frame_column, .. } => {
                let fc = *frame_column as usize;
                // Extract column fc from the row-major frame (n_rows × n_cols_frame).
                let mut col_vals: Vec<f64> = (0..u_rows)
                    .map(|r| frame.data[r * n_cols_frame + fc])
                    .collect();
                // Sort ascending. total_cmp is NaN-safe (sorts NaN deterministically
                // rather than panicking); contract validate() already requires finite data.
                col_vals.sort_by(f64::total_cmp);
                // Write into upload_normal column j (row-major).
                for r in 0..u_rows {
                    upload_normal[r * n_nf + j] = col_vals[r];
                }
                j += 1;
            }
            ColumnSpec::Synthetic { .. } => {
                j += 1; // leave zeros, advance j
            }
            ColumnSpec::FactorSynthetic { .. } | ColumnSpec::FactorFromFrame { .. } => {
                // Not a non-factor column — do not advance j.
            }
        }
    }

    let upload_normal_shape = (frame.n_rows, n_nf as u32);
    (
        upload_normal,
        upload_normal_shape,
        frame.data.clone(),
        (frame.n_rows, frame.n_cols),
        Vec::new(),
    )
}

fn translate_effect_sizes(
    c: &SimulationContract,
    n_non_factor: u32,
    n_factor_dummies: u32,
) -> Vec<f64> {
    let n_interactions = c
        .design_generation
        .terms
        .iter()
        .filter(|t| matches!(t, DesignTerm::Interaction { .. }))
        .count() as u32;
    let n_total = (1 + n_non_factor + n_factor_dummies + n_interactions) as usize;
    let mut effects = vec![0.0; n_total];

    let mut non_factor_col_index: Vec<i32> = vec![-1; c.generation.columns.len()];
    let mut factor_dummy_base: Vec<i32> = vec![-1; c.generation.columns.len()];
    let mut nf_cursor = 1u32;
    let mut fd_cursor = 1u32 + n_non_factor;
    for (i, col) in c.generation.columns.iter().enumerate() {
        match col {
            ColumnSpec::Synthetic { .. }
            | ColumnSpec::Resampled { .. }
            | ColumnSpec::ResampledBinary { .. } => {
                non_factor_col_index[i] = nf_cursor as i32;
                nf_cursor += 1;
            }
            ColumnSpec::FactorSynthetic { n_levels, .. }
            | ColumnSpec::FactorFromFrame { n_levels, .. } => {
                factor_dummy_base[i] = fd_cursor as i32;
                fd_cursor += n_levels.saturating_sub(1);
            }
        }
    }

    let interaction_base = 1 + n_non_factor + n_factor_dummies;
    let mut interaction_idx = 0u32;
    for (term, &beta) in c
        .design_generation
        .terms
        .iter()
        .zip(c.outcome.coefficients.iter())
    {
        let col = match term {
            DesignTerm::Const => 0usize,
            DesignTerm::Direct { column } => non_factor_col_index[column.0 as usize] as usize,
            DesignTerm::DummyOf {
                column,
                level_index,
            } => (factor_dummy_base[column.0 as usize] + (*level_index as i32) - 1) as usize,
            DesignTerm::Interaction { .. } => {
                let kc = (interaction_base + interaction_idx) as usize;
                interaction_idx += 1;
                kc
            }
        };
        effects[col] = beta;
    }
    // The Const term's coefficient is always 0.0; the intercept lives in
    // `outcome.intercept`. The data-generation kernel reads the intercept from
    // `effect_sizes[0]` (its convention — see the data_gen/glm unit tests), so
    // fold it in here. OLS/Mle keep intercept == 0.0 (no-op); for Glm this
    // restores the baseline log-odds, without which every sim runs at p=0.5.
    effects[0] = c.outcome.intercept;
    effects
}

/// Translate a design-test term index to its kernel column position.
fn term_to_kernel_col(
    c: &SimulationContract,
    term: u32,
    n_non_factor: u32,
    n_factor_dummies: u32,
    design_test: &DesignSpec,
) -> u32 {
    match &design_test.terms[term as usize] {
        DesignTerm::Const => 0,
        DesignTerm::Direct { column } => column_position_for_continuous(c, column.0),
        DesignTerm::DummyOf {
            column,
            level_index,
        } => column_position_for_dummy(c, column.0, *level_index, n_non_factor),
        DesignTerm::Interaction { components } => {
            interaction_kernel_col(c, components, n_non_factor, n_factor_dummies)
        }
    }
}

/// Returns `(target_indices, target_report_indices, contrast_pairs,
/// contrast_report_pairs)`, the kernel-space and host-report-space twins of the
/// test targets, produced **in lockstep** (same order, same length) so either
/// space pairs with the power-vector layout:
///
/// - `target_indices` / `contrast_pairs` — **generation-kernel** column
///   positions the kernel reads (full-fit β̂ columns; the reduced-fit path remaps
///   them through `excl_col_remap`). These go into `SimulationSpec`.
/// - `target_report_indices` / `contrast_report_pairs` — **design_test term**
///   positions, the space the host's effect skeleton is indexed in
///   (`skeleton[report_index]` names the effect). For a correctly-specified model
///   (`design_test == design_generation`) each term position equals its kernel
///   column, so the two spaces coincide; they diverge only when a `test_formula`
///   drops or reorders a generation term. The orchestrator echoes these into
///   `PowerResult.{target_indices, contrast_pairs}`.
///
/// `Marginal` targets and reference-collapsed `Contrast`s feed the marginal
/// channel — keyed by kernel column (so the emitted order matches the kernel's
/// ascending-column marginal layout), valued by the contract `term` (resp.
/// `positive`). Non-reference `Contrast`s feed the contrast channel in target
/// order.
#[allow(clippy::type_complexity)]
fn translate_targets_and_contrasts(
    c: &SimulationContract,
    n_non_factor: u32,
    n_factor_dummies: u32,
    design_test: &DesignSpec,
) -> (Vec<u32>, Vec<u32>, Vec<(u32, u32)>, Vec<(u32, u32)>) {
    // kernel column → design_test term position. The BTreeMap key (kernel column)
    // drives the sort/dedup exactly as the old BTreeSet did; the value carries the
    // report (skeleton) index emitted in the same order.
    let mut marginal_map: std::collections::BTreeMap<u32, u32> = Default::default();
    let mut contrasts: Vec<(u32, u32)> = Vec::new();
    let mut contrasts_report: Vec<(u32, u32)> = Vec::new();

    for target in &c.test.targets {
        match target {
            TestTarget::Marginal { term } => {
                let col = term_to_kernel_col(c, *term, n_non_factor, n_factor_dummies, design_test);
                marginal_map.insert(col, *term);
            }
            TestTarget::Joint { .. } => {
                // Joint targets are handled by the report_overall flag, not
                // by target_indices. Nothing to do here.
            }
            TestTarget::Contrast { positive, negative } => {
                let n_term = &design_test.terms[*negative as usize];
                if matches!(n_term, DesignTerm::Const) {
                    // Reference-level collapse: β_ref = 0 in dummy coding,
                    // so the contrast reduces to a Marginal on positive.
                    let p_col = term_to_kernel_col(
                        c,
                        *positive,
                        n_non_factor,
                        n_factor_dummies,
                        design_test,
                    );
                    marginal_map.insert(p_col, *positive);
                } else {
                    let p_col = term_to_kernel_col(
                        c,
                        *positive,
                        n_non_factor,
                        n_factor_dummies,
                        design_test,
                    );
                    let n_col = term_to_kernel_col(
                        c,
                        *negative,
                        n_non_factor,
                        n_factor_dummies,
                        design_test,
                    );
                    contrasts.push((p_col, n_col));
                    contrasts_report.push((*positive, *negative));
                }
            }
        }
    }

    (
        marginal_map.keys().copied().collect(),
        marginal_map.values().copied().collect(),
        contrasts,
        contrasts_report,
    )
}

/// Host-facing report indices for `c`'s marginal targets and contrast pairs, in
/// **design_test term-position** space — the space the host's effect skeleton is
/// indexed in. Returned in the same order as `SimulationSpec.{target_indices,
/// contrast_pairs}` (the generation-kernel space the engine reads), so the
/// orchestrator can echo these into `PowerResult` without reordering the power
/// vectors. Equals the kernel indices for a correctly-specified model; differs
/// only when a `test_formula` drops or reorders a generation term.
///
/// Assumes a validated contract (`contract_to_simulation_spec` is called first
/// in `lower_contracts`); indexes `design_test.terms` without re-checking bounds.
/// Checklist: the report and kernel vectors are produced together by
/// `translate_targets_and_contrasts` — keep them lockstep there, not here.
pub fn report_targets_and_contrasts(c: &SimulationContract) -> (Vec<u32>, Vec<(u32, u32)>) {
    let design_test = c.design_test.as_ref().unwrap_or(&c.design_generation);
    // Predictor tallies (mirror translate_generation; trivial at lowering time).
    let mut n_non_factor = 0u32;
    let mut n_factor_dummies = 0u32;
    for col in &c.generation.columns {
        match col {
            ColumnSpec::Synthetic { .. }
            | ColumnSpec::Resampled { .. }
            | ColumnSpec::ResampledBinary { .. } => n_non_factor += 1,
            ColumnSpec::FactorSynthetic { n_levels, .. }
            | ColumnSpec::FactorFromFrame { n_levels, .. } => {
                n_factor_dummies += n_levels.saturating_sub(1)
            }
        }
    }
    let (_, target_report, _, contrast_report) =
        translate_targets_and_contrasts(c, n_non_factor, n_factor_dummies, design_test);
    (target_report, contrast_report)
}

/// Kernel column lists for each appended interaction, in design_generation
/// term order. Component columns resolve via the same continuous/dummy
/// position helpers used for marginal terms.
fn translate_interactions(c: &SimulationContract, n_non_factor: u32) -> Vec<Vec<u32>> {
    c.design_generation
        .terms
        .iter()
        .filter_map(|term| {
            if let DesignTerm::Interaction { components } = term {
                Some(
                    components
                        .iter()
                        .map(|comp| match comp {
                            DesignTerm::Direct { column } => {
                                column_position_for_continuous(c, column.0)
                            }
                            DesignTerm::DummyOf {
                                column,
                                level_index,
                            } => column_position_for_dummy(c, column.0, *level_index, n_non_factor),
                            _ => unreachable!("invariant 18 guarantees Direct/DummyOf components"),
                        })
                        .collect(),
                )
            } else {
                None
            }
        })
        .collect()
}

/// Kernel column of an interaction whose components equal `components`,
/// matched against the generation design (which is what the kernel
/// materialises). Used for targets that reference a test-design Interaction.
fn interaction_kernel_col(
    c: &SimulationContract,
    components: &[DesignTerm],
    n_non_factor: u32,
    n_factor_dummies: u32,
) -> u32 {
    let base = 1 + n_non_factor + n_factor_dummies;
    let mut j = 0u32;
    for term in &c.design_generation.terms {
        if let DesignTerm::Interaction { components: gen } = term {
            if gen.as_slice() == components {
                return base + j;
            }
            j += 1;
        }
    }
    unreachable!("test interaction has no matching generation interaction")
}

fn column_position_for_continuous(c: &SimulationContract, gen_idx: u32) -> u32 {
    let mut cursor = 1u32;
    for (i, col) in c.generation.columns.iter().enumerate() {
        if let ColumnSpec::Synthetic { .. }
        | ColumnSpec::Resampled { .. }
        | ColumnSpec::ResampledBinary { .. } = col
        {
            if i as u32 == gen_idx {
                return cursor;
            }
            cursor += 1;
        }
    }
    unreachable!("validate() ensured Direct references a continuous column")
}

fn column_position_for_dummy(
    c: &SimulationContract,
    gen_idx: u32,
    level_index: u32,
    n_non_factor: u32,
) -> u32 {
    let mut cursor = 1u32 + n_non_factor;
    for (i, col) in c.generation.columns.iter().enumerate() {
        if let ColumnSpec::FactorSynthetic { n_levels, .. }
        | ColumnSpec::FactorFromFrame { n_levels, .. } = col
        {
            if i as u32 == gen_idx {
                return cursor + (level_index - 1);
            }
            cursor += n_levels.saturating_sub(1);
        }
    }
    unreachable!("validate() ensured DummyOf references a factor column")
}

/// Expand `generation.cluster_level_columns` (predictor ColumnIds) into the
/// flat list of **kernel** column indices the broadcast step consumes:
/// a continuous predictor → its single `x_full` index; a factor → all of its
/// dummy `x_full` indices. Empty in ⇒ empty out (today's behaviour).
fn expand_cluster_level_columns(c: &SimulationContract, n_non_factor: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for col_id in &c.generation.cluster_level_columns {
        let idx = col_id.0;
        // D4: a Resampled*/FactorFromFrame target is rejected upstream in the
        // Python port; assert it here rather than re-validating (release no-op).
        debug_assert!(
            matches!(
                c.generation.columns[idx as usize],
                ColumnSpec::Synthetic { .. } | ColumnSpec::FactorSynthetic { .. }
            ),
            "cluster-level column must be synthetic (D4: uploaded/resampled rejected upstream)"
        );
        // Resampled*/FactorFromFrame share the same position math as their
        // synthetic counterparts, so the arms are grouped rather than unreachable.
        match &c.generation.columns[idx as usize] {
            ColumnSpec::Synthetic { .. }
            | ColumnSpec::Resampled { .. }
            | ColumnSpec::ResampledBinary { .. } => {
                out.push(column_position_for_continuous(c, idx));
            }
            ColumnSpec::FactorSynthetic { n_levels, .. }
            | ColumnSpec::FactorFromFrame { n_levels, .. } => {
                for level_index in 1..*n_levels {
                    out.push(column_position_for_dummy(c, idx, level_index, n_non_factor));
                }
            }
        }
    }
    out
}

fn factor_position_for(c: &SimulationContract, factor_col: u32) -> u32 {
    let mut k = 0u32;
    for (i, col) in c.generation.columns.iter().enumerate() {
        if matches!(
            col,
            ColumnSpec::FactorSynthetic { .. } | ColumnSpec::FactorFromFrame { .. }
        ) {
            if i as u32 == factor_col {
                return k;
            }
            k += 1;
        }
    }
    unreachable!("validate() ensured posthoc.factor_column is a factor")
}

fn translate_posthoc(
    c: &SimulationContract,
    design_test: &DesignSpec,
) -> (Vec<KernelPosthoc>, Option<f64>) {
    if c.posthoc.is_empty() {
        return (Vec::new(), None);
    }
    let n_non_factor = c
        .generation
        .columns
        .iter()
        .filter(|col| {
            matches!(
                col,
                ColumnSpec::Synthetic { .. }
                    | ColumnSpec::Resampled { .. }
                    | ColumnSpec::ResampledBinary { .. }
            )
        })
        .count() as u32;
    let mut blocks = Vec::with_capacity(c.posthoc.len());
    for ph in &c.posthoc {
        let factor_index = factor_position_for(c, ph.factor_column.0);
        let target_indices = ph
            .target_term_indices
            .iter()
            .map(|&t| match &design_test.terms[t as usize] {
                DesignTerm::DummyOf {
                    column,
                    level_index,
                } => column_position_for_dummy(c, column.0, *level_index, n_non_factor),
                _ => unreachable!("validate() ensured posthoc targets are DummyOf"),
            })
            .collect();
        blocks.push(KernelPosthoc {
            factor_index,
            target_indices,
        });
    }
    // posthoc_alpha is shared across all blocks; use the first block's value.
    let posthoc_alpha = c.posthoc[0].posthoc_alpha;
    (blocks, posthoc_alpha)
}

fn translate_heteroskedasticity_driver(c: &SimulationContract) -> Option<u32> {
    // The contract driver is a generation.columns `ColumnId`; the kernel
    // driver is the `x_full` column index (intercept = 0, first continuous =
    // 1). NOTE: no `-1` — `column_position_for_continuous` already returns
    // that 1-based x_full index (it starts its cursor at 1, skipping the
    // intercept). λ never appears here — it is scenario-only.
    c.outcome
        .heteroskedasticity_driver
        .map(|col| column_position_for_continuous(c, col.0))
}

fn synthetic_to_distribution_only(k: &SyntheticKind) -> Distribution {
    synthetic_to_distribution(k).0
}

fn translate_scenario(s: &engine_contract::ScenarioPerturbations) -> KernelScenario {
    KernelScenario {
        name: s.name.clone(),
        heterogeneity: s.heterogeneity,
        heteroskedasticity_ratio: s.heteroskedasticity_ratio,
        correlation_noise_sd: s.correlation_noise_sd,
        distribution_change_prob: s.distribution_change_prob,
        new_distributions: s
            .new_distributions
            .iter()
            .map(synthetic_to_distribution_only)
            .collect(),
        residual_change_prob: s.residual_change_prob,
        // Contract `ResidualDist` IS the kernel `ResidualDist` (engine-core
        // re-exports it from engine-contract), so no translation needed.
        residual_dists: s.residual_dists.clone(),
        residual_df: s.residual_df,
        sampled_factor_proportions: s.sampled_factor_proportions,
        truth_start: s.truth_start,
        // `LmeScenarioPerturbations` is re-exported verbatim from
        // engine_contract (see crate::spec), so the perturbations clone through.
        lme: s.lme.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine_contract::fixtures::example1_simple_ols;
    use engine_contract::{
        ClusterSizing, ClusterSpec, ColumnId, CorrectionMethod, DesignSpec, EstimatorSpec,
        GenerationSpec, LmeScenarioPerturbations, OutcomeKind, OutcomeSpec, PosthocSpec,
        ResidualSpec, ScenarioPerturbations, SimulationContract, SlopeTerm, TestSpec, TestTarget,
        UploadedFrame, WaldSe,
    };

    #[test]
    fn example1_round_trips_generation_block() {
        let c = example1_simple_ols();
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.n_non_factor, 2);
        assert_eq!(s.n_factor_dummies, 0);
        assert_eq!(
            s.var_types,
            vec![Distribution::Normal, Distribution::Normal]
        );
        assert_eq!(s.var_params, vec![0.0, 0.0]);
        assert_eq!(s.factor_n_levels, Vec::<i32>::new());
        assert_eq!(s.factor_proportions, Vec::<f64>::new());
        assert_eq!(s.correlation, vec![1.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.upload_normal_shape, (0, 0));
        assert_eq!(s.upload_data_shape, (0, 0));
    }

    #[test]
    fn effect_sizes_layout_matches_design_matrix_columns() {
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::FactorSynthetic {
                        n_levels: 3,
                        proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                        sampled_proportions: None,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 2,
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.3, 0.5, 0.4],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 2,
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 3 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        // Engine layout: [intercept, x1, group[2], group[3]]
        assert_eq!(s.effect_sizes, vec![0.0, 0.4, 0.3, 0.5]);
    }

    #[test]
    fn logit_intercept_folds_into_effect_sizes_slot0() {
        // Regression: the Logit data-gen kernel reads the baseline log-odds
        // from effect_sizes[0]. The Const term's coefficient is always 0.0 and
        // the intercept lives in model.intercept; the adapter must fold it in,
        // else every logit sim runs at sigmoid(0) = p = 0.5 regardless of the
        // requested baseline probability.
        let logit_intercept = (0.3f64 / 0.7f64).ln(); // logit(0.3) ≈ -0.8473
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![ColumnSpec::Synthetic {
                    kind: SyntheticKind::Normal,
                    pinned: false,
                }],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Binary,
                intercept: logit_intercept,
                coefficients: vec![0.0, 0.5],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                ],
            }),
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 1 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.effect_sizes[0], logit_intercept);
        assert_eq!(s.effect_sizes[1], 0.5);
    }

    #[test]
    fn target_indices_use_design_test_column_positions() {
        let c = example1_simple_ols();
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.target_indices, vec![1, 2]);
        assert_eq!(s.correction_method, CorrectionMethod::None);
        assert_eq!(s.crit_values.alpha, 0.05);
        assert_eq!(s.crit_values.posthoc_alpha, None);
    }

    #[test]
    fn fit_columns_derived_from_reduced_design_test() {
        // Generation fits y ~ x1 + x2; the test design drops x2, so the engine
        // must fit only {intercept, x1} (omitted-variable / `test_formula`).
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.1, 0.8],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 1 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        // kernel cols: 0 intercept, 1 x1, 2 x2; design_test keeps {0, 1}.
        assert_eq!(
            s.fit_columns,
            vec![0u32, 1],
            "reduced test design ⇒ fit only intercept + x1"
        );
    }

    #[test]
    fn fit_columns_empty_when_design_test_covers_full_generation() {
        // design_test covers every generation column ⇒ no reduction ⇒
        // fit_columns stays empty (the full-fit / current behaviour).
        let c = example1_simple_ols();
        let s = contract_to_simulation_spec(&c).unwrap();
        assert!(
            s.fit_columns.is_empty(),
            "full test design must not request a reduced fit"
        );
    }

    /// Build a `y ~ x1 + x2` contract whose `test_formula` drops the **leading**
    /// term x1 and keeps x2 (the omitted-variable / confounding case). The kept
    /// term x2 is generation-kernel column 2 but reduced-design term position 1.
    fn drop_leading_continuous_contract() -> SimulationContract {
        SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.3, 0.0],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                // keep x2 (ColumnId(1)), drop the leading x1.
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                // term 1 in design_test names x2.
                targets: vec![TestTarget::Marginal { term: 1 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        }
    }

    #[test]
    fn report_indices_name_reduced_term_position_when_leading_term_dropped() {
        // Regression for the `test_formula` target-index bug: dropping a leading
        // generation term and keeping a later one made the host-facing target
        // index land past the reduced effect skeleton, crashing `.summary()`.
        let c = drop_leading_continuous_contract();
        // The kernel read path stays in generation-kernel space (x2 = column 2);
        // batch.rs remaps it through excl_col_remap during the reduced fit.
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(
            s.target_indices,
            vec![2],
            "kernel read index must stay generation-kernel (x2 = column 2)"
        );
        // The host-facing report index is x2's position in the reduced test
        // design (term 1) — in range for the length-2 skeleton (Const + x2).
        let (report_targets, report_contrasts) = report_targets_and_contrasts(&c);
        assert_eq!(
            report_targets,
            vec![1],
            "report index = reduced term position"
        );
        assert!(report_contrasts.is_empty());
        let skeleton_len = c.design_test.as_ref().unwrap().terms.len();
        assert!(
            report_targets.iter().all(|&i| (i as usize) < skeleton_len),
            "every report index must be in range for the reduced skeleton (len {skeleton_len})"
        );
    }

    #[test]
    fn report_indices_handle_factor_target_after_dropped_leading_continuous() {
        // Generation y ~ x1 + g (g a 3-level factor): kernel cols intercept 0,
        // x1 = 1, g[1] = 2, g[2] = 3. The test design drops the leading x1 and
        // keeps the factor, so the kept dummies land at generation-kernel cols
        // {2, 3} — past the reduced skeleton's end (len 3, valid 0..2) — while
        // their reduced term positions are {1, 2}.
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::FactorSynthetic {
                        n_levels: 3,
                        proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                        sampled_proportions: None,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 2,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.0, 0.3, 0.5],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                // keep the factor (both dummies), drop the leading x1.
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 2,
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![
                    TestTarget::Marginal { term: 1 },
                    TestTarget::Marginal { term: 2 },
                ],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(
            s.target_indices,
            vec![2, 3],
            "kernel read = factor dummy columns"
        );
        let (report_targets, _) = report_targets_and_contrasts(&c);
        assert_eq!(
            report_targets,
            vec![1, 2],
            "report = reduced term positions"
        );
        let skeleton_len = c.design_test.as_ref().unwrap().terms.len(); // 3
        assert!(report_targets.iter().all(|&i| (i as usize) < skeleton_len));
    }

    #[test]
    fn posthoc_splits_alpha_into_crit_values_and_indices_into_posthoc_spec() {
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::FactorSynthetic {
                        n_levels: 3,
                        proportions: vec![0.4, 0.3, 0.3],
                        sampled_proportions: None,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 2,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.5, 0.2, 0.4],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 2,
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![
                    TestTarget::Marginal { term: 2 },
                    TestTarget::Marginal { term: 3 },
                ],
                correction: CorrectionMethod::Holm,
                alpha: 0.05,
            },
            posthoc: vec![PosthocSpec {
                factor_column: ColumnId(1),
                target_term_indices: vec![2, 3],
                posthoc_alpha: Some(0.025),
            }],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        c.validate().unwrap();
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.correction_method, CorrectionMethod::Holm);
        assert_eq!(s.crit_values.posthoc_alpha, Some(0.025));
        assert_eq!(s.posthoc.len(), 1);
        let ph = &s.posthoc[0];
        assert_eq!(ph.factor_index, 0);
        assert_eq!(ph.target_indices, vec![2, 3]);
    }

    #[test]
    fn glm_translates_with_intercept() {
        let mut c = example1_simple_ols();
        c.outcome.kind = OutcomeKind::Binary;
        c.estimator = EstimatorSpec::Glm;
        c.outcome.intercept = -1.3863;
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.estimator, EstimatorSpec::Glm);
        assert_eq!(s.intercept, -1.3863);
    }

    #[test]
    fn heteroskedasticity_driver_maps_column_to_xfull_index() {
        let mut c = example1_simple_ols();
        c.outcome.heteroskedasticity_driver = Some(ColumnId(1));
        let s = contract_to_simulation_spec(&c).unwrap();
        // ColumnId(1) is the 2nd continuous predictor → x_full column index 2
        // (intercept = 0, first continuous = 1). No `-1` offset: the kernel
        // driver is the x_full index, used to read col_mean/col_std and x_full.
        assert_eq!(s.heteroskedasticity_driver, Some(2));
    }

    #[test]
    fn residual_dist_and_pin_pass_through_as_typed_enum() {
        let mut c = example1_simple_ols();
        c.outcome.residual = ResidualSpec {
            distribution: ResidualDist::HighKurtosis,
            pinned: true,
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.residual_dist, ResidualDist::HighKurtosis);
        assert!(s.residual_pinned);
    }

    #[test]
    fn lme_cluster_and_scenario_translate() {
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Mle;
        c.generation.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![],
        });
        c.scenario = ScenarioPerturbations {
            name: "realistic".into(),
            heterogeneity: 0.1,
            heteroskedasticity_ratio: 0.05,
            correlation_noise_sd: 0.15,
            distribution_change_prob: 0.2,
            new_distributions: vec![SyntheticKind::RightSkewed],
            residual_change_prob: 0.5,
            residual_dists: vec![ResidualDist::HighKurtosis],
            residual_df: 8.0,
            sampled_factor_proportions: true,
            truth_start: true,
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::Normal,
                random_effect_df: 5.0,
                icc_noise_sd: 0.05,
            }),
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        assert!(
            s.scenario.sampled_factor_proportions,
            "sampled_factor_proportions must map through the adapter"
        );
        assert!(
            s.scenario.truth_start,
            "truth_start must map through the adapter"
        );
        let cluster = s.cluster.expect("cluster set for LME");
        assert_eq!(
            cluster.sizing,
            ClusterSizing::FixedClusters { n_clusters: 20 }
        );
        assert_eq!(cluster.tau_squared, 0.25);
        assert_eq!(s.scenario.name, "realistic");
        assert_eq!(
            s.scenario.new_distributions,
            vec![Distribution::RightSkewed]
        );
        assert_eq!(s.scenario.residual_dists, vec![ResidualDist::HighKurtosis]);
        let lme = s.scenario.lme.unwrap();
        assert_eq!(lme.random_effect_dist, ResidualDist::Normal);
        assert_eq!(lme.icc_noise_sd, 0.05);
    }

    #[test]
    fn joint_all_non_intercept_sets_report_overall() {
        // The v1-parity omnibus shape: Joint{terms = all non-intercept
        // positions of design_test} routes to SimulationSpec.report_overall.
        let mut c = example1_simple_ols();
        c.test.targets = vec![TestTarget::Joint { terms: vec![1, 2] }];
        let s = contract_to_simulation_spec(&c).unwrap();
        assert!(s.report_overall);
        assert_eq!(s.target_indices, Vec::<u32>::new());
    }

    #[test]
    fn joint_all_non_intercept_combined_with_marginals() {
        // Mixed Joint(omnibus) + Marginals: both channels populated.
        let mut c = example1_simple_ols();
        c.test.targets = vec![
            TestTarget::Marginal { term: 1 },
            TestTarget::Marginal { term: 2 },
            TestTarget::Joint { terms: vec![1, 2] },
        ];
        let s = contract_to_simulation_spec(&c).unwrap();
        assert!(s.report_overall);
        assert_eq!(s.target_indices, vec![1, 2]);
    }

    #[test]
    fn joint_subset_target_is_rejected_as_unsupported() {
        // Arbitrary Joint subsets (not the full omnibus shape) still error —
        // we don't yet have a per-joint contrast surface.
        let mut c = example1_simple_ols();
        // design_test has 3 terms (Const, x1, x2). Add a third column so a
        // proper "subset" Joint exists: {1} alone covers a strict subset of
        // non-intercept positions {1, 2}.
        c.test.targets = vec![TestTarget::Joint { terms: vec![1] }];
        match contract_to_simulation_spec(&c) {
            // validate() requires Joint.len() >= 2, so the rejection here
            // comes through `ContractInvalid` rather than `JointNotSupported`.
            Err(AdapterError::ContractInvalid(_)) => {}
            other => panic!("unexpected: {other:?}"),
        }

        // For a true subset rejection, use a contract with >=3 non-intercept
        // terms in design_test and a Joint covering only some of them.
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(2),
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.5, 0.3, 0.2],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(2),
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Joint { terms: vec![1, 2] }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        c.validate().unwrap();
        match contract_to_simulation_spec(&c) {
            Err(AdapterError::JointNotSupported { got }) => assert_eq!(got, vec![1, 2]),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adapted_spec_msgpack_roundtrips() {
        let c = example1_simple_ols();
        let s = contract_to_simulation_spec(&c).unwrap();
        let bytes = rmp_serde::to_vec_named(&s).unwrap();
        let back: SimulationSpec = rmp_serde::from_slice(&bytes).unwrap();
        assert_eq!(s, back);
    }

    /// `TestTarget::Contrast { positive: B, negative: Const(0) }` where 0 is
    /// the Const term (dummy-coding reference) must collapse to
    /// `Marginal { term: B }` in the adapter — same `target_indices`, empty
    /// `contrast_pairs`. Reference-level β = 0 in dummy coding, so the contrast
    /// reduces to a plain marginal test on the positive term.
    ///
    /// Design: 3-level factor (A=ref, B, C) in design_test:
    ///   term 0 = Const (intercept / reference absorber)
    ///   term 1 = DummyOf{factor, 1}  (= B)
    ///   term 2 = DummyOf{factor, 2}  (= C)
    #[test]
    fn contrast_with_reference_collapses_to_marginal() {
        let contract_with_marginal = SimulationContract {
            generation: GenerationSpec {
                columns: vec![ColumnSpec::FactorSynthetic {
                    n_levels: 3,
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    sampled_proportions: None,
                }],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 2,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.4, 0.2],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 2,
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 1 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };

        // Same contract but using Contrast { positive: 1, negative: 0 }
        // where term 0 = Const (reference). Must collapse to Marginal { 1 }.
        let mut contract_with_contrast = contract_with_marginal.clone();
        contract_with_contrast.test.targets = vec![TestTarget::Contrast {
            positive: 1,
            negative: 0,
        }];

        let spec_marginal = contract_to_simulation_spec(&contract_with_marginal).unwrap();
        let spec_contrast = contract_to_simulation_spec(&contract_with_contrast).unwrap();

        // The adapter must produce identical target_indices and empty contrast_pairs.
        assert_eq!(
            spec_marginal.target_indices, spec_contrast.target_indices,
            "Contrast{{B, Const}} must collapse to the same target_indices as Marginal{{B}}"
        );
        assert!(
            spec_contrast.contrast_pairs.is_empty(),
            "reference-collapse should produce no contrast_pairs"
        );
    }

    /// A non-reference `Contrast { positive: 1, negative: 2 }` (B vs C, both
    /// non-reference dummies) must be routed to `contrast_pairs` in the spec,
    /// not to `target_indices`.
    #[test]
    fn contrast_non_reference_routes_to_contrast_pairs() {
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![ColumnSpec::FactorSynthetic {
                    n_levels: 3,
                    proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                    sampled_proportions: None,
                }],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 2,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.4, 0.2],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 2,
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Contrast {
                    positive: 1,
                    negative: 2,
                }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        let spec = contract_to_simulation_spec(&c).unwrap();
        // Both B and C are real dummies (not Const) → goes to contrast_pairs.
        assert!(
            spec.target_indices.is_empty(),
            "non-reference contrast must not add to target_indices"
        );
        assert_eq!(
            spec.contrast_pairs.len(),
            1,
            "non-reference contrast must produce one contrast_pair"
        );
        // Kernel positions: n_non_factor=0, factor dummy layout:
        // [intercept(0), DummyOf{col=0, level=1}(1), DummyOf{col=0, level=2}(2)]
        // positive term=1 → DummyOf{col=0, level=1} → cursor=1+(0)+(1-1)=1
        // negative term=2 → DummyOf{col=0, level=2} → cursor=1+(0)+(2-1)=2
        let (p_col, n_col) = spec.contrast_pairs[0];
        assert_eq!(p_col, 1, "positive contrast column must be 1 (first dummy)");
        assert_eq!(
            n_col, 2,
            "negative contrast column must be 2 (second dummy)"
        );
    }

    /// `contract_to_simulation_spec` returns `Err(ContractInvalid)` when the
    /// underlying `c.validate()` fails. Here we corrupt the coefficient length
    /// so it no longer matches the design terms, and assert the adapter surfaces
    /// a ContractInvalid rather than translating a malformed contract.
    #[test]
    fn invalid_contract_rejected() {
        let mut c = example1_simple_ols();
        // Break the coefficient/term length invariant.
        c.outcome.coefficients.push(0.123);
        assert!(
            c.validate().is_err(),
            "fixture mutation must make validate() fail"
        );
        match contract_to_simulation_spec(&c) {
            Err(AdapterError::ContractInvalid(_)) => {}
            other => panic!("expected ContractInvalid, got {other:?}"),
        }
    }

    /// A continuous×continuous interaction contract maps to one
    /// appended kernel column, a correctly-placed effect size, and a target on
    /// that column.
    ///
    /// Design: y = x1 + x2 + x1:x2 with two Normal predictors.
    /// Kernel layout: [intercept(0), x1(1), x2(2), x1:x2(3)]
    #[test]
    fn continuous_interaction_maps_to_appended_kernel_column() {
        use engine_contract::{
            ColumnId, CorrectionMethod, DesignSpec, EstimatorSpec, GenerationSpec, OutcomeKind,
            OutcomeSpec, ResidualDist, ResidualSpec, ScenarioPerturbations, TestSpec, TestTarget,
        };
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                    DesignTerm::Interaction {
                        components: vec![
                            DesignTerm::Direct {
                                column: ColumnId(0),
                            },
                            DesignTerm::Direct {
                                column: ColumnId(1),
                            },
                        ],
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                // [intercept=0.0, x1=0.5, x2=0.3, x1:x2=0.2]
                coefficients: vec![0.0, 0.5, 0.3, 0.2],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            // No test formula → mirrors the real spec-builder contract, where
            // the adapter falls back to design_generation (term index 3 below).
            design_test: None,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                // Target the interaction term (term index 3 in design_generation)
                targets: vec![TestTarget::Marginal { term: 3 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        c.validate().unwrap();
        let s = contract_to_simulation_spec(&c).unwrap();

        // Kernel layout: [intercept, x1, x2, x1:x2] → n_non_factor=2, n_factor_dummies=0
        assert_eq!(s.n_non_factor, 2);
        assert_eq!(s.n_factor_dummies, 0);
        // interactions[0] = [kernel_col(x1), kernel_col(x2)] = [1, 2]
        assert_eq!(s.interactions, vec![vec![1u32, 2u32]]);
        // effect_sizes: [intercept(0)=0.0, x1(1)=0.5, x2(2)=0.3, x1:x2(3)=0.2]
        assert_eq!(s.effect_sizes, vec![0.0, 0.5, 0.3, 0.2]);
        // target on interaction → appended column 3
        assert_eq!(s.target_indices, vec![3u32]);
    }

    // -----------------------------------------------------------------
    // B1: translate_uploaded_frame populates upload_normal
    // -----------------------------------------------------------------

    /// Helpers to build a minimal valid contract containing an uploaded frame
    /// with one Resampled column. Used by B1 tests.
    fn minimal_uploaded_contract(
        frame_data: Vec<f64>,
        n_rows: u32,
        n_cols: u32,
        frame_column: u32,
    ) -> SimulationContract {
        SimulationContract {
            generation: GenerationSpec {
                columns: vec![ColumnSpec::Resampled { frame_column }],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: Some(UploadedFrame {
                    data: frame_data,
                    n_rows,
                    n_cols,
                    bootstrap: false,
                }),
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.5],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: None,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 1 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        }
    }

    #[test]
    fn b1_upload_normal_populated_and_sorted_ascending() {
        // Frame: 3 rows × 1 col, values [0.3, -1.2, 0.9].
        // Expected upload_normal col 0 (sorted ascending): [-1.2, 0.3, 0.9].
        let frame_data = vec![0.3, -1.2, 0.9]; // row-major 3×1
        let c = minimal_uploaded_contract(frame_data, 3, 1, 0);
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.n_non_factor, 1);
        assert_eq!(s.upload_normal_shape, (3, 1));
        assert_eq!(s.upload_normal.len(), 3);
        // Column 0 in row-major (U, n_nf) → upload_normal[r * 1 + 0] for r in 0..3
        let col0: Vec<f64> = (0..3).map(|r| s.upload_normal[r]).collect();
        assert_eq!(col0, vec![-1.2, 0.3, 0.9]);
        // upload_data must still be populated (Phase-2 bootstrap reads it)
        assert_eq!(s.upload_data_shape, (3, 1));
        assert_eq!(s.upload_data, vec![0.3, -1.2, 0.9]);
    }

    #[test]
    fn b1_upload_normal_shape_matches_n_non_factor() {
        // 4 rows × 2 cols frame; only col 0 is Resampled → n_nf = 1.
        // upload_normal_shape.1 must equal n_non_factor.
        let frame_data: Vec<f64> = vec![1.0, 10.0, 2.0, 11.0, 3.0, 12.0, 4.0, 13.0]; // 4×2
        let c = minimal_uploaded_contract(frame_data, 4, 2, 0);
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.n_non_factor, 1);
        assert_eq!(s.upload_normal_shape, (4, 1));
    }

    // -----------------------------------------------------------------
    // P2-3: strict (bootstrap) mode — upload_data + bootstrap_frame_map,
    // upload_normal empty.
    // -----------------------------------------------------------------

    /// NORTA (bootstrap == false) path: bootstrap_frame_map is empty even though
    /// upload_normal / upload_data are populated. Confirms the existing partial
    /// path is unchanged by the strict wiring.
    #[test]
    fn norta_leaves_bootstrap_frame_map_empty() {
        let frame_data = vec![0.3, -1.2, 0.9]; // 3×1
        let c = minimal_uploaded_contract(frame_data, 3, 1, 0);
        let s = contract_to_simulation_spec(&c).unwrap();
        assert!(!s.upload_normal.is_empty(), "NORTA fills upload_normal");
        assert!(
            s.bootstrap_frame_map.is_empty(),
            "NORTA must leave bootstrap_frame_map empty"
        );
    }

    /// Strict (bootstrap == true): upload_data populated, upload_normal EMPTY,
    /// and bootstrap_frame_map == [Some(0)] for one uploaded non-factor column.
    #[test]
    fn strict_populates_upload_data_and_frame_map_leaves_upload_normal_empty() {
        let frame_data = vec![0.3, -1.2, 0.9]; // 3×1
        let mut c = minimal_uploaded_contract(frame_data.clone(), 3, 1, 0);
        // Flip the frame to strict (bootstrap) mode.
        c.generation.uploaded_frame.as_mut().unwrap().bootstrap = true;
        let s = contract_to_simulation_spec(&c).unwrap();
        // upload_normal MUST be empty in strict mode.
        assert!(
            s.upload_normal.is_empty(),
            "strict must leave upload_normal empty"
        );
        assert_eq!(s.upload_normal_shape, (0, 0));
        // upload_data carries the raw frame for row-sampling.
        assert_eq!(s.upload_data, frame_data);
        assert_eq!(s.upload_data_shape, (3, 1));
        // One non-factor uploaded column → frame column 0; no factors.
        assert_eq!(s.bootstrap_frame_map, vec![Some(0)]);
    }

    /// `cluster_level_columns` expands predictor → kernel columns:
    /// a continuous predictor at non-factor position k → one index `1 + k`;
    /// a factor → ALL its dummy kernel indices.
    ///
    /// Design: cols [Synthetic(0), Synthetic(1), Factor3(2)].
    /// Kernel layout: [intercept(0), x0(1), x1(2), f[1](3), f[2](4)].
    /// Mark {ColumnId(1) (=x1), ColumnId(2) (=factor)} → [2, 3, 4].
    #[test]
    fn cluster_level_columns_expand_to_kernel_indices() {
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::FactorSynthetic {
                        n_levels: 3,
                        proportions: vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                        sampled_proportions: None,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: Some(ClusterSpec {
                    sizing: ClusterSizing::FixedClusters { n_clusters: 10 },
                    tau_squared: 0.2,
                    slopes: vec![],
                    extra_groupings: vec![],
                }),
                uploaded_frame: None,
                cluster_level_columns: vec![ColumnId(1), ColumnId(2)],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(2),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(2),
                        level_index: 2,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.5, 0.3, 0.2, 0.4],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: None,
            estimator: EstimatorSpec::Mle,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 2 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        c.validate().unwrap();
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.between_var_indices, vec![2u32, 3, 4]);
    }

    /// Strict (bootstrap == true) with a **mixed** column layout:
    ///   col 0 = Synthetic{Normal}     → None  (pass-1 non-factor, not from frame)
    ///   col 1 = Resampled{frame_col=0} → Some(0) (pass-1 non-factor, from frame col 0)
    ///   col 2 = FactorFromFrame{frame_col=1, n_levels=2} → Some(1) (pass-2 factor)
    ///
    /// This exercises the Synthetic→None emission and the pass-2 factor extension
    /// that the single-column test did not cover.
    #[test]
    fn strict_mixed_columns_bootstrap_frame_map_has_none_and_both_passes() {
        // Frame: 4 rows × 2 cols (col 0 continuous, col 1 factor).
        // Row-major: row0=[1.0,0.0], row1=[2.0,1.0], row2=[3.0,0.0], row3=[4.0,1.0]
        let frame_data = vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0];
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::Synthetic {
                        kind: SyntheticKind::Normal,
                        pinned: false,
                    },
                    ColumnSpec::Resampled { frame_column: 0 },
                    ColumnSpec::FactorFromFrame {
                        frame_column: 1,
                        n_levels: 2,
                        proportions: vec![0.5, 0.5],
                        sampled_proportions: None,
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: Some(UploadedFrame {
                    data: frame_data,
                    n_rows: 4,
                    n_cols: 2,
                    bootstrap: true,
                }),
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::Direct {
                        column: ColumnId(0),
                    },
                    DesignTerm::Direct {
                        column: ColumnId(1),
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(2),
                        level_index: 1,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.5, 0.3, 0.2],
                residual: ResidualSpec {
                    distribution: engine_contract::ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: None,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![TestTarget::Marginal { term: 1 }],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        c.validate().unwrap();
        let s = contract_to_simulation_spec(&c).unwrap();
        // Strict mode: upload_normal must be empty.
        assert!(
            s.upload_normal.is_empty(),
            "strict must leave upload_normal empty"
        );
        assert_eq!(s.upload_normal_shape, (0, 0));
        // Mixed layout: Synthetic→None, Resampled→Some(0), FactorFromFrame→Some(1).
        assert_eq!(s.bootstrap_frame_map, vec![None, Some(0), Some(1)]);
    }

    // -----------------------------------------------------------------------
    // cluster_slope_design_cols resolution tests (Task 2)
    // -----------------------------------------------------------------------

    /// Build a valid `(1 + x1 | g)` contract from example1_simple_ols:
    /// ColumnId(0) is the first continuous predictor → x_full index 1.
    fn contract_with_primary_slope() -> SimulationContract {
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Mle;
        c.generation.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![SlopeTerm {
                column: ColumnId(0),
                variance: 0.16,
                corr_with_intercept: 0.0,
                corr_with: vec![],
            }],
            extra_groupings: vec![],
        });
        c
    }

    /// Build a valid `(1 + x1 + x2 | g)` contract: both continuous predictors
    /// as slopes. ColumnId(0) → x_full 1; ColumnId(1) → x_full 2.
    fn contract_with_two_primary_slopes() -> SimulationContract {
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Mle;
        c.generation.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![
                SlopeTerm {
                    column: ColumnId(0),
                    variance: 0.16,
                    corr_with_intercept: 0.0,
                    corr_with: vec![],
                },
                SlopeTerm {
                    column: ColumnId(1),
                    variance: 0.09,
                    corr_with_intercept: 0.0,
                    corr_with: vec![0.0], // corr with slope 0; len == k == 1
                },
            ],
            extra_groupings: vec![],
        });
        c
    }

    /// Intercept-only cluster `(1 | g)`: slopes empty → design cols empty.
    fn contract_intercept_only_cluster() -> SimulationContract {
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Mle;
        c.generation.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 20 },
            tau_squared: 0.25,
            slopes: vec![],
            extra_groupings: vec![],
        });
        c
    }

    /// A `(1 + x1 | g)` spec resolves the slope covariate (generation col 0,
    /// the first continuous → x_full index 1) onto cluster_slope_design_cols.
    #[test]
    fn slope_design_cols_resolve_to_x_full_index() {
        let c = contract_with_primary_slope();
        let spec = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(spec.cluster_slope_design_cols, vec![1u32]);
    }

    /// `(1 + x1 + x2 | g)` resolves both, in declaration order (x_full 1, 2).
    #[test]
    fn multi_slope_design_cols_resolve_in_order() {
        let c = contract_with_two_primary_slopes();
        let spec = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(spec.cluster_slope_design_cols, vec![1u32, 2u32]);
    }

    #[test]
    fn no_slope_leaves_design_cols_empty() {
        let c = contract_intercept_only_cluster();
        let spec = contract_to_simulation_spec(&c).unwrap();
        assert!(spec.cluster_slope_design_cols.is_empty());
    }

    /// Two synthetic factors with explicit `sampled_proportions` overrides:
    /// the adapter must lower one entry per factor into `factor_sampled`,
    /// aligned with `factor_n_levels` (one per factor, NOT one per dummy).
    #[test]
    fn translate_generation_lowers_per_factor_sampled() {
        let c = SimulationContract {
            generation: GenerationSpec {
                columns: vec![
                    ColumnSpec::FactorSynthetic {
                        n_levels: 2,
                        proportions: vec![0.7, 0.3],
                        sampled_proportions: Some(false),
                    },
                    ColumnSpec::FactorSynthetic {
                        n_levels: 2,
                        proportions: vec![0.5, 0.5],
                        sampled_proportions: Some(true),
                    },
                ],
                correlations: Correlations::Identity,
                cluster: None,
                uploaded_frame: None,
                cluster_level_columns: vec![],
            },
            design_generation: DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 1,
                    },
                ],
            },
            outcome: OutcomeSpec {
                kind: OutcomeKind::Continuous,
                intercept: 0.0,
                coefficients: vec![0.0, 0.3, 0.4],
                residual: ResidualSpec {
                    distribution: ResidualDist::Normal,
                    pinned: false,
                },
                heteroskedasticity_driver: None,
                link: None,
            },
            design_test: Some(DesignSpec {
                terms: vec![
                    DesignTerm::Const,
                    DesignTerm::DummyOf {
                        column: ColumnId(0),
                        level_index: 1,
                    },
                    DesignTerm::DummyOf {
                        column: ColumnId(1),
                        level_index: 1,
                    },
                ],
            }),
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            nagq: 1,
            test: TestSpec {
                targets: vec![
                    TestTarget::Marginal { term: 1 },
                    TestTarget::Marginal { term: 2 },
                ],
                correction: CorrectionMethod::None,
                alpha: 0.05,
            },
            posthoc: vec![],
            scenario: ScenarioPerturbations::default(),
            max_failed_fraction: 0.03,
        };
        let s = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(s.factor_sampled, vec![Some(false), Some(true)]);
    }

    #[test]
    fn adapter_threads_wald_se() {
        let mut c = example1_simple_ols();
        c.wald_se = WaldSe::Hessian;
        let spec = contract_to_simulation_spec(&c).unwrap();
        assert_eq!(spec.wald_se, WaldSe::Hessian);
    }
}
