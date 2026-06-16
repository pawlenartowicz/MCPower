//! `SimulationContract::validate()` and helpers; numbered invariants (01–21) are the sole gate before the kernel sees a contract.

use crate::contract::SimulationContract;
use crate::design::DesignTerm;
use crate::error::ContractError;
use crate::estimator::EstimatorSpec;
use crate::generation::{
    ClusterSizing, ColumnSpec, Correlations, GroupingRelation, MAX_EXTRA_GROUPINGS, MAX_PRIMARY_Q,
};
use crate::outcome::OutcomeKind;
use crate::test_spec::TestTarget;

/// Free-function form of `SimulationContract::validate`. Hosts that don't
/// want to take `self` by reference (FFI bindings, pre-flight checks) call
/// this instead. Same logic — single source of truth.
pub fn validate(contract: &SimulationContract) -> Result<(), ContractError> {
    contract.validate()
}

impl SimulationContract {
    pub fn validate(&self) -> Result<(), ContractError> {
        self.invariant_01_coefficient_length()?;
        self.invariant_02_test_target_indices_in_range()?;
        self.invariant_03_test_target_well_formed()?;
        self.invariant_04_column_ids_in_range()?;
        self.invariant_05_correlation_dimensions()?;
        self.invariant_06_correlation_only_continuous()?;
        self.invariant_07_factor_proportions_well_formed()?;
        self.invariant_08_dummy_level_in_range()?;
        self.invariant_09_direct_only_on_continuous()?;
        self.invariant_10_heteroskedasticity_on_continuous()?;
        self.invariant_11_uploaded_frame_referenced_when_required()?;
        self.invariant_12_estimator_outcome_matrix()?;
        self.invariant_13_lme_scenario_requires_mle()?;
        self.invariant_14_max_failed_fraction()?;
        self.invariant_15_alpha()?;
        self.invariant_16_correlation_psd()?;
        self.invariant_17_posthoc_consistency()?;
        self.invariant_18_interaction_well_formed()?;
        self.invariant_19_extra_slopes_unsupported()?;
        self.invariant_20_extra_grouping_structure()?;
        self.invariant_21_primary_slope_structure()?;
        Ok(())
    }

    /// Like [`validate`] but skips `invariant_13_lme_scenario_requires_mle`.
    /// Use this for template contracts where the estimator is a temporary
    /// placeholder (`Ols`) that will be overwritten before the final
    /// `validate()` call — specifically the inner per-scenario pass in
    /// `build_linear_contract_with_skeleton` (engine-spec-builder).
    pub fn validate_template(&self) -> Result<(), ContractError> {
        self.invariant_01_coefficient_length()?;
        self.invariant_02_test_target_indices_in_range()?;
        self.invariant_03_test_target_well_formed()?;
        self.invariant_04_column_ids_in_range()?;
        self.invariant_05_correlation_dimensions()?;
        self.invariant_06_correlation_only_continuous()?;
        self.invariant_07_factor_proportions_well_formed()?;
        self.invariant_08_dummy_level_in_range()?;
        self.invariant_09_direct_only_on_continuous()?;
        self.invariant_10_heteroskedasticity_on_continuous()?;
        self.invariant_11_uploaded_frame_referenced_when_required()?;
        self.invariant_12_estimator_outcome_matrix()?;
        // invariant_13_lme_scenario_requires_mle intentionally omitted:
        // the estimator is still the Ols placeholder here; the outer
        // build_contract_with_skeleton pass calls full validate() after
        // writing the real estimator.
        self.invariant_14_max_failed_fraction()?;
        self.invariant_15_alpha()?;
        self.invariant_16_correlation_psd()?;
        self.invariant_17_posthoc_consistency()?;
        self.invariant_18_interaction_well_formed()?;
        self.invariant_19_extra_slopes_unsupported()?;
        self.invariant_20_extra_grouping_structure()?;
        self.invariant_21_primary_slope_structure()?;
        Ok(())
    }

    fn design_test_or_generation(&self) -> &crate::design::DesignSpec {
        self.design_test.as_ref().unwrap_or(&self.design_generation)
    }

    /// Returns the set of DesignSpecs that need to be checked for structural
    /// invariants: always `design_generation`; plus `design_test` when present.
    fn designs_to_check(&self) -> Vec<&crate::design::DesignSpec> {
        let mut v: Vec<&crate::design::DesignSpec> = vec![&self.design_generation];
        if let Some(dt) = &self.design_test {
            v.push(dt);
        }
        v
    }

    fn invariant_01_coefficient_length(&self) -> Result<(), ContractError> {
        let coeffs = self.outcome.coefficients.len();
        let terms = self.design_generation.terms.len();
        if coeffs != terms {
            return Err(ContractError::CoefficientLengthMismatch { coeffs, terms });
        }
        Ok(())
    }

    fn invariant_02_test_target_indices_in_range(&self) -> Result<(), ContractError> {
        let design_test = self.design_test_or_generation();
        let n_terms = design_test.terms.len();
        let check_term = |t: u32| -> Result<(), ContractError> {
            if (t as usize) >= n_terms {
                Err(ContractError::TestTargetTermOutOfRange { term: t, n_terms })
            } else {
                Ok(())
            }
        };
        for target in &self.test.targets {
            match target {
                TestTarget::Marginal { term } => check_term(*term)?,
                TestTarget::Joint { terms } => {
                    for &t in terms {
                        check_term(t)?;
                    }
                }
                TestTarget::Contrast { positive, negative } => {
                    check_term(*positive)?;
                    check_term(*negative)?;
                }
            }
        }
        Ok(())
    }

    fn invariant_03_test_target_well_formed(&self) -> Result<(), ContractError> {
        // Empty targets is valid when posthoc requests are present
        // (posthoc-only runs with n_targets == 0).
        if self.test.targets.is_empty() && self.posthoc.is_empty() {
            return Err(ContractError::InvalidTestSpec(
                "test.targets must be non-empty",
            ));
        }
        let mut seen_marginal: std::collections::BTreeSet<u32> = Default::default();
        // Track (positive, negative) pairs canonically (smaller first) so that
        // {A, B} and {B, A} are considered duplicates.
        let mut seen_contrast: std::collections::BTreeSet<(u32, u32)> = Default::default();
        for target in &self.test.targets {
            match target {
                TestTarget::Marginal { term } => {
                    if !seen_marginal.insert(*term) {
                        return Err(ContractError::InvalidTestSpec("duplicate Marginal target"));
                    }
                }
                TestTarget::Joint { terms } => {
                    if terms.len() < 2 {
                        return Err(ContractError::InvalidTestSpec(
                            "Joint target must have >= 2 terms",
                        ));
                    }
                    let mut local: std::collections::BTreeSet<u32> = Default::default();
                    for t in terms {
                        if !local.insert(*t) {
                            return Err(ContractError::InvalidTestSpec(
                                "duplicate term inside Joint target",
                            ));
                        }
                    }
                }
                TestTarget::Contrast { positive, negative } => {
                    if positive == negative {
                        return Err(ContractError::InvalidTestSpec(
                            "Contrast positive and negative must differ",
                        ));
                    }
                    let key = if positive < negative {
                        (*positive, *negative)
                    } else {
                        (*negative, *positive)
                    };
                    if !seen_contrast.insert(key) {
                        return Err(ContractError::InvalidTestSpec("duplicate Contrast pair"));
                    }
                }
            }
        }
        Ok(())
    }

    fn invariant_04_column_ids_in_range(&self) -> Result<(), ContractError> {
        let n = self.generation.columns.len();
        let check = |id: u32| -> Result<(), ContractError> {
            if (id as usize) >= n {
                Err(ContractError::ColumnIdOutOfRange { id, n_columns: n })
            } else {
                Ok(())
            }
        };
        for design in self.designs_to_check() {
            for term in &design.terms {
                match term {
                    DesignTerm::Const => {}
                    DesignTerm::Direct { column } | DesignTerm::DummyOf { column, .. } => {
                        check(column.0)?;
                    }
                    DesignTerm::Interaction { components } => {
                        for comp in components {
                            if let DesignTerm::Direct { column }
                            | DesignTerm::DummyOf { column, .. } = comp
                            {
                                check(column.0)?;
                            }
                        }
                    }
                }
            }
        }
        if let Some(column) = &self.outcome.heteroskedasticity_driver {
            check(column.0)?;
        }
        if let Correlations::Matrix {
            continuous_columns, ..
        } = &self.generation.correlations
        {
            for c in continuous_columns {
                check(c.0)?;
            }
        }
        for ph in &self.posthoc {
            check(ph.factor_column.0)?;
        }
        for col_id in &self.generation.cluster_level_columns {
            check(col_id.0)?;
        }
        Ok(())
    }

    fn invariant_05_correlation_dimensions(&self) -> Result<(), ContractError> {
        if let Correlations::Matrix {
            continuous_columns,
            values,
        } = &self.generation.correlations
        {
            let n = continuous_columns.len();
            let expected = n * n;
            if values.len() != expected {
                return Err(ContractError::CorrelationDimensionMismatch {
                    got: values.len(),
                    expected,
                    n,
                });
            }
        }
        Ok(())
    }

    fn invariant_06_correlation_only_continuous(&self) -> Result<(), ContractError> {
        if let Correlations::Matrix {
            continuous_columns, ..
        } = &self.generation.correlations
        {
            for c in continuous_columns {
                let col = &self.generation.columns[c.0 as usize];
                match col {
                    ColumnSpec::Synthetic { .. }
                    | ColumnSpec::Resampled { .. }
                    | ColumnSpec::ResampledBinary { .. } => {}
                    ColumnSpec::FactorSynthetic { .. } | ColumnSpec::FactorFromFrame { .. } => {
                        return Err(ContractError::CorrelationOnNonContinuous { id: c.0 });
                    }
                }
            }
        }
        Ok(())
    }

    fn invariant_07_factor_proportions_well_formed(&self) -> Result<(), ContractError> {
        for col in &self.generation.columns {
            let (n_levels, proportions) = match col {
                ColumnSpec::FactorSynthetic {
                    n_levels,
                    proportions,
                    ..
                }
                | ColumnSpec::FactorFromFrame {
                    n_levels,
                    proportions,
                    ..
                } => (*n_levels, proportions),
                _ => continue,
            };
            if n_levels < 2 {
                return Err(ContractError::InvalidFactorProportions(
                    "n_levels must be >= 2",
                ));
            }
            if proportions.len() != n_levels as usize {
                return Err(ContractError::InvalidFactorProportions(
                    "proportions.len must equal n_levels",
                ));
            }
            // Proportion sum guard (mirrors validate_pre_projection tolerance 1e-6).
            let sum: f64 = proportions.iter().sum();
            if (sum - 1.0).abs() > 1e-6 {
                return Err(ContractError::InvalidFactorProportions(
                    "proportions must sum to 1.0",
                ));
            }
        }
        Ok(())
    }

    fn invariant_08_dummy_level_in_range(&self) -> Result<(), ContractError> {
        let check_dummy = |col_id: u32, level_index: u32| -> Result<(), ContractError> {
            let col = &self.generation.columns[col_id as usize];
            let n_levels = match col {
                ColumnSpec::FactorSynthetic { n_levels, .. }
                | ColumnSpec::FactorFromFrame { n_levels, .. } => *n_levels,
                _ => {
                    return Err(ContractError::DirectOnFactor { id: col_id });
                }
            };
            if level_index == 0 || level_index >= n_levels {
                return Err(ContractError::DummyLevelOutOfRange {
                    column: col_id,
                    level_index,
                    n_levels,
                });
            }
            Ok(())
        };
        for design in self.designs_to_check() {
            for term in &design.terms {
                match term {
                    DesignTerm::DummyOf {
                        column,
                        level_index,
                    } => {
                        check_dummy(column.0, *level_index)?;
                    }
                    DesignTerm::Interaction { components } => {
                        for comp in components {
                            if let DesignTerm::DummyOf {
                                column,
                                level_index,
                            } = comp
                            {
                                check_dummy(column.0, *level_index)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn invariant_09_direct_only_on_continuous(&self) -> Result<(), ContractError> {
        let check_direct = |col_id: u32| -> Result<(), ContractError> {
            match &self.generation.columns[col_id as usize] {
                ColumnSpec::Synthetic { .. }
                | ColumnSpec::Resampled { .. }
                | ColumnSpec::ResampledBinary { .. } => Ok(()),
                ColumnSpec::FactorSynthetic { .. } | ColumnSpec::FactorFromFrame { .. } => {
                    Err(ContractError::DirectOnFactor { id: col_id })
                }
            }
        };
        for design in self.designs_to_check() {
            for term in &design.terms {
                match term {
                    DesignTerm::Direct { column } => check_direct(column.0)?,
                    DesignTerm::Interaction { components } => {
                        for comp in components {
                            if let DesignTerm::Direct { column } = comp {
                                check_direct(column.0)?;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn invariant_10_heteroskedasticity_on_continuous(&self) -> Result<(), ContractError> {
        if let Some(column) = &self.outcome.heteroskedasticity_driver {
            match &self.generation.columns[column.0 as usize] {
                ColumnSpec::Synthetic { .. }
                | ColumnSpec::Resampled { .. }
                | ColumnSpec::ResampledBinary { .. } => {}
                _ => return Err(ContractError::HeteroskedasticityOnFactor { id: column.0 }),
            }
        }
        Ok(())
    }

    fn invariant_11_uploaded_frame_referenced_when_required(&self) -> Result<(), ContractError> {
        let mut max_frame_col: Option<u32> = None;
        for col in &self.generation.columns {
            let fc = match col {
                ColumnSpec::Resampled { frame_column }
                | ColumnSpec::ResampledBinary { frame_column, .. }
                | ColumnSpec::FactorFromFrame { frame_column, .. } => Some(*frame_column),
                _ => None,
            };
            if let Some(fc) = fc {
                max_frame_col = Some(max_frame_col.map_or(fc, |m| m.max(fc)));
            }
        }
        let Some(max_col) = max_frame_col else {
            return Ok(());
        };
        let Some(frame) = &self.generation.uploaded_frame else {
            return Err(ContractError::UploadedFrameMissing {
                frame_column: max_col,
            });
        };
        if max_col >= frame.n_cols {
            return Err(ContractError::UploadedFrameMissing {
                frame_column: max_col,
            });
        }
        Ok(())
    }

    // replaces invariant_12 + invariant_13
    fn invariant_12_estimator_outcome_matrix(&self) -> Result<(), ContractError> {
        match self.estimator {
            EstimatorSpec::Ols => {} // accepts any outcome; clustered generation allowed (ignore-clustering OLS)
            EstimatorSpec::Glm => {
                if self.outcome.kind != OutcomeKind::Binary {
                    return Err(ContractError::GlmRequiresBinary);
                }
            }
            EstimatorSpec::Mle => {
                if self.generation.cluster.is_none() {
                    return Err(ContractError::MleWithoutCluster);
                }
            }
        }
        Ok(())
    }

    // replaces invariant_13's lme-scenario guard: scenario.lme (the RE perturbation
    // knobs) is meaningful for the mixed-model estimators that draw random effects —
    // Gaussian LMM (Mle) and clustered logistic GLMM (Glm + cluster). Reject for plain
    // Ols / Glm-without-cluster. The variant name LmeScenarioRequiresMle is kept (renaming
    // an engine-contract error is a wider blast radius than this GLMM relax warrants).
    fn invariant_13_lme_scenario_requires_mle(&self) -> Result<(), ContractError> {
        if self.scenario.lme.is_none() {
            return Ok(());
        }
        let mle = self.estimator == EstimatorSpec::Mle;
        let glmm = self.estimator == EstimatorSpec::Glm && self.generation.cluster.is_some();
        if mle || glmm {
            Ok(())
        } else {
            Err(ContractError::LmeScenarioRequiresMle)
        }
    }

    fn invariant_14_max_failed_fraction(&self) -> Result<(), ContractError> {
        let v = self.max_failed_fraction;
        if v.is_nan() || !(0.0..=1.0).contains(&v) {
            return Err(ContractError::MaxFailedFractionOutOfRange { got: v });
        }
        Ok(())
    }

    fn invariant_15_alpha(&self) -> Result<(), ContractError> {
        let v = self.test.alpha;
        if !(v > 0.0 && v < 1.0) {
            return Err(ContractError::AlphaOutOfRange { got: v });
        }
        Ok(())
    }

    /// Invariant 16: `Correlations::Matrix` values must form a positive
    /// semi-definite matrix. Uses a pure-Rust Cholesky attempt (no external dep)
    /// on the n×n column-major flat buffer, mirroring the algorithm in
    /// `engine-spec-builder::correlation::is_psd` (same tolerance -1e-8 so
    /// singular-but-PSD matrices, e.g. perfectly-correlated pairs, are accepted).
    fn invariant_16_correlation_psd(&self) -> Result<(), ContractError> {
        let Correlations::Matrix {
            continuous_columns,
            values,
        } = &self.generation.correlations
        else {
            return Ok(());
        };
        let n = continuous_columns.len();
        if n == 0 {
            return Ok(());
        }
        // Dimension check already done by invariant_05; only proceed if sizes match.
        if values.len() != n * n {
            return Ok(());
        }
        // Tolerance matches spec-builder's `is_psd` (correlation.rs) so a matrix
        // that is PSD but singular (smallest eigenvalue ≈ 0) is not falsely rejected.
        const EPS: f64 = -1e-8;
        let mut l = vec![0.0_f64; n * n];
        for j in 0..n {
            let mut s = values[j * n + j];
            for k in 0..j {
                s -= l[k * n + j] * l[k * n + j];
            }
            if s < EPS {
                return Err(ContractError::CorrelationNotPsd);
            }
            let s = s.max(0.0).sqrt();
            l[j * n + j] = s;
            if s == 0.0 {
                // Zero pivot: column below must also be zero for PSD.
                for i in (j + 1)..n {
                    let mut sum = values[j * n + i];
                    for k in 0..j {
                        sum -= l[k * n + i] * l[k * n + j];
                    }
                    if sum.abs() > 1e-8 {
                        return Err(ContractError::CorrelationNotPsd);
                    }
                }
                continue;
            }
            for i in (j + 1)..n {
                let mut sum = values[j * n + i];
                for k in 0..j {
                    sum -= l[k * n + i] * l[k * n + j];
                }
                l[j * n + i] = sum / s;
            }
        }
        Ok(())
    }

    /// Invariant 18: every `Interaction` has ≥2 components and each component
    /// is a `Direct` or `DummyOf` (no `Const`, no nested `Interaction`).
    fn invariant_18_interaction_well_formed(&self) -> Result<(), ContractError> {
        for design in self.designs_to_check() {
            for term in &design.terms {
                if let DesignTerm::Interaction { components } = term {
                    if components.len() < 2 {
                        return Err(ContractError::InteractionTooFewComponents {
                            n: components.len(),
                        });
                    }
                    for comp in components {
                        match comp {
                            DesignTerm::Direct { .. } | DesignTerm::DummyOf { .. } => {}
                            _ => return Err(ContractError::InteractionBadComponent),
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // Primary slopes are live capability (validated structurally by invariant_21).
    // Slopes on EXTRA groupings stay rejected — supporting multi-grouping
    // random slopes requires non-trivial composition changes and is not yet
    // implemented.
    fn invariant_19_extra_slopes_unsupported(&self) -> Result<(), ContractError> {
        if let Some(cluster) = &self.generation.cluster {
            for g in &cluster.extra_groupings {
                if !g.slopes.is_empty() {
                    return Err(ContractError::ExtraSlopesUnsupported);
                }
            }
        }
        Ok(())
    }

    /// Structural rules for the primary random slopes (§2.1/§3.2): one or more,
    /// composable with intercept-only crossed/nested extra groupings (slopes ON an
    /// extra grouping stay rejected by invariant_19); each on a continuous column
    /// that is a Direct fixed effect; sane variance/correlation; a non-zero
    /// intercept variance to correlate against; a corr_with lower triangle of the
    /// right length; and a PSD covariance D.
    fn invariant_21_primary_slope_structure(&self) -> Result<(), ContractError> {
        let Some(cluster) = &self.generation.cluster else {
            return Ok(());
        };
        if cluster.slopes.is_empty() {
            return Ok(());
        }
        // q_p = 1 + #slopes; must fit the solver's stack-allocated θ truth-start
        // buffer (MAX_THETA in lmm.rs = MAX_PRIMARY_Q*(MAX_PRIMARY_Q+1)/2 + MAX_EXTRA_GROUPINGS).
        if 1 + cluster.slopes.len() > MAX_PRIMARY_Q {
            return Err(ContractError::TooManySlopes {
                got: (1 + cluster.slopes.len()) as u32,
                max: MAX_PRIMARY_Q as u32,
            });
        }
        if !(cluster.tau_squared.is_finite() && cluster.tau_squared > 0.0) {
            return Err(ContractError::SlopeInterceptVarianceMissing {
                got: cluster.tau_squared,
            });
        }
        for (k, slope) in cluster.slopes.iter().enumerate() {
            let id = slope.column.0;
            // Range (mirrors invariant_04's continuous range guard).
            if (id as usize) >= self.generation.columns.len() {
                return Err(ContractError::ColumnIdOutOfRange {
                    id,
                    n_columns: self.generation.columns.len(),
                });
            }
            // Continuous (Synthetic / Resampled / ResampledBinary), not a factor.
            match &self.generation.columns[id as usize] {
                ColumnSpec::Synthetic { .. }
                | ColumnSpec::Resampled { .. }
                | ColumnSpec::ResampledBinary { .. } => {}
                _ => return Err(ContractError::SlopeColumnNotContinuous { id }),
            }
            // Must appear as a Direct fixed effect (so x_full carries its values,
            // the s-Gram recovery works, and the model matches the lmer oracle).
            let has_direct = self.designs_to_check().iter().any(|d| {
                d.terms
                    .iter()
                    .any(|t| matches!(t, DesignTerm::Direct { column } if column.0 == id))
            });
            if !has_direct {
                return Err(ContractError::SlopeColumnNotInDesign { id });
            }
            if !(slope.variance.is_finite() && slope.variance >= 0.0) {
                return Err(ContractError::SlopeVarianceInvalid { got: slope.variance });
            }
            if !(slope.corr_with_intercept.is_finite()
                && slope.corr_with_intercept.abs() <= 1.0)
            {
                return Err(ContractError::SlopeCorrInvalid {
                    got: slope.corr_with_intercept,
                });
            }
            // corr_with carries exactly k entries (correlations with slopes 0..k).
            if slope.corr_with.len() != k {
                return Err(ContractError::SlopeCorrWithLenInvalid {
                    slope: k as u32,
                    expected: k as u32,
                    got: slope.corr_with.len() as u32,
                });
            }
            for &c in &slope.corr_with {
                if !(c.is_finite() && c.abs() <= 1.0) {
                    return Err(ContractError::SlopeCorrInvalid { got: c });
                }
            }
        }
        // The full RE covariance must admit a Cholesky (data-gen needs it).
        if !cluster.re_covariance_is_psd() {
            return Err(ContractError::SlopeCovarianceNotPSD);
        }
        Ok(())
    }

    // Structural rules for extra groupings (§2.2): the total count must fit the
    // solver's fixed id buffer (MAX_EXTRA_GROUPINGS — else add_rows_multi/
    // truth-start overflow their stack arrays), crossed factors need the
    // FixedClusters primary (the FixedSize interaction is ambiguous — honest
    // deferral), at most one nesting level below the primary AND at most one
    // nested factor (sibling nested factors would need within-family fill the
    // solver doesn't carry), Regime-B nesting must divide the cluster size,
    // counts must be real groupings (>= 2 levels), variances must be sane.
    fn invariant_20_extra_grouping_structure(&self) -> Result<(), ContractError> {
        let Some(cluster) = &self.generation.cluster else {
            return Ok(());
        };
        if cluster.extra_groupings.len() > MAX_EXTRA_GROUPINGS {
            return Err(ContractError::TooManyExtraGroupings {
                got: cluster.extra_groupings.len() as u32,
                max: MAX_EXTRA_GROUPINGS as u32,
            });
        }
        let mut n_nested = 0u32;
        for g in &cluster.extra_groupings {
            if !(g.tau_squared.is_finite() && g.tau_squared >= 0.0) {
                return Err(ContractError::GroupingTauSquaredInvalid {
                    got: g.tau_squared,
                });
            }
            match &g.relation {
                GroupingRelation::Crossed { n_clusters } => {
                    if *n_clusters < 2 {
                        return Err(ContractError::GroupingCountTooSmall {
                            got: *n_clusters,
                        });
                    }
                    if !matches!(cluster.sizing, ClusterSizing::FixedClusters { .. }) {
                        return Err(ContractError::CrossedRequiresFixedClusters);
                    }
                }
                GroupingRelation::NestedWithin { n_per_parent } => {
                    if *n_per_parent < 2 {
                        return Err(ContractError::GroupingCountTooSmall {
                            got: *n_per_parent,
                        });
                    }
                    n_nested += 1;
                    if n_nested > 1 {
                        return Err(ContractError::MultipleNestedUnsupported);
                    }
                    if let ClusterSizing::FixedSize { cluster_size } = cluster.sizing {
                        if cluster_size % n_per_parent != 0 {
                            return Err(ContractError::NestedSizeIndivisible {
                                cluster_size,
                                n_per_parent: *n_per_parent,
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn invariant_17_posthoc_consistency(&self) -> Result<(), ContractError> {
        if self.posthoc.is_empty() {
            return Ok(());
        }
        // 17c: posthoc is only supported for Ols.
        if self.estimator != EstimatorSpec::Ols {
            return Err(ContractError::PosthocRequiresOls);
        }
        let design_test = self.design_test_or_generation();
        for ph in &self.posthoc {
            // 17a: factor_column must be a factor.
            match &self.generation.columns[ph.factor_column.0 as usize] {
                ColumnSpec::FactorSynthetic { .. } | ColumnSpec::FactorFromFrame { .. } => {}
                _ => {
                    return Err(ContractError::InvalidPosthoc(
                        "factor_column is not a factor",
                    ))
                }
            }
            // 17b: every target_term_index must point at a DummyOf of the same factor.
            for &t in &ph.target_term_indices {
                let term =
                    design_test
                        .terms
                        .get(t as usize)
                        .ok_or(ContractError::InvalidPosthoc(
                            "posthoc target_term_index out of range",
                        ))?;
                match term {
                    DesignTerm::DummyOf { column, .. } if *column == ph.factor_column => {}
                    _ => {
                        return Err(ContractError::InvalidPosthoc(
                            "posthoc target must be a DummyOf of factor_column",
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}
