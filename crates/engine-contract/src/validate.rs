//! `SimulationContract::validate()` and helpers; numbered invariants (01–25) are the sole gate before the kernel sees a contract.

use crate::contract::SimulationContract;
use crate::design::DesignTerm;
use crate::error::ContractError;
use crate::estimator::EstimatorSpec;
use crate::generation::{
    ClusterSizing, ColumnSpec, Correlations, GroupingRelation, SlopeTerm, MAX_EXTRA_GROUPINGS,
    MAX_EXTRA_Q, MAX_PRIMARY_Q,
};
use crate::outcome::OutcomeKind;
use crate::scenarios::ScenarioPerturbations;
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
        self.invariant_19_extra_grouping_slope_structure()?;
        self.invariant_20_extra_grouping_structure()?;
        self.invariant_21_primary_slope_structure()?;
        self.invariant_22_scenario_perturbations_well_formed()?;
        self.invariant_23_binary_probabilities_in_range()?;
        self.invariant_24_link_matches_kind()?;
        self.invariant_25_nagq_backstop()?;
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
        self.invariant_19_extra_grouping_slope_structure()?;
        self.invariant_20_extra_grouping_structure()?;
        self.invariant_21_primary_slope_structure()?;
        self.invariant_22_scenario_perturbations_well_formed()?;
        self.invariant_23_binary_probabilities_in_range()?;
        // invariant_24_link_matches_kind / invariant_25_nagq_backstop
        // intentionally omitted (mirrors the invariant_13 omission above):
        // outcome.kind, outcome.link, and generation.cluster are still
        // placeholders here (Ols/Continuous/no-cluster), while nagq is already
        // set from the LinearSpec — running the backstops now would reject an
        // eligible nagq>1 against the un-patched template. The outer
        // build_contract_with_skeleton pass patches those fields and calls full
        // validate(), which enforces both.
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
                // GLM/GLMM fits the non-Gaussian kinds: Binary (logit/probit)
                // and Count (Poisson log). Continuous stays OLS/LMM.
                if !matches!(self.outcome.kind, OutcomeKind::Binary | OutcomeKind::Count) {
                    return Err(ContractError::GlmRequiresBinaryOrCount);
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

    // invariant_19: random slopes on EXTRA groupings are SUPPORTED (gated kernel
    // path). Each extra factor's RE width q_g = 1 + #slopes must fit the solver's
    // per-factor vech(Λ_g) block (MAX_EXTRA_Q — else the θ packing / tail scratch
    // overflow), and every slope column gets the same structural validation as a
    // primary slope (continuous Direct fixed effect, sane variance/correlation,
    // corr_with lower triangle). The slope↔intercept and slope↔slope correlations
    // form each factor's own D_g; PSD of D_g is the data-gen layer's concern.
    fn invariant_19_extra_grouping_slope_structure(&self) -> Result<(), ContractError> {
        let Some(cluster) = &self.generation.cluster else {
            return Ok(());
        };
        for g in &cluster.extra_groupings {
            let q_g = 1 + g.slopes.len();
            if q_g > MAX_EXTRA_Q {
                return Err(ContractError::ExtraGroupingQTooLarge {
                    got: q_g as u32,
                    max: MAX_EXTRA_Q as u32,
                });
            }
            self.check_slope_terms(&g.slopes)?;
        }
        Ok(())
    }

    /// Per-slope structural checks shared by the primary factor (invariant_21)
    /// and the extra groupings (invariant_19): each slope sits on an in-range,
    /// continuous column that appears as a Direct fixed effect (so x_full carries
    /// its values, the s-Gram recovery works, and the model matches the lmer
    /// oracle); finite variance ≥ 0; finite correlations in [-1, 1]; and a
    /// corr_with lower triangle of length exactly `k` (the slope's own index).
    /// The error vocabulary is identical for both factors.
    fn check_slope_terms(&self, slopes: &[SlopeTerm]) -> Result<(), ContractError> {
        for (k, slope) in slopes.iter().enumerate() {
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
                return Err(ContractError::SlopeVarianceInvalid {
                    got: slope.variance,
                });
            }
            if !(slope.corr_with_intercept.is_finite() && slope.corr_with_intercept.abs() <= 1.0) {
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
        Ok(())
    }

    /// Structural rules for the primary random slopes: one or more,
    /// composable with crossed/nested extra groupings (their own slopes are
    /// validated by invariant_19); each on a continuous column
    /// that is a Direct fixed effect; sane variance/correlation; a non-zero
    /// intercept variance to correlate against; a corr_with lower triangle of the
    /// right length; and a PSD covariance D.
    fn invariant_21_primary_slope_structure(&self) -> Result<(), ContractError> {
        let Some(cluster) = &self.generation.cluster else {
            return Ok(());
        };
        if cluster.slopes.is_empty() {
            // Intercept-only: tau_squared may be 0.0 (ICC = 0 is a valid
            // control-arm value) but must be finite and non-negative. The
            // slopes path tightens this to tau_squared > 0.0 (a slope needs a
            // non-zero intercept variance to correlate against).
            if !(cluster.tau_squared.is_finite() && cluster.tau_squared >= 0.0) {
                return Err(ContractError::InterceptOnlyTauSquaredInvalid {
                    got: cluster.tau_squared,
                });
            }
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
        self.check_slope_terms(&cluster.slopes)?;
        // The full RE covariance must admit a Cholesky (data-gen needs it).
        if !cluster.re_covariance_is_psd() {
            return Err(ContractError::SlopeCovarianceNotPSD);
        }
        Ok(())
    }

    // Structural rules for extra groupings: the total count must fit the
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
                return Err(ContractError::GroupingTauSquaredInvalid { got: g.tau_squared });
            }
            match &g.relation {
                GroupingRelation::Crossed { n_clusters } => {
                    if *n_clusters < 2 {
                        return Err(ContractError::GroupingCountTooSmall { got: *n_clusters });
                    }
                    if !matches!(cluster.sizing, ClusterSizing::FixedClusters { .. }) {
                        return Err(ContractError::CrossedRequiresFixedClusters);
                    }
                }
                GroupingRelation::NestedWithin { n_per_parent } => {
                    if *n_per_parent < 2 {
                        return Err(ContractError::GroupingCountTooSmall { got: *n_per_parent });
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

    /// Invariant 22: all numeric fields in `ScenarioPerturbations` and its
    /// `lme` sub-block (`LmeScenarioPerturbations`) are finite and within the
    /// domain that the kernel can consume:
    ///
    /// - `heterogeneity` ≥ 0.0  (additive jitter SD; 0.0 = off)
    /// - `correlation_noise_sd` ≥ 0.0  (0.0 = off)
    /// - `heteroskedasticity_ratio` ≥ 0.0  (1.0 = homoskedastic; the kernel
    ///   clamps sub-1.0 ratios via `.max(1.0)`, so they are defined-as-off, not
    ///   malformed — only NaN/Inf/negative are rejected)
    /// - `distribution_change_prob` ∈ [0.0, 1.0]  (probability; 0.0 = off)
    /// - `residual_change_prob` ∈ [0.0, 1.0]  (probability; 0.0 = off)
    /// - `residual_df` is finite — 0.0 is the inactive sentinel; no positivity
    ///   check (the kernel interprets 0.0 as "Normal fallback")
    /// - `lme.random_effect_df` is finite and ≥ 0.0 — 0.0 is the Normal-path
    ///   sentinel; negative values are meaningless
    /// - `lme.icc_noise_sd` is finite and ≥ 0.0 — exactly 0.0 is a valid
    ///   control-arm value meaning "no ICC perturbation"
    fn invariant_22_scenario_perturbations_well_formed(&self) -> Result<(), ContractError> {
        let s: &ScenarioPerturbations = &self.scenario;

        let check = |v: f64, field: &'static str, ok: bool| -> Result<(), ContractError> {
            if !ok {
                Err(ContractError::ScenarioPerturbationOutOfRange { field, got: v })
            } else {
                Ok(())
            }
        };

        check(
            s.heterogeneity,
            "heterogeneity",
            s.heterogeneity.is_finite() && s.heterogeneity >= 0.0,
        )?;
        check(
            s.correlation_noise_sd,
            "correlation_noise_sd",
            s.correlation_noise_sd.is_finite() && s.correlation_noise_sd >= 0.0,
        )?;
        check(
            s.heteroskedasticity_ratio,
            "heteroskedasticity_ratio",
            s.heteroskedasticity_ratio.is_finite() && s.heteroskedasticity_ratio >= 0.0,
        )?;
        check(
            s.distribution_change_prob,
            "distribution_change_prob",
            s.distribution_change_prob.is_finite()
                && (0.0..=1.0).contains(&s.distribution_change_prob),
        )?;
        check(
            s.residual_change_prob,
            "residual_change_prob",
            s.residual_change_prob.is_finite() && (0.0..=1.0).contains(&s.residual_change_prob),
        )?;
        // residual_df: 0.0 is the inactive sentinel; only reject non-finite.
        check(s.residual_df, "residual_df", s.residual_df.is_finite())?;

        if let Some(lme) = &s.lme {
            let lme_check = |v: f64, field: &'static str, ok: bool| -> Result<(), ContractError> {
                if !ok {
                    Err(ContractError::LmePerturbationOutOfRange { field, got: v })
                } else {
                    Ok(())
                }
            };
            // random_effect_df: 0.0 = Normal-path sentinel; >= 0.0 required.
            lme_check(
                lme.random_effect_df,
                "random_effect_df",
                lme.random_effect_df.is_finite() && lme.random_effect_df >= 0.0,
            )?;
            // icc_noise_sd: 0.0 = no perturbation (valid for a control arm).
            lme_check(
                lme.icc_noise_sd,
                "icc_noise_sd",
                lme.icc_noise_sd.is_finite() && lme.icc_noise_sd >= 0.0,
            )?;
        }

        Ok(())
    }

    /// Invariant 23: `SyntheticKind::Binary { p }` must be a valid probability
    /// on the closed unit interval, and `ColumnSpec::ResampledBinary { proportion }`
    /// must likewise be finite and in [0.0, 1.0].  Both endpoints are allowed:
    /// `p = 0.0`/`p = 1.0` are degenerate but structurally valid marginals;
    /// `proportion = 0.0` arises from a real all-zero uploaded column.
    fn invariant_23_binary_probabilities_in_range(&self) -> Result<(), ContractError> {
        for col in &self.generation.columns {
            match col {
                ColumnSpec::Synthetic {
                    kind: crate::generation::SyntheticKind::Binary { p },
                    ..
                } if !(p.is_finite() && (0.0..=1.0).contains(p)) => {
                    return Err(ContractError::BinaryProbabilityOutOfRange { got: *p });
                }
                ColumnSpec::ResampledBinary { proportion, .. }
                    if !(proportion.is_finite() && (0.0..=1.0).contains(proportion)) =>
                {
                    return Err(ContractError::ResampledBinaryProportionOutOfRange {
                        got: *proportion,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// A non-canonical link override is only defined for one kind: probit on
    /// Binary. Reject `Some(Probit)` on Continuous/Count (Count is log-only,
    /// Continuous has no link).
    fn invariant_24_link_matches_kind(&self) -> Result<(), ContractError> {
        if self.outcome.link == Some(crate::outcome::LinkKind::Probit)
            && self.outcome.kind != OutcomeKind::Binary
        {
            return Err(ContractError::ProbitRequiresBinary);
        }
        Ok(())
    }

    /// Backstop for the AGQ node count. `nagq` must be odd and in `1..=25`
    /// always; `nagq > 1` additionally requires an eligible shape: a
    /// Binary/Count outcome, a clustered spec, no crossed/nested extra
    /// groupings, and ≤ 3 REs per group (intercept + slopes). Hosts strip
    /// ineligible `nagq > 1` to 1 with a warning before building the contract,
    /// so reaching either error here indicates a host bug. Mirrors glmm's
    /// `fit.rs::assert_model_shape` and each port's spec-builder eligibility
    /// check — change all three together.
    fn invariant_25_nagq_backstop(&self) -> Result<(), ContractError> {
        let k = self.nagq;
        if k == 0 || k > 25 || k % 2 == 0 {
            return Err(ContractError::NagqOutOfRange { got: k });
        }
        if k == 1 {
            return Ok(());
        }
        let kind_ok = matches!(self.outcome.kind, OutcomeKind::Binary | OutcomeKind::Count);
        let Some(cluster) = &self.generation.cluster else {
            return Err(ContractError::NagqIneligibleShape { got: k });
        };
        let single_grouping = cluster.extra_groupings.is_empty();
        let re_count = 1 + cluster.slopes.len(); // intercept + random slopes
        if !(kind_ok && single_grouping && re_count <= 3) {
            return Err(ContractError::NagqIneligibleShape { got: k });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixtures::example1_simple_ols;
    use crate::generation::{ClusterSizing, ClusterSpec, ColumnSpec, SyntheticKind};
    use crate::outcome::ResidualDist;
    use crate::scenarios::LmeScenarioPerturbations;

    // ---- helpers ----------------------------------------------------------------

    /// Base OLS contract with a clustered (intercept-only) generation spec;
    /// used by the tau_squared intercept-only tests.
    fn clustered_contract(tau: f64) -> SimulationContract {
        let mut c = example1_simple_ols();
        c.generation.cluster = Some(ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters { n_clusters: 10 },
            tau,
        ));
        c
    }

    /// Base OLS contract with a `Binary { p }` synthetic column.
    fn binary_contract(p: f64) -> SimulationContract {
        let mut c = example1_simple_ols();
        // replace the first synthetic column with Binary
        c.generation.columns[0] = ColumnSpec::Synthetic {
            kind: SyntheticKind::Binary { p },
            pinned: false,
        };
        c
    }

    /// Base OLS contract with a `ResampledBinary` column backed by a minimal
    /// uploaded frame.
    fn resampled_binary_contract(proportion: f64) -> SimulationContract {
        use crate::generation::{Correlations, UploadedFrame};
        let mut c = example1_simple_ols();
        c.generation.columns[0] = ColumnSpec::ResampledBinary {
            frame_column: 0,
            proportion,
        };
        c.generation.uploaded_frame = Some(UploadedFrame {
            data: vec![0.0, 1.0, 0.0, 1.0],
            n_rows: 2,
            n_cols: 2,
            bootstrap: false,
        });
        // Remove correlation matrix that references col 0 as continuous
        // (invariant_06 would reject a ResampledBinary in the matrix, but
        // example1 uses Identity so nothing to do).
        c.generation.correlations = Correlations::Identity;
        c
    }

    // ---- invariant_22: ScenarioPerturbations --------------------------------

    #[test]
    fn inv22_default_scenario_passes() {
        // ScenarioPerturbations::default() must pass all guards.
        assert!(example1_simple_ols().validate().is_ok());
    }

    // heterogeneity

    #[test]
    fn inv22_heterogeneity_nan_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.heterogeneity = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "heterogeneity",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_heterogeneity_negative_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.heterogeneity = -0.1;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "heterogeneity",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_heterogeneity_zero_passes() {
        let mut c = example1_simple_ols();
        c.scenario.heterogeneity = 0.0; // inactive sentinel
        assert!(c.validate().is_ok());
    }

    // correlation_noise_sd

    #[test]
    fn inv22_correlation_noise_sd_nan_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.correlation_noise_sd = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "correlation_noise_sd",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_correlation_noise_sd_negative_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.correlation_noise_sd = -1.0;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "correlation_noise_sd",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_correlation_noise_sd_zero_passes() {
        let mut c = example1_simple_ols();
        c.scenario.correlation_noise_sd = 0.0;
        assert!(c.validate().is_ok());
    }

    // heteroskedasticity_ratio

    #[test]
    fn inv22_heteroskedasticity_ratio_nan_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.heteroskedasticity_ratio = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "heteroskedasticity_ratio",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_heteroskedasticity_ratio_negative_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.heteroskedasticity_ratio = -0.5;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "heteroskedasticity_ratio",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_heteroskedasticity_ratio_below_one_passes() {
        // Sub-1.0 ratios are clamped to 1.0 (off) by the kernel, not malformed —
        // an engine-core translation test exercises 0.05, so the contract must
        // accept it rather than reject a legitimate host value.
        let mut c = example1_simple_ols();
        c.scenario.heteroskedasticity_ratio = 0.05;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inv22_heteroskedasticity_ratio_one_passes() {
        let mut c = example1_simple_ols();
        c.scenario.heteroskedasticity_ratio = 1.0; // homoskedastic / off sentinel
        assert!(c.validate().is_ok());
    }

    // distribution_change_prob

    #[test]
    fn inv22_distribution_change_prob_nan_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.distribution_change_prob = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "distribution_change_prob",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_distribution_change_prob_above_one_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.distribution_change_prob = 1.1;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "distribution_change_prob",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_distribution_change_prob_zero_passes() {
        let mut c = example1_simple_ols();
        c.scenario.distribution_change_prob = 0.0;
        assert!(c.validate().is_ok());
    }

    // residual_change_prob

    #[test]
    fn inv22_residual_change_prob_nan_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.residual_change_prob = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "residual_change_prob",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_residual_change_prob_negative_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.residual_change_prob = -0.5;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "residual_change_prob",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_residual_change_prob_zero_passes() {
        let mut c = example1_simple_ols();
        c.scenario.residual_change_prob = 0.0;
        assert!(c.validate().is_ok());
    }

    // residual_df: 0.0 is the inactive sentinel — only NaN/inf should reject.

    #[test]
    fn inv22_residual_df_nan_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.residual_df = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "residual_df",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_residual_df_inf_rejects() {
        let mut c = example1_simple_ols();
        c.scenario.residual_df = f64::INFINITY;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ScenarioPerturbationOutOfRange {
                    field: "residual_df",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_residual_df_zero_passes() {
        // 0.0 = inactive sentinel; must not be rejected.
        let mut c = example1_simple_ols();
        c.scenario.residual_df = 0.0;
        assert!(c.validate().is_ok());
    }

    // LmeScenarioPerturbations.random_effect_df

    fn lme_contract() -> SimulationContract {
        use crate::estimator::EstimatorSpec;
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Mle;
        c.generation.cluster = Some(ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters { n_clusters: 10 },
            0.25,
        ));
        c.scenario.lme = Some(LmeScenarioPerturbations {
            random_effect_dist: ResidualDist::Normal,
            random_effect_df: 5.0,
            icc_noise_sd: 0.1,
        });
        c
    }

    #[test]
    fn inv22_lme_random_effect_df_nan_rejects() {
        let mut c = lme_contract();
        c.scenario.lme.as_mut().unwrap().random_effect_df = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::LmePerturbationOutOfRange {
                    field: "random_effect_df",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_lme_random_effect_df_negative_rejects() {
        let mut c = lme_contract();
        c.scenario.lme.as_mut().unwrap().random_effect_df = -1.0;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::LmePerturbationOutOfRange {
                    field: "random_effect_df",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_lme_random_effect_df_zero_passes() {
        // 0.0 = Normal-path sentinel; must not be rejected.
        let mut c = lme_contract();
        c.scenario.lme.as_mut().unwrap().random_effect_df = 0.0;
        assert!(c.validate().is_ok());
    }

    // LmeScenarioPerturbations.icc_noise_sd

    #[test]
    fn inv22_lme_icc_noise_sd_nan_rejects() {
        let mut c = lme_contract();
        c.scenario.lme.as_mut().unwrap().icc_noise_sd = f64::NAN;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::LmePerturbationOutOfRange {
                    field: "icc_noise_sd",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_lme_icc_noise_sd_negative_rejects() {
        let mut c = lme_contract();
        c.scenario.lme.as_mut().unwrap().icc_noise_sd = -0.05;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::LmePerturbationOutOfRange {
                    field: "icc_noise_sd",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv22_lme_icc_noise_sd_zero_passes() {
        // 0.0 = no ICC perturbation (valid for a control arm).
        let mut c = lme_contract();
        c.scenario.lme.as_mut().unwrap().icc_noise_sd = 0.0;
        assert!(c.validate().is_ok());
    }

    // ---- invariant_21 (extended): tau_squared intercept-only ----------------

    #[test]
    fn inv21_tau_squared_nan_rejects_on_intercept_only() {
        let c = clustered_contract(f64::NAN);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::InterceptOnlyTauSquaredInvalid { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv21_tau_squared_negative_rejects_on_intercept_only() {
        let c = clustered_contract(-0.1);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::InterceptOnlyTauSquaredInvalid { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv21_tau_squared_zero_passes_on_intercept_only() {
        // ICC = 0 → tau_squared = 0.0 is a valid control-arm value.
        let c = clustered_contract(0.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inv21_tau_squared_positive_passes_on_intercept_only() {
        let c = clustered_contract(0.25);
        assert!(c.validate().is_ok());
    }

    // ---- invariant_23: SyntheticKind::Binary { p } --------------------------

    #[test]
    fn inv23_binary_p_nan_rejects() {
        let c = binary_contract(f64::NAN);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::BinaryProbabilityOutOfRange { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv23_binary_p_above_one_rejects() {
        let c = binary_contract(1.1);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::BinaryProbabilityOutOfRange { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv23_binary_p_negative_rejects() {
        let c = binary_contract(-0.1);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::BinaryProbabilityOutOfRange { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv23_binary_p_zero_passes() {
        // p = 0.0 is a degenerate but structurally valid marginal (closed interval).
        let c = binary_contract(0.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inv23_binary_p_one_passes() {
        // p = 1.0 likewise.
        let c = binary_contract(1.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inv23_binary_p_mid_passes() {
        let c = binary_contract(0.3);
        assert!(c.validate().is_ok());
    }

    // ---- invariant_23: ResampledBinary { proportion } -----------------------

    #[test]
    fn inv23_resampled_binary_proportion_nan_rejects() {
        let c = resampled_binary_contract(f64::NAN);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ResampledBinaryProportionOutOfRange { .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv23_resampled_binary_proportion_above_one_rejects() {
        let c = resampled_binary_contract(1.5);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(
                err,
                ContractError::ResampledBinaryProportionOutOfRange { .. }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn inv23_resampled_binary_proportion_zero_passes() {
        // All-zero uploaded column → proportion = 0.0; must not be rejected.
        let c = resampled_binary_contract(0.0);
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inv23_resampled_binary_proportion_one_passes() {
        let c = resampled_binary_contract(1.0);
        assert!(c.validate().is_ok());
    }

    // ---- invariant_24: probit link only on Binary ---------------------------

    #[test]
    fn inv24_probit_on_continuous_rejects() {
        let mut c = example1_simple_ols();
        // example1 is Continuous; force a probit link override.
        c.outcome.link = Some(crate::outcome::LinkKind::Probit);
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::ProbitRequiresBinary),
            "got {err:?}"
        );
    }

    #[test]
    fn inv24_probit_on_binary_passes() {
        use crate::estimator::EstimatorSpec;
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Glm;
        c.outcome.kind = OutcomeKind::Binary;
        c.outcome.link = Some(crate::outcome::LinkKind::Probit);
        assert!(c.validate().is_ok(), "{:?}", c.validate());
    }

    // ---- invariant_25: nagq backstop ----------------------------------------

    /// Eligible AGQ base: a clustered Binary GLMM with a single grouping factor
    /// and an intercept-only random-effect structure (1 RE ≤ 3).
    fn eligible_glmm() -> SimulationContract {
        use crate::estimator::EstimatorSpec;
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Mle;
        c.outcome.kind = OutcomeKind::Binary;
        c.generation.cluster = Some(ClusterSpec::intercept_only(
            ClusterSizing::FixedClusters { n_clusters: 10 },
            0.25,
        ));
        c
    }

    #[test]
    fn inv25_nagq_three_eligible_passes() {
        let mut c = eligible_glmm();
        c.nagq = 3;
        assert!(c.validate().is_ok(), "{:?}", c.validate());
    }

    #[test]
    fn inv25_nagq_upper_bound_25_eligible_passes() {
        // 25 is the top of the odd `1..=25` range; an off-by-one such as
        // `k >= 25` would wrongly reject the boundary value itself.
        let mut c = eligible_glmm();
        c.nagq = 25;
        assert!(c.validate().is_ok(), "{:?}", c.validate());
    }

    #[test]
    fn inv25_nagq_three_res_eligible_passes() {
        // Intercept + 2 slopes = 3 REs, the max allowed; an off-by-one such
        // as `re_count < 3` would wrongly reject this shape.
        use crate::generation::SlopeTerm;
        use crate::ids::ColumnId;
        let mut c = eligible_glmm();
        let cluster = c.generation.cluster.as_mut().unwrap();
        for (i, col) in [0u32, 1].into_iter().enumerate() {
            cluster.slopes.push(SlopeTerm {
                column: ColumnId(col),
                variance: 0.2,
                corr_with_intercept: 0.0,
                corr_with: vec![0.0; i],
            });
        }
        c.nagq = 3;
        assert!(c.validate().is_ok(), "{:?}", c.validate());
    }

    #[test]
    fn inv25_nagq_one_on_ineligible_shape_passes() {
        // nagq = 1 is the Laplace default and always allowed, even on an
        // ineligible (OLS/Continuous) shape.
        let mut c = example1_simple_ols();
        c.nagq = 1;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn inv25_nagq_even_rejects() {
        let mut c = eligible_glmm();
        c.nagq = 4;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::NagqOutOfRange { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv25_nagq_above_range_rejects() {
        let mut c = eligible_glmm();
        c.nagq = 27;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::NagqOutOfRange { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv25_nagq_on_ols_continuous_rejects() {
        let mut c = example1_simple_ols();
        c.nagq = 3;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::NagqIneligibleShape { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv25_nagq_on_unclustered_glm_rejects() {
        use crate::estimator::EstimatorSpec;
        let mut c = example1_simple_ols();
        c.estimator = EstimatorSpec::Glm;
        c.outcome.kind = OutcomeKind::Binary;
        // No cluster → ineligible shape for nagq > 1.
        c.nagq = 3;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::NagqIneligibleShape { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv25_nagq_with_second_grouping_rejects() {
        use crate::generation::{GroupingRelation, GroupingSpec};
        let mut c = eligible_glmm();
        c.generation
            .cluster
            .as_mut()
            .unwrap()
            .extra_groupings
            .push(GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 8 },
                tau_squared: 0.1,
                slopes: vec![],
            });
        c.nagq = 3;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::NagqIneligibleShape { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn inv25_nagq_with_too_many_res_rejects() {
        use crate::generation::SlopeTerm;
        use crate::ids::ColumnId;
        let mut c = eligible_glmm();
        // intercept + 3 slopes = 4 REs > 3. Each slope's `corr_with` lists the
        // correlations with the earlier slopes, so lengths grow 0, 1, 2.
        let cluster = c.generation.cluster.as_mut().unwrap();
        for (i, col) in [0u32, 1, 0].into_iter().enumerate() {
            cluster.slopes.push(SlopeTerm {
                column: ColumnId(col),
                variance: 0.2,
                corr_with_intercept: 0.0,
                corr_with: vec![0.0; i],
            });
        }
        c.nagq = 3;
        let err = c.validate().unwrap_err();
        assert!(
            matches!(err, ContractError::NagqIneligibleShape { .. }),
            "got {err:?}"
        );
    }
}
