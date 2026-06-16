//! Upload-data helpers: standardization and contract population.
//!
//! Standardization is single-sourced here so every host (Python, R, Tauri,
//! WASM) and every test uses the same numeric transformations. The contract
//! populator (`apply_upload`) mutates a `GenerationSpec` template in-place,
//! switching matched predictor columns from synthetic to frame-backed variants
//! and installing the standardized frame data. Correlation measurement
//! (`measure_correlations`) runs immediately after and overwrites the
//! correlation field with empirical Spearman rank correlation converted to the
//! latent-Gaussian scale, subject to explicit user pairs taking precedence.

use engine_contract::{ColumnId, ColumnSpec, Correlations, GenerationSpec, UploadedFrame};

use crate::error::SpecError;
use crate::input::{CorrelationPair, UploadColumn, UploadColumnType, UploadInput};
use crate::variables::PredictorTable;

/// Z-score a slice with population SD (ddof=0).
///
/// Returns an empty `Vec` for empty input and all-zeros when SD == 0.
/// Single-sourced here so every host (Python, R, Tauri, WASM) uses identical
/// numeric transformations — including `driver.rs`'s outcome z-scoring path.
pub fn standardize_continuous(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let variance: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let sd = variance.sqrt();
    if sd == 0.0 {
        return vec![0.0; n];
    }
    values.iter().map(|&x| (x - mean) / sd).collect()
}

/// The canonical effect-recovery design built from typed upload columns.
#[derive(Debug, Clone)]
pub struct RecoveryDesign {
    /// Column-major flat design: `[Intercept, non-factors, factor dummies, interactions]`.
    pub design_flat: Vec<f64>,
    /// Parallel index-keyed semantic names ("intercept", predictor names, "f[level]", interaction names).
    pub semantic_names: Vec<String>,
    /// Number of design columns (`design_flat.len() == nrow * ncol`).
    pub ncol: usize,
}

/// Build the canonical effect-recovery design from typed upload columns — the
/// single source for `get_effects_from_data` across Tauri, Python, and R.
///
/// Column order: `[Intercept, non-factors (standardized via [`standardize_column`]),
/// factor 0/1 dummies, interaction products]`, column-major flat, with a parallel
/// semantic-name list (e.g. "intercept", "cyl[6]").
///
/// `columns` are the typed uploaded columns: factor `values` are level codes
/// `0..k-1` with `labels[code]` the level name; non-factor `values` are raw
/// numerics. RULE-1 safe: label strings only decode which numeric code maps to
/// which dummy, never as model semantics. Callers should pre-validate column
/// presence for friendly host-side errors; the `Recovery*` errors here are the
/// defensive backstop.
///
/// # Errors
/// - [`SpecError::RecoveryColumnMissing`]: a modeled predictor (or interaction component) has no matching uploaded column.
/// - [`SpecError::RecoveryColumnLength`]: a matched column has a different row count than `nrow`.
pub fn build_recovery_design(
    table: &PredictorTable,
    columns: &[UploadColumn],
    nrow: usize,
) -> Result<RecoveryDesign, SpecError> {
    use std::collections::HashMap;

    let col_by_name: HashMap<&str, &UploadColumn> =
        columns.iter().map(|c| (c.name.as_str(), c)).collect();

    let require_col = |name: &str| -> Result<&UploadColumn, SpecError> {
        let col =
            col_by_name
                .get(name)
                .copied()
                .ok_or_else(|| SpecError::RecoveryColumnMissing {
                    name: name.to_string(),
                })?;
        if col.values.len() != nrow {
            return Err(SpecError::RecoveryColumnLength {
                name: name.to_string(),
                expected: nrow,
                got: col.values.len(),
            });
        }
        Ok(col)
    };

    // Standardized component columns keyed by semantic name (predictor names for
    // non-factors, "f[level]" for dummies). Built once, reused for interactions.
    let mut std_columns: HashMap<String, Vec<f64>> = HashMap::new();

    // Non-factor predictors: z-score (continuous) / centre (binary) / pass-through.
    for name in &table.non_factor_names {
        let col = require_col(name)?;
        std_columns.insert(name.clone(), standardize_column(col));
    }

    // Factor dummies: 0/1 indicator for each non-reference level.
    for (fi, factor_name) in table.factor_names.iter().enumerate() {
        let col = require_col(factor_name)?;
        let raw_vals = &col.values;
        for level in &table.factor_levels[fi] {
            let dummy_name = format!("{factor_name}[{level}]");
            let indicator: Vec<f64> = if !col.labels.is_empty() {
                // values are level codes; find the code for this level name.
                let code = col.labels.iter().position(|l| l == level).map(|i| i as f64);
                raw_vals
                    .iter()
                    .map(|&v| if Some(v) == code { 1.0 } else { 0.0 })
                    .collect()
            } else {
                raw_vals
                    .iter()
                    .map(|&v| if v.to_string() == *level { 1.0 } else { 0.0 })
                    .collect()
            };
            std_columns.insert(dummy_name, indicator);
        }
    }

    // Design column-major: [Intercept, non-factors, factor dummies, interactions].
    let mut semantic_names: Vec<String> = vec!["intercept".into()];
    let mut design_cols: Vec<Vec<f64>> = vec![vec![1.0_f64; nrow]];

    for name in &table.non_factor_names {
        semantic_names.push(name.clone());
        design_cols.push(std_columns[name.as_str()].clone());
    }
    for (fi, factor_name) in table.factor_names.iter().enumerate() {
        for level in &table.factor_levels[fi] {
            let dummy_name = format!("{factor_name}[{level}]");
            semantic_names.push(dummy_name.clone());
            design_cols.push(std_columns[&dummy_name].clone());
        }
    }
    // Interactions in effect order — elementwise product of component columns.
    for (eff_idx, eff_name) in table.effect_names.iter().enumerate() {
        let comps = &table.interaction_components[eff_idx];
        if comps.is_empty() {
            continue; // main effect already emitted above
        }
        let mut prod = vec![1.0_f64; nrow];
        for (pred_name, level_opt) in comps {
            let comp_name = match level_opt {
                Some(level) => format!("{pred_name}[{level}]"),
                None => pred_name.clone(),
            };
            let comp_col =
                std_columns
                    .get(&comp_name)
                    .ok_or_else(|| SpecError::RecoveryColumnMissing {
                        name: comp_name.clone(),
                    })?;
            for (r, &v) in comp_col.iter().enumerate() {
                prod[r] *= v;
            }
        }
        semantic_names.push(eff_name.clone());
        design_cols.push(prod);
    }

    let ncol = design_cols.len();
    let mut design_flat: Vec<f64> = Vec::with_capacity(nrow * ncol);
    for col in &design_cols {
        design_flat.extend_from_slice(col);
    }
    Ok(RecoveryDesign {
        design_flat,
        semantic_names,
        ncol,
    })
}

/// Standardize a single uploaded column.
///
/// - `Continuous`: z-score with population SD (ddof=0). If SD == 0, returns all zeros.
/// - `Binary`: center at empirical proportion; `x - p` where `p = mean(values)`.
/// - `Factor`: returns values unchanged (level codes 0..k-1 need no standardization).
pub fn standardize_column(col: &UploadColumn) -> Vec<f64> {
    match col.col_type {
        UploadColumnType::Continuous => standardize_continuous(&col.values),
        UploadColumnType::Binary => {
            let n = col.values.len();
            if n == 0 {
                return Vec::new();
            }
            let p: f64 = col.values.iter().sum::<f64>() / n as f64;
            col.values.iter().map(|&x| x - p).collect()
        }
        UploadColumnType::Factor => col.values.clone(),
    }
}

/// Compute empirical proportion (mean of 0/1 values) for a binary column.
fn binary_proportion(col: &UploadColumn) -> f64 {
    if col.values.is_empty() {
        return 0.5;
    }
    col.values.iter().sum::<f64>() / col.values.len() as f64
}

/// Compute level proportions for a factor column.
/// Returns a `Vec<f64>` of length `n_levels` (= `labels.len()`), where each
/// entry is the fraction of rows with that level code.
fn level_proportions(col: &UploadColumn) -> Vec<f64> {
    let n_levels = col.labels.len();
    if n_levels == 0 || col.values.is_empty() {
        return vec![];
    }
    let n = col.values.len() as f64;
    let mut counts = vec![0u32; n_levels];
    for &v in &col.values {
        let code = v as usize;
        if code < n_levels {
            counts[code] += 1;
        }
    }
    counts.iter().map(|&c| c as f64 / n).collect()
}

/// Compute Pearson r over two already-standardized vectors (mean ≈ 0).
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let sum_ab: f64 = a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum();
    let sum_a2: f64 = a.iter().map(|ai| ai * ai).sum();
    let sum_b2: f64 = b.iter().map(|bi| bi * bi).sum();
    let denom = (sum_a2 * sum_b2).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        sum_ab / denom
    }
}

/// Fractional (average-rank) ranks of `x`. Each tie group shares the mean of the
/// 1-based ranks it spans, so the downstream Pearson-on-ranks is the tie-correct
/// Spearman — the `1 − 6Σd²/(n(n²−1))` shortcut is wrong under ties. Ranks of a
/// z-scored column equal ranks of the raw column (standardization is monotone),
/// and equal raw values stay bit-equal after z-scoring, so exact-equality tie
/// detection is correct here.
fn rank_avg(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    while i < n {
        // Span the tie group [i, j) of equal values, then share its mean rank.
        let mut j = i + 1;
        while j < n && x[order[j]] == x[order[i]] {
            j += 1;
        }
        // Mean of the 1-based ranks (i+1)..=j over the group.
        let avg = ((i + 1 + j) as f64) / 2.0;
        for &k in &order[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// Measure Spearman rank correlation ρ_S of two columns and convert it to the
/// latent-Gaussian Pearson that reproduces it through a Gaussian copula:
/// `ρ_z = 2·sin(π·ρ_S/6)`. With that ρ_z installed, the generated data's
/// population Spearman is exactly ρ_S — the rank dependence this upload path
/// exists to reproduce.
///
/// Ranks are centered before `pearson` because `pearson` assumes mean-zero
/// inputs and ranks have mean `(n+1)/2`. A zero-variance (constant) column has
/// all-equal ranks → centered to zero → `pearson` denom 0 → ρ_S = 0 (latent 0),
/// matching the prior zero-variance behavior.
fn spearman_to_latent(a: &[f64], b: &[f64]) -> f64 {
    let ra = rank_avg(a);
    let rb = rank_avg(b);
    let mean_a = ra.iter().sum::<f64>() / ra.len() as f64;
    let mean_b = rb.iter().sum::<f64>() / rb.len() as f64;
    let ca: Vec<f64> = ra.iter().map(|&r| r - mean_a).collect();
    let cb: Vec<f64> = rb.iter().map(|&r| r - mean_b).collect();
    let rho_s = pearson(&ca, &cb);
    2.0 * (std::f64::consts::PI * rho_s / 6.0).sin()
}

/// Smallest symmetric ridge-shrink toward identity that restores PSD: returns
/// `(1−ε)·R + ε·I` for the smallest ε ∈ [0, 1] passing [`crate::correlation::is_psd`],
/// found by bisection; returns `R` unchanged when it is already PSD (ε = 0). The
/// unit diagonal is preserved (`(1−ε)·1 + ε·1 = 1`), so the result stays a valid
/// correlation matrix.
///
/// PSD-ness is monotone in ε — every eigenvalue maps to `(1−ε)λ + ε`, which rises
/// with ε for λ < 1 (and a unit-diagonal matrix always has λ_min ≤ 1), with ε = 1
/// giving the identity — so a single threshold separates non-PSD from PSD and
/// bisection finds it. 2×2 measured blocks never enter the bisection branch:
/// `|2·sin(π·ρ_S/6)| ≤ 1`, so they are PSD by construction.
fn ridge_shrink_to_psd(r: &[f64], n: usize) -> Vec<f64> {
    if crate::correlation::is_psd(r, n) {
        return r.to_vec();
    }
    let shrink = |eps: f64| -> Vec<f64> {
        let mut m: Vec<f64> = r.iter().map(|&v| (1.0 - eps) * v).collect();
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        m
    };
    let mut lo = 0.0_f64; // known non-PSD (R itself, this branch)
    let mut hi = 1.0_f64; // identity, PSD
    while hi - lo > 1e-9 {
        let mid = 0.5 * (lo + hi);
        if crate::correlation::is_psd(&shrink(mid), n) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    shrink(hi)
}

/// Apply the upload data to a `GenerationSpec` template.
///
/// For each predictor name in `predictor_names` (non-factors first, then
/// factors — matching contract column order), if a same-named column exists
/// in `upload.columns`, the contract `ColumnSpec` is replaced with a
/// frame-backed variant:
/// - `Continuous` → `ColumnSpec::Resampled { frame_column }`
/// - `Binary`     → `ColumnSpec::ResampledBinary { frame_column, proportion }`
/// - `Factor`     → `ColumnSpec::FactorFromFrame { frame_column, n_levels, proportions }`
///
/// Only matched columns are stored in the frame (row-major, standardized).
/// Unmatched predictor columns keep their synthetic spec untouched.
/// Extra upload columns (not in the model) are ignored.
///
/// `predictor_names` must be `non_factor_names` concatenated with `factor_names`
/// from the `PredictorTable` — this matches the contract column index layout.
///
/// Returns a parallel `Vec<Option<Vec<f64>>>` over `predictor_names[0..n_non_factor]`:
/// `Some(std_values)` for each matched *continuous* non-factor column, `None`
/// otherwise (binary, factor, or unmatched). Correlation is continuous-only by
/// design, so binary columns are excluded here. Passed directly to
/// `measure_correlations` so standardization is computed once.
pub fn apply_upload(
    gen: &mut GenerationSpec,
    predictor_names: &[String],
    n_non_factor: usize,
    upload: &UploadInput,
) -> Vec<Option<Vec<f64>>> {
    let n_rows = upload.n_rows as usize;
    // Build a name → upload column lookup
    let col_by_name: std::collections::HashMap<&str, &UploadColumn> = upload
        .columns
        .iter()
        .map(|c| (c.name.as_str(), c))
        .collect();

    // Standardized data for matched columns, in contract column order.
    // We collect (frame_col_index, standardized_values) pairs.
    let mut frame_cols: Vec<Vec<f64>> = Vec::new();
    let mut frame_col_cursor: u32 = 0;

    // Parallel vec over predictor_names[0..n_non_factor]: standardized values
    // for matched continuous/binary columns; None for unmatched or factor.
    let mut non_factor_std: Vec<Option<Vec<f64>>> = vec![None; n_non_factor];

    for (contract_col_idx, pred_name) in predictor_names.iter().enumerate() {
        if let Some(uc) = col_by_name.get(pred_name.as_str()) {
            let fc = frame_col_cursor;
            let std_vals = standardize_column(uc);
            // Capture standardized values for matched *continuous* non-factor
            // columns only. Correlation is continuous-only by design — binary and
            // factor columns are generated from their marginals (independent); their
            // joint dependence is preserved only by strict-mode resampling, not the
            // partial-mode copula path.
            if contract_col_idx < n_non_factor {
                match uc.col_type {
                    UploadColumnType::Continuous => {
                        non_factor_std[contract_col_idx] = Some(std_vals.clone());
                    }
                    UploadColumnType::Binary | UploadColumnType::Factor => {}
                }
            }
            frame_cols.push(std_vals);
            // Replace the contract column spec
            match uc.col_type {
                UploadColumnType::Continuous => {
                    gen.columns[contract_col_idx] = ColumnSpec::Resampled { frame_column: fc };
                }
                UploadColumnType::Binary => {
                    let proportion = binary_proportion(uc);
                    gen.columns[contract_col_idx] = ColumnSpec::ResampledBinary {
                        frame_column: fc,
                        proportion,
                    };
                }
                UploadColumnType::Factor => {
                    let props = level_proportions(uc);
                    let n_levels = props.len() as u32;
                    // Preserve the per-factor override set on the synthetic column
                    // this rewrite replaces (set_variable_type runs before upload).
                    let sampled_proportions = match &gen.columns[contract_col_idx] {
                        ColumnSpec::FactorSynthetic {
                            sampled_proportions,
                            ..
                        } => *sampled_proportions,
                        _ => None,
                    };
                    gen.columns[contract_col_idx] = ColumnSpec::FactorFromFrame {
                        frame_column: fc,
                        n_levels,
                        proportions: props,
                        sampled_proportions,
                    };
                }
            }
            frame_col_cursor += 1;
        }
    }

    if !frame_cols.is_empty() {
        // Build row-major frame data: n_rows × n_matched.
        let n_matched = frame_cols.len();
        let mut data = vec![0.0_f64; n_rows * n_matched];
        for (col_idx, col_vals) in frame_cols.iter().enumerate() {
            for (row_idx, &val) in col_vals.iter().enumerate().take(n_rows) {
                data[row_idx * n_matched + col_idx] = val;
            }
        }
        gen.uploaded_frame = Some(UploadedFrame {
            data,
            n_rows: upload.n_rows,
            n_cols: n_matched as u32,
            // Strict (bootstrap) signal: the kernel row-samples whole frame rows
            // when set. Only `UploadMode::Strict` carries it; none/partial use NORTA.
            bootstrap: matches!(upload.mode, crate::input::UploadMode::Strict),
        });
    }

    non_factor_std
}

/// Measure empirical **Spearman** rank correlations among matched continuous
/// columns, convert each to the latent-Gaussian scale (`2·sin(π·ρ_S/6)`), and
/// install as the contract correlation matrix, then overlay explicit user pairs
/// (user wins). Partial upload regenerates each continuous column through its
/// empirical margin — a monotone transform that preserves only rank dependence —
/// so reproducing the uploaded **rank** correlation is the faithful target;
/// realized Pearson stays approximate (whatever the Gaussian copula implies).
///
/// Explicit user pairs are installed **raw** (latent-Pearson) and are *not*
/// converted — declared correlations keep latent-Pearson semantics; only the
/// upload-measured entries go through the rank→latent conversion.
///
/// Called after `apply_upload` for `UploadMode::Partial`.
/// `non_factor_names` is the slice of non-factor predictor names (positions 0..n).
/// `spec_correlations` are the user-explicit pairs (binary-named pairs are
/// already rejected upstream in `build_contract_correlations`).
/// `non_factor_std` is the parallel vec returned by `apply_upload`:
/// `Some(std_values)` for matched continuous columns, `None` otherwise (binary,
/// factor, or unmatched) — so binary/factor columns get identity rows here.
///
/// The sin-inflation of the conversion can break PSD on a near-degenerate
/// measured block, so it is ridge-shrunk toward identity (smallest ε) before the
/// overlay. Returns `Err(SpecError::CorrelationNotPsd)` only when the *user
/// overlay* leaves the combined matrix non–positive-semi-definite.
pub fn measure_correlations(
    gen: &mut GenerationSpec,
    non_factor_names: &[String],
    spec_correlations: &[CorrelationPair],
    non_factor_std: &[Option<Vec<f64>>],
) -> Result<(), crate::error::SpecError> {
    let n = non_factor_names.len();
    if n == 0 {
        return Ok(());
    }

    // Build identity + measure Pearson r for matched pairs.
    let mut values = vec![0.0_f64; n * n];
    for i in 0..n {
        values[i * n + i] = 1.0;
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if let (Some(ci), Some(cj)) = (&non_factor_std[i], &non_factor_std[j]) {
                let r = spearman_to_latent(ci, cj);
                // column-major: values[j * n + i] = values[i * n + j] = r
                values[j * n + i] = r;
                values[i * n + j] = r;
            }
        }
    }

    // The Spearman→latent map slightly inflates each entry, which can tip a
    // near-degenerate measured block (3+ strongly-correlated columns) to
    // indefinite — the NORTA defect. Repair it with the smallest ridge-shrink
    // toward identity *before* the user overlay, so the explicit pairs below land
    // on a PSD measured block. (No-op in the common case: already-PSD matrices,
    // and every 2×2, return unchanged.)
    let mut values = ridge_shrink_to_psd(&values, n);

    // Build name → non-factor index map for user-pair overlay.
    let name_to_idx: std::collections::HashMap<&str, usize> = non_factor_names
        .iter()
        .enumerate()
        .map(|(i, name)| (name.as_str(), i))
        .collect();

    // Overlay explicit user pairs (user wins).
    for pair in spec_correlations {
        if let (Some(&i), Some(&j)) = (
            name_to_idx.get(pair.a.as_str()),
            name_to_idx.get(pair.b.as_str()),
        ) {
            values[j * n + i] = pair.value;
            values[i * n + j] = pair.value;
        }
    }

    // Re-check PSD on the combined matrix (measured + user overlay). The measured
    // block was just ridge-shrunk to PSD, but an explicit user-pair overlay can
    // still break it — catch that here and fail loud. (The kernel additionally
    // spectral-clips every draw, so a slip here would not panic, only silently
    // de-calibrate; the up-front check keeps user-induced non-PSD explicit.)
    if !crate::correlation::is_psd(&values, n) {
        return Err(crate::error::SpecError::CorrelationNotPsd);
    }

    // Non-factor columns occupy positions 0..n_non_factor in generation.columns
    // (factors are appended after), matching what build_contract_correlations emits.
    let continuous_columns: Vec<ColumnId> = (0..n as u32).map(ColumnId).collect();
    gen.correlations = Correlations::Matrix {
        continuous_columns,
        values,
    };
    Ok(())
}

#[cfg(test)]
mod recovery_tests {
    use super::*;
    use crate::formula::parse_formula;
    use crate::input::{PredictorSpec, VarKind};
    use crate::variables::build_predictor_table;

    /// Golden: intercept + one z-scored continuous + one 2-level factor dummy,
    /// column-major, with the canonical semantic-name order. Pins the assembly
    /// the Tauri/py/R ports all bind to.
    #[test]
    fn recovery_design_intercept_continuous_factor() {
        let parsed = parse_formula("y ~ x + g").unwrap();
        let predictors = vec![
            PredictorSpec {
                name: "x".into(),
                pinned: false,
                kind: VarKind::Normal,
            },
            PredictorSpec {
                name: "g".into(),
                pinned: false,
                kind: VarKind::Factor {
                    levels: vec!["a".into(), "b".into()],
                    proportions: vec![0.5, 0.5],
                    reference: "a".into(),
                    sampled_proportions: None,
                },
            },
        ];
        let table = build_predictor_table(&parsed, &predictors).unwrap();
        let columns = vec![
            UploadColumn {
                name: "x".into(),
                col_type: UploadColumnType::Continuous,
                values: vec![1.0, 2.0, 3.0],
                labels: vec![],
            },
            UploadColumn {
                name: "g".into(),
                col_type: UploadColumnType::Factor,
                values: vec![0.0, 1.0, 0.0], // codes: a=0, b=1
                labels: vec!["a".into(), "b".into()],
            },
        ];
        let rd = build_recovery_design(&table, &columns, 3).unwrap();
        assert_eq!(rd.semantic_names, vec!["intercept", "x", "g[b]"]);
        assert_eq!(rd.ncol, 3);
        // Column-major: intercept | z-scored x | g[b] dummy.
        assert_eq!(&rd.design_flat[0..3], &[1.0, 1.0, 1.0]);
        let z = 1.0_f64 / (2.0_f64 / 3.0).sqrt(); // |z-score| of {1,2,3}
        assert!((rd.design_flat[3] + z).abs() < 1e-12);
        assert!(rd.design_flat[4].abs() < 1e-12);
        assert!((rd.design_flat[5] - z).abs() < 1e-12);
        assert_eq!(&rd.design_flat[6..9], &[0.0, 1.0, 0.0]);
    }

    #[test]
    fn recovery_design_missing_column_errors() {
        let parsed = parse_formula("y ~ x").unwrap();
        let predictors = vec![PredictorSpec {
            name: "x".into(),
            pinned: false,
            kind: VarKind::Normal,
        }];
        let table = build_predictor_table(&parsed, &predictors).unwrap();
        let err = build_recovery_design(&table, &[], 3).unwrap_err();
        assert!(matches!(err, SpecError::RecoveryColumnMissing { ref name } if name == "x"));
    }
}

#[cfg(test)]
mod correlation_tests {
    use super::*;

    /// Tied values share the mean of the ranks they span (fractional ranking),
    /// and the leading sort means unordered input is ranked correctly.
    #[test]
    fn rank_avg_shares_mean_rank_across_ties() {
        // x = [20, 30, 10, 20]: 10→1, the two 20s tie at 1-based ranks 2,3 → 2.5,
        // 30→4. Indexed back to original positions: [2.5, 4.0, 1.0, 2.5].
        let ranks = rank_avg(&[20.0, 30.0, 10.0, 20.0]);
        assert_eq!(ranks, vec![2.5, 4.0, 1.0, 2.5]);
    }

    /// ρ_S is measured on ranks and returned on the latent scale `2·sin(π·ρ_S/6)`.
    /// a=[1..5], b=[1,2,3,5,4] → ρ_S = 0.9; latent = 0.9079809994790935 ≠ 0.9.
    #[test]
    fn spearman_to_latent_applies_norta_inversion() {
        let lat = spearman_to_latent(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 2.0, 3.0, 5.0, 4.0]);
        let expected = 2.0 * (std::f64::consts::PI * 0.9 / 6.0).sin();
        assert!((lat - expected).abs() < 1e-12, "got {lat}, want {expected}");
    }

    /// A constant (zero-variance) column ranks to all-equal → centered to zero →
    /// `pearson` denom 0 → ρ_S = 0 (latent 0). Must not panic or NaN.
    #[test]
    fn spearman_to_latent_zero_variance_is_zero() {
        let lat = spearman_to_latent(&[2.0, 2.0, 2.0], &[1.0, 2.0, 3.0]);
        assert!(lat.abs() < 1e-12, "got {lat}");
    }

    /// An already-PSD matrix is returned unchanged (ε = 0) — no de-calibration in
    /// the common case.
    #[test]
    fn ridge_shrink_leaves_psd_unchanged() {
        let r = vec![1.0, 0.5, 0.5, 1.0]; // 2×2, PSD
        assert_eq!(ridge_shrink_to_psd(&r, 2), r);
    }

    /// An indefinite measured block is repaired: the result is PSD, keeps a unit
    /// diagonal, and has off-diagonals shrunk toward zero (smaller magnitude).
    #[test]
    fn ridge_shrink_repairs_indefinite() {
        // det = -2.888 → indefinite.
        #[rustfmt::skip]
        let r = vec![
            1.0,  0.9, -0.9,
            0.9,  1.0,  0.9,
           -0.9,  0.9,  1.0,
        ];
        assert!(!crate::correlation::is_psd(&r, 3), "fixture must start indefinite");
        let repaired = ridge_shrink_to_psd(&r, 3);
        assert!(crate::correlation::is_psd(&repaired, 3), "repaired must be PSD");
        // Unit diagonal preserved.
        for i in 0..3 {
            assert!((repaired[i * 3 + i] - 1.0).abs() < 1e-12);
        }
        // Off-diagonals shrunk toward zero.
        assert!(repaired[1].abs() < 0.9 && repaired[1].abs() > 0.0);
        assert!((repaired[2].abs()) < 0.9);
    }
}
