//! Counter-pooling merge for `single_core_*` results.
//!
//! Hosts running multiple `single_core_find_power` workers (WASM Web Workers,
//! Python multiprocessing) pass the per-worker `ScenarioResult<PowerResult>`
//! slices to `merge_power_results`. The result is statistically equivalent to
//! `find_power(contract, sum(n_sims))` within MC noise — bit-equality is
//! intentionally not guaranteed because workers walk independent RNG paths.

use crate::aggregation::wilson_ci;
use crate::grid::derive_sample_size_outputs;
use crate::result::{
    ByValue, Ci, EstimatorExtras, OrchestratorError, PosthocPower, PowerResult, SampleSizeMethod,
    SampleSizeResult, ScenarioResult,
};

/// Pool per-worker `PowerResult`s into a single result equivalent to running
/// `find_power(contract, sum(n_sims))`. All parts must share the same
/// scenario labels, sample size `n`, target_indices, and estimator.
///
/// Re-derived fields (`power_*`, `ci_*`, `convergence_rate`, estimator-extra
/// rates and means) are recomputed from the pooled counters. `boundary_hit`
/// vectors are concatenated.
///
/// Counter-pool mirror of `aggregate_batch` (aggregation.rs): every raw
/// counter that fold emits must be pooled here — change together.
pub fn merge_power_results(
    parts: &[ScenarioResult<PowerResult>],
) -> Result<ScenarioResult<PowerResult>, OrchestratorError> {
    if parts.is_empty() {
        return Err(OrchestratorError::IncompatibleMerge(
            "merge_power_results: parts is empty".into(),
        ));
    }
    let n_scenarios = parts[0].scenarios.len();
    for (i, p) in parts.iter().enumerate() {
        if p.scenarios.len() != n_scenarios {
            return Err(OrchestratorError::IncompatibleMerge(format!(
                "part {i} has {} scenarios, expected {}",
                p.scenarios.len(),
                n_scenarios
            )));
        }
    }

    let mut merged_scenarios = Vec::with_capacity(n_scenarios);
    for s_idx in 0..n_scenarios {
        let label = &parts[0].scenarios[s_idx].0;
        for (i, p) in parts.iter().enumerate() {
            if &p.scenarios[s_idx].0 != label {
                return Err(OrchestratorError::IncompatibleMerge(format!(
                    "scenario label mismatch at idx {s_idx}, part {i}: {:?} vs {label:?}",
                    p.scenarios[s_idx].0
                )));
            }
        }
        let prs: Vec<&PowerResult> = parts.iter().map(|p| &p.scenarios[s_idx].1).collect();
        let merged_pr = merge_one_power_result(&prs)?;
        merged_scenarios.push((label.clone(), merged_pr));
    }
    Ok(ScenarioResult {
        scenarios: merged_scenarios,
    })
}

fn merge_one_power_result(prs: &[&PowerResult]) -> Result<PowerResult, OrchestratorError> {
    let head = prs[0];
    // Entry count of the per-target vectors: marginals + contrast pairs.
    // Pool over the counts' own length — `target_indices.len()` alone would
    // silently drop the contrast entries.
    let n_targets = head.target_indices.len() + head.contrast_pairs.len();
    // Guard the head's own counter lengths against n_targets up front. The
    // per-part loop below only checks each part agrees with `head`, so a head
    // whose success_counts are shorter than n_targets (e.g. built without the
    // raw counters) would pass every part check and then panic indexing
    // `[t]` in the pooling loop — return the clean error instead.
    if head.success_counts_uncorrected.len() != n_targets
        || head.success_counts_corrected.len() != n_targets
    {
        return Err(OrchestratorError::IncompatibleMerge(
            "success_counts length does not match target_indices + contrast_pairs — \
             part may have been built without raw counters"
                .into(),
        ));
    }
    // Joint-significance histogram length: targets + post-hoc contrasts + 1. A real
    // aggregated part always carries this length; an empty (len-0) histogram is a part
    // built without the per-sim counters. All parts must agree — mixing an empty and a
    // populated histogram silently breaks the bucket-sum invariant (Σ buckets == n_sims),
    // so reject the mismatch like the sibling success_counts vectors (an all-empty merge
    // stays empty).
    let hist_len = head.success_count_histogram_uncorrected.len();
    for p in &prs[1..] {
        if p.n != head.n {
            return Err(OrchestratorError::IncompatibleMerge(format!(
                "n mismatch: {} vs {}",
                p.n, head.n
            )));
        }
        if p.target_indices != head.target_indices {
            return Err(OrchestratorError::IncompatibleMerge(
                "target_indices mismatch".into(),
            ));
        }
        if p.contrast_pairs != head.contrast_pairs {
            return Err(OrchestratorError::IncompatibleMerge(
                "contrast_pairs mismatch".into(),
            ));
        }
        if std::mem::discriminant(&p.estimator_extras)
            != std::mem::discriminant(&head.estimator_extras)
        {
            return Err(OrchestratorError::IncompatibleMerge(
                "estimator_extras variant mismatch".into(),
            ));
        }
        if p.success_counts_uncorrected.len() != head.success_counts_uncorrected.len() {
            return Err(OrchestratorError::IncompatibleMerge(
                "success_counts_uncorrected length mismatch — \
                 part may have been built without raw counters"
                    .into(),
            ));
        }
        if p.success_counts_corrected.len() != head.success_counts_corrected.len() {
            return Err(OrchestratorError::IncompatibleMerge(
                "success_counts_corrected length mismatch".into(),
            ));
        }
        if p.factor_exclusion_counts.len() != head.factor_exclusion_counts.len() {
            return Err(OrchestratorError::IncompatibleMerge(
                "factor_exclusion_counts length mismatch across parts".into(),
            ));
        }
        if p.factor_separation_counts.len() != head.factor_separation_counts.len() {
            return Err(OrchestratorError::IncompatibleMerge(
                "factor_separation_counts length mismatch across parts".into(),
            ));
        }
        if p.success_count_histogram_uncorrected.len() != hist_len
            || p.success_count_histogram_corrected.len() != hist_len
        {
            return Err(OrchestratorError::IncompatibleMerge(
                "success_count_histogram length mismatch across parts — a part was \
                 built without the per-sim joint-significance counters"
                    .into(),
            ));
        }
        if p.overall_significant_rate.is_some() != head.overall_significant_rate.is_some() {
            return Err(OrchestratorError::IncompatibleMerge(
                "overall_significant_rate presence mismatch across parts".into(),
            ));
        }
    }

    let n_factors = head.factor_exclusion_counts.len();
    let mut pooled_unc = vec![0u64; n_targets];
    let mut pooled_cor = vec![0u64; n_targets];
    let mut pooled_conv = 0u64;
    let mut pooled_overall = 0u64;
    let mut pooled_n_sims = 0u64;
    let mut pooled_boundary: Vec<u8> = Vec::new();
    let mut pooled_hist_unc = vec![0u64; hist_len];
    let mut pooled_hist_cor = vec![0u64; hist_len];
    let mut pooled_excl = vec![0u64; n_factors];
    let mut pooled_sep = vec![0u64; n_factors];

    for p in prs {
        for t in 0..n_targets {
            pooled_unc[t] += p.success_counts_uncorrected[t];
            pooled_cor[t] += p.success_counts_corrected[t];
        }
        pooled_conv += p.convergence_count;
        pooled_overall += p.overall_significant_count;
        pooled_n_sims += p.n_sims;
        pooled_boundary.extend_from_slice(&p.boundary_hit);
        for b in 0..hist_len {
            pooled_hist_unc[b] += p.success_count_histogram_uncorrected[b];
            pooled_hist_cor[b] += p.success_count_histogram_corrected[b];
        }
        for f in 0..n_factors {
            pooled_excl[f] += p.factor_exclusion_counts[f];
            pooled_sep[f] += p.factor_separation_counts[f];
        }
    }

    if pooled_n_sims == 0 {
        return Err(OrchestratorError::IncompatibleMerge(
            "merged n_sims is zero — all parts carry n_sims=0".into(),
        ));
    }
    let denom = pooled_n_sims as f64;
    let power_unc: Vec<f64> = pooled_unc.iter().map(|&k| k as f64 / denom).collect();
    let power_cor: Vec<f64> = pooled_cor.iter().map(|&k| k as f64 / denom).collect();
    let ci_unc: Vec<Ci> = pooled_unc
        .iter()
        .map(|&k| wilson_ci(k, pooled_n_sims))
        .collect();
    let ci_cor: Vec<Ci> = pooled_cor
        .iter()
        .map(|&k| wilson_ci(k, pooled_n_sims))
        .collect();

    let merged_extras = merge_estimator_extras(prs, pooled_n_sims)?;

    // Posthoc merge: pool block-by-block when all parts share the same structure.
    // The `n_blocks == 0` case (no posthoc requested) yields empty; a block-count
    // or per-block contrast-count mismatch is rejected like every sibling field.
    let posthoc = merge_posthoc_blocks(prs, pooled_n_sims)?;

    Ok(PowerResult {
        n: head.n,
        n_sims: pooled_n_sims,
        target_indices: head.target_indices.clone(),
        contrast_pairs: head.contrast_pairs.clone(),
        power_uncorrected: power_unc,
        power_corrected: power_cor,
        ci_uncorrected: ci_unc,
        ci_corrected: ci_cor,
        convergence_rate: pooled_conv as f64 / denom,
        boundary_hit: pooled_boundary,
        estimator_extras: merged_extras,
        overall_significant_rate: head
            .overall_significant_rate
            .map(|_| pooled_overall as f64 / denom),
        success_counts_uncorrected: pooled_unc,
        success_counts_corrected: pooled_cor,
        convergence_count: pooled_conv,
        overall_significant_count: pooled_overall,
        overall_significant_ci: head
            .overall_significant_rate
            .map(|_| wilson_ci(pooled_overall, pooled_n_sims)),
        success_count_histogram_uncorrected: pooled_hist_unc,
        success_count_histogram_corrected: pooled_hist_cor,
        grid_warnings: head.grid_warnings.clone(),
        posthoc,
        factor_exclusion_counts: pooled_excl,
        factor_separation_counts: pooled_sep,
    })
}

/// Pool per-worker `SampleSizeResult`s into a single result. Grid is the only
/// sample-size method; per-N grid points pool exactly like `merge_power_results`.
pub fn merge_sample_size_results(
    parts: &[ScenarioResult<SampleSizeResult>],
) -> Result<ScenarioResult<SampleSizeResult>, OrchestratorError> {
    if parts.is_empty() {
        return Err(OrchestratorError::IncompatibleMerge(
            "merge_sample_size_results: parts is empty".into(),
        ));
    }
    let n_scenarios = parts[0].scenarios.len();
    for (i, p) in parts.iter().enumerate() {
        if p.scenarios.len() != n_scenarios {
            return Err(OrchestratorError::IncompatibleMerge(format!(
                "part {i} has {} scenarios, expected {}",
                p.scenarios.len(),
                n_scenarios
            )));
        }
    }

    let head_method = parts[0].scenarios.first().map(|(_, r)| r.method);

    let head_target_power = parts[0]
        .scenarios
        .first()
        .map(|(_, r)| r.target_power)
        .unwrap_or(0.0);
    let head_method = head_method.unwrap_or(SampleSizeMethod::Grid {
        by: ByValue::Fixed(1),
        mode: crate::result::GridMode::Linear,
    });
    // Parts come from one spec, so atoms agree by construction — asserted
    // below alongside the grid-mismatch check to pin it explicitly (the fit
    // recompute ceils n_achievable to this atom).
    let head_cluster_atom = parts[0]
        .scenarios
        .first()
        .map(|(_, r)| r.cluster_atom)
        .unwrap_or(1);

    let mut merged_scenarios = Vec::with_capacity(n_scenarios);
    for s_idx in 0..n_scenarios {
        let label = &parts[0].scenarios[s_idx].0;
        for (i, p) in parts.iter().enumerate() {
            if &p.scenarios[s_idx].0 != label {
                return Err(OrchestratorError::IncompatibleMerge(format!(
                    "scenario label mismatch at idx {s_idx}, part {i}: {:?} vs {label:?}",
                    p.scenarios[s_idx].0
                )));
            }
            let r = &p.scenarios[s_idx].1;
            if r.method != head_method {
                return Err(OrchestratorError::IncompatibleMerge(
                    "method mismatch across parts".into(),
                ));
            }
            if (r.target_power - head_target_power).abs() > 0.0 {
                return Err(OrchestratorError::IncompatibleMerge(format!(
                    "target_power mismatch: {} vs {}",
                    r.target_power, head_target_power
                )));
            }
            if r.cluster_atom != head_cluster_atom {
                return Err(OrchestratorError::IncompatibleMerge(format!(
                    "cluster_atom mismatch: {} vs {}",
                    r.cluster_atom, head_cluster_atom
                )));
            }
        }

        // Grid-aligned pooling: each part's `grid_or_trace[i]` corresponds to
        // the same N. Confirm the N grids match before pooling.
        let head_grid: Vec<usize> = parts[0].scenarios[s_idx]
            .1
            .grid_or_trace
            .iter()
            .map(|pr| pr.n)
            .collect();
        for (i, p) in parts.iter().enumerate() {
            let grid_i: Vec<usize> = p.scenarios[s_idx]
                .1
                .grid_or_trace
                .iter()
                .map(|pr| pr.n)
                .collect();
            if grid_i != head_grid {
                return Err(OrchestratorError::IncompatibleMerge(format!(
                    "grid mismatch at scenario {s_idx}, part {i}"
                )));
            }
        }

        // Pool per-N PowerResult.
        let n_points = head_grid.len();
        let mut pooled_per_n = Vec::with_capacity(n_points);
        for n_idx in 0..n_points {
            let prs: Vec<&PowerResult> = parts
                .iter()
                .map(|p| &p.scenarios[s_idx].1.grid_or_trace[n_idx])
                .collect();
            pooled_per_n.push(merge_one_power_result(&prs)?);
        }

        // Recompute all four derivations from the pooled counts via the
        // shared helper. Per-worker `fitted` values are computed-then-
        // discarded here, mirroring how `first_achieved` is recomputed —
        // the fit is deterministic over counts, so this is exact.
        let d = derive_sample_size_outputs(
            &pooled_per_n,
            &head_grid,
            head_target_power,
            head_cluster_atom,
        );

        let head_grid_warnings = parts[0].scenarios[s_idx].1.grid_warnings.clone();
        merged_scenarios.push((
            label.clone(),
            SampleSizeResult {
                grid_or_trace: pooled_per_n,
                first_achieved: d.first_achieved,
                first_joint_achieved: d.first_joint_achieved,
                fitted: d.fitted,
                fitted_joint: d.fitted_joint,
                first_overall_achieved: d.first_overall_achieved,
                fitted_overall: d.fitted_overall,
                cluster_atom: head_cluster_atom,
                target_power: head_target_power,
                method: head_method,
                grid_warnings: head_grid_warnings,
            },
        ));
    }
    Ok(ScenarioResult {
        scenarios: merged_scenarios,
    })
}

/// Pool posthoc blocks across parts. Returns `Ok(vec![])` only for the
/// legitimate "no posthoc requested" case (`n_blocks == 0`); a block-count or
/// per-block contrast-count mismatch is rejected with `IncompatibleMerge`, like
/// every sibling pooled-vector guard (mixing these would silently corrupt the
/// pooled counts on the WASM wire).
fn merge_posthoc_blocks(
    prs: &[&PowerResult],
    pooled_n_sims: u64,
) -> Result<Vec<PosthocPower>, OrchestratorError> {
    let n_blocks = prs[0].posthoc.len();
    if n_blocks == 0 {
        return Ok(Vec::new());
    }
    // All parts must have the same block count and per-block contrast count.
    for p in &prs[1..] {
        if p.posthoc.len() != n_blocks {
            return Err(OrchestratorError::IncompatibleMerge(
                "posthoc block count mismatch across parts".into(),
            ));
        }
        for b in 0..n_blocks {
            if p.posthoc[b].success_counts_uncorrected.len()
                != prs[0].posthoc[b].success_counts_uncorrected.len()
            {
                return Err(OrchestratorError::IncompatibleMerge(
                    "posthoc contrast count mismatch across parts".into(),
                ));
            }
        }
    }
    let denom = pooled_n_sims as f64;
    Ok((0..n_blocks)
        .map(|b| {
            let n_contrasts = prs[0].posthoc[b].success_counts_uncorrected.len();
            let mut pooled_unc = vec![0u64; n_contrasts];
            let mut pooled_cor = vec![0u64; n_contrasts];
            for p in prs {
                for c in 0..n_contrasts {
                    pooled_unc[c] += p.posthoc[b].success_counts_uncorrected[c];
                    pooled_cor[c] += p.posthoc[b].success_counts_corrected[c];
                }
            }
            let power_unc: Vec<f64> = pooled_unc.iter().map(|&k| k as f64 / denom).collect();
            let power_cor: Vec<f64> = pooled_cor.iter().map(|&k| k as f64 / denom).collect();
            let ci_unc: Vec<Ci> = pooled_unc
                .iter()
                .map(|&k| wilson_ci(k, pooled_n_sims))
                .collect();
            let ci_cor: Vec<Ci> = pooled_cor
                .iter()
                .map(|&k| wilson_ci(k, pooled_n_sims))
                .collect();
            PosthocPower {
                n_levels: prs[0].posthoc[b].n_levels,
                power_uncorrected: power_unc,
                power_corrected: power_cor,
                ci_uncorrected: ci_unc,
                ci_corrected: ci_cor,
                success_counts_uncorrected: pooled_unc,
                success_counts_corrected: pooled_cor,
            }
        })
        .collect())
}

fn merge_estimator_extras(
    prs: &[&PowerResult],
    pooled_n_sims: u64,
) -> Result<EstimatorExtras, OrchestratorError> {
    match &prs[0].estimator_extras {
        EstimatorExtras::Ols {} => Ok(EstimatorExtras::Ols {}),
        EstimatorExtras::Glm { .. } => {
            let mut sum = 0.0f64;
            let mut n = 0u64;
            let mut singular_count = 0u64;
            let mut singular_n = 0u64;
            let mut tau_sum = 0.0f64;
            let mut tau_n = 0u64;
            for p in prs {
                if let EstimatorExtras::Glm {
                    baseline_prob_sum,
                    baseline_prob_n,
                    singular_count: sc,
                    singular_n: sn,
                    tau_squared_hat_sum: ts,
                    tau_squared_hat_n: tn,
                    ..
                } = &p.estimator_extras
                {
                    sum += baseline_prob_sum;
                    n += baseline_prob_n;
                    singular_count += sc;
                    singular_n += sn;
                    tau_sum += ts;
                    tau_n += tn;
                }
            }
            Ok(EstimatorExtras::Glm {
                baseline_prob_realized: if n > 0 { sum / n as f64 } else { 0.0 },
                baseline_prob_sum: sum,
                baseline_prob_n: n,
                singular_fit_rate: if singular_n > 0 {
                    singular_count as f64 / singular_n as f64
                } else {
                    0.0
                },
                singular_count,
                singular_n,
                tau_squared_hat_mean: if tau_n > 0 {
                    tau_sum / tau_n as f64
                } else {
                    f64::NAN
                },
                tau_squared_hat_sum: tau_sum,
                tau_squared_hat_n: tau_n,
            })
        }
        EstimatorExtras::Mle { .. } => {
            let mut tau_sum = 0.0f64;
            let mut tau_n = 0u64;
            let mut boundary_hits = 0u64;
            let mut joint_unc = 0u64;
            let mut joint_cor = 0u64;
            let mut singular_count = 0u64;
            let mut singular_n = 0u64;
            let mut comp_counts: Vec<u64> = Vec::new();
            let mut comp_counts_init = false;
            for p in prs {
                if let EstimatorExtras::Mle {
                    tau_sum: ts,
                    tau_n: tn,
                    boundary_hits: bh,
                    joint_uncorrected_count: ju,
                    joint_corrected_count: jc,
                    singular_count: sc,
                    singular_n: sn,
                    boundary_component_counts: bcc,
                    ..
                } = &p.estimator_extras
                {
                    tau_sum += ts;
                    tau_n += tn;
                    boundary_hits += bh;
                    joint_unc += ju;
                    joint_cor += jc;
                    singular_count += sc;
                    singular_n += sn;
                    // boundary_component_counts: elementwise Σ. All workers share the
                    // same n_variance_components, so every part's vector has the same
                    // length — reject a mismatch like every other pooled count vector
                    // rather than silently concatenating a longer one.
                    if !comp_counts_init {
                        comp_counts = vec![0u64; bcc.len()];
                        comp_counts_init = true;
                    } else if bcc.len() != comp_counts.len() {
                        return Err(OrchestratorError::IncompatibleMerge(
                            "boundary_component_counts length mismatch across parts".into(),
                        ));
                    }
                    for (i, &c) in bcc.iter().enumerate() {
                        comp_counts[i] += c;
                    }
                }
            }
            // Recompute per-component rates against the pooled singular_n.
            let comp_rates: Vec<f64> = comp_counts
                .iter()
                .map(|&c| {
                    if singular_n > 0 {
                        c as f64 / singular_n as f64
                    } else {
                        0.0
                    }
                })
                .collect();
            let denom = pooled_n_sims as f64;
            Ok(EstimatorExtras::Mle {
                tau_estimate: if tau_n > 0 {
                    tau_sum / tau_n as f64
                } else {
                    0.0
                },
                boundary_hits,
                joint_uncorrected_rate: joint_unc as f64 / denom,
                joint_corrected_rate: joint_cor as f64 / denom,
                tau_sum,
                tau_n,
                joint_uncorrected_count: joint_unc,
                joint_corrected_count: joint_cor,
                singular_fit_rate: if singular_n > 0 {
                    singular_count as f64 / singular_n as f64
                } else {
                    0.0
                },
                singular_count,
                singular_n,
                boundary_rate_per_component: comp_rates,
                boundary_component_counts: comp_counts,
            })
        }
    }
}
