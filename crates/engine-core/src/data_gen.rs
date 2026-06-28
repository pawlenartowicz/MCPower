//! Per-sim data generation — the per-sim hot path.
//!
//! For fixed `(base_seed, sim_id)`, row `i` is deterministic and independent of
//! `max_n`. Prefix `X_full[:N]` is bit-identical across runs with different
//! `max_n`. The scenario design (Σ, L, var_types, residual choice) is drawn
//! fresh every sim from the scenario stream.
//!
//! Generation is split by draw shape: the hot block-shaped draws run as planar
//! column passes over class-addressed Philox sub-streams — continuous X
//! (`CLASS_XNORM`: per-predictor normal columns, then the L·z Cholesky mix and
//! marginal transform as a column sweep) and residuals (`CLASS_RESID`,
//! slot-addressed: Normal/Binary slot 0; t/χ² use slot 0 = z plus χ²
//! accumulator slots — see the residual pass). The remaining scattered work
//! (factor/categorical draws, bootstrap row-picks, cluster REs, broadcast and
//! interaction fills) stays in a row loop on the sequential per-sim stream.
//!
//! Cluster random intercepts are generated whenever a ClusterSpec is present,
//! independent of outcome_kind and estimator (generate-clustered/solve-OLS is valid).

use crate::distributions::{marginal_uniform, phi, sample_t};
use crate::rng::{fill_normal_column, fill_uniform_column, SimRng, CLASS_RESID, CLASS_XNORM};
use crate::scenarios::{
    perturb_correlation, perturb_var_types, pick_residual, scenario_rng, STREAM_TAG_HET,
};
use crate::spec::{
    ClusterSizing, Distribution, EngineError, OutcomeKind, ResidualDist, SimulationSpec,
};
use crate::workspace::SimWorkspace;
use crate::FLOAT_NEAR_ZERO;

// Censored Exp(1) at E ≤ EXP_CAP, standardized to exact unit variance; they
// standardize the Right/LeftSkewed marginals in apply_marginal.
// Closed form (E ~ Exp(1)):  mean_c = 1 − e^{−c};  E[min(E,c)²] = 2 − (2c+2)e^{−c};
//   var_c = E[min²] − mean_c².  EXP_CAP solves (c − mean_c)/sd_c = 6 (support = 6 SD).
// MEAN/STD are derived from the EXP_CAP literal below (not the unrounded root),
// so the three constants are mutually consistent and variance is exactly 1.
const EXP_CAP: f64 = 6.959_255_993_647_11;
const EXP_CENSORED_MEAN: f64 = 0.999_050_197_028_828_9;
const EXP_CENSORED_STD: f64 = 0.993_367_632_769_713_4;
const SQRT3: f64 = 1.732_050_807_568_877_2;

/// Generate one full simulation into `ws.x_full` / `ws.y_full`.
///
/// Layout of `ws.x_full` columns (row-major in faer column-major Mat):
///   `[ intercept (1.0) | non_factor (n_non_factor) | factor_dummies (n_factor_dummies) | interactions ]`
///
/// Interaction columns are appended last: column `1 + n_non_factor + n_factor_dummies + j`
/// holds the elementwise product of the component columns listed in `spec.interactions[j]`.
///
/// On error: returns `EngineError` — currently only when correlation factoring
/// fails (scenario perturbations push the matrix off the PSD cone).
pub fn generate_sim_data(
    spec: &SimulationSpec,
    sim_id: u64,
    base_seed: u64,
    ws: &mut SimWorkspace,
) -> Result<(), EngineError> {
    // Cluster random intercepts are generated whenever a ClusterSpec is present,
    // independent of outcome_kind and estimator (generate-clustered/solve-OLS).
    // (No guard needed: the contract's validate() already gates Mle⇒cluster.)

    let n_non_factor = spec.n_non_factor as usize;
    let n_factor_dummies = spec.n_factor_dummies as usize;
    let n_factors = spec.factor_n_levels.len();
    let max_n = ws.x_full.nrows();
    let n_predictors = ws.x_full.ncols();
    debug_assert_eq!(
        n_predictors,
        1 + n_non_factor + n_factor_dummies + spec.interactions.len(),
        "x_full column count must match intercept + non_factor + factor_dummies + interactions"
    );

    // 1. Draw the per-sim scenario design.
    populate_design(spec, base_seed, sim_id, ws)?;

    // Bootstrap (strict mode) precompute. Active when bootstrap_frame_map is
    // non-empty; empty in NORTA/none/partial. See the bootstrap arm in the row
    // loop below for the row-copy semantics.
    let is_bootstrap = !spec.bootstrap_frame_map.is_empty();
    // Bootstrap-only locals (see the bootstrap arm below).
    let (n_cols_frame, u_rows, factor_col_starts) = if is_bootstrap {
        debug_assert_eq!(
            spec.bootstrap_frame_map.len(),
            n_non_factor + n_factors,
            "bootstrap_frame_map length must equal n_non_factor + n_factors"
        );
        let n_cols_frame = spec.upload_data_shape.1 as usize;
        let u_rows = spec.upload_data_shape.0 as usize;
        // Each factor's starting dummy column (absolute index into x_full).
        // Invariant: must stay in sync with the col_offset walk in the (b) factor
        // block below — same start (1 + n_non_factor) and increment (n_levels - 1)
        // per factor. Update both together.
        let mut starts = Vec::with_capacity(n_factors);
        let mut off = 1 + n_non_factor;
        for f in 0..n_factors {
            starts.push(off);
            off += (spec.factor_n_levels[f].max(0) as usize).saturating_sub(1);
        }
        (n_cols_frame, u_rows, starts)
    } else {
        (0, 0, Vec::new())
    };

    // 2. Per-sim RNG stream — fresh for each sim, independent of the scenario stream.
    let mut rng = SimRng::new(base_seed, sim_id);

    // 2a. Cluster random-effect draws — one per cluster, fresh every sim.
    // Placed before the row loop so the RNG call order is: cluster draws first,
    // then per-row X/residual draws; this makes the per-row draws independent of
    // n_clusters. Draws happen whenever cluster.is_some().
    if let Some(cluster_spec) = &spec.cluster {
        // τ_eff is set per sim by populate_design (above); read it out as
        // a Copy f64 before the mutable cluster_u_draws writes.
        let tau = ws.tau_squared_design.max(0.0).sqrt();
        // RE distribution comes from the scenario's lme block (default Normal).
        // draw_residual already returns unit-variance draws, so τ² is preserved.
        let (re_dist, re_df) = spec
            .scenario
            .lme
            .as_ref()
            .map(|l| (l.random_effect_dist, l.random_effect_df))
            .unwrap_or((ResidualDist::Normal, 0.0));
        let n_clusters = cluster_spec.sizing.n_clusters_at(max_n);
        for c in 0..n_clusters {
            ws.cluster_u_draws[c] = (draw_residual(&mut rng, re_dist, re_df) * tau) as f32;
        }

        // 2a′. Primary random-slope draw: per cluster, (u_0,…,u_{q-1}) ~ N(0, D)
        // via the q_p×q_p Cholesky L of D = diag(τ)·R·diag(τ). u_0 is already
        // drawn above as τ₀·z₀ and left in cluster_u_draws (an f32 store); recover
        // the f32-rounded z₀ = u₀/τ₀ (f32 precision, consistent with the f32 plane),
        // draw z_1..z_{q-1} per cluster, and set u_k = Σ_{j≤k} L[k][j]·z_j for
        // k ≥ 1 (lower-triangular L), stored row-major in
        // cluster_slope_u_draws[c·#slopes + (k−1)]. Specs without a slope skip
        // this entirely — their RNG stream is unchanged.
        if !cluster_spec.slopes.is_empty() {
            let n_sl = cluster_spec.slopes.len();
            let q = 1 + n_sl;
            // τ₀ reuses `tau` from the primary draw above (= √tau_squared_design),
            // so the z₀ = u₀/τ₀ recovery divides by the EXACT τ₀ that scaled u₀ —
            // structurally, not by a re-derived value that merely happens to match.
            let (qd, r) = cluster_spec.re_correlation_matrix();
            debug_assert_eq!(qd, q);
            let mut tau_vec = Vec::with_capacity(q);
            tau_vec.push(tau);
            for s in &cluster_spec.slopes {
                tau_vec.push(s.variance.max(0.0).sqrt());
            }
            let mut dmat = vec![0.0f64; q * q];
            for i in 0..q {
                for j in 0..q {
                    dmat[i * q + j] = tau_vec[i] * r[i * q + j] * tau_vec[j];
                }
            }
            let l = glmm::linalg::chol_lower(&dmat, q);
            let tau0 = tau_vec[0];
            let n_clusters_sl = cluster_spec.sizing.n_clusters_at(max_n);
            let mut z = vec![0.0f64; q];
            for c in 0..n_clusters_sl {
                // validate() requires τ₀² > 0 when slopes exist, so tau0 > 0.
                z[0] = if tau0 > 0.0 {
                    ws.cluster_u_draws[c] as f64 / tau0
                } else {
                    0.0
                };
                #[allow(clippy::needless_range_loop)]
                for j in 1..q {
                    z[j] = draw_residual(&mut rng, re_dist, re_df);
                }
                for k in 1..q {
                    let mut uk = 0.0;
                    for j in 0..=k {
                        uk += l[k * q + j] * z[j];
                    }
                    ws.cluster_slope_u_draws[c * n_sl + (k - 1)] = uk as f32;
                }
            }
        }

        // 2b. Extra-grouping draws — one block per extra grouping, declaration
        // order, drawn straight after the primary so per-row draws stay
        // independent of the grouping structure. Old specs (no extras) consume
        // an identical RNG stream — no golden shift. Same re_dist/re_df knobs
        // for every grouping (per-grouping uniformity); per-grouping effective τ² (ICC
        // jitter) comes from extra_tau_sq_design, set with tau_squared_design by
        // populate_design — change together.
        for g in 0..ws.extra_u_draws.len() {
            let tau_g = ws.extra_tau_sq_design[g].max(0.0).sqrt();
            let k_g = ws.extra_u_draws[g].len();
            for c in 0..k_g {
                ws.extra_u_draws[g][c] = (draw_residual(&mut rng, re_dist, re_df) * tau_g) as f32;
            }
        }
    }

    // Intercept (column 0) is always 1.0.
    for i in 0..max_n {
        ws.x_full[(i, 0)] = 1.0f32;
    }

    // Per-factor proportion sampling: a Some override wins; None (or an absent
    // entry, e.g. empty factor_sampled) inherits the scenario default. Resolved
    // per factor below; precomputed here only to decide the alloc-counts reset.
    let scenario_sampled = spec.scenario.sampled_factor_proportions;
    let any_exact = (0..n_factors).any(|f| {
        !spec
            .factor_sampled
            .get(f)
            .copied()
            .flatten()
            .unwrap_or(scenario_sampled)
    });
    // Fixed-allocation walk state — running per-level counts for every factor
    // (factor_proportions layout). Reset each sim whenever at least one factor
    // uses the deterministic walk; the resize is a no-op after the first sim.
    if any_exact && !spec.factor_proportions.is_empty() {
        ws.factor_alloc_counts.clear();
        ws.factor_alloc_counts
            .resize(spec.factor_proportions.len(), 0);
    }

    // (a) Continuous block — planar, class-addressed (CLASS_XNORM, column =
    // predictor j). Replaces the former per-row z draws: values come from the
    // (key, class, col, row) address, so they are prefix-stable in max_n and
    // independent of every other draw class by construction.
    let key = rng.key();
    for j in 0..n_non_factor {
        let col = &mut ws.z_planar[j * max_n..j * max_n + max_n];
        fill_normal_column(
            key,
            CLASS_XNORM,
            j as u32,
            &mut ws.gen_words,
            &mut ws.gen_tail_idx,
            col,
        );
    }
    // Lower-tri mix, column sweep: x[:, 1+j] = Σ_{k≤j} L[j,k]·z[:, k] with an
    // f64 accumulator column — same k-ascending add order per element as the
    // old per-row loop. The marginal transform runs per column (one var_type
    // dispatch per column, not per element) on the f32-narrowed mix value:
    // x_full is the f32 generation plane, and the narrow keeps the
    // transform input identical to what the old read-back-from-x_full saw.
    for j in 0..n_non_factor {
        ws.mix_acc[..max_n].fill(0.0);
        for k in 0..=j {
            let l_jk = ws.l_design[(j, k)];
            // Bounds: z_planar is max_n * n_non_factor; k < n_non_factor and
            // max_n is the x_full row count — bounds are consistent by construction.
            let zk = &ws.z_planar[k * max_n..k * max_n + max_n];
            let acc = &mut ws.mix_acc[..max_n];
            for i in 0..max_n {
                acc[i] += l_jk * zk[i] as f64;
            }
        }
        let vt = ws.var_types_design[j];
        let param = spec.var_params.get(j).copied().unwrap_or(0.5);
        if matches!(vt, Distribution::Normal) {
            for i in 0..max_n {
                ws.x_full[(i, 1 + j)] = ws.mix_acc[i] as f32;
            }
        } else {
            // f32-narrow then widen mirrors the old read-back from x_full
            // (old code stored acc as f32 then re-read as f64 for apply_marginal).
            for i in 0..max_n {
                ws.mix_acc[i] = ws.mix_acc[i] as f32 as f64;
            }
            apply_marginal_column(&mut ws.mix_acc[..max_n], vt, param, j, spec);
            for i in 0..max_n {
                ws.x_full[(i, 1 + j)] = ws.mix_acc[i] as f32;
            }
        }
    }

    // (c) Residual columns — CLASS_RESID, slot-addressed. Slot layout per
    // block dist: Normal/Binary → slot 0 only; T/HighKurtosis → slot 0 = z,
    // slots 1..=df = χ² accumulator normals; Right/LeftSkewed → slots
    // 0..df-1 = χ² accumulator normals. Mirrors draw_residual's math
    // (T standardization df/(df−2), χ² centering (χ²−df)/√(2df), LeftSkewed
    // sign flip) — change together. df is clamped ≥ 3 exactly as there, so
    // sample_t's degenerate-df fallback has no batched counterpart (the clamp
    // makes it unreachable). One harmless cast difference vs draw_residual:
    // .max(1) here is applied before the u32 cast rather than at the loop
    // bound — equivalent under the df ≥ 3 clamp.
    match spec.outcome_kind {
        OutcomeKind::Binary => {
            fill_uniform_column(
                key,
                CLASS_RESID,
                0,
                &mut ws.gen_words,
                &mut ws.residuals[..max_n],
            );
        }
        OutcomeKind::Continuous => match ws.residual_dist_design {
            ResidualDist::Normal => {
                fill_normal_column(
                    key,
                    CLASS_RESID,
                    0,
                    &mut ws.gen_words,
                    &mut ws.gen_tail_idx,
                    &mut ws.residuals[..max_n],
                );
            }
            ResidualDist::HighKurtosis => {
                let df = ws.residual_df_design.max(3.0);
                let df_int = (df.round() as i64).max(1) as u32;
                let scale = 1.0 / (df / (df - 2.0)).sqrt();
                fill_normal_column(
                    key,
                    CLASS_RESID,
                    0,
                    &mut ws.gen_words,
                    &mut ws.gen_tail_idx,
                    &mut ws.residuals[..max_n],
                );
                ws.mix_acc[..max_n].fill(0.0);
                for s in 0..df_int {
                    fill_normal_column(
                        key,
                        CLASS_RESID,
                        1 + s,
                        &mut ws.gen_words,
                        &mut ws.gen_tail_idx,
                        &mut ws.resid_g[..max_n],
                    );
                    for i in 0..max_n {
                        let g = ws.resid_g[i] as f64;
                        ws.mix_acc[i] += g * g;
                    }
                }
                for i in 0..max_n {
                    let z = ws.residuals[i] as f64;
                    let denom = (ws.mix_acc[i] / df).sqrt();
                    let t = if denom <= 0.0 { z } else { z / denom };
                    ws.residuals[i] = (t * scale) as f32;
                }
            }
            ResidualDist::RightSkewed | ResidualDist::LeftSkewed => {
                let df = ws.residual_df_design.max(3.0);
                let df_int = (df.round() as i64).max(1) as u32;
                let scale = 1.0 / (2.0 * df).sqrt();
                let sign = if matches!(ws.residual_dist_design, ResidualDist::LeftSkewed) {
                    -1.0
                } else {
                    1.0
                };
                ws.mix_acc[..max_n].fill(0.0);
                for s in 0..df_int {
                    fill_normal_column(
                        key,
                        CLASS_RESID,
                        s,
                        &mut ws.gen_words,
                        &mut ws.gen_tail_idx,
                        &mut ws.resid_g[..max_n],
                    );
                    for i in 0..max_n {
                        let g = ws.resid_g[i] as f64;
                        ws.mix_acc[i] += g * g;
                    }
                }
                for i in 0..max_n {
                    ws.residuals[i] = (sign * (ws.mix_acc[i] - df) * scale) as f32;
                }
            }
            ResidualDist::Uniform => {
                // Unit-SD uniform U(−√3, √3): u ∈ [0,1) → (2u − 1)·√3 (var =
                // (2√3)²/12 = 1). Bounded support, no tail-bound work. Slot 0
                // uniform fill, same slot the Binary arm uses. Mirrors
                // draw_residual's Uniform arm — change together.
                fill_uniform_column(
                    key,
                    CLASS_RESID,
                    0,
                    &mut ws.gen_words,
                    &mut ws.residuals[..max_n],
                );
                for i in 0..max_n {
                    let u = ws.residuals[i] as f64;
                    ws.residuals[i] = ((2.0 * u - 1.0) * SQRT3) as f32;
                }
            }
        },
    }

    // 3. Row loop — factor/bootstrap/broadcast/interaction blocks. Continuous
    // columns and residuals are filled by the planar passes above; the
    // sequential rng stream below serves only categorical draws, the
    // bootstrap row pick, and cluster-RE draws (drawn before this loop).
    for i in 0..max_n {
        // (b) Factor block — categorical draws per factor, reference-coded.
        let mut col_offset = 1 + n_non_factor;
        let mut prop_offset = 0usize;
        for f in 0..n_factors {
            let n_levels = spec.factor_n_levels[f].max(0) as usize;
            if n_levels == 0 {
                continue;
            }
            let probs = &spec.factor_proportions[prop_offset..prop_offset + n_levels];
            // Inherit the scenario default unless this factor overrides it.
            let sampled = spec
                .factor_sampled
                .get(f)
                .copied()
                .flatten()
                .unwrap_or(scenario_sampled);
            let lvl = if !sampled {
                // Deterministic allocation — consumes no RNG. The shared per-sim
                // stream therefore sits at a different position than in sampled
                // mode from here on; in mixed-mode designs the per-factor choice
                // determines RNG consumption (intentional, per-sim deterministic).
                fixed_level_next(
                    probs,
                    &mut ws.factor_alloc_counts[prop_offset..prop_offset + n_levels],
                )
            } else {
                // Simple randomization — one categorical draw per factor per row.
                rng.next_categorical(probs)
            };
            // Zero the dummies in this factor first.
            for d in 0..(n_levels.saturating_sub(1)) {
                ws.x_full[(i, col_offset + d)] = 0.0f32;
            }
            // Reference-coded: level 0 = reference (all zero); lvl > 0 sets dummy lvl-1 to 1.
            // Dummies must remain exact 0.0/1.0 — update_factor_exclusions (batch.rs) recovers
            // the level via == 1.0 comparison; any deviation would silently miscount.
            if lvl > 0 && lvl - 1 < n_levels.saturating_sub(1) {
                ws.x_full[(i, col_offset + lvl - 1)] = 1.0f32;
            }
            prop_offset += n_levels;
            col_offset += n_levels.saturating_sub(1);
        }

        // (b.4) Bootstrap arm — strict mode. Draw ONE source row index from the
        // per-sim RNG and overwrite every uploaded design column with that source
        // row's values. One source index per design row is what preserves the exact
        // empirical joint across columns (the point of strict). Synthetic columns
        // keep the values they already drew above; only uploaded columns are
        // overwritten. The continuous/factor blocks above still ran for uploaded
        // columns (their drawn values are discarded here) — this keeps the per-row
        // RNG call count constant ⇒ row-stable. Runs AFTER the factor block and
        // BEFORE the interaction block, so interactions use the copied values.
        if is_bootstrap {
            debug_assert!(u_rows > 0, "bootstrap mode requires a non-empty frame");
            let r = ((rng.next_uniform() as f64 * u_rows as f64) as usize)
                .min(u_rows.saturating_sub(1));
            // Non-factor uploaded columns.
            for j in 0..n_non_factor {
                if let Some(fc) = spec.bootstrap_frame_map[j] {
                    let v = spec.upload_data[r * n_cols_frame + fc as usize];
                    // Use the ORIGINAL contract type (spec.var_types[j]), not the
                    // scenario-perturbed var_types_design — an uploaded column copies
                    // real data regardless of scenario perturbation.
                    let x = match spec.var_types[j] {
                        Distribution::UploadedBinary => {
                            // stored = x - p; recover 0/1 (consistent with the NORTA
                            // 0/1 marginal).
                            let p = spec.var_params.get(j).copied().unwrap_or(0.0);
                            if v + p >= 0.5 {
                                1.0
                            } else {
                                0.0
                            }
                        }
                        // UploadedData / continuous: copy the standardized value directly.
                        _ => v,
                    };
                    ws.x_full[(i, 1 + j)] = x as f32;
                }
            }
            // Factor uploaded columns: re-expand the copied level code to
            // reference-coded dummies.
            for (f, &start) in factor_col_starts.iter().enumerate() {
                if let Some(fc) = spec.bootstrap_frame_map[n_non_factor + f] {
                    let level = spec.upload_data[r * n_cols_frame + fc as usize].round() as usize;
                    let n_dummies = (spec.factor_n_levels[f].max(0) as usize).saturating_sub(1);
                    for d in 0..n_dummies {
                        ws.x_full[(i, start + d)] = 0.0f32;
                    }
                    if level > 0 && level - 1 < n_dummies {
                        ws.x_full[(i, start + level - 1)] = 1.0f32;
                    }
                }
            }
        }

        // (b.45) Cluster-level broadcast — overwrite marked columns on
        // non-representative rows with their cluster's representative-row value.
        // Runs AFTER the factor/bootstrap block and BEFORE interactions, so a
        // cross-level interaction uses the broadcast value. The representative
        // row's value is already final (rep < i in both layouts, rows ascend)
        // and resident in x_full, so no per-cluster cache buffer is needed. The
        // per-row draw above already ran for these columns (value discarded on
        // non-rep rows), so the RNG call count is unchanged ⇒ empty
        // between_var_indices is byte-identical to today.
        //
        // The (cluster → rep-row) mapping below MUST define the same clusters as
        // ws.cluster_ids (which keys the u_c add in the assembly loop) — both
        // derive from cluster.sizing; keep them in lockstep.
        if !spec.between_var_indices.is_empty() {
            if let Some(cluster) = &spec.cluster {
                let rep = match &cluster.sizing {
                    ClusterSizing::FixedClusters { n_clusters } => {
                        i % ((*n_clusters).max(1) as usize)
                    }
                    ClusterSizing::FixedSize { cluster_size } => {
                        let cs = (*cluster_size).max(1) as usize;
                        i - (i % cs)
                    }
                };
                if i != rep {
                    for &col_u in &spec.between_var_indices {
                        ws.x_full[(i, col_u as usize)] = ws.x_full[(rep, col_u as usize)];
                    }
                }
            }
        }

        // (b.5) Interaction block — raw elementwise product of component
        // columns. Components are continuous/dummy columns already
        // filled above (nested interactions rejected by contract invariant 18),
        // so every index is < col_offset. Intercept (col 0) is never a component.
        // At this point col_offset == 1 + n_non_factor + n_factor_dummies.
        for (j, comps) in spec.interactions.iter().enumerate() {
            let mut prod = 1.0f64;
            for &c in comps {
                prod *= ws.x_full[(i, c as usize)] as f64;
            }
            ws.x_full[(i, col_offset + j)] = prod as f32;
        }
    }

    // 4. Compute y = Xβ + scaled residual — fused single row pass.
    //
    // β layout: effect_sizes has length n_predictors (intercept + non_factor + factor_dummies).
    // We don't add intercept separately — it's column 0 of X.
    //
    // Per-study (per-simulation) β-jitter — heterogeneity as one realized study.
    // When het is active a single shift vector δ (length n_predictors) is drawn
    // ONCE per sim (in `beta_eff`, just below) and the SAME jittered effects apply
    // to every row:
    //   lp_clean = Σⱼ βⱼ·xᵢⱼ            — drives heteroskedasticity scaling
    //   lp_eff   = Σⱼ β_effⱼ·xᵢⱼ        — drives y / Bernoulli assembly
    //   δⱼ = N(0, sⱼ²) per sim,  β_effⱼ = clip_to_sign(βⱼ, βⱼ + δⱼ)
    // where sⱼ = het·|βⱼ| for effect coefficients (j ≥ 1): the true effect varies
    // study-to-study with between-study SD τ = het·|β| (the Many Labs / Holzmeister
    // calibration, h = τ/|β|). Per study, not per observation: the variation does
    // NOT average out, so power is bounded by ≈ Φ(1/het) regardless of N. The clip
    // pulls a realized effect toward 0 but never across it — a study that would
    // flip sign becomes exactly 0 (a true null, rejected only at rate α), so the
    // two-sided tally in batch.rs needs no change and the ceiling emerges as
    // Φ(1/het) + (1−Φ(1/het))·α. Cost: the censored mean is nudged up by a factor
    // Φ(1/h)+h·φ(1/h) (×1.0008 at h=0.4, ×1.08 at h=1.0) — negligible at the presets.
    // For the intercept (j = 0) sⱼ is outcome-dependent and is NOT clipped:
    //   - Continuous: s₀ = 0 — intercept jitter excluded (v1 parity,
    //     effects-only; every host sends β₀ = 0 here anyway).
    //   - Binary: s₀ = het — additive log-odds jitter on the baseline (a nuisance
    //     shift with no tested direction), i.e. multiplicative odds wobble of
    //     ≈ ±het·100%. NOT scaled by |β₀|: that would vanish at p = 0.5 and explode
    //     for rare outcomes — an artifact of intercept-as-column-0.
    // δ is drawn from a domain-separated stream (sim_id ^ STREAM_TAG_HET), so it
    // never perturbs X / residuals: het = 0 is byte-identical and δ is N-stable
    // (one draw per sim, independent of max_n). When use_het_jitter is false,
    // lp_clean is the only accumulator (lp_eff stays 0 and is unused).
    //
    // Heteroskedasticity: residual variance follows the renormalized-multiplicative
    // monotone model Var(εᵢ) = σ²·exp(γ·zᵢ)/exp(γ²/2), where z is the standardized
    // driver (the linear predictor Xβ when driver=None, else a chosen x_full column,
    // using the population moments in spec.het_coeffs) and γ = ln(λ)/4. λ is the
    // model ratio if pinned, else the scenario's ratio (1.0 = homoskedastic).
    // Applies to ALL continuous outcomes including LME, where it scales the level-1
    // residual only — the cluster draw u_c (added below) stays homoskedastic. Binary
    // variance is fixed by p_i, so het never applies there. The /exp(γ²/2) factor
    // preserves mean residual variance (exact for a normal driver), so dialing λ
    // does not change total noise. Uses lp_clean, not lp_eff: het scaling is about
    // the *true* signal magnitude, not jittered draws.
    //
    // Accepted approximation: spec.het_coeffs (the driver population moments
    // used to standardize z) is computed once per batch from the *base* spec.
    // Scenario perturbations (distribution swaps, correlation noise) shift the
    // realized driver moments, so under perturbed scenarios the λ calibration
    // is anchored to the unperturbed design. The drift is second-order next to
    // the perturbations themselves; recomputing per sim would be per-sim
    // O(p²) machinery and a result-moving change for marginal statistical gain.
    //
    // Invariant preserved: y rows are row-stable in (base_seed, sim_id) — no
    // window-dependent normalisation (v1's empirical-SD residual rescaling is dropped).
    let het_total = spec.scenario.heterogeneity.max(0.0);
    let use_het_jitter = het_total > FLOAT_NEAR_ZERO;
    // Intercept (j = 0) jitter SD — see the comment block above.
    let het_intercept_sd = match spec.outcome_kind {
        OutcomeKind::Continuous => 0.0,
        OutcomeKind::Binary => het_total,
    };

    // Per-study effect draw β_eff: one δ per sim from the domain-separated het
    // stream, clipped to sign for effect columns. Empty (and the stream untouched)
    // when het is off, which is what keeps het = 0 byte-identical. One next_normal
    // is consumed per predictor — including sⱼ = 0 columns (β = 0, or the
    // continuous intercept) — so predictor j always pairs with the j-th draw.
    let beta_eff: Vec<f64> = if use_het_jitter {
        let mut het_rng = SimRng::new(base_seed, sim_id ^ STREAM_TAG_HET);
        (0..n_predictors)
            .map(|j| {
                let beta_j = spec.effect_sizes[j];
                let s_j = if j == 0 {
                    het_intercept_sd
                } else {
                    het_total * beta_j.abs()
                };
                let raw = beta_j + het_rng.next_normal() as f64 * s_j;
                // Clip effect columns toward zero, never across it. j = 0 is the
                // intercept (nuisance shift, unclipped); β = 0 ⇒ s = 0 ⇒ raw = 0.
                if j == 0 || beta_j == 0.0 {
                    raw
                } else if beta_j > 0.0 {
                    raw.max(0.0)
                } else {
                    raw.min(0.0)
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    // Resolve λ → γ once per sim. λ is scenario-only (the model contributes
    // the driver, never the magnitude); the .max(1.0) backstop makes a stray
    // λ<1 read as "off" rather than inverse het.
    let lambda = spec.scenario.heteroskedasticity_ratio.max(1.0);
    let gamma = lambda.ln() / 4.0;
    // Driver moments (full-design layout): linear predictor when driver=None,
    // else the chosen x_full column. .get() guards default()-empty coeffs (a
    // missing column reads std=0 → apply_hsk false, never an index panic).
    let hsk_driver = spec.heteroskedasticity_driver;
    let (hsk_center, hsk_std) = match hsk_driver {
        None => (spec.het_coeffs.lp_pop_mean, spec.het_coeffs.lp_pop_std),
        Some(idx) => {
            let idx = idx as usize;
            (
                spec.het_coeffs.col_mean.get(idx).copied().unwrap_or(0.0),
                spec.het_coeffs.col_std.get(idx).copied().unwrap_or(0.0),
            )
        }
    };
    let apply_hsk = matches!(spec.outcome_kind, OutcomeKind::Continuous)
        && gamma > FLOAT_NEAR_ZERO
        && hsk_std > FLOAT_NEAR_ZERO;
    let hsk_inv_std = if apply_hsk { 1.0 / hsk_std } else { 0.0 };
    let hsk_inv_norm = (-gamma * gamma / 2.0).exp(); // = 1 / exp(γ²/2)

    // cluster_sizing for the assembly loop: present whenever cluster.is_some().
    let cluster_sizing = spec.cluster.as_ref().map(|s| &s.sizing);

    for i in 0..max_n {
        let mut lp_clean = 0.0;
        let mut lp_eff = 0.0;
        #[allow(clippy::needless_range_loop)]
        for j in 0..n_predictors {
            // `n_predictors == x_full.ncols() == effect_sizes.len() == beta_eff.len()`
            // (the batch validator enforces the equality), so direct indexing is in
            // bounds. β_eff was drawn once per sim above — the same per-study effect
            // applies to every row, so no RNG is consumed here.
            let xij = ws.x_full[(i, j)] as f64;
            lp_clean += xij * spec.effect_sizes[j];
            if use_het_jitter {
                lp_eff += xij * beta_eff[j];
            }
        }
        let lp = if use_het_jitter { lp_eff } else { lp_clean };

        if apply_hsk {
            // Stage the exp argument γ·z into mix_acc (free here — the X column
            // sweep is done; Binary's η uses eta_acc, no clash); the exp runs as
            // one SIMD column pass after this loop, the residual scaling and the
            // y assembly follow it.
            let driver_i = match hsk_driver {
                None => lp_clean,
                Some(idx) => ws.x_full[(i, idx as usize)] as f64,
            };
            let z = (driver_i - hsk_center) * hsk_inv_std;
            ws.mix_acc[i] = gamma * z;
        }

        // Random-effect contribution on the LATENT scale: primary intercept +
        // extra groupings + primary slopes. Outcome-independent — Continuous adds
        // it to y, Binary puts it inside the link. Zero when there is no cluster.
        let mut u_re = if cluster_sizing.is_some() {
            ws.cluster_u_draws[ws.cluster_ids[i] as usize] as f64
        } else {
            0.0
        };
        for g in 0..ws.extra_grouping_ids.len() {
            u_re += ws.extra_u_draws[g][ws.extra_grouping_ids[g][i] as usize] as f64;
        }
        let gprim = ws.cluster_ids[i] as usize;
        let n_sl = spec.cluster_slope_design_cols.len();
        for (k, &sc) in spec.cluster_slope_design_cols.iter().enumerate() {
            u_re += ws.cluster_slope_u_draws[gprim * n_sl + k] as f64
                * ws.x_full[(i, sc as usize)] as f64;
        }

        match spec.outcome_kind {
            OutcomeKind::Continuous => {
                if apply_hsk {
                    // Residuals are still un-scaled here (the hsk exp pass runs
                    // after this loop), so y assembly moves to a pass after it;
                    // stage η = lp + u_re. The (lp + u_re) + ε association
                    // differs from the old (lp + ε) + u_re by ≤1 ULP — hsk is
                    // result-moving through the owned exp anyway.
                    ws.eta_acc[i] = lp + u_re;
                } else {
                    // y = Xβ + ε + u_re. When no cluster, u_re == 0.0, identical
                    // to old OLS path.
                    ws.y_full[i] = (lp + ws.residuals[i] as f64 + u_re) as f32;
                }
            }
            OutcomeKind::Binary => {
                // η staged into eta_acc; the sigmoid runs as one SIMD column pass
                // after this loop instead of a per-row libm exp. residuals[i] holds
                // the pre-drawn Uniform[0,1). lp carries the log-odds β-jitter when on.
                ws.eta_acc[i] = lp + u_re;
            }
        }
    }

    if apply_hsk {
        // mult_i = exp(γ·z_i)·hsk_inv_norm via the owned SIMD exp (≤1 ULP of the
        // old libm .exp() — result-moving, rides the golden campaign), then the
        // same per-element scale-and-narrow as the old in-loop form.
        glmm::simd_transcendental::exp_fill(&mut ws.mix_acc[..max_n]);
        for i in 0..max_n {
            let mult = ws.mix_acc[i] * hsk_inv_norm;
            ws.residuals[i] = (ws.residuals[i] as f64 * mult.sqrt()) as f32;
        }
        for i in 0..max_n {
            ws.y_full[i] = (ws.eta_acc[i] + ws.residuals[i] as f64) as f32;
        }
    }

    if matches!(spec.outcome_kind, OutcomeKind::Binary) {
        // p_i = σ(η_i) via the owned SIMD kernel (≤2 ULP of the old libm
        // sigmoid_stable — a y flip needs the uniform inside that gap, ~2⁻⁵²/row;
        // formally result-moving, rides the golden campaign).
        glmm::simd_transcendental::sigmoid_fill(&mut ws.eta_acc[..max_n]);
        for i in 0..max_n {
            ws.y_full[i] = if (ws.residuals[i] as f64) < ws.eta_acc[i] {
                1.0f32
            } else {
                0.0f32
            };
        }
    }

    Ok(())
}

/// Deterministic factor-level assignment for `fixed_allocation` scenarios: an
/// incremental largest-remainder walk (weighted round-robin) over the row
/// index. `counts` holds this factor's per-level totals for the rows already
/// assigned this sim; the next row gets the level with the largest running
/// deficit `p_g·(rows+1) − counts[g]`, ties broken by lowest level index.
///
/// Properties (tests below):
/// - consumes no RNG — the assignment is a pure function of (row index, probs);
/// - row-stable: row i's level depends only on rows 0..i, so `X_full[:N]`
///   stays a nested prefix across the sample-size grid;
/// - at every prefix N: Σ counts == N; each count stays within 1 of N·p_g
///   for two-level factors (the exact Hamilton split) and within ~1 in
///   general — transient deviations slightly above 1 are possible only for
///   many-level, highly skewed proportion vectors;
/// - equal proportions reduce to round-robin `i % k`, matching the cluster
///   round-robin precedent.
fn fixed_level_next(probs: &[f64], counts: &mut [u32]) -> usize {
    debug_assert!(!probs.is_empty(), "fixed_level_next: empty probs");
    let total: f64 = probs.iter().sum();
    debug_assert!(total > 0.0, "fixed_level_next: proportions sum to zero");
    let assigned: u32 = counts.iter().sum();
    let t = (assigned + 1) as f64;
    let mut best = 0usize;
    let mut best_deficit = f64::NEG_INFINITY;
    for (g, &p) in probs.iter().enumerate() {
        // Strict > keeps the lowest index on ties.
        let deficit = (p / total) * t - counts[g] as f64;
        if deficit > best_deficit {
            best_deficit = deficit;
            best = g;
        }
    }
    counts[best] += 1;
    best
}

/// Deterministic level counts the fixed-allocation walk produces for the
/// first `n` rows — the same largest-remainder walk the row loop runs.
/// Used by the orchestrator's pre-run sparse-factor validator.
pub fn fixed_allocation_counts(probs: &[f64], n: usize) -> Vec<u32> {
    let mut counts = vec![0u32; probs.len()];
    for _ in 0..n {
        let _ = fixed_level_next(probs, &mut counts);
    }
    counts
}

/// Smallest N ≤ n_max at which every level's fixed-allocation count
/// reaches `k_min`, or None if not reached within n_max. Single forward
/// walk — O(n_max · k).
pub fn min_inclusion_n(probs: &[f64], k_min: u32, n_max: usize) -> Option<usize> {
    let mut counts = vec![0u32; probs.len()];
    for n in 1..=n_max {
        let _ = fixed_level_next(probs, &mut counts);
        if counts.iter().all(|&c| c >= k_min) {
            return Some(n);
        }
    }
    None
}

/// Populate the workspace design slots for `sim_id`: either the optimistic
/// fast path (copy from spec, Cholesky) or the full per-sim scenario draw.
fn populate_design(
    spec: &SimulationSpec,
    base_seed: u64,
    sim_id: u64,
    ws: &mut SimWorkspace,
) -> Result<(), EngineError> {
    let n_nf = spec.n_non_factor as usize;

    if spec.scenario.is_optimistic() {
        // Optimistic baseline: copy correlation into l_design scratch and Cholesky.
        for j in 0..n_nf {
            for i in 0..n_nf {
                ws.l_design[(i, j)] = spec.correlation[j * n_nf + i];
            }
        }
        crate::correlation::factor_only(ws.l_design.as_mut())?;
        // Var types and residual choice come straight from the spec.
        for j in 0..n_nf {
            ws.var_types_design[j] = spec
                .var_types
                .get(j)
                .copied()
                .unwrap_or(Distribution::Normal);
        }
        ws.residual_dist_design = spec.residual_dist;
        ws.residual_df_design = spec.scenario.residual_df;
        ws.tau_squared_design = spec.cluster.as_ref().map(|c| c.tau_squared).unwrap_or(0.0);
        if let Some(c) = spec.cluster.as_ref() {
            for (g, gs) in c.extra_groupings.iter().enumerate() {
                ws.extra_tau_sq_design[g] = gs.tau_squared;
            }
        }
        return Ok(());
    }

    // Scenario path: draw fresh design.
    let mut scen_rng = scenario_rng(base_seed, sim_id);

    // Refill the workspace-owned base correlation from spec.correlation —
    // zero-alloc. The contents are identical every sim, but refilling per
    // draw keeps the buffer from ever serving a stale spec.
    for j in 0..n_nf {
        for i in 0..n_nf {
            ws.corr_base[(i, j)] = spec.correlation[j * n_nf + i];
        }
    }
    perturb_correlation(
        &spec.scenario,
        ws.corr_base.as_ref(),
        &mut scen_rng,
        ws.sigma_design.as_mut(),
        &mut ws.perturb_noise_buf,
    );
    // Copy sigma_design → l_design and PSD-repair-and-factor in-place.
    for j in 0..n_nf {
        for i in 0..n_nf {
            ws.l_design[(i, j)] = ws.sigma_design[(i, j)];
        }
    }
    crate::correlation::psd_repair_and_factor(ws.l_design.as_mut(), &mut ws.evd_scratch)?;

    perturb_var_types(
        &spec.scenario,
        &spec.var_types,
        &spec.var_pinned,
        &mut scen_rng,
        &mut ws.var_types_design,
    );
    ws.residual_dist_design = pick_residual(
        &spec.scenario,
        spec.residual_dist,
        spec.residual_pinned,
        &mut scen_rng,
    );
    ws.residual_df_design = spec.scenario.residual_df;
    // Effective τ² per sim (D6): base cluster variance plus an optional ICC
    // jitter drawn from the scenario RNG (after the design draws above). Clamped
    // ≥ 0 (D5). No cluster ⇒ 0.0; no jitter ⇒ base τ² (so re_dist=Normal,
    // icc=0 stays byte-identical to the optimistic path's cluster draw).
    let base_tau_sq = spec.cluster.as_ref().map(|c| c.tau_squared).unwrap_or(0.0);
    ws.tau_squared_design = match &spec.scenario.lme {
        Some(lme) if lme.icc_noise_sd > 0.0 => {
            let jitter = scen_rng.next_normal() as f64 * lme.icc_noise_sd;
            (base_tau_sq + jitter).max(0.0)
        }
        _ => base_tau_sq,
    };
    // Extra groupings: same D6 rule, one INDEPENDENT jitter draw per grouping
    // (uniform = same knob semantics per grouping, not one shared jitter),
    // drawn after the primary's in declaration order so old specs' scenario
    // stream is unchanged. Clamped ≥ 0 (D5).
    if let Some(c) = spec.cluster.as_ref() {
        for (g, gs) in c.extra_groupings.iter().enumerate() {
            ws.extra_tau_sq_design[g] = match &spec.scenario.lme {
                Some(lme) if lme.icc_noise_sd > 0.0 => {
                    let jitter = scen_rng.next_normal() as f64 * lme.icc_noise_sd;
                    (gs.tau_squared + jitter).max(0.0)
                }
                _ => gs.tau_squared,
            };
        }
    }
    Ok(())
}

/// Apply a marginal transform to a single value. `z` is post-Cholesky (correlated
/// standard-normal). Matches v1's `transform_distribution` except the
/// Right/LeftSkewed arms, which are a tail-censored standardized Exp(1)
/// (v1 mis-standardized an Exp(1) deviate with lognormal constants).
///
/// Part of the reproducibility contract (see the `distributions.rs` module
/// docs) — golden-pinned per variant by `golden_apply_marginal_all_variants`.
/// Production runs the column twin `apply_marginal_column` (bit-identical,
/// pinned by `apply_marginal_column_matches_scalar`); this per-element form
/// is the contract reference the golden/parity tests call.
#[cfg_attr(not(test), allow(dead_code))]
fn apply_marginal(
    z: f64,
    dist_type: Distribution,
    param: f64,
    var_idx: usize,
    spec: &SimulationSpec,
) -> f64 {
    match dist_type {
        Distribution::Normal => z,
        Distribution::Binary => {
            // Binary: u = Φ(z); 1 in the HIGH-z tail (u ≥ 1 − param) so the marginal
            // is monotone-INCREASING in the latent z, like every other marginal.
            // P(X=1) = param is preserved either way; the direction matters only for
            // consistency and any future copula use — predictor correlation is
            // continuous-only by design, so binary is generated independently. We do
            // NOT center here per-row; v1 centers the whole column at the end (a small
            // hypothesis-testing bias, not a hot-path requirement).
            let u = phi(z);
            if u < 1.0 - param {
                0.0
            } else {
                1.0
            }
        }
        Distribution::RightSkewed => {
            // Standardized, tail-censored Exp(1): skew +1.90, exkurt 4.85, support [-1, +6] SD.
            // E = -ln(Φ(-z)) is monotone-increasing in z, unbounded right tail; censor at
            // EXP_CAP and standardize by the CENSORED Exp(1) moments (exact unit variance —
            // the t3.rs philosophy, NOT an output clamp). Replaces the σ=1 lognormal, whose
            // +104 SD tail made this shape heavier than HighKurtosis. Same Exp(1) inverse-CDF
            // family v1 drew (v1's -ln(Φ(z))), standardized correctly + capped.
            // ln is the owned engine kernel (`ln_owned`, ≤2 ULP of libm); its 2^−11 input
            // floor plus the .min(EXP_CAP) keep the censor total: phi underflow → clamped
            // u → −ln ≥ 7.6 → min returns the cap, so DIST-08 holds with no extra clamp —
            // don't restructure it away. phi's A&S tail error only matters where
            // Φ(−z) ≥ e^{−EXP_CAP} ≈ 9.5e-4 (|z| ≤ 3.1); beyond that the cap censors anyway.
            let e = (-glmm::simd_transcendental::ln_owned(phi(-z))).min(EXP_CAP);
            (e - EXP_CENSORED_MEAN) / EXP_CENSORED_STD
        }
        Distribution::LeftSkewed => {
            // Mirror: -RightSkewed(-z). Heavy tail on the left, monotone-increasing in z.
            let e = (-glmm::simd_transcendental::ln_owned(phi(z))).min(EXP_CAP);
            (EXP_CENSORED_MEAN - e) / EXP_CENSORED_STD
        }
        Distribution::HighKurtosis => {
            let table = spec.t3_table.as_ref()
                .expect("Distribution::HighKurtosis used but t3_table not initialised; run_batch must populate this");
            table.lookup(phi(z))
        }
        Distribution::Uniform => marginal_uniform(z, -SQRT3, SQRT3),
        Distribution::UploadedData => {
            // Gaussian-copula NORTA marginal: map correlated z → uniform u via Φ,
            // then map u → data value via the inverse empirical CDF.
            let u = phi(z).clamp(0.0, 1.0);
            empirical_quantile(spec, var_idx, u)
        }
        Distribution::UploadedBinary => {
            // Probit-threshold marginal: empirical proportion in `param` (set by the
            // adapter from ResampledBinary). 1 in the HIGH-z tail (u ≥ 1 − param) so
            // the marginal is monotone-increasing in z and P(X=1) = param. Uploaded
            // binary is never part of the correlation matrix (correlation is
            // continuous-only by design; binary/factor joint dependence is preserved
            // only by strict-mode resampling), so it is generated independently here.
            let u = phi(z);
            if u < 1.0 - param {
                0.0
            } else {
                1.0
            }
        }
        Distribution::UploadedFactor => {
            // Factor marginal/proportions come from the FactorFromFrame multinomial
            // path, not apply_marginal. Return identity as a no-op fallback.
            z
        }
    }
}

/// Column-wise twin of `apply_marginal`: transforms a whole f64 column in
/// place via the SIMD phi kernel, bit-identical per element to the scalar fn
/// (pinned by `apply_marginal_column_matches_scalar`). The scalar fn is the
/// reproducibility-contract reference; this is the production path. Each arm
/// mirrors the scalar op order exactly; where the scalar calls `phi(-z)`
/// (RightSkewed), the column pass negates first — same value, same bits.
fn apply_marginal_column(
    col: &mut [f64],
    dist_type: Distribution,
    param: f64,
    var_idx: usize,
    spec: &SimulationSpec,
) {
    use glmm::simd_transcendental::phi_fill;
    match dist_type {
        // Caller fast-paths Normal; identity here keeps the dispatch total.
        Distribution::Normal | Distribution::UploadedFactor => {}
        Distribution::Binary | Distribution::UploadedBinary => {
            phi_fill(col);
            for u in col.iter_mut() {
                *u = if *u < 1.0 - param { 0.0 } else { 1.0 };
            }
        }
        Distribution::Uniform => {
            phi_fill(col);
            // Mirror marginal_uniform(z, -SQRT3, SQRT3) = a + (b−a)·Φ op-for-op.
            let (a, b) = (-SQRT3, SQRT3);
            for u in col.iter_mut() {
                *u = a + (b - a) * *u;
            }
        }
        Distribution::RightSkewed => {
            for z in col.iter_mut() {
                *z = -*z;
            }
            phi_fill(col);
            glmm::simd_transcendental::ln_fill(col);
            for u in col.iter_mut() {
                let e = (-*u).min(EXP_CAP);
                *u = (e - EXP_CENSORED_MEAN) / EXP_CENSORED_STD;
            }
        }
        Distribution::LeftSkewed => {
            phi_fill(col);
            glmm::simd_transcendental::ln_fill(col);
            for u in col.iter_mut() {
                let e = (-*u).min(EXP_CAP);
                *u = (EXP_CENSORED_MEAN - e) / EXP_CENSORED_STD;
            }
        }
        Distribution::HighKurtosis => {
            let table = spec.t3_table.as_ref().expect(
                "Distribution::HighKurtosis used but t3_table not initialised; run_batch must populate this",
            );
            phi_fill(col);
            for u in col.iter_mut() {
                *u = table.lookup(*u);
            }
        }
        Distribution::UploadedData => {
            phi_fill(col);
            for u in col.iter_mut() {
                *u = empirical_quantile(spec, var_idx, u.clamp(0.0, 1.0));
            }
        }
    }
}

/// Linear-interpolation empirical quantile (R type-7 / numpy linear style).
///
/// Reads column `col` from `spec.upload_normal` (row-major `(U, n_nf)` table
/// where column `col` holds sorted-ascending standardised uploaded values).
/// Maps `u ∈ [0, 1]` to a value in `[sorted[0], sorted[U-1]]` via linear
/// interpolation at position `pos = u * (U - 1)`.
///
/// Type-7 semantics: identical to `quantile(x, type=7)` in R and
/// `numpy.quantile(x, interpolation="linear")` — no probability-mass adjustment,
/// fractional position linearly interpolated between adjacent order statistics.
fn empirical_quantile(spec: &SimulationSpec, col: usize, u: f64) -> f64 {
    let (u_rows, n_cols) = spec.upload_normal_shape;
    let u_rows = u_rows as usize;
    let n_cols = n_cols as usize;
    if u_rows == 0 {
        return 0.0;
    }
    // Column `col`'s sorted values: upload_normal[r * n_cols + col] for r in 0..u_rows.
    let pos = u * (u_rows - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(u_rows - 1);
    let frac = pos - lo as f64;
    let val_lo = spec.upload_normal[lo * n_cols + col];
    let val_hi = spec.upload_normal[hi * n_cols + col];
    val_lo + frac * (val_hi - val_lo)
}

/// Draw one residual from the per-sim RNG.
///
/// Part of the reproducibility contract (see the `distributions.rs` module
/// docs) — the RNG-draw structure (chi² accumulator count, standardization
/// scale, LeftSkewed sign flip) is golden-pinned by
/// `golden_draw_residual_all_variants`.
///
/// Outcome residuals are filled by the batched pass in `generate_sim_data`
/// (which mirrors this math — change together); this scalar path now serves
/// the per-cluster RE draws and tests only.
fn draw_residual(rng: &mut SimRng, residual_dist: ResidualDist, residual_df: f64) -> f64 {
    match residual_dist {
        ResidualDist::HighKurtosis => {
            let df = residual_df.max(3.0);
            // Standardize so empirical SD ≈ 1 — t(df) has variance df/(df-2).
            let scale = 1.0 / (df / (df - 2.0)).sqrt();
            sample_t(rng, df) * scale
        }
        ResidualDist::RightSkewed | ResidualDist::LeftSkewed => {
            // Chi-squared centered & scaled.
            let df = residual_df.max(3.0);
            let mut chi2 = 0.0;
            let df_int = df.round() as i64;
            for _ in 0..df_int.max(1) {
                let g = rng.next_normal() as f64;
                chi2 += g * g;
            }
            let scale = 1.0 / (2.0 * df).sqrt();
            // LeftSkewed flips the sign so the heavy tail is on the left.
            let centered = (chi2 - df) * scale;
            if matches!(residual_dist, ResidualDist::LeftSkewed) {
                -centered
            } else {
                centered
            }
        }
        ResidualDist::Normal => rng.next_normal() as f64,
        // Unit-SD uniform U(−√3, √3) — mirrors the batched Uniform arm in
        // generate_sim_data; change together.
        ResidualDist::Uniform => (2.0 * rng.next_uniform() as f64 - 1.0) * SQRT3,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{
        CorrectionMethod, CritValues, Distribution, EstimatorSpec, HeteroskedasticityCoeffs,
        OutcomeKind, ResidualDist, ScenarioPerturbations, SimulationSpec,
    };

    fn ols_spec_2x(
        n_factor_dummies: u32,
        factor_n_levels: Vec<i32>,
        factor_props: Vec<f64>,
    ) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 2,
            n_factor_dummies,
            correlation: vec![1.0, 0.3, 0.3, 1.0],
            var_types: vec![Distribution::Normal, Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0, 0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels,
            factor_proportions: factor_props,
            factor_sampled: Vec::new(),
            // β layout: intercept, x1, x2, (optional factor dummies)
            effect_sizes: {
                let mut v = vec![0.0, 0.5, 0.3];
                v.extend(std::iter::repeat_n(0.0, n_factor_dummies as usize));
                v
            },
            target_indices: vec![1, 2],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    fn ols_spec_simple() -> SimulationSpec {
        ols_spec_2x(0, vec![], vec![])
    }

    // ------------------------------------------------------------------
    // fixed_allocation — deterministic largest-remainder walk (R1 + R2)
    // and the factor-block branch.
    // ------------------------------------------------------------------

    #[test]
    fn fixed_walk_exact_counts_at_every_prefix() {
        // R1: at every prefix N, Σ counts == N and each count is within 1 of
        // N·p_g; at N = 100 a 70/30 request is exactly [70, 30].
        let probs = [0.7, 0.3];
        let mut counts = [0u32; 2];
        for n in 1..=100usize {
            fixed_level_next(&probs, &mut counts);
            let total: u32 = counts.iter().sum();
            assert_eq!(total as usize, n, "counts must sum to N at every prefix");
            for (g, &c) in counts.iter().enumerate() {
                assert!(
                    (c as f64 - n as f64 * probs[g]).abs() <= 1.0 + 1e-12,
                    "count[{g}]={c} too far from {n}·{}",
                    probs[g]
                );
            }
        }
        assert_eq!(counts, [70, 30]);
    }

    #[test]
    fn fixed_walk_equal_props_reduce_to_round_robin() {
        // Ties break to the lowest level index, so equal proportions are
        // exactly round-robin i % k — the same tie-break rule used by
        // ClusterSizing::cluster_of_row.
        let probs = [1.0 / 3.0; 3];
        let mut counts = [0u32; 3];
        for i in 0..30usize {
            let lvl = fixed_level_next(&probs, &mut counts);
            assert_eq!(lvl, i % 3, "row {i}");
        }
    }

    #[test]
    fn fixed_walk_normalises_weights_and_permits_empty_cells() {
        // Unnormalised proportions use normalised weights; a level with
        // N·p_g < 0.5 receives zero rows at small N (correct apportionment).
        let probs = [7.0, 3.0]; // sums to 10, behaves as 0.7/0.3
        let mut counts = [0u32; 2];
        for _ in 0..100 {
            fixed_level_next(&probs, &mut counts);
        }
        assert_eq!(counts, [70, 30]);

        let probs = [0.9, 0.1];
        let mut counts = [0u32; 2];
        for _ in 0..4 {
            fixed_level_next(&probs, &mut counts);
        }
        assert_eq!(counts, [4, 0], "minority level is empty until N·p_g grows");
    }

    #[test]
    fn fixed_allocation_counts_matches_row_loop_walk() {
        // Same probs/N as the existing minority-empty test: [0.9, 0.1] at N=4 → [4, 0].
        assert_eq!(fixed_allocation_counts(&[0.9, 0.1], 4), vec![4, 0]);
        // Within-1 of N·p at a healthy N.
        let c = fixed_allocation_counts(&[0.7, 0.1, 0.1, 0.1], 40);
        assert_eq!(c.iter().sum::<u32>(), 40);
        assert_eq!(c, vec![28, 4, 4, 4]);
    }

    #[test]
    fn min_inclusion_n_finds_threshold_crossing() {
        // p_min = 0.1, k_min = 5 → ceil-ish 50 region; exact via the walk.
        let n = min_inclusion_n(&[0.9, 0.1], 5, 200).unwrap();
        let c = fixed_allocation_counts(&[0.9, 0.1], n);
        assert!(c.iter().all(|&x| x >= 5));
        let c_prev = fixed_allocation_counts(&[0.9, 0.1], n - 1);
        assert!(c_prev.iter().any(|&x| x < 5));
        assert_eq!(min_inclusion_n(&[0.99, 0.01], 5, 100), None);
    }

    fn factor_spec_70_30() -> SimulationSpec {
        // 2 continuous + one 2-level factor (70/30). Default scenario ⇒
        // sampled_factor_proportions == false.
        ols_spec_2x(1, vec![2], vec![0.7, 0.3])
    }

    #[test]
    fn fixed_allocation_design_counts_exact_and_constant_across_sims() {
        // The requested design IS the simulated design — level-1 count is
        // exactly N·p on every sim, and the whole allocation pattern is
        // identical across sims (deterministic X′X for pure-factor designs).
        let spec = factor_spec_70_30();
        let n = 100;
        let n_pred = 4;
        let mut ws = SimWorkspace::new(n, n_pred, 2, 1, None);
        let mut first: Option<Vec<f32>> = None;
        for sim_id in 0..5u64 {
            generate_sim_data(&spec, sim_id, 42, &mut ws).unwrap();
            let col: Vec<f32> = (0..n).map(|i| ws.x_full[(i, 3)]).collect();
            let ones: f32 = col.iter().sum();
            assert_eq!(
                ones, 30.0_f32,
                "level-1 count must be exactly N·p at sim {sim_id}"
            );
            match &first {
                None => first = Some(col),
                Some(prev) => assert_eq!(prev, &col, "allocation must be identical across sims"),
            }
        }
    }

    #[test]
    fn fixed_allocation_prefix_stable_across_max_n() {
        // R2 (hard invariant): X_full[:N] is a nested prefix across the grid
        // in fixed mode — factor, continuous, and y all bit-identical.
        let spec = factor_spec_70_30();
        let n_pred = 4;
        let mut ws_big = SimWorkspace::new(1000, n_pred, 2, 1, None);
        let mut ws_small = SimWorkspace::new(200, n_pred, 2, 1, None);
        generate_sim_data(&spec, 5, 42, &mut ws_big).unwrap();
        generate_sim_data(&spec, 5, 42, &mut ws_small).unwrap();
        for i in 0..200 {
            for j in 0..n_pred {
                assert_eq!(
                    ws_big.x_full[(i, j)],
                    ws_small.x_full[(i, j)],
                    "x_full[{i},{j}] differs across max_n"
                );
            }
            assert_eq!(ws_big.y_full[i], ws_small.y_full[i], "y differs at i={i}");
        }
    }

    #[test]
    fn fixed_vs_sampled_first_row_continuous_byte_identical() {
        // All draws share one sequential per-sim stream, and within a row
        // the continuous block draws BEFORE the factor block — so row 0's
        // continuous values are byte-identical across modes (the fixed branch
        // consumes no RNG before them). From row 1 onward the missing
        // categorical draw shifts the stream, so later rows' continuous
        // values and all residuals diverge — deliberately not compared.
        let spec_fixed = factor_spec_70_30();
        let mut spec_sampled = factor_spec_70_30();
        spec_sampled.scenario.sampled_factor_proportions = true;
        let n = 100;
        let n_pred = 4;
        let mut ws_f = SimWorkspace::new(n, n_pred, 2, 1, None);
        let mut ws_s = SimWorkspace::new(n, n_pred, 2, 1, None);
        generate_sim_data(&spec_fixed, 3, 42, &mut ws_f).unwrap();
        generate_sim_data(&spec_sampled, 3, 42, &mut ws_s).unwrap();
        for j in 0..=2 {
            assert_eq!(
                ws_f.x_full[(0, j)],
                ws_s.x_full[(0, j)],
                "row-0 continuous x[0,{j}] must match across modes"
            );
        }
    }

    #[test]
    fn sampled_mode_retains_count_jitter() {
        // sampled_factor_proportions == true keeps the per-row categorical draw:
        // realized counts vary across sims (Multinomial jitter retained as an
        // explicit realism perturbation). Deterministic under the fixed seed.
        let mut spec = factor_spec_70_30();
        spec.scenario.sampled_factor_proportions = true;
        let n = 100;
        let n_pred = 4;
        let mut ws = SimWorkspace::new(n, n_pred, 2, 1, None);
        let mut counts = Vec::new();
        for sim_id in 0..20u64 {
            generate_sim_data(&spec, sim_id, 42, &mut ws).unwrap();
            let ones: f32 = (0..n).map(|i| ws.x_full[(i, 3)]).sum();
            counts.push(ones as u32);
        }
        assert!(
            counts.iter().any(|&c| c != counts[0]),
            "sampled mode must retain count jitter, got constant {counts:?}"
        );
    }

    fn factor_spec_2factors() -> SimulationSpec {
        // Two 2-level factors (70/30 and 50/50). n_factor_dummies = 2.
        // Column layout: intercept(0) | x1(1) | x2(2) | f0_dummy(3) | f1_dummy(4)
        ols_spec_2x(2, vec![2, 2], vec![0.7, 0.3, 0.5, 0.5])
    }

    #[test]
    fn mixed_mode_exact_factor_constant_sampled_factor_jitters() {
        // Factor 0 forced EXACT (Some(false)), factor 1 forced SAMPLED (Some(true)).
        // Scenario default = false (so f1's override is the active change).
        // Dummy columns: col 3 = factor 0 (70/30), col 4 = factor 1 (50/50).
        let mut spec = factor_spec_2factors();
        spec.factor_sampled = vec![Some(false), Some(true)];
        let n = 100;
        let n_pred = 5; // intercept + 2 continuous + 2 factor dummies
        let mut ws = SimWorkspace::new(n, n_pred, 2, 1, None);

        let mut f0_first: Option<Vec<f32>> = None;
        let mut f1_counts = Vec::new();
        for sim_id in 0..20u64 {
            generate_sim_data(&spec, sim_id, 42, &mut ws).unwrap();
            let f0: Vec<f32> = (0..n).map(|i| ws.x_full[(i, 3)]).collect();
            let f1_ones: f32 = (0..n).map(|i| ws.x_full[(i, 4)]).sum();
            assert_eq!(
                f0.iter().sum::<f32>(),
                30.0,
                "exact factor count at sim {sim_id}"
            );
            match &f0_first {
                None => f0_first = Some(f0),
                Some(prev) => assert_eq!(prev, &f0, "exact factor must not jitter"),
            }
            f1_counts.push(f1_ones);
        }
        let f1_min = f1_counts.iter().cloned().fold(f32::INFINITY, f32::min);
        let f1_max = f1_counts.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(f1_max > f1_min, "sampled factor must jitter across sims");
    }

    #[test]
    fn per_factor_override_beats_scenario_default() {
        // Scenario default = SAMPLED, but factor 0 overridden EXACT (Some(false))
        // must stay constant; factor 1 inherits (None → sampled).
        let mut spec = factor_spec_2factors();
        spec.scenario.sampled_factor_proportions = true;
        spec.factor_sampled = vec![Some(false), None];
        let n = 100;
        let n_pred = 5;
        let mut ws = SimWorkspace::new(n, n_pred, 2, 1, None);
        let mut f0_first: Option<Vec<f32>> = None;
        for sim_id in 0..10u64 {
            generate_sim_data(&spec, sim_id, 42, &mut ws).unwrap();
            let f0: Vec<f32> = (0..n).map(|i| ws.x_full[(i, 3)]).collect();
            assert_eq!(f0.iter().sum::<f32>(), 30.0);
            match &f0_first {
                None => f0_first = Some(f0),
                Some(prev) => assert_eq!(prev, &f0, "overridden-exact factor must not jitter"),
            }
        }
    }

    #[test]
    fn fixed_allocation_leaves_binary_continuous_stochastic() {
        // Distribution::Binary goes through the continuous block
        // (threshold marginal), NOT the factor block — it keeps stochastic
        // per-row values under fixed allocation (multi-threshold extension is out of
        // scope). Deterministic under the fixed seed.
        let mut spec = factor_spec_70_30();
        spec.var_types = vec![Distribution::Normal, Distribution::Binary];
        spec.var_params = vec![0.0, 0.5];
        let n = 100;
        let n_pred = 4;
        let mut ws = SimWorkspace::new(n, n_pred, 2, 1, None);
        let mut sums = Vec::new();
        for sim_id in 0..20u64 {
            generate_sim_data(&spec, sim_id, 42, &mut ws).unwrap();
            let s: f32 = (0..n).map(|i| ws.x_full[(i, 2)]).sum();
            sums.push(s);
        }
        assert!(
            sums.iter().any(|&s| s != sums[0]),
            "binary continuous column must stay stochastic under fixed allocation"
        );
    }

    #[test]
    fn clustered_ols_generates_without_error() {
        // Continuous + cluster=Some + Ols is the free pairing — must succeed
        // AND produce a well-shaped design: x_full is (n, 1+n_non_factor),
        // y_full has length n, and the intercept column is exactly 1.0.
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = ols_spec_simple();
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 10 },
            tau_squared: 0.1,
            slopes: vec![],
            extra_groupings: vec![],
        });
        let n = 50;
        let n_predictors = 3;
        let mut ws = SimWorkspace::new(
            n,
            n_predictors,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters: 10 },
                0.25,
            )),
        );
        generate_sim_data(&spec, 0, 42, &mut ws)
            .expect("clustered OLS should generate without error");
        assert_eq!(ws.x_full.nrows(), n);
        assert_eq!(ws.x_full.ncols(), n_predictors);
        for i in 0..n {
            assert_eq!(
                ws.x_full[(i, 0)],
                1.0,
                "intercept column must be 1.0 at row {i}"
            );
            assert!(ws.y_full[i].is_finite(), "y[{i}] must be finite");
        }
    }
    #[test]
    fn binary_cluster_icc_noise_sd_jitter_is_live() {
        // Settled decision (05b): icc_noise_sd jitters τ² in raw τ²-space once per
        // sim for every outcome kind including Binary. Pins that the jitter path
        // is reached on the binary+cluster (GLMM) arm — the realized per-sim τ²
        // (ws.tau_squared_design) moves between sims. Asymmetry note (docs Phase 5):
        // the knob operates on the latent (log-odds) τ², not the probability scale.
        //
        // Base τ²=1.0 with icc_noise_sd=0.3 keeps both sims' jittered τ² clear of
        // the ≥0 clamp, so the "sims differ" assertion is deterministic (two
        // independent per-sim normal draws added to 1.0, neither clamped to 0).
        use crate::spec::{ClusterSizing, ClusterSpec, LmeScenarioPerturbations};
        let base_tau_sq = 1.0;
        let mut spec = ols_spec_simple();
        spec.outcome_kind = OutcomeKind::Binary;
        spec.estimator = EstimatorSpec::Glm;
        spec.intercept = 0.0; // logit(0.5) — balanced baseline ⇒ non-degenerate outcomes
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 10 },
            tau_squared: base_tau_sq,
            slopes: vec![],
            extra_groupings: vec![],
        });
        spec.scenario.lme = Some(LmeScenarioPerturbations {
            random_effect_dist: ResidualDist::Normal,
            random_effect_df: 0.0,
            icc_noise_sd: 0.3,
        });

        let n = 200;
        let n_predictors = 3;
        let mut ws = SimWorkspace::new(
            n,
            n_predictors,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters: 10 },
                base_tau_sq,
            )),
        );

        // sim 0 and sim 1 — distinct sim_id ⇒ distinct scenario RNG ⇒
        // independent jitter draws.
        let realized_tau = |ws: &SimWorkspace| ws.tau_squared_design;
        let outcome_sum = |ws: &SimWorkspace| (0..n).map(|i| ws.y_full[i]).sum::<f32>();

        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();
        let tau0 = realized_tau(&ws);
        let sum0 = outcome_sum(&ws);

        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        let tau1 = realized_tau(&ws);
        let sum1 = outcome_sum(&ws);

        // (a) both realized per-sim τ² finite and ≥ 0
        assert!(
            tau0.is_finite() && tau0 >= 0.0,
            "tau0={tau0} must be finite ≥0"
        );
        assert!(
            tau1.is_finite() && tau1 >= 0.0,
            "tau1={tau1} must be finite ≥0"
        );
        // (b) the jitter is live: the two sims' realized τ² differ
        assert!(
            (tau0 - tau1).abs() > 1e-9,
            "icc_noise_sd jitter must move per-sim τ² on the binary path: tau0={tau0} tau1={tau1}"
        );
        // (c) binary outcomes are non-degenerate (not all-0 / all-1) in both sims
        for (b, s) in [(0u64, sum0), (1u64, sum1)] {
            assert!(
                s > 0.0 && s < n as f32,
                "binary outcomes degenerate at sim {b}: sum={s} (n={n})"
            );
        }
    }

    // Note: lme_without_cluster_spec_rejected is removed — that rule now lives
    // in the contract's validate(), and the kernel guard it asserted is deleted.

    // `lme_intercept_clusters_drawn_per_block` deleted: its only assertion was
    // an empirical ICC band (|icc_est − 0.5| < 0.1 over 2000 sims) — smuggled
    // statistical correctness (statistical-band tests belong at L3 only). The
    // deterministic mechanic (cluster draws precede the row loop / are added to y)
    // is covered by `cluster_draws_precede_row_loop` (DGEN-08); empirical ICC
    // consistency (DGEN-09) is L3-only.

    #[test]
    fn d1_step2_prefix_stable_across_max_n() {
        // Generate at max_n=1000 and at max_n=200. The first 200 rows must be
        // bitwise identical for X and y (per the row-stable RNG guarantee).
        let spec = ols_spec_simple();
        let n_pred = 3;
        let mut ws_big = SimWorkspace::new(1000, n_pred, 2, 0, None);
        let mut ws_small = SimWorkspace::new(200, n_pred, 2, 0, None);
        generate_sim_data(&spec, 5, 42, &mut ws_big).unwrap();
        generate_sim_data(&spec, 5, 42, &mut ws_small).unwrap();

        for i in 0..200 {
            for j in 0..n_pred {
                let big = ws_big.x_full[(i, j)];
                let small = ws_small.x_full[(i, j)];
                assert_eq!(
                    big, small,
                    "x_full[{i},{j}] differs: big={big}, small={small}"
                );
            }
            assert_eq!(ws_big.y_full[i], ws_small.y_full[i], "y differs at i={i}");
        }
    }

    // `d1_step3_marginal_sanity_normal_columns` and `d1_step3_correlation_preserved`
    // deleted: both asserted only statistical bands over 5000 simulated draws
    // (normal column mean≈0/var≈1; sample correlation r≈0.3) — smuggled
    // statistical correctness (statistical-band tests belong at L3 only). The
    // deterministic mechanics survive elsewhere: intercept column == 1.0 is
    // asserted by `clustered_ols_generates_without_error` (DGEN-04); the Normal
    // marginal-identity by `apply_marginal_normal_is_identity` (DIST-06); and the
    // Cholesky design being bit-identical across runs by
    // `d1_step4_optimistic_no_perturbation`.

    #[test]
    fn d1_step4_optimistic_no_perturbation() {
        // With ScenarioPerturbations::default(), the design must match a hand-built
        // baseline: Cholesky of spec.correlation, original var_types, original
        // residual_dist. Run twice and ensure X is bit-identical.
        let spec = ols_spec_simple();
        let n_pred = 3;
        let mut ws_a = SimWorkspace::new(100, n_pred, 2, 0, None);
        let mut ws_b = SimWorkspace::new(100, n_pred, 2, 0, None);
        generate_sim_data(&spec, 0, 42, &mut ws_a).unwrap();
        generate_sim_data(&spec, 0, 42, &mut ws_b).unwrap();
        for i in 0..100 {
            for j in 0..n_pred {
                assert_eq!(ws_a.x_full[(i, j)], ws_b.x_full[(i, j)]);
            }
        }
        assert_eq!(
            ws_a.evd_scratch.evd_repair_count, 0,
            "optimistic path must not touch EVD"
        );
    }

    #[test]
    fn d1_step5_scenario_stream_independence() {
        // High-noise scenario: two runs at same (base_seed, sim_id) produce identical X.
        let mut spec = ols_spec_simple();
        spec.scenario = ScenarioPerturbations {
            name: "realistic".into(),
            correlation_noise_sd: 0.2,
            distribution_change_prob: 0.0,
            ..Default::default()
        };
        let n_pred = 3;
        let mut ws_a = SimWorkspace::new(100, n_pred, 2, 0, None);
        let mut ws_b = SimWorkspace::new(100, n_pred, 2, 0, None);
        generate_sim_data(&spec, 7, 42, &mut ws_a).unwrap();
        generate_sim_data(&spec, 7, 42, &mut ws_b).unwrap();
        for i in 0..100 {
            for j in 0..n_pred {
                assert_eq!(ws_a.x_full[(i, j)], ws_b.x_full[(i, j)]);
            }
        }

        // Different scenario, same data-seed sim_id → X differs.
        let mut spec2 = ols_spec_simple();
        spec2.scenario = ScenarioPerturbations {
            name: "doomer".into(),
            correlation_noise_sd: 0.4,
            ..Default::default()
        };
        let mut ws_c = SimWorkspace::new(100, n_pred, 2, 0, None);
        generate_sim_data(&spec2, 7, 42, &mut ws_c).unwrap();
        let mut differed = false;
        for i in 0..100 {
            for j in 0..n_pred {
                if ws_a.x_full[(i, j)] != ws_c.x_full[(i, j)] {
                    differed = true;
                    break;
                }
            }
            if differed {
                break;
            }
        }
        assert!(differed, "different scenarios should produce different X");
    }

    #[test]
    fn d1_factor_dummy_layout_3_levels() {
        // Factor with 3 levels and proportions [0.5, 0.3, 0.2] → 2 dummy columns.
        let mut spec = ols_spec_2x(2, vec![3], vec![0.5, 0.3, 0.2]);
        spec.effect_sizes = vec![0.0, 0.5, 0.3, 0.0, 0.0];

        let n_pred = 5;
        let n = 2000;
        let mut ws = SimWorkspace::new(n, n_pred, 2, 1, None);
        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();

        // Deterministic reference-coding structure (DGEN-05): each dummy is 0/1,
        // every row is at most one-hot (reference level → both zero). The realized
        // level *frequencies* are a sampling property (statistical, L3 seed) and are
        // intentionally not asserted here — at least one row per level confirms the
        // levels are reachable without pinning a proportion band.
        let mut seen = [false; 3];
        for i in 0..n {
            let d0 = ws.x_full[(i, 3)];
            let d1 = ws.x_full[(i, 4)];
            assert!(d0 == 0.0 || d0 == 1.0);
            assert!(d1 == 0.0 || d1 == 1.0);
            let sum = d0 + d1;
            assert!(sum == 0.0 || sum == 1.0, "row {i}: d0={d0}, d1={d1}");
            if sum == 0.0 {
                seen[0] = true;
            } else if d0 == 1.0 {
                seen[1] = true;
            } else {
                seen[2] = true;
            }
        }
        assert!(
            seen.iter().all(|&s| s),
            "every factor level (incl. reference) must be reachable: {seen:?}"
        );
    }

    // -----------------------------------------------------------------
    // Binary outcome arm tests
    // -----------------------------------------------------------------

    fn logit_spec_simple(intercept: f64, beta1: f64) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: 1,
            n_factor_dummies: 0,
            correlation: vec![1.0],
            var_types: vec![Distribution::Normal],
            var_pinned: vec![],
            var_params: vec![0.0],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![intercept, beta1],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Binary,
            estimator: EstimatorSpec::Glm,
            wald_se: Default::default(),
            intercept,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    #[test]
    fn logit_y_is_binary() {
        let spec = logit_spec_simple(0.0, 0.5);
        let n = 5000;
        let mut ws = SimWorkspace::new(n, 2, 1, 0, None);
        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        for i in 0..n {
            let y = ws.y_full[i];
            assert!(y == 0.0 || y == 1.0, "y[{i}] = {y} is not binary");
        }
    }

    #[test]
    fn logit_prefix_stable_across_max_n() {
        // First 200 rows at max_n=1000 must equal the 200 rows at max_n=200
        // bit-for-bit (both X and y).
        let spec = logit_spec_simple(0.0, 0.5);
        let n_pred = 2;
        let mut ws_big = SimWorkspace::new(1000, n_pred, 1, 0, None);
        let mut ws_small = SimWorkspace::new(200, n_pred, 1, 0, None);
        generate_sim_data(&spec, 5, 42, &mut ws_big).unwrap();
        generate_sim_data(&spec, 5, 42, &mut ws_small).unwrap();

        for i in 0..200 {
            for j in 0..n_pred {
                let big = ws_big.x_full[(i, j)];
                let small = ws_small.x_full[(i, j)];
                assert_eq!(
                    big, small,
                    "logit x_full[{i},{j}] differs: big={big}, small={small}"
                );
            }
            assert_eq!(
                ws_big.y_full[i], ws_small.y_full[i],
                "logit y differs at i={i}: big={}, small={}",
                ws_big.y_full[i], ws_small.y_full[i]
            );
        }
    }

    #[test]
    fn logit_predictor_increases_probability() {
        // intercept = 0, β₁ = 2 → strong monotone link: large positive x ⇒ p ≈ 1.
        let spec = logit_spec_simple(0.0, 2.0);
        let n = 5000;
        let mut ws = SimWorkspace::new(n, 2, 1, 0, None);
        generate_sim_data(&spec, 1, 42, &mut ws).unwrap();
        let mut pos_y = 0.0_f64;
        let mut pos_n = 0usize;
        let mut neg_y = 0.0_f64;
        let mut neg_n = 0usize;
        for i in 0..n {
            let x1 = ws.x_full[(i, 1)];
            let y = ws.y_full[i] as f64;
            if x1 > 0.0 {
                pos_y += y;
                pos_n += 1;
            } else if x1 < 0.0 {
                neg_y += y;
                neg_n += 1;
            }
        }
        let p_pos = pos_y / pos_n as f64;
        let p_neg = neg_y / neg_n as f64;
        assert!(
            p_pos > 0.7,
            "rows with x>0 should have p>0.7, got {p_pos} ({pos_n} rows)"
        );
        assert!(
            p_neg < 0.3,
            "rows with x<0 should have p<0.3, got {p_neg} ({neg_n} rows)"
        );
    }

    // -----------------------------------------------------------------
    // Marginal / residual transform property tests (DIST-06/08/10/11)
    // -----------------------------------------------------------------

    /// DIST-06: `apply_marginal` with `Distribution::Normal` is the identity
    /// transform — the input z is returned unchanged for any finite z.
    #[test]
    fn apply_marginal_normal_is_identity() {
        let spec = ols_spec_simple();
        for z in [-3.5_f64, -1.0, 0.0, 0.4, 2.7, 8.0] {
            let out = apply_marginal(z, Distribution::Normal, 0.0, 0, &spec);
            assert_eq!(out, z, "Normal marginal must be identity at z={z}");
        }
    }

    /// DIST-08: `apply_marginal` with Right/LeftSkewed stays finite for any
    /// finite z: when Φ underflows to 0 the raw deviate is −ln(0) = +∞, and
    /// the `.min(EXP_CAP)` censor maps it to the cap (f64::min drops the ∞ arm).
    #[test]
    fn apply_marginal_skewed_is_finite() {
        let spec = ols_spec_simple();
        for z in [-50.0_f64, -8.0, -1.0, 0.0, 1.0, 8.0, 50.0] {
            for dist in [Distribution::RightSkewed, Distribution::LeftSkewed] {
                let out = apply_marginal(z, dist, 0.0, 0, &spec);
                assert!(
                    out.is_finite(),
                    "{dist:?} marginal must be finite at z={z}, got {out}"
                );
            }
        }
    }

    /// DIST-10: `draw_residual` for HighKurtosis enforces df ≥ 3 — a
    /// requested df below 3 must produce the *same* draw as df == 3 on a
    /// shared RNG state (the floor clamp). A broken kernel that used the raw
    /// df would diverge.
    #[test]
    fn draw_residual_t_enforces_df_floor() {
        let mut rng_low = SimRng::new(42, 1);
        let mut rng_floor = SimRng::new(42, 1);
        let low = draw_residual(&mut rng_low, ResidualDist::HighKurtosis, 2.0); // clamps to 3
        let floor = draw_residual(&mut rng_floor, ResidualDist::HighKurtosis, 3.0);
        assert_eq!(
            low, floor,
            "df<3 must clamp to df=3 (identical draw on shared seed)"
        );
        assert!(low.is_finite(), "standardized t draw must be finite");
    }

    /// DIST-11: `draw_residual` LeftSkewed is the negation of RightSkewed from
    /// the *same* RNG state — only the sign is flipped.
    #[test]
    fn draw_residual_left_is_negated_right() {
        let mut rng_right = SimRng::new(7, 3);
        let mut rng_left = SimRng::new(7, 3);
        let right = draw_residual(&mut rng_right, ResidualDist::RightSkewed, 8.0);
        let left = draw_residual(&mut rng_left, ResidualDist::LeftSkewed, 8.0);
        assert_eq!(
            left, -right,
            "LeftSkewed must be −RightSkewed on identical RNG state"
        );
    }

    #[test]
    fn interaction_column_is_elementwise_product() {
        // [intercept, x1, x2, x1:x2]; interactions[0] = product of cols 1,2.
        // ols_spec_2x gives n_non_factor=2, n_factor_dummies=0, effect_sizes=[0.0,0.5,0.3].
        // We extend to 4 columns by appending one interaction and its effect size.
        let mut spec = ols_spec_2x(0, vec![], vec![]);
        spec.effect_sizes = vec![0.0, 0.5, 0.3, 0.0]; // 4 columns now
        spec.interactions = vec![vec![1, 2]];
        spec.target_indices = vec![1, 2, 3];

        let n = 256;
        let n_pred = 4; // intercept + x1 + x2 + x1:x2
                        // SimWorkspace::new(n_rows, n_predictors, n_non_factor, n_factors, cluster)
        let mut ws = SimWorkspace::new(n, n_pred, 2, 0, None);
        // generate_sim_data(spec, sim_id, base_seed, ws)
        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();

        assert_eq!(ws.x_full.ncols(), 4);
        for i in 0..n {
            assert_eq!(ws.x_full[(i, 0)], 1.0, "intercept untouched at row {i}");
            let prod = ws.x_full[(i, 1)] * ws.x_full[(i, 2)];
            // f32 arithmetic: tolerance loosened from 1e-12 to 1e-6.
            assert!(
                (ws.x_full[(i, 3)] - prod).abs() < 1e-6,
                "row {i}: interaction col {} != x1*x2 {}",
                ws.x_full[(i, 3)],
                prod
            );
        }
    }

    /// Right/LeftSkewed marginals are STANDARDIZED tail-censored Exp(1):
    /// mean 0, var 1, skew ±1.90, excess kurtosis 4.85, support [−1, +6] SD
    /// (right; mirrored for left), checked by numerical integration against
    /// the N(0,1) latent density (no RNG). Guards the centring/scale of the
    /// swap pool — a mis-centred or mis-scaled candidate leaks a mean shift /
    /// effect shrink into every distribution_change_prob scenario — and the
    /// 6-SD bound (an output clamp instead of latent censoring would shrink
    /// the variance ~12% silently).
    #[test]
    fn skew_marginals_are_standardized() {
        let spec = ols_spec_simple();
        let h = 1e-3_f64;
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let (mut m0, mut m1, mut m2, mut m3, mut m4) = (0.0_f64, 0.0, 0.0, 0.0, 0.0);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut z = -10.0_f64;
        while z < 10.0 {
            let zm = z + 0.5 * h; // midpoint rule
            let w = inv_sqrt_2pi * (-0.5 * zm * zm).exp() * h;
            let t = apply_marginal(zm, Distribution::RightSkewed, 0.0, 0, &spec);
            // Mirror identity: LeftSkewed(z) is exactly −RightSkewed(−z).
            let t_left = apply_marginal(-zm, Distribution::LeftSkewed, 0.0, 0, &spec);
            assert_eq!(t_left, -t, "left(−z) must equal −right(z) at z={zm}");
            m0 += w;
            m1 += w * t;
            m2 += w * t * t;
            m3 += w * t * t * t;
            m4 += w * t * t * t * t;
            lo = lo.min(t);
            hi = hi.max(t);
            z += h;
        }
        assert!((m0 - 1.0).abs() < 1e-6, "total weight {m0}");
        assert!(m1.abs() < 1e-4, "mean {m1}");
        assert!((m2 - 1.0).abs() < 1e-3, "var {m2}");
        // mean≈0, var≈1 ⇒ third/fourth raw moments ≈ skew / kurtosis.
        assert!((m3 - 1.8995).abs() < 0.02, "skew {m3}");
        assert!((m4 - 3.0 - 4.8457).abs() < 0.05, "exkurt {}", m4 - 3.0);
        // Support: censored at exactly +6 SD; left edge is (0 − mean)/sd ≈ −1.006.
        assert!((hi - 6.0).abs() < 1e-9, "max {hi}, want +6.0 SD");
        assert!((lo + 1.0057).abs() < 1e-3, "min {lo}, want ≈ −1.006 SD");
    }

    /// HighKurtosis marginal is STANDARDIZED censored t(3): mean 0, var 1,
    /// excess kurtosis ≈ 6.39 (the censored 2048-knot table's own law — full
    /// t(3) kurtosis is infinite), checked by numerical integration against
    /// the N(0,1) latent density (no RNG). Guards the build-time normalization
    /// in `T3PpfTable::build` — the v1-parity /√3 scaling standardized the
    /// uncensored t(3) and left this marginal at var ≈ 0.858.
    #[test]
    fn high_kurtosis_marginal_is_standardized() {
        use crate::marginals::T3PpfTable;
        let mut spec = ols_spec_simple();
        spec.t3_table = Some(T3PpfTable::build_default());
        let h = 1e-3_f64;
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let (mut m0, mut m1, mut m2, mut m4) = (0.0_f64, 0.0, 0.0, 0.0);
        let mut z = -10.0_f64;
        while z < 10.0 {
            let zm = z + 0.5 * h; // midpoint rule
            let w = inv_sqrt_2pi * (-0.5 * zm * zm).exp() * h;
            let t = apply_marginal(zm, Distribution::HighKurtosis, 0.0, 0, &spec);
            m0 += w;
            m1 += w * t;
            m2 += w * t * t;
            m4 += w * t * t * t * t;
            z += h;
        }
        assert!((m0 - 1.0).abs() < 1e-6, "total weight {m0}");
        assert!(m1.abs() < 1e-4, "mean {m1}");
        assert!((m2 - 1.0).abs() < 1e-3, "var {m2}");
        // mean≈0, var≈1 ⇒ the fourth raw moment − 3 ≈ excess kurtosis.
        assert!((m4 - 3.0 - 6.39).abs() < 0.1, "exkurt {}", m4 - 3.0);
    }

    // -----------------------------------------------------------------
    // Golden bit patterns — reproducibility contract. These are the private
    // half of the golden surface; the public half (SimRng, scenario_rng, phi,
    // marginal_uniform, sample_t) lives in tests/golden_rng.rs. A failure
    // means a result-moving change: it must land in all ports simultaneously
    // and carry a version bump, never as a silent improvement. If deliberate,
    // regenerate the constants from the new stream and bump the version.
    // -----------------------------------------------------------------

    /// Golden bits for `apply_marginal` on a fixed z-grid, one row per
    /// `Distribution` variant. Pins the polynomial constants, the clamp
    /// bounds, the t(3) PPF table, and the empirical-quantile interpolation.
    #[test]
    fn golden_apply_marginal_all_variants() {
        use crate::marginals::T3PpfTable;
        let mut spec = ols_spec_simple();
        spec.t3_table = Some(T3PpfTable::build_default());
        // Fabricated sorted-ascending standardized upload table, shape (5, 2);
        // var_idx = 1 below reads column 1: [-2, -1, 0, 1, 2].
        spec.upload_normal = vec![
            -1.5, -2.0, //
            -0.5, -1.0, //
            0.0, 0.0, //
            0.5, 1.0, //
            1.5, 2.0,
        ];
        spec.upload_normal_shape = (5, 2);

        const Z_GRID: [f64; 7] = [-8.0, -3.0, -1.0, 0.0, 1.0, 3.0, 8.0];
        #[rustfmt::skip]
        const GOLDEN: [(Distribution, f64, [u64; 7]); 9] = [
            (
                Distribution::Normal,
                0.0,
                [
                    0xc020000000000000, 0xc008000000000000, 0xbff0000000000000,
                    0x0000000000000000, 0x3ff0000000000000, 0x4008000000000000,
                    0x4020000000000000,
                ],
            ),
            (
                Distribution::Binary,
                0.3,
                [
                    0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
                    0x0000000000000000, 0x3ff0000000000000, 0x3ff0000000000000,
                    0x3ff0000000000000,
                ],
            ),
            (
                Distribution::RightSkewed,
                0.0,
                [
                    0xbff0176e624ce734, 0xbff011dc6d48e547, 0xbfea9e36eba39761,
                    0xbfd3b560b7c0d86b, 0x3feb1f7b370920a9, 0x40169593c77fa84a,
                    0x4018000000000001,
                ],
            ),
            (
                Distribution::LeftSkewed,
                0.0,
                [
                    0xc018000000000001, 0xc0169593c77fa84a, 0xbfeb1f7b370920a9,
                    0x3fd3b560b7c0d86b, 0x3fea9e36eba39761, 0x3ff011dc6d48e547,
                    0x3ff0176e624ce734,
                ],
            ),
            (
                Distribution::HighKurtosis,
                0.0,
                [
                    0xc017fd51eba89eef, 0xc0173bf17001a661, 0xbfe8000bfda55b32,
                    0xbe0d4a3598980000, 0x3fe8000bfda55b37, 0x40173bf17001a74e,
                    0x4017fd51eba89eef,
                ],
            ),
            (
                Distribution::Uniform,
                0.0,
                [
                    0xbffbb67ae8584ca0, 0xbffba35352610c22, 0xbff2eb53ae7dfbdc,
                    0xbe1dc1a3c0000000, 0x3ff2eb53ae7dfbdc, 0x3ffba35352610c22,
                    0x3ffbb67ae8584ca0,
                ],
            ),
            (
                Distribution::UploadedData,
                0.0,
                [
                    0xbffffffffffffff5, 0xbfffe9e1d3ab44f2, 0xbff5d89797a01c9d,
                    0xbe212e0be0000000, 0x3ff5d89797a01c9c, 0x3fffe9e1d3ab44f2,
                    0x3ffffffffffffff4,
                ],
            ),
            (
                Distribution::UploadedBinary,
                0.4,
                [
                    0x0000000000000000, 0x0000000000000000, 0x0000000000000000,
                    0x0000000000000000, 0x3ff0000000000000, 0x3ff0000000000000,
                    0x3ff0000000000000,
                ],
            ),
            (
                Distribution::UploadedFactor,
                0.0,
                [
                    0xc020000000000000, 0xc008000000000000, 0xbff0000000000000,
                    0x0000000000000000, 0x3ff0000000000000, 0x4008000000000000,
                    0x4020000000000000,
                ],
            ),
        ];

        for (dist, param, want) in GOLDEN {
            for (&z, w) in Z_GRID.iter().zip(want) {
                let got = apply_marginal(z, dist, param, 1, &spec).to_bits();
                assert_eq!(
                    got,
                    w,
                    "apply_marginal({dist:?}) diverged at z={z}: \
                     got {got:#018x} ({}), want {w:#018x} ({})",
                    f64::from_bits(got),
                    f64::from_bits(w),
                );
            }
        }
    }

    #[test]
    fn apply_marginal_column_matches_scalar() {
        // Column path == scalar path, bit for bit, every variant — the same
        // spec fixture and (Distribution, param) table as
        // golden_apply_marginal_all_variants, on a denser z-grid.
        use crate::marginals::T3PpfTable;
        let mut spec = ols_spec_simple();
        spec.t3_table = Some(T3PpfTable::build_default());
        spec.upload_normal = vec![
            -1.5, -2.0, //
            -0.5, -1.0, //
            0.0, 0.0, //
            0.5, 1.0, //
            1.5, 2.0,
        ];
        spec.upload_normal_shape = (5, 2);

        const CASES: [(Distribution, f64); 9] = [
            (Distribution::Normal, 0.0),
            (Distribution::Binary, 0.3),
            (Distribution::RightSkewed, 0.0),
            (Distribution::LeftSkewed, 0.0),
            (Distribution::HighKurtosis, 0.0),
            (Distribution::Uniform, 0.0),
            (Distribution::UploadedData, 0.0),
            (Distribution::UploadedBinary, 0.4),
            (Distribution::UploadedFactor, 0.0),
        ];
        let zs: Vec<f64> = (0..4_001)
            .map(|k| -4.0 + 8.0 * k as f64 / 4_000.0)
            .collect();
        for (dist, param) in CASES {
            let mut col = zs.clone();
            apply_marginal_column(&mut col, dist, param, 1, &spec);
            for (i, &z) in zs.iter().enumerate() {
                assert_eq!(
                    col[i].to_bits(),
                    apply_marginal(z, dist, param, 1, &spec).to_bits(),
                    "{dist:?} diverged at z={z}"
                );
            }
        }
    }

    /// Golden bits for `draw_residual`, one stream per `ResidualDist` variant
    /// (df = 8.0). Pins the RNG-draw structure (chi² accumulator draw count,
    /// standardization scale, LeftSkewed sign flip).
    #[test]
    fn golden_draw_residual_all_variants() {
        #[rustfmt::skip]
        const GOLDEN: [(ResidualDist, u64, [u64; 8]); 5] = [
            (
                ResidualDist::Normal,
                100,
                [
                    0x3fa629a4c0000000, 0xbff24c47e0000000, 0x3fcb363640000000,
                    0xbff43d6640000000, 0x40039d58a0000000, 0x3fd47a3e00000000,
                    0xbfa15ff560000000, 0xbff0547340000000,
                ],
            ),
            (
                ResidualDist::Uniform,
                101,
                // Frozen from the shipped kernel: (2u − 1)·√3, u = next_uniform().
                [
                    0x3ff71156344c81d7, 0xbff1d74abe4a4664, 0xbff03095d68912ad,
                    0x3fdd8b8d1fc270fc, 0xbff99a37672c2b12, 0xbff09be8ed4e5a83,
                    0x3ff8ccd0f046865e, 0xbff8ded1ecff983b,
                ],
            ),
            (
                ResidualDist::HighKurtosis,
                102,
                [
                    0x3fcea5ba8cb57920, 0xbfc8aed270dbc2a4, 0x3feba35a279cbe64,
                    0xbfbff0c3790ddda2, 0xbfff092a366650b5, 0x4003c523e2e28bf4,
                    0xbfdfbb7807ec5bd8, 0x3fe80d67d4761013,
                ],
            ),
            (
                ResidualDist::RightSkewed,
                103,
                [
                    0xbff172c3c01c95df, 0xbfdc5fdc45875818, 0x3fdc613633264a30,
                    0x3fdc02e1fac8d1f0, 0xbfe3b5684a6a0a82, 0xbfd1257e89af2910,
                    0xbfc6802b716e6750, 0x3ff318a727303a08,
                ],
            ),
            (
                ResidualDist::LeftSkewed,
                104,
                [
                    0x3fe4066577104b4c, 0x3fc9bf8d45edb1f0, 0xbfe930442f9217ec,
                    0xc006211ab7d09342, 0xbfd013b27590c3e8, 0x3fe08663a7fa3374,
                    0x3fdf84c033c56d20, 0xbfb130c775781b40,
                ],
            ),
        ];

        for (rd, sim_id, want) in GOLDEN {
            let mut rng = SimRng::new(42, sim_id);
            for (i, w) in want.into_iter().enumerate() {
                let got = draw_residual(&mut rng, rd, 8.0).to_bits();
                assert_eq!(
                    got,
                    w,
                    "draw_residual({rd:?}) diverged at draw {i}: \
                     got {got:#018x} ({}), want {w:#018x} ({})",
                    f64::from_bits(got),
                    f64::from_bits(w),
                );
            }
        }
    }

    /// DGEN-08: cluster random-effect draws are taken as a fixed-size block
    /// *before* the per-row X/residual loop — their count is set by
    /// `n_clusters`, not by `tau_squared`. Two specs differing only in
    /// `tau_squared` (same `n_clusters`) therefore consume the cluster block
    /// identically and produce bit-identical `x_full`: the per-row X draws are
    /// not interleaved with the cluster draws. A broken kernel that drew
    /// cluster effects inside the row loop (count varying with the loop) would
    /// shift the X stream.
    ///
    /// This also guards CRN / curve quality, not just determinism: drawing the
    /// cluster block before the row loop is what keeps `find_sample_size`'s
    /// grid datasets nested across N, so the power-vs-N curve stays smooth.
    #[test]
    fn cluster_draws_precede_row_loop() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let n = 40;
        let n_pred = 3;
        let n_clusters = 4;

        let mut spec_a = ols_spec_simple();
        spec_a.estimator = EstimatorSpec::Mle;
        spec_a.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        });
        let mut spec_b = ols_spec_simple();
        spec_b.estimator = EstimatorSpec::Mle;
        spec_b.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: 1.5,
            slopes: vec![],
            extra_groupings: vec![],
        });

        let mut ws_a = SimWorkspace::new(
            n,
            n_pred,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters },
                0.25,
            )),
        );
        let mut ws_b = SimWorkspace::new(
            n,
            n_pred,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedClusters { n_clusters },
                0.25,
            )),
        );
        generate_sim_data(&spec_a, 3, 42, &mut ws_a).unwrap();
        generate_sim_data(&spec_b, 3, 42, &mut ws_b).unwrap();

        // All cluster effects populated (block drawn up front).
        for c in 0..n_clusters as usize {
            assert!(
                ws_a.cluster_u_draws[c].is_finite(),
                "cluster_u_draws[{c}] must be drawn"
            );
        }
        // X stream identical: tau only scales the cluster draws, it does not
        // change how many RNG values the per-row loop consumes.
        for i in 0..n {
            for j in 0..n_pred {
                assert_eq!(
                    ws_a.x_full[(i, j)],
                    ws_b.x_full[(i, j)],
                    "X stream diverges at ({i},{j}) when only tau_squared differs — \
                     per-row X draws must not interleave with the cluster block"
                );
            }
        }
    }

    /// DGEN-CL1 (parity): with an empty `between_var_indices`, a clustered spec
    /// performs NO broadcast — the (would-be) marked continuous column still
    /// varies within a cluster, and the run is deterministic.
    #[test]
    fn empty_between_var_indices_does_not_broadcast() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 4 },
            tau_squared: 0.2,
            slopes: vec![],
            extra_groupings: vec![],
        });
        // between_var_indices stays vec![] (from ols_spec_simple).
        let n = 40;
        let n_pred = 3;
        let sizing = ClusterSizing::FixedClusters { n_clusters: 4 };
        let mut ws = SimWorkspace::new(
            n,
            n_pred,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();
        // Cluster 0 = rows {0,4,8,...}; col 1 must NOT be constant across them.
        let v0 = ws.x_full[(0, 1)];
        let mut varies = false;
        let mut i = 4;
        while i < n {
            if ws.x_full[(i, 1)] != v0 {
                varies = true;
                break;
            }
            i += 4;
        }
        assert!(varies, "empty between_var_indices must NOT broadcast col 1");
    }

    /// DGEN-CL2: a marked continuous column is CONSTANT within each cluster and
    /// varies across clusters (FixedClusters round-robin layout).
    #[test]
    fn marked_continuous_column_constant_within_cluster() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let n_clusters = 4u32;
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: 0.2,
            slopes: vec![],
            extra_groupings: vec![],
        });
        spec.between_var_indices = vec![1]; // mark first continuous column
        let n = 40;
        let n_pred = 3;
        let sizing = ClusterSizing::FixedClusters { n_clusters };
        let mut ws = SimWorkspace::new(
            n,
            n_pred,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();
        // Guard the broadcast/u_c seam: cluster_ids (which keys u_c) must group
        // rows the same way the broadcast formula does (rep = i % n_clusters).
        for i in 0..n {
            assert_eq!(
                ws.cluster_ids[i] as usize,
                i % (n_clusters as usize),
                "cluster_ids must match the broadcast grouping at row {i}"
            );
        }
        // Each cluster c = rows {c, c+nc, c+2nc, ...}; col 1 constant within.
        for c in 0..n_clusters as usize {
            let rep = ws.x_full[(c, 1)];
            let mut i = c + n_clusters as usize;
            while i < n {
                assert_eq!(ws.x_full[(i, 1)], rep, "col 1 not constant in cluster {c}");
                i += n_clusters as usize;
            }
        }
        // Col 1 must differ between at least two clusters (distinct rep draws).
        let reps: Vec<f32> = (0..n_clusters as usize)
            .map(|c| ws.x_full[(c, 1)])
            .collect();
        assert!(
            reps.iter().any(|&v| v != reps[0]),
            "marked col must vary across clusters"
        );
        // Col 2 (unmarked) still varies within cluster 0 (rows {0, nc, 2nc, …}).
        let mut varies = false;
        let mut i = n_clusters as usize;
        while i < n {
            if ws.x_full[(i, 2)] != ws.x_full[(0, 2)] {
                varies = true;
                break;
            }
            i += n_clusters as usize;
        }
        assert!(varies, "unmarked col 2 must still vary within a cluster");
    }

    /// DGEN-CL3: a cross-level interaction (marked × within) is the elementwise
    /// product of the BROADCAST value and the within-cluster value, row-by-row.
    #[test]
    fn cross_level_interaction_uses_broadcast_value() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        // [intercept, x1, x2, x1:x2]; x1 marked cluster-level.
        let mut spec = ols_spec_2x(0, vec![], vec![]);
        spec.effect_sizes = vec![0.0, 0.5, 0.3, 0.2];
        spec.interactions = vec![vec![1, 2]];
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters: 4 },
            tau_squared: 0.2,
            slopes: vec![],
            extra_groupings: vec![],
        });
        spec.between_var_indices = vec![1];
        let n = 40;
        let n_pred = 4;
        let sizing = ClusterSizing::FixedClusters { n_clusters: 4 };
        let mut ws = SimWorkspace::new(
            n,
            n_pred,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();
        for i in 0..n {
            let prod = ws.x_full[(i, 1)] * ws.x_full[(i, 2)];
            // f32 arithmetic: tolerance loosened from 1e-12 to 1e-6.
            assert!(
                (ws.x_full[(i, 3)] - prod).abs() < 1e-6,
                "row {i}: interaction != broadcast(x1)*x2"
            );
        }
        // x1 constant within cluster 0; x2 varies within cluster 0.
        let v1 = ws.x_full[(0, 1)];
        assert_eq!(
            ws.x_full[(4, 1)],
            v1,
            "x1 must be broadcast (constant in cluster)"
        );
        assert!(
            ws.x_full[(4, 2)] != ws.x_full[(0, 2)],
            "x2 must vary within cluster"
        );
    }

    /// DGEN-CL4: a marked FACTOR broadcasts a single valid one-hot per cluster —
    /// every dummy of the factor is overwritten from the same representative row,
    /// so all rows of a cluster share one reference-coded pattern.
    #[test]
    fn marked_factor_broadcasts_single_one_hot_per_cluster() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let n_clusters = 5u32;
        // 3-level factor → dummy cols 3,4. n_pred = 1 + 2 + 2 = 5.
        let mut spec = ols_spec_2x(2, vec![3], vec![0.5, 0.3, 0.2]);
        spec.effect_sizes = vec![0.0, 0.5, 0.3, 0.0, 0.0];
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: 0.2,
            slopes: vec![],
            extra_groupings: vec![],
        });
        spec.between_var_indices = vec![3, 4]; // both dummies of the factor
        let n = 50;
        let n_pred = 5;
        let sizing = ClusterSizing::FixedClusters { n_clusters };
        let mut ws = SimWorkspace::new(
            n,
            n_pred,
            2,
            1,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        generate_sim_data(&spec, 0, 42, &mut ws).unwrap();
        for c in 0..n_clusters as usize {
            let (d0, d1) = (ws.x_full[(c, 3)], ws.x_full[(c, 4)]);
            // Valid reference-coded one-hot.
            assert!(d0 == 0.0 || d0 == 1.0);
            assert!(d1 == 0.0 || d1 == 1.0);
            assert!(d0 + d1 == 0.0 || d0 + d1 == 1.0, "cluster {c}: not one-hot");
            // Constant across all rows of the cluster.
            let mut i = c + n_clusters as usize;
            while i < n {
                assert_eq!(ws.x_full[(i, 3)], d0, "dummy0 not constant in cluster {c}");
                assert_eq!(ws.x_full[(i, 4)], d1, "dummy1 not constant in cluster {c}");
                i += n_clusters as usize;
            }
        }
    }

    /// DGEN-CL5 (parity + determinism): on the OPTIMISTIC path (lme None), the
    /// per-cluster draw equals `next_normal() · τ` from a clone of the same
    /// per-sim RNG — byte-identical to the pre-Phase-2 kernel.
    #[test]
    fn cluster_draw_normal_matches_next_normal_times_tau() {
        use crate::rng::SimRng;
        use crate::spec::{ClusterSizing, ClusterSpec};
        let n_clusters = 4u32;
        let tau_sq = 0.3_f64;
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: tau_sq,
            slopes: vec![],
            extra_groupings: vec![],
        });
        let n = 40;
        let (base_seed, sim_id) = (42u64, 3u64);
        let sizing = ClusterSizing::FixedClusters { n_clusters };
        let mut ws = SimWorkspace::new(
            n,
            3,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        generate_sim_data(&spec, sim_id, base_seed, &mut ws).unwrap();
        // Cluster draws are the FIRST draws of the per-sim RNG (before the row
        // loop), so a fresh clone reproduces them exactly.
        let mut check = SimRng::new(base_seed, sim_id);
        let tau = tau_sq.sqrt();
        for c in 0..n_clusters as usize {
            // next_normal() is f32; cluster_u_draws is f32 — compute in f64 then narrow.
            let expected = (check.next_normal() as f64 * tau) as f32;
            assert_eq!(ws.cluster_u_draws[c], expected, "cluster {c} draw");
        }
    }

    /// DGEN-CL6: a non-Normal RE distribution uses `draw_residual`, NOT
    /// `next_normal()` — a clone using `draw_residual(T, df)·τ` must match, and a
    /// `next_normal()·τ` reference must diverge.
    #[test]
    fn cluster_draw_uses_draw_residual_for_non_normal_re() {
        use crate::rng::SimRng;
        use crate::spec::{ClusterSizing, ClusterSpec, LmeScenarioPerturbations};
        let n_clusters = 4u32;
        let tau_sq = 0.3_f64;
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: tau_sq,
            slopes: vec![],
            extra_groupings: vec![],
        });
        // Non-Normal REs, NO ICC jitter (icc=0) ⇒ τ_eff == spec τ, no scen_rng
        // jitter draw.
        spec.scenario = ScenarioPerturbations {
            name: "re_t".into(),
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::HighKurtosis,
                random_effect_df: 8.0,
                icc_noise_sd: 0.0,
            }),
            ..Default::default()
        };
        let n = 40;
        let (base_seed, sim_id) = (42u64, 3u64);
        let sizing = ClusterSizing::FixedClusters { n_clusters };
        let mut ws = SimWorkspace::new(
            n,
            3,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        generate_sim_data(&spec, sim_id, base_seed, &mut ws).unwrap();
        let tau = tau_sq.sqrt();
        let mut check = SimRng::new(base_seed, sim_id);
        let mut ref_normal = SimRng::new(base_seed, sim_id);
        for c in 0..n_clusters as usize {
            // cluster_u_draws is f32; expected and normal_ref narrowed from f64 arithmetic.
            let expected =
                (draw_residual(&mut check, ResidualDist::HighKurtosis, 8.0) * tau) as f32;
            assert_eq!(ws.cluster_u_draws[c], expected, "cluster {c} (t-draw)");
            let normal_ref = (ref_normal.next_normal() as f64 * tau) as f32;
            assert_ne!(
                ws.cluster_u_draws[c], normal_ref,
                "must NOT be a normal draw"
            );
        }
    }

    /// DGEN-CL7 (D5): a non-zero `icc_noise_sd` perturbs τ²_eff per sim from
    /// the scenario RNG, and the effective τ² is clamped ≥ 0. With a tiny base
    /// τ² and a large jitter SD, some sims must clamp to exactly 0.0.
    #[test]
    fn icc_noise_clamps_effective_tau_squared_non_negative() {
        use crate::spec::{ClusterSizing, ClusterSpec, LmeScenarioPerturbations};
        let n_clusters = 4u32;
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedClusters { n_clusters },
            tau_squared: 0.0001,
            slopes: vec![],
            extra_groupings: vec![],
        });
        spec.scenario = ScenarioPerturbations {
            name: "icc_jitter".into(),
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::Normal,
                random_effect_df: 0.0,
                icc_noise_sd: 5.0, // huge relative to base ⇒ ~half the blocks clamp
            }),
            ..Default::default()
        };
        let n = 40;
        let sizing = ClusterSizing::FixedClusters { n_clusters };
        let mut ws = SimWorkspace::new(
            n,
            3,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                sizing.clone(),
                0.25,
            )),
        );
        let mut saw_clamp = false;
        let mut saw_positive = false;
        // Every sim draws its own jitter.
        for s in 0..20u64 {
            generate_sim_data(&spec, s, 42, &mut ws).unwrap();
            assert!(ws.tau_squared_design >= 0.0, "τ²_eff must be clamped ≥ 0");
            if ws.tau_squared_design == 0.0 {
                saw_clamp = true;
            } else {
                saw_positive = true;
            }
        }
        // saw_positive fails before the kernel wires τ²_eff (field stays 0.0):
        // proves the jitter is actually applied, not just that the field is unset.
        assert!(
            saw_positive,
            "ICC jitter must move τ²_eff above 0 in some sims"
        );
        assert!(
            saw_clamp,
            "a negative jitter must have clamped τ²_eff to 0 at least once"
        );
    }

    /// DGEN-FS: FixedSize layout produces contiguous blocks (`cluster_ids[i] == i / cluster_size`)
    /// and the `cluster_u_draws` buffer is sized to `max_n / cluster_size`.
    #[test]
    fn fixed_size_block_layout_is_nested_across_prefixes() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        // Use a standard simple OLS spec with a FixedSize cluster override.
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        spec.cluster = Some(ClusterSpec {
            sizing: ClusterSizing::FixedSize { cluster_size: 5 },
            tau_squared: 0.3,
            slopes: vec![],
            extra_groupings: vec![],
        });
        // max_n = 40; expected clusters = 40 / 5 = 8.
        let n = 40;
        let n_pred = 3;
        let mut ws = SimWorkspace::new(
            n,
            n_pred,
            2,
            0,
            Some(&engine_contract::ClusterSpec::intercept_only(
                ClusterSizing::FixedSize { cluster_size: 5 },
                0.25,
            )),
        );
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        for i in 0..n {
            assert_eq!(ws.cluster_ids[i] as usize, i / 5, "block layout at row {i}");
        }
        assert_eq!(
            ws.cluster_u_draws.len(),
            8,
            "buffer sized to max_n/cluster_size"
        );
    }

    /// Multi-grouping assembly identity: y decomposes exactly into
    /// lp + residual + primary u + Σ extra u (the assembly loop's own parts,
    /// read back from the workspace buffers).
    #[test]
    fn crossed_nested_assembly_adds_every_u_term() {
        use crate::spec::{ClusterSizing, ClusterSpec, EstimatorSpec};
        use engine_contract::{GroupingRelation, GroupingSpec};
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 4 }, 0.20);
        cluster.extra_groupings = vec![
            GroupingSpec {
                relation: GroupingRelation::Crossed { n_clusters: 3 },
                tau_squared: 0.15,
                slopes: vec![],
            },
            GroupingSpec {
                relation: GroupingRelation::NestedWithin { n_per_parent: 2 },
                tau_squared: 0.05,
                slopes: vec![],
            },
        ];
        spec.cluster = Some(cluster.clone());
        let max_n = 48; // 2 atom blocks (atom = 4·3·2 = 24)
        let mut ws = SimWorkspace::new(max_n, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        for i in 0..max_n {
            // x_full, residuals, cluster_u_draws, extra_u_draws, y_full are f32 — widen.
            let mut lp = 0.0_f64;
            for j in 0..3 {
                lp += ws.x_full[(i, j)] as f64 * spec.effect_sizes[j];
            }
            let mut expect =
                lp + ws.residuals[i] as f64 + ws.cluster_u_draws[ws.cluster_ids[i] as usize] as f64;
            for g in 0..2 {
                expect += ws.extra_u_draws[g][ws.extra_grouping_ids[g][i] as usize] as f64;
            }
            // f32 tolerance.
            assert!(
                (ws.y_full[i] as f64 - expect).abs() <= 1e-6,
                "row {i}: y={} expected {expect}",
                ws.y_full[i]
            );
        }
        // Draw buffers were actually populated with τ-scaled values.
        let v0: f64 = ws.extra_u_draws[0]
            .iter()
            .map(|u| (*u * *u) as f64)
            .sum::<f64>()
            / 3.0;
        assert!(v0 > 0.0, "crossed draws must be non-degenerate");
    }

    /// Per-grouping τ² slots: optimistic path copies the spec values; the
    /// extra draws scale by their own τ_g (zero τ ⇒ exactly zero draws).
    #[test]
    fn extra_tau_blocks_scale_their_own_grouping() {
        use crate::spec::{ClusterSizing, ClusterSpec, EstimatorSpec};
        use engine_contract::{GroupingRelation, GroupingSpec};
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 4 }, 0.20);
        cluster.extra_groupings = vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 3 },
            tau_squared: 0.0,
            slopes: vec![],
        }];
        spec.cluster = Some(cluster.clone());
        let mut ws = SimWorkspace::new(24, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        assert_eq!(ws.extra_tau_sq_design, vec![0.0]);
        assert!(ws.extra_u_draws[0].iter().all(|&u| u == 0.0));
        assert!(ws.cluster_u_draws[..4].iter().any(|&u| u != 0.0));
    }

    /// Uniformity (engine-level): the scenario ICC jitter is an
    /// INDEPENDENT draw per grouping (the primary and each extra get their own
    /// scenario-RNG realization, not one shared jitter), deterministic per
    /// (seed, sim), and a zero `icc_noise_sd` recovers the exact base τ² for
    /// every grouping. This is the uniformity validation while the LME scenario knobs
    /// are not yet exposed in the R port (the L5 `scen_re_multi` harness covers
    /// the same law once they are).
    #[test]
    fn scenario_icc_jitter_is_independent_per_grouping() {
        use crate::spec::{ClusterSizing, ClusterSpec, EstimatorSpec, LmeScenarioPerturbations};
        use engine_contract::{GroupingRelation, GroupingSpec};
        let (base_tau_p, base_tau_x) = (0.20_f64, 0.15_f64);
        let mut spec = ols_spec_simple();
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 6 }, base_tau_p);
        cluster.extra_groupings = vec![GroupingSpec {
            relation: GroupingRelation::Crossed { n_clusters: 4 },
            tau_squared: base_tau_x,
            slopes: vec![],
        }];
        spec.cluster = Some(cluster.clone());
        // lme present ⇒ non-optimistic ⇒ the scenario jitter branch runs.
        spec.scenario = ScenarioPerturbations {
            name: "re_jit".into(),
            lme: Some(LmeScenarioPerturbations {
                random_effect_dist: ResidualDist::Normal,
                random_effect_df: 0.0,
                icc_noise_sd: 0.05,
            }),
            ..Default::default()
        };
        let mut ws = SimWorkspace::new(48, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        let jit_p = ws.tau_squared_design - base_tau_p;
        let jit_x = ws.extra_tau_sq_design[0] - base_tau_x;
        assert!(jit_p != 0.0, "primary τ² must be jittered");
        assert!(jit_x != 0.0, "extra τ² must be jittered (per-grouping)");
        assert!(
            jit_p != jit_x,
            "the jitters must be independent draws, not one shared realization"
        );
        // Deterministic per (seed, sim).
        let mut ws2 = SimWorkspace::new(48, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws2).unwrap();
        assert_eq!(ws.extra_tau_sq_design, ws2.extra_tau_sq_design);
        assert_eq!(ws.tau_squared_design, ws2.tau_squared_design);
        // Control: icc_noise_sd = 0 ⇒ exact base τ² for every grouping.
        spec.scenario.lme.as_mut().unwrap().icc_noise_sd = 0.0;
        let mut ws3 = SimWorkspace::new(48, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws3).unwrap();
        assert_eq!(ws3.extra_tau_sq_design[0], base_tau_x);
        assert_eq!(ws3.tau_squared_design, base_tau_p);
    }

    // ------------------------------------------------------------------
    // Task 4 — D-correlated random-slope draw + assembly.
    // ------------------------------------------------------------------

    /// Spec with `n_cont` continuous predictors; n_predictors = 1 + n_cont
    /// (intercept + predictor columns). Identity correlation; all β = 0.25.
    fn minimal_spec(n_cont: usize) -> SimulationSpec {
        SimulationSpec {
            n_non_factor: n_cont as u32,
            n_factor_dummies: 0,
            correlation: {
                let mut v = vec![0.0f64; n_cont * n_cont];
                for d in 0..n_cont {
                    v[d * n_cont + d] = 1.0;
                }
                v
            },
            var_types: vec![Distribution::Normal; n_cont],
            var_pinned: vec![],
            var_params: vec![0.0; n_cont],
            upload_normal: vec![],
            upload_normal_shape: (0, 0),
            upload_data: vec![],
            upload_data_shape: (0, 0),
            bootstrap_frame_map: vec![],
            between_var_indices: vec![],
            factor_n_levels: vec![],
            factor_proportions: vec![],
            factor_sampled: Vec::new(),
            effect_sizes: vec![0.25f64; 1 + n_cont],
            target_indices: vec![1],
            contrast_pairs: vec![],
            interactions: vec![],
            correction_method: CorrectionMethod::None,
            crit_values: CritValues {
                alpha: 0.05,
                posthoc_alpha: None,
            },
            heteroskedasticity_driver: None,
            residual_dist: ResidualDist::Normal,
            residual_pinned: false,
            outcome_kind: OutcomeKind::Continuous,
            estimator: EstimatorSpec::Ols,
            wald_se: Default::default(),
            intercept: 0.0,
            posthoc: vec![],
            max_failed_fraction: 0.1,
            cluster: None,
            scenario: ScenarioPerturbations::optimistic(),
            t3_table: None,
            het_coeffs: HeteroskedasticityCoeffs::default(),
            report_overall: false,
            factor_min_level_count: 0,
            cluster_slope_design_cols: vec![],
            fit_columns: Vec::new(),
        }
    }

    fn slope(col: u32, var: f64, corr_int: f64, corr_with: Vec<f64>) -> engine_contract::SlopeTerm {
        engine_contract::SlopeTerm {
            column: engine_contract::ColumnId(col),
            variance: var,
            corr_with_intercept: corr_int,
            corr_with,
        }
    }

    /// Single-slope assembly identity: y = lp + ε + u₀[g] + u₁[g]·x_slope.
    #[test]
    fn random_slope_assembly_adds_u0_plus_u1_x() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = minimal_spec(2); // 2 continuous predictors → x_full cols 1,2
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 8 }, 0.20);
        cluster.slopes = vec![slope(0, 0.10, 0.3, vec![])];
        spec.cluster = Some(cluster.clone());
        spec.cluster_slope_design_cols = vec![1]; // col 0 → x_full index 1
        let max_n = 96;
        let mut ws = SimWorkspace::new(max_n, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        let sc = 1usize;
        for i in 0..max_n {
            // x_full, residuals, *_u_draws, y_full are f32 — widen.
            let mut lp = 0.0_f64;
            for j in 0..3 {
                lp += ws.x_full[(i, j)] as f64 * spec.effect_sizes[j];
            }
            let g = ws.cluster_ids[i] as usize;
            let expect = lp
                + ws.residuals[i] as f64
                + ws.cluster_u_draws[g] as f64
                + ws.cluster_slope_u_draws[g] as f64 * ws.x_full[(i, sc)] as f64; // stride 1
            assert!((ws.y_full[i] as f64 - expect).abs() <= 1e-6, "row {i}");
        }
        let v1: f64 = ws
            .cluster_slope_u_draws
            .iter()
            .map(|u| (*u * *u) as f64)
            .sum::<f64>()
            / 8.0;
        assert!(v1 > 0.0);
    }

    /// Multi-slope assembly: y = lp + ε + u₀[g] + u₁[g]·x1 + u₂[g]·x2, read from
    /// the stride-2 slope buffer (`[g·2 + k]`).
    #[test]
    fn random_multislope_assembly_adds_all_terms() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = minimal_spec(2); // x_full cols 1,2
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 8 }, 0.20);
        cluster.slopes = vec![slope(0, 0.10, 0.3, vec![]), slope(1, 0.08, 0.1, vec![0.2])];
        spec.cluster = Some(cluster.clone());
        spec.cluster_slope_design_cols = vec![1, 2];
        let max_n = 96;
        let mut ws = SimWorkspace::new(max_n, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        for i in 0..max_n {
            // x_full, residuals, *_u_draws, y_full are f32 — widen.
            let mut lp = 0.0_f64;
            for j in 0..3 {
                lp += ws.x_full[(i, j)] as f64 * spec.effect_sizes[j];
            }
            let g = ws.cluster_ids[i] as usize;
            let expect = lp
                + ws.residuals[i] as f64
                + ws.cluster_u_draws[g] as f64
                + ws.cluster_slope_u_draws[g * 2] as f64 * ws.x_full[(i, 1)] as f64
                + ws.cluster_slope_u_draws[g * 2 + 1] as f64 * ws.x_full[(i, 2)] as f64;
            assert!((ws.y_full[i] as f64 - expect).abs() <= 1e-6, "row {i}");
        }
    }

    /// Intercept↔slope correlation sign: ρ > 0 ⇒ (u₀, u₁) covary positively.
    #[test]
    fn slope_draw_respects_correlation_sign() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = minimal_spec(1);
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 4000 }, 0.40);
        cluster.slopes = vec![slope(0, 0.25, 0.6, vec![])];
        spec.cluster = Some(cluster.clone());
        spec.cluster_slope_design_cols = vec![1];
        let max_n = 8000;
        let mut ws = SimWorkspace::new(max_n, 2, 1, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        let k = 4000usize;
        let (mut s0, mut s1, mut s01) = (0.0f64, 0.0f64, 0.0f64);
        for c in 0..k {
            let (u0, u1) = (
                ws.cluster_u_draws[c] as f64,
                ws.cluster_slope_u_draws[c] as f64,
            );
            s0 += u0;
            s1 += u1;
            s01 += u0 * u1;
        }
        let cov = s01 / k as f64 - (s0 / k as f64) * (s1 / k as f64);
        assert!(cov > 0.0, "ρ>0 ⇒ positive (u₀,u₁) covariance; got {cov}");
    }

    /// Clustered-binary identity: p_i = σ(lp_i + u₀[g] + u₁[g]·x1), Bernoulli
    /// draw against the pre-drawn Uniform. Single random slope.
    #[test]
    fn binary_re_assembly_uses_sigmoid_of_lp_plus_u() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = minimal_spec(1); // 1 continuous predictor → x_full col 1
        spec.estimator = EstimatorSpec::Glm;
        spec.outcome_kind = OutcomeKind::Binary;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 8 }, 0.20);
        cluster.slopes = vec![engine_contract::SlopeTerm {
            column: engine_contract::ColumnId(0),
            variance: 0.10,
            corr_with_intercept: 0.3,
            corr_with: vec![],
        }];
        spec.cluster = Some(cluster.clone());
        spec.cluster_slope_design_cols = vec![1]; // col 0 of slopes → x_full index 1
        let max_n = 96;
        let mut ws = SimWorkspace::new(max_n, 2, 1, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        for i in 0..max_n {
            let mut lp = 0.0_f64;
            for j in 0..2 {
                lp += ws.x_full[(i, j)] as f64 * spec.effect_sizes[j];
            }
            let g = ws.cluster_ids[i] as usize;
            // n_sl == 1, so cluster_slope_u_draws[g * 1 + 0] == cluster_slope_u_draws[g]
            let u_re = ws.cluster_u_draws[g] as f64
                + ws.cluster_slope_u_draws[g] as f64 * ws.x_full[(i, 1)] as f64;
            let p = glmm::mcpower::sigmoid_stable(lp + u_re);
            let expect = if (ws.residuals[i] as f64) < p {
                1.0f32
            } else {
                0.0f32
            };
            assert_eq!(ws.y_full[i], expect, "row {i}");
        }
        // y is a real Bernoulli mix (not all-0/all-1) under this signal.
        let s: f32 = ws.y_full[..max_n].iter().sum();
        assert!(s > 0.0 && (s as usize) < max_n);
    }

    /// Slope↔slope correlation sign: corr_with[0] > 0 ⇒ (u₁, u₂) covary positively.
    #[test]
    fn multislope_respects_slope_slope_correlation_sign() {
        use crate::spec::{ClusterSizing, ClusterSpec};
        let mut spec = minimal_spec(2);
        spec.estimator = EstimatorSpec::Mle;
        let mut cluster =
            ClusterSpec::intercept_only(ClusterSizing::FixedClusters { n_clusters: 4000 }, 0.30);
        cluster.slopes = vec![slope(0, 0.25, 0.0, vec![]), slope(1, 0.25, 0.0, vec![0.6])];
        spec.cluster = Some(cluster.clone());
        spec.cluster_slope_design_cols = vec![1, 2];
        let max_n = 8000;
        let mut ws = SimWorkspace::new(max_n, 3, 2, 0, Some(&cluster));
        generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
        let k = 4000usize;
        let (mut s1, mut s2, mut s12) = (0.0f64, 0.0f64, 0.0f64);
        for c in 0..k {
            let (u1, u2) = (
                ws.cluster_slope_u_draws[c * 2] as f64,
                ws.cluster_slope_u_draws[c * 2 + 1] as f64,
            );
            s1 += u1;
            s2 += u2;
            s12 += u1 * u2;
        }
        let cov = s12 / k as f64 - (s1 / k as f64) * (s2 / k as f64);
        assert!(
            cov > 0.0,
            "corr_with>0 ⇒ positive (u₁,u₂) covariance; got {cov}"
        );
    }

    /// Stage-2 batched residuals: HighKurtosis t(5) → var≈1, symmetric;
    /// RightSkewed χ²(5) → var≈1, positive skew; LeftSkewed mirrors; Uniform
    /// U(−√3,√3) → var≈1, symmetric. 50k draws → var SE ≈ 0.9%; the skew
    /// gates are intentionally loose (0.15 ≈ 14·SE for symmetry, 0.3 ≈ a
    /// quarter of χ²(5)'s population skew √(8/5)≈1.26) — they catch
    /// slot/formula mix-ups, not calibration drift.
    #[test]
    fn batched_t_and_chi2_residuals_have_unit_variance_and_correct_shape() {
        for (dist, want_skew_sign) in [
            (ResidualDist::HighKurtosis, 0.0_f64),
            (ResidualDist::RightSkewed, 1.0),
            (ResidualDist::LeftSkewed, -1.0),
            (ResidualDist::Uniform, 0.0),
        ] {
            let mut spec = ols_spec_2x(0, vec![], vec![]);
            spec.residual_dist = dist;
            spec.scenario.residual_df = 5.0;
            let max_n = 50_000usize;
            let mut ws = SimWorkspace::new(max_n, 3, 2, 0, None);
            generate_sim_data(&spec, 0, 2137, &mut ws).unwrap();
            let n = max_n as f64;
            let mean: f64 = ws.residuals.iter().map(|&r| r as f64).sum::<f64>() / n;
            let var: f64 = ws
                .residuals
                .iter()
                .map(|&r| (r as f64 - mean).powi(2))
                .sum::<f64>()
                / n;
            let m3: f64 = ws
                .residuals
                .iter()
                .map(|&r| (r as f64 - mean).powi(3))
                .sum::<f64>()
                / n;
            let skew = m3 / var.powf(1.5);
            assert!(mean.abs() < 0.05, "{dist:?} mean={mean}");
            assert!((var - 1.0).abs() < 0.06, "{dist:?} var={var}");
            if want_skew_sign == 0.0 {
                assert!(skew.abs() < 0.15, "{dist:?} skew={skew}");
            } else {
                assert!(skew * want_skew_sign > 0.3, "{dist:?} skew={skew}");
            }
        }
    }
}
