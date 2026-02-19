"""
Simulation execution for MCPower framework.

This module contains the core Monte Carlo simulation logic, extracted
from the original base.py for better separation of concerns.
"""

import warnings
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..backends import get_backend
from ..stats.data_generation import _generate_factors


def _generate_cluster_id_array(sample_size: int, cluster_specs: Dict) -> Optional[np.ndarray]:
    """Generate a cluster-membership array for mixed models.

    Args:
        sample_size: Total number of observations.
        cluster_specs: Dict of cluster specifications (keyed by grouping
            variable name).

    Returns:
        1-D integer array of cluster IDs (e.g. ``[0,0,0, 1,1,1, ...]``)
        or ``None`` if *cluster_specs* is empty.
    """
    if not cluster_specs:
        return None
    first_spec = next(iter(cluster_specs.values()))
    # Compute cluster_size from sample_size if not provided
    cluster_size = first_spec["cluster_size"]
    if cluster_size is None:
        cluster_size = sample_size // first_spec["n_clusters"]
    return np.repeat(np.arange(first_spec["n_clusters"]), cluster_size)


class SimulationRunner:
    """Executes Monte Carlo simulations for power analysis.

    Each iteration generates a predictor matrix ``X`` (via the active
    compute backend), extends it with interaction columns, generates
    the dependent variable ``y``, and runs the appropriate statistical
    analysis (OLS or LME). Failed iterations (e.g. LME convergence
    failures) are tracked and the analysis aborts if the failure rate
    exceeds the configured threshold.
    """

    def __init__(
        self,
        n_simulations: int,
        seed: Optional[int] = None,
        alpha: float = 0.05,
        parallel: Union[bool, str] = False,
        n_cores: int = 1,
        max_failed_simulations: float = 0.03,
    ):
        """Initialise the simulation runner.

        Args:
            n_simulations: Number of Monte Carlo iterations.
            seed: Base random seed. Each iteration uses
                ``seed + 4 * sim_id``.
            alpha: Significance level for hypothesis tests.
            parallel: Parallel processing mode (unused inside the
                runner itself; parallelism is handled at the
                sample-size loop level).
            n_cores: Number of CPU cores (reserved for future use).
            max_failed_simulations: Maximum acceptable proportion of
                failed iterations (0–1). Exceeding this raises
                ``RuntimeError`` for mixed models.
        """
        self.n_simulations = n_simulations
        self.seed = seed
        self.alpha = alpha
        self.parallel = parallel
        self.n_cores = n_cores
        self.max_failed_simulations = max_failed_simulations

    def run_power_simulations(
        self,
        sample_size: int,
        metadata: "SimulationMetadata",
        generate_y_func: Callable,
        analyze_func: Callable,
        create_X_extended_func: Callable,
        scenario_config: Optional[Dict] = None,
        apply_perturbations_func: Optional[Callable] = None,
        progress=None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run the full Monte Carlo simulation loop.

        Args:
            sample_size: Number of observations per iteration.
            metadata: Pre-computed ``SimulationMetadata``.
            generate_y_func: Callback to generate the dependent variable.
            analyze_func: Callback to run OLS analysis.
            create_X_extended_func: Callback to build the extended design
                matrix (adding interaction columns).
            scenario_config: Optional scenario-perturbation dict.
            apply_perturbations_func: Callback to perturb correlations and
                distributions per iteration (scenario mode only).
            progress: Optional ``ProgressReporter`` (advanced by 1 per
                iteration).
            cancel_check: Optional callable returning ``True`` to abort.

        Returns:
            Dict with keys ``"all_results"``, ``"all_results_corrected"``,
            ``"n_simulations_used"``, ``"n_simulations_failed"``, and
            optionally ``"diagnostics"`` and ``"failure_reasons"`` when
            *verbose* is enabled.

        Raises:
            RuntimeError: If all simulations fail or the failure rate
                exceeds ``max_failed_simulations`` for mixed models.
        """
        all_results = []
        all_results_corrected = []
        n_wald_fallbacks = 0
        collected_diagnostics: Optional[List[Dict[str, Any]]] = [] if metadata.verbose else None
        failure_reasons: Optional[Dict[str, int]] = {} if metadata.verbose else None

        # Phase 2 Optimization: Precompute values that are constant across simulations
        # This eliminates redundant computations in the simulation loop

        # Precompute cluster IDs (8-12% speedup)
        # For slopes/nested models, cluster IDs are generated per-simulation
        # inside _generate_random_effects (they depend on the model type).
        if metadata.cluster_specs and not metadata.has_random_slopes and not metadata.has_nested:
            metadata.cluster_ids_template = _generate_cluster_id_array(sample_size, metadata.cluster_specs)

        # Precompute fixed effect mask (3-5% speedup)
        if metadata.cluster_effect_indices:
            all_indices = np.arange(len(metadata.effect_sizes))
            metadata.fixed_effect_mask = ~np.isin(all_indices, metadata.cluster_effect_indices)
            metadata.fixed_effect_sizes_cached = metadata.effect_sizes[metadata.fixed_effect_mask]
        else:
            metadata.fixed_effect_sizes_cached = metadata.effect_sizes

        # Precompute LME critical values (custom solver)
        if metadata.cluster_specs:
            from ..stats.lme_solver import compute_lme_critical_values

            n_fixed = len(metadata.target_indices)
            # n_fixed_effects = number of columns in X_expanded (excluding intercept)
            # This equals the total effect count minus cluster effects
            n_fixed_total = len(metadata.effect_sizes)
            if metadata.cluster_effect_indices:
                n_fixed_total -= len(metadata.cluster_effect_indices)
            chi2_crit, z_crit, correction_z_crits = compute_lme_critical_values(
                self.alpha, n_fixed_total, n_fixed, metadata.correction_method
            )
            metadata.lme_chi2_crit = chi2_crit  # type: ignore[assignment]
            metadata.lme_z_crit = z_crit  # type: ignore[assignment]
            metadata.lme_correction_z_crits = correction_z_crits  # type: ignore[assignment]

        for sim_id in range(self.n_simulations):
            if cancel_check is not None and cancel_check():
                from ..progress import SimulationCancelled

                raise SimulationCancelled("Simulation cancelled by user")

            sim_seed = self.seed + 4 * sim_id if self.seed is not None else None

            # Apply perturbations if in scenario mode
            if scenario_config is not None and apply_perturbations_func is not None:
                perturbed_corr, perturbed_types = apply_perturbations_func(
                    metadata.correlation_matrix,
                    metadata.var_types,
                    scenario_config,
                    sim_seed,
                )
            else:
                perturbed_corr = metadata.correlation_matrix
                perturbed_types = metadata.var_types

            result = self._single_simulation(
                sim_id=sim_id,
                sample_size=sample_size,
                metadata=metadata,
                correlation_matrix=perturbed_corr,
                var_types=perturbed_types,
                generate_y_func=generate_y_func,
                analyze_func=analyze_func,
                create_X_extended_func=create_X_extended_func,
                sim_seed=sim_seed,
                scenario_config=scenario_config,
            )

            if result is not None:
                # Check if verbose mode returned a dict
                if metadata.verbose and isinstance(result, dict):
                    if result.get("failed"):
                        # Track failure
                        reason = result.get("failure_reason", "Unknown")
                        if failure_reasons is not None:
                            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                    else:
                        # Extract results and diagnostics
                        sim_significant, sim_significant_corrected = result["results"]
                        all_results.append(sim_significant)
                        all_results_corrected.append(sim_significant_corrected)
                        if result.get("wald_fallback"):
                            n_wald_fallbacks += 1
                        if "diagnostics" in result and collected_diagnostics is not None:
                            diag = result["diagnostics"].copy()
                            diag["sim_id"] = sim_id
                            collected_diagnostics.append(diag)
                elif isinstance(result, tuple) and len(result) == 3:
                    # Normal mode with Wald flag (mixed models)
                    sim_significant, sim_significant_corrected, wald_flag = result
                    all_results.append(sim_significant)
                    all_results_corrected.append(sim_significant_corrected)
                    if wald_flag:
                        n_wald_fallbacks += 1
                else:
                    # Normal mode without Wald flag (OLS) — 2-tuple
                    sim_significant, sim_significant_corrected = result
                    all_results.append(sim_significant)
                    all_results_corrected.append(sim_significant_corrected)
            elif metadata.verbose and failure_reasons is not None:
                # Track as unknown failure
                failure_reasons["Unknown (returned None)"] = failure_reasons.get("Unknown (returned None)", 0) + 1

            if progress is not None:
                progress.advance(1)

        if not all_results:
            raise RuntimeError("All simulations failed")

        # Check failure rate
        n_failed = self.n_simulations - len(all_results)
        failed_pct = n_failed / self.n_simulations

        # Only enforce strict threshold for LME models (cluster_ids present)
        # For OLS models, just warn but don't fail (to preserve backward compatibility)
        has_clusters = metadata.cluster_specs and len(metadata.cluster_specs) > 0

        if has_clusters and failed_pct > self.max_failed_simulations:
            raise RuntimeError(
                f"Too many failed simulations: {n_failed}/{self.n_simulations} "
                f"({failed_pct:.1%}), threshold: {self.max_failed_simulations:.1%}"
            )
        elif n_failed > 0:
            if has_clusters:
                warnings.warn(f"{n_failed} simulations failed ({failed_pct:.1%})")
            else:
                # For OLS models, only warn if failure rate is very high
                if failed_pct > 0.1:  # > 10%
                    warnings.warn(f"{n_failed} simulations failed ({failed_pct:.1%}) - check data/model specification")

        # Warn if Wald fallback rate exceeds 10% (mixed models only)
        if n_wald_fallbacks > 0 and len(all_results) > 0:
            wald_pct = n_wald_fallbacks / len(all_results)
            if wald_pct > 0.10:
                warnings.warn(
                    f"Wald test fallback used in {n_wald_fallbacks}/{len(all_results)} "
                    f"simulations ({wald_pct:.1%}). The likelihood-ratio test failed "
                    f"to converge in these cases. Overall significance results may "
                    f"be less reliable. Consider increasing sample size or "
                    f"simplifying the model."
                )

        # ICC comparison: warn if estimated ICC differs substantially from specified
        if metadata.verbose and collected_diagnostics and metadata.cluster_specs:
            icc_estimates: List[float] = [
                d["icc_estimated"]
                for d in collected_diagnostics
                if d.get("icc_estimated") is not None and not np.isnan(d.get("icc_estimated", np.nan))
            ]
            if icc_estimates:
                mean_icc = float(np.mean(icc_estimates))
                _warn_icc_mismatch(metadata, mean_icc)

        result_dict = {
            "all_results": all_results,
            "all_results_corrected": all_results_corrected,
            "n_simulations_used": len(all_results),
            "n_simulations_failed": n_failed,
            "n_wald_fallbacks": n_wald_fallbacks,
        }

        if metadata.verbose:
            result_dict["diagnostics"] = collected_diagnostics
            result_dict["failure_reasons"] = failure_reasons

        return result_dict

    def _single_simulation(
        self,
        sim_id: int,
        sample_size: int,
        metadata: "SimulationMetadata",
        correlation_matrix: np.ndarray,
        var_types: np.ndarray,
        generate_y_func: Callable,
        analyze_func: Callable,
        create_X_extended_func: Callable,
        sim_seed: Optional[int] = None,
        scenario_config: Optional[Dict] = None,
    ) -> Any:
        """
        Execute a single Monte Carlo simulation.

        Args:
            sim_id: Simulation identifier
            sample_size: Number of observations
            metadata: Simulation metadata
            correlation_matrix: Correlation matrix (possibly perturbed)
            var_types: Variable types (possibly perturbed)
            generate_y_func: Y generation function
            analyze_func: Statistical analysis function
            create_X_extended_func: Design matrix extension function
            sim_seed: Random seed for this simulation
            scenario_config: Optional scenario config for LME perturbations

        Returns:
            Tuple of (uncorrected_significant, corrected_significant) arrays,
            or None if simulation failed
        """
        try:
            # Adjust sample_size for clusters if needed
            if metadata.cluster_specs:
                first_spec = next(iter(metadata.cluster_specs.values()))
                sample_size = first_spec["n_clusters"] * first_spec["cluster_size"]

            # Check if strict mode with uploaded data
            if metadata.preserve_correlation == "strict" and metadata.uploaded_raw_data is not None:
                # Strict mode: bootstrap uploaded data + generate created variables separately
                from ..stats.data_generation import bootstrap_uploaded_data

                # Bootstrap uploaded data (whole rows)
                X_uploaded_non_factors, X_uploaded_factors = bootstrap_uploaded_data(
                    sample_size,
                    metadata.uploaded_raw_data,
                    metadata.uploaded_var_metadata,
                    sim_seed,
                )

                # Merge uploaded and created non-factor variables
                if X_uploaded_non_factors.shape[1] == metadata.n_non_factor_vars:
                    # All non-factor variables are uploaded
                    X_non_factors = X_uploaded_non_factors
                else:
                    # Mixed: generate all non-factor vars, replace uploaded columns
                    X_non_factors = get_backend().generate_X(
                        sample_size,
                        metadata.n_non_factor_vars,
                        correlation_matrix,
                        var_types,
                        metadata.var_params,
                        metadata.upload_normal_values,
                        metadata.upload_data_values,
                        sim_seed if sim_seed is not None else -1,
                    )
                    # Overwrite uploaded columns with bootstrapped data
                    uploaded_nf_indices = [i for i, t in enumerate(var_types) if t in (98, 99)]
                    for j, full_idx in enumerate(uploaded_nf_indices):
                        X_non_factors[:, full_idx] = X_uploaded_non_factors[:, j]

                # Merge uploaded and created factor variables
                total_expected_dummies = sum(s["n_levels"] - 1 for s in metadata.factor_specs) if metadata.factor_specs else 0
                if X_uploaded_factors.shape[1] == total_expected_dummies:
                    # All factors are uploaded (or no factors at all)
                    X_factors = X_uploaded_factors
                else:
                    # Mixed: generate all factors, replace uploaded factor columns
                    X_factors = _generate_factors(sample_size, metadata.factor_specs, sim_seed)
                    # Overwrite uploaded factor dummy columns with bootstrapped data
                    if X_uploaded_factors.shape[1] > 0:
                        col_offset = 0
                        uploaded_col_offset = 0
                        for fname, spec in zip(metadata.factor_names, metadata.factor_specs, strict=False):
                            n_dummies = spec["n_levels"] - 1
                            # Check if this factor was uploaded
                            if fname in metadata.uploaded_var_metadata and metadata.uploaded_var_metadata[fname].get("type") == "factor":
                                X_factors[:, col_offset : col_offset + n_dummies] = X_uploaded_factors[
                                    :, uploaded_col_offset : uploaded_col_offset + n_dummies
                                ]
                                uploaded_col_offset += n_dummies
                            col_offset += n_dummies

            else:
                # Normal mode ('no' or 'partial'): standard pipeline
                # Generate non-factor variables
                if metadata.n_non_factor_vars > 0:
                    X_non_factors = get_backend().generate_X(
                        sample_size,
                        metadata.n_non_factor_vars,
                        correlation_matrix,
                        var_types,
                        metadata.var_params,
                        metadata.upload_normal_values,
                        metadata.upload_data_values,
                        sim_seed if sim_seed is not None else -1,
                    )
                else:
                    X_non_factors = np.empty((sample_size, 0), dtype=float)

                # Generate factor variables (as dummy variables)
                X_factors = _generate_factors(sample_size, metadata.factor_specs, sim_seed)

            # Compute LME perturbations (ICC jitter, non-normal RE dist)
            lme_perturbations = None
            if metadata.cluster_specs and scenario_config is not None:
                from ..core.scenarios import apply_lme_perturbations

                lme_perturbations = apply_lme_perturbations(metadata.cluster_specs, scenario_config, sim_seed)

            # Generate cluster random effects (independent of upload mode)
            re_result = None  # Phase 2: random effects result for slopes/nesting
            if metadata.cluster_specs:
                if metadata.has_random_slopes or metadata.has_nested:
                    from ..stats.data_generation import _generate_random_effects

                    re_result = _generate_random_effects(
                        sample_size=sample_size,
                        cluster_specs=metadata.cluster_specs,
                        X_non_factors=X_non_factors,
                        non_factor_names=metadata.non_factor_names,
                        sim_seed=sim_seed,
                        lme_perturbations=lme_perturbations,
                    )
                    X_cluster = re_result.intercept_columns
                else:
                    from ..stats.data_generation import _generate_cluster_effects

                    X_cluster = _generate_cluster_effects(
                        sample_size=sample_size,
                        cluster_specs=metadata.cluster_specs,
                        sim_seed=sim_seed,
                        lme_perturbations=lme_perturbations,
                    )
            else:
                X_cluster = np.empty((sample_size, 0), dtype=float)

            # Merge: non-factors first, then cluster effects, then factors
            parts = [p for p in [X_non_factors, X_cluster, X_factors] if p.shape[1] > 0]
            if len(parts) > 1:
                X = np.column_stack(parts)
            elif len(parts) == 1:
                X = parts[0]
            else:
                X = np.empty((sample_size, 0), dtype=float)

            # Create extended design matrix with interactions (excludes cluster effects)
            X_expanded = create_X_extended_func(X)

            # Split effect sizes: fixed effects vs cluster effects
            # Use precomputed values (Phase 2 optimization)
            if metadata.cluster_effect_indices:
                fixed_effect_sizes = metadata.fixed_effect_sizes_cached
                cluster_effect_sizes = metadata.effect_sizes[metadata.cluster_effect_indices]
            else:
                fixed_effect_sizes = metadata.fixed_effect_sizes_cached
                cluster_effect_sizes = None

            # Generate dependent variable with fixed effects only
            y = generate_y_func(
                X_expanded=X_expanded,
                effect_sizes_expanded=fixed_effect_sizes,
                heterogeneity=metadata.heterogeneity,
                heteroskedasticity=metadata.heteroskedasticity,
                sim_seed=sim_seed,
            )

            # Add cluster random effects contribution
            if cluster_effect_sizes is not None:
                # Extract cluster effect columns from X using cluster_column_indices (positions in X)
                X_cluster_effects = X[:, metadata.cluster_column_indices]
                # Add cluster contribution: cluster_effects * effect_sizes (typically 1.0)
                cluster_contribution = X_cluster_effects @ cluster_effect_sizes
                y = y + cluster_contribution

            # Phase 2: add random slope contributions to y
            if re_result is not None and not np.allclose(re_result.slope_contribution, 0):
                y = y + re_result.slope_contribution

            # Apply LME residual perturbations (non-normal residuals)
            if metadata.cluster_specs and scenario_config is not None:
                from ..core.scenarios import apply_lme_residual_perturbations

                y = apply_lme_residual_perturbations(y, scenario_config, sim_seed)

            # Determine cluster IDs for the solver
            cluster_ids: Optional[np.ndarray]
            if re_result is not None:
                # Slopes/nested: use primary grouping var from re_result
                # For nested, the wrapper will use the full cluster_ids_dict
                first_gv = next(iter(re_result.cluster_ids_dict))
                cluster_ids = re_result.cluster_ids_dict[first_gv]
            else:
                # Simple intercept: use precomputed template (Phase 2 optimization)
                cluster_ids = metadata.cluster_ids_template

            # Route to correct analysis method
            if cluster_ids is not None:
                # Mixed model path (LME)
                from ..stats.mixed_models import _lme_analysis_wrapper

                lme_result = _lme_analysis_wrapper(
                    X_expanded,
                    y,
                    metadata.target_indices,
                    cluster_ids,
                    metadata.cluster_column_indices,
                    metadata.correction_method,
                    self.alpha,
                    backend="custom",
                    verbose=metadata.verbose,
                    chi2_crit=getattr(metadata, "lme_chi2_crit", None),
                    z_crit=getattr(metadata, "lme_z_crit", None),
                    correction_z_crits=getattr(metadata, "lme_correction_z_crits", None),
                    re_result=re_result,
                )

                # Check if LME convergence failed
                if metadata.verbose and isinstance(lme_result, dict):
                    if lme_result["results"] is None:
                        # Failed convergence
                        return {
                            "failed": True,
                            "failure_reason": lme_result.get("failure_reason", "LME convergence failed"),
                            "sim_id": sim_id,
                        }
                    else:
                        # Success with diagnostics
                        results = lme_result["results"]
                        diagnostics = lme_result.get("diagnostics")
                elif lme_result is None:
                    # Non-verbose mode failure
                    return None
                else:
                    # Non-verbose mode success
                    results = lme_result
                    diagnostics = None
            else:
                # Standard OLS path
                results = analyze_func(
                    X_expanded,
                    y,
                    metadata.target_indices,
                    self.alpha,
                    metadata.correction_method,
                )
                diagnostics = None

            # Extract results: [f_sig, uncorr..., corr..., (wald_flag)]
            n_targets = len(metadata.target_indices)
            f_significant = bool(results[0])
            uncorrected = results[1 : 1 + n_targets].astype(bool)
            corrected = results[1 + n_targets : 1 + 2 * n_targets].astype(bool)

            # Extract Wald-fallback flag if present (mixed models append it)
            wald_flag = False
            expected_len = 1 + 2 * n_targets
            if len(results) > expected_len:
                wald_flag = bool(results[expected_len])

            # Post-hoc pairwise contrasts (OLS path only)
            if metadata.posthoc_specs and cluster_ids is None:
                from ..stats.ols import compute_posthoc_contrasts

                ph_uncorr, ph_corr, regular_override = compute_posthoc_contrasts(
                    X_expanded,
                    y,
                    metadata.posthoc_specs,
                    metadata.posthoc_method,
                    metadata.posthoc_t_crit,
                    metadata.posthoc_tukey_crits,
                    target_indices=metadata.target_indices,
                    correction_method=metadata.correction_method,
                    correction_t_crits_combined=getattr(metadata, "posthoc_correction_t_crits_combined", None),
                )

                # If FDR/Holm combined correction was applied, override regular corrected
                if regular_override is not None:
                    corrected = regular_override

                uncorrected = np.concatenate([uncorrected, ph_uncorr])
                corrected = np.concatenate([corrected, ph_corr])

            # Add F-test to beginning
            sim_significant = np.concatenate([[f_significant], uncorrected])
            sim_significant_corrected = np.concatenate([[f_significant], corrected])

            if metadata.verbose and diagnostics is not None:
                return {
                    "results": (sim_significant, sim_significant_corrected),
                    "diagnostics": diagnostics,
                    "wald_fallback": wald_flag,
                    "sim_id": sim_id,
                }
            else:
                return sim_significant, sim_significant_corrected, wald_flag

        except ImportError:
            raise  # Don't swallow missing-backend errors
        except Exception as e:
            if metadata.verbose:
                return {"failed": True, "failure_reason": f"{type(e).__name__}: {str(e)}", "error_type": type(e).__name__, "sim_id": sim_id}
            else:
                return None


def _warn_icc_mismatch(metadata: "SimulationMetadata", mean_estimated_icc: float) -> None:
    """Warn if estimated ICC differs substantially from the user-specified ICC."""
    for gv, spec_dict in metadata.cluster_specs.items():
        specified_icc = spec_dict.get("icc")
        if specified_icc is None or specified_icc <= 0:
            continue
        if mean_estimated_icc <= 0:
            continue
        relative_diff = abs(mean_estimated_icc - specified_icc) / specified_icc
        if relative_diff > 0.50:
            warnings.warn(
                f"Estimated ICC ({mean_estimated_icc:.3f}) differs from specified "
                f"ICC ({specified_icc:.3f}) by {relative_diff:.0%} for grouping "
                f"variable '{gv}'. This may indicate model misspecification, "
                f"insufficient cluster size, or extreme effect sizes."
            )


class SimulationMetadata:
    """Pre-computed metadata for simulation execution.

    Holds all static data needed for running simulations, avoiding
    repeated extraction from the model on every iteration.  Created once
    by ``prepare_metadata`` and passed to ``SimulationRunner``.

    Attributes:
        target_indices: Effect-order indices of the effects being tested.
        n_non_factor_vars: Number of continuous/binary predictor columns.
        correlation_matrix: Correlation matrix for non-factor predictors.
        var_types: Integer-coded distribution types for each non-factor
            predictor (0=normal, 1=binary, 2=right_skewed, …, 99=uploaded).
        var_params: Per-variable parameters (binary proportions, etc.).
        factor_specs: List of dicts describing each factor (n_levels,
            proportions).
        upload_normal_values: Normal quantiles for uploaded-data lookup.
        upload_data_values: Empirical quantiles for uploaded-data lookup.
        effect_sizes: Full array of standardised effect sizes.
        correction_method: Encoded multiple-comparison correction
            (0=none, 1=Bonferroni, 2=BH, 3=Holm).
        heterogeneity: SD of random effect-size multiplier.
        heteroskedasticity: Correlation between first predictor and error SD.
        preserve_correlation: Upload correlation mode
            (``"no"``/``"partial"``/``"strict"``).
        uploaded_raw_data: Normalised raw data for strict-mode bootstrap.
        uploaded_var_metadata: Per-variable metadata from uploaded data.
        cluster_specs: Dict of cluster specifications for mixed models.
        n_cluster_effect_vars: Number of cluster random-effect columns.
        cluster_column_indices: Positions of cluster-effect columns in ``X``.
        cluster_effect_indices: Positions of cluster effects in
            ``effect_sizes``.
        factor_names: Ordered list of factor variable names.
        verbose: Whether to collect per-simulation diagnostics.
    """

    def __init__(
        self,
        target_indices: np.ndarray,
        n_non_factor_vars: int,
        correlation_matrix: np.ndarray,
        var_types: np.ndarray,
        var_params: np.ndarray,
        factor_specs: List[Dict],
        upload_normal_values: np.ndarray,
        upload_data_values: np.ndarray,
        effect_sizes: np.ndarray,
        correction_method: int,
        heterogeneity: float = 0.0,
        heteroskedasticity: float = 0.0,
        preserve_correlation: str = "partial",
        uploaded_raw_data: Optional[np.ndarray] = None,
        uploaded_var_metadata: Optional[dict] = None,
        cluster_specs: Optional[Dict[str, Dict]] = None,
        n_cluster_effect_vars: int = 0,
        cluster_column_indices: Optional[List[int]] = None,
        cluster_effect_indices: Optional[List[int]] = None,
        factor_names: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.target_indices = target_indices
        self.n_non_factor_vars = n_non_factor_vars
        self.correlation_matrix = correlation_matrix
        self.var_types = var_types
        self.var_params = var_params
        self.factor_specs = factor_specs
        self.upload_normal_values = upload_normal_values
        self.upload_data_values = upload_data_values
        self.effect_sizes = effect_sizes
        self.correction_method = correction_method
        self.heterogeneity = heterogeneity
        self.heteroskedasticity = heteroskedasticity
        self.preserve_correlation = preserve_correlation
        self.uploaded_raw_data = uploaded_raw_data
        self.uploaded_var_metadata = uploaded_var_metadata or {}
        self.cluster_specs = cluster_specs or {}
        self.n_cluster_effect_vars = n_cluster_effect_vars
        self.cluster_column_indices = cluster_column_indices or []
        self.cluster_effect_indices = cluster_effect_indices or []
        self.factor_names = factor_names or []
        self.verbose = verbose

        # LME scenario config (stored when scenario mode + cluster_specs)
        self.lme_scenario_config: Optional[Dict] = None

        # Precomputed values for performance (Phase 2 optimizations)
        self.cluster_ids_template: Optional[np.ndarray] = None  # Precomputed cluster ID array
        self.fixed_effect_mask: Optional[np.ndarray] = None  # Precomputed boolean mask for fixed effects
        self.fixed_effect_sizes_cached: Optional[np.ndarray] = None  # Precomputed fixed effect sizes

        # Precomputed LME critical values (custom solver)
        self.lme_chi2_crit = None
        self.lme_z_crit = None
        self.lme_correction_z_crits = None

        # Phase 2: random slopes and nesting metadata
        self.has_random_slopes: bool = False
        self.has_nested: bool = False
        self.non_factor_names: List[str] = []

        # Post-hoc comparison fields
        self.posthoc_specs: List = []
        self.posthoc_method: str = "t-test"
        self.posthoc_tukey_crits: Dict[str, float] = {}
        self.posthoc_t_crit: float = 0.0


def _compute_fixed_effect_variance(registry) -> float:
    """
    Compute Σ β²_j × Var(X_j) for all fixed effects (excluding cluster effects).

    This is needed to correct the ICC-to-τ² conversion. The naive formula
    τ² = ICC / (1 - ICC) assumes within-cluster variance equals σ²_error = 1.
    In reality, within-cluster variance = σ²_error + Var(Xβ), so the correct
    formula is: τ² = ICC / (1 - ICC) × (1 + Σ β²_j × Var(X_j)).

    Variance by variable type:
    - Normal/skewed/uniform/uploaded/high-kurtosis: Var = 1.0 (standardized)
    - Binary / uploaded binary: Var = p(1-p)
    - Factor dummy for level k: Var = p_k(1-p_k)
    - Interaction: product of component variances (valid when means ≈ 0)

    Args:
        registry: VariableRegistry instance

    Returns:
        Total fixed-effect variance Σ β²_j × Var(X_j)
    """
    cluster_effect_names = set(registry.cluster_effect_names)

    # Build predictor → variance map
    predictor_variance = {}

    # Non-factor variables
    non_factor_names = registry.non_factor_names
    var_types = registry.get_var_types()
    var_params = registry.get_var_params()

    for i, name in enumerate(non_factor_names):
        vtype = int(var_types[i])
        if vtype in (1, 98):  # binary or uploaded binary
            p = var_params[i]
            predictor_variance[name] = p * (1.0 - p)
        else:
            # Normal (0), right_skewed (2), left_skewed (3),
            # high_kurtosis (4), uniform (5), uploaded_data (99):
            # all standardized to Var = 1.0 in the DGP
            predictor_variance[name] = 1.0

    # Factor dummy variables
    for dummy_name, dummy_info in registry._factor_dummies.items():
        factor_name = dummy_info["factor_name"]
        level = dummy_info["level"]
        factor_info = registry._factors[factor_name]
        proportions = factor_info.get("proportions")
        if proportions is not None:
            # level is 1-indexed; proportions list is 0-indexed
            p_k = proportions[level - 1]
        else:
            # Equal proportions (default)
            n_levels = factor_info["n_levels"]
            p_k = 1.0 / n_levels
        predictor_variance[dummy_name] = p_k * (1.0 - p_k)

    # Sum β²_j × Var(X_j) over all fixed effects
    total_variance = 0.0
    for effect_name, effect in registry._effects.items():
        if effect_name in cluster_effect_names:
            continue

        beta_sq = effect.effect_size**2
        if beta_sq == 0.0:
            continue

        if effect.effect_type == "main":
            var_x = predictor_variance.get(effect_name, 1.0)
            total_variance += beta_sq * var_x
        else:
            # Interaction: product of component variances
            var_product = 1.0
            for var_name in effect.var_names:
                var_product *= predictor_variance.get(var_name, 1.0)
            total_variance += beta_sq * var_product

    return total_variance


def prepare_metadata(
    model,
    target_tests: List[str],
    correction: Optional[str] = None,
) -> SimulationMetadata:
    """
    Prepare simulation metadata from model state.

    This function extracts and pre-computes all data needed for running
    simulations, converting model state into efficient numpy arrays.

    Args:
        model: MCPowerModel instance
        target_tests: List of effects to test
        correction: Multiple comparison correction method

    Returns:
        SimulationMetadata instance
    """
    registry = model._registry

    # Get effect order and target indices from registry
    target_indices = registry.get_target_indices(target_tests)

    # Get non-factor variables only for data generation
    non_factor_vars = registry.non_factor_names
    n_non_factor_vars = len(non_factor_vars)

    # Correlation matrix - only for non-factor variables
    if n_non_factor_vars == 0:
        correlation_matrix = np.eye(1)
    else:
        corr = registry.get_correlation_matrix()
        if corr is None:
            correlation_matrix = np.eye(n_non_factor_vars)
        else:
            correlation_matrix = corr

    # Variable types and parameters directly from registry
    if n_non_factor_vars > 0:
        var_types = registry.get_var_types()
        var_params = registry.get_var_params()
    else:
        var_types = np.zeros(1, dtype=np.int64)
        var_params = np.zeros(1, dtype=np.float64)

    # Factor specifications from registry
    factor_specs = registry.get_factor_specs()

    # Effect sizes from registry
    effect_sizes = registry.get_effect_sizes()

    # Correction method encoding
    correction_method = 0
    is_tukey_correction = False
    if correction:
        method = correction.lower().replace("-", "_").replace(" ", "_")
        if method == "bonferroni":
            correction_method = 1
        elif method in ["benjamini_hochberg", "bh", "fdr"]:
            correction_method = 2
        elif method == "holm":
            correction_method = 3
        elif method == "tukey":
            correction_method = 0  # No correction for regular effects
            is_tukey_correction = True

    # Extract cluster specifications (including Phase 2 fields)
    cluster_specs = {}
    has_random_slopes = False
    has_nested = False
    for gv, spec in registry._cluster_specs.items():
        cluster_specs[gv] = {
            "grouping_var": spec.grouping_var,
            "n_clusters": spec.n_clusters,
            "cluster_size": spec.cluster_size,
            "tau_squared": spec.tau_squared,
            "icc": spec.icc,
            "id_effect_name": spec.id_effect_name,
            "random_slope_vars": list(spec.random_slope_vars),
            "slope_variance": spec.slope_variance,
            "slope_intercept_corr": spec.slope_intercept_corr,
            "G_matrix": spec.G_matrix.copy() if spec.G_matrix is not None else None,
            "parent_var": spec.parent_var,
            "n_per_parent": spec.n_per_parent,
            "q": spec.q,
        }
        if spec.q > 1:
            has_random_slopes = True
        if spec.parent_var is not None:
            has_nested = True

    # Adjust τ² for fixed-effect variance contribution
    if cluster_specs:
        sigma_sq_fixed = _compute_fixed_effect_variance(registry)
        sigma_sq_within = 1.0 + sigma_sq_fixed
        for gv, spec_dict in cluster_specs.items():
            original_icc = registry._cluster_specs[gv].icc
            if original_icc > 0:
                adjusted_tau_sq = (original_icc / (1.0 - original_icc)) * sigma_sq_within
                spec_dict["tau_squared"] = adjusted_tau_sq

                # If G_matrix exists (slope model), rebuild with adjusted intercept variance
                if spec_dict.get("G_matrix") is not None and spec_dict["q"] > 1:
                    new_tau_int = np.sqrt(adjusted_tau_sq)
                    tau_slope = np.sqrt(spec_dict["slope_variance"]) if spec_dict["slope_variance"] > 0 else 0.0
                    rho = spec_dict["slope_intercept_corr"]
                    q = spec_dict["q"]
                    G = np.zeros((q, q))
                    G[0, 0] = adjusted_tau_sq
                    for i in range(1, q):
                        G[i, i] = spec_dict["slope_variance"]
                        G[0, i] = rho * new_tau_int * tau_slope
                        G[i, 0] = rho * new_tau_int * tau_slope
                    spec_dict["G_matrix"] = G

    n_cluster_effect_vars = len(registry.cluster_effect_names)

    # Compute cluster column indices in TWO spaces:
    # 1. Positions in X (for extracting cluster effect values)
    # 2. Positions in effect_order (for indexing effect_sizes)
    cluster_column_indices_X = []
    cluster_effect_indices_in_effect_order = []

    if cluster_specs:
        # Positions in X: after non-factor vars, before factor dummies
        start_idx = n_non_factor_vars
        end_idx = start_idx + n_cluster_effect_vars
        cluster_column_indices_X = list(range(start_idx, end_idx))

        # Positions in effect_order: find cluster effects by name
        effect_order = list(registry._effects.keys())
        cluster_effect_names = registry.cluster_effect_names
        cluster_effect_indices_in_effect_order = [i for i, name in enumerate(effect_order) if name in cluster_effect_names]

    # No adjustment needed for target_indices anymore!
    # X_expanded now excludes cluster effects (see _create_X_extended),
    # so target_indices already reference the correct positions in X_expanded

    metadata = SimulationMetadata(
        target_indices=target_indices,
        n_non_factor_vars=n_non_factor_vars,
        correlation_matrix=correlation_matrix,
        var_types=var_types,
        var_params=var_params,
        factor_specs=factor_specs,
        upload_normal_values=model.upload_normal_values if model.upload_normal_values is not None else np.zeros((2, 2), dtype=np.float64),
        upload_data_values=model.upload_data_values if model.upload_data_values is not None else np.zeros((2, 2), dtype=np.float64),
        effect_sizes=effect_sizes,
        correction_method=correction_method,
        heterogeneity=model.heterogeneity,
        heteroskedasticity=model.heteroskedasticity,
        preserve_correlation=model._preserve_correlation,
        uploaded_raw_data=model._uploaded_raw_data,
        uploaded_var_metadata=model._uploaded_var_metadata,
        cluster_specs=cluster_specs,
        n_cluster_effect_vars=n_cluster_effect_vars,
        cluster_column_indices=cluster_column_indices_X,
        cluster_effect_indices=cluster_effect_indices_in_effect_order,
        factor_names=list(registry.factor_names),
    )

    # Phase 2: set slope/nesting flags and non-factor names
    metadata.has_random_slopes = has_random_slopes
    metadata.has_nested = has_nested
    metadata.non_factor_names = list(non_factor_vars)

    # Post-hoc specs from model
    if hasattr(model, "_posthoc_specs") and model._posthoc_specs:
        metadata.posthoc_specs = model._posthoc_specs
        metadata.posthoc_method = "tukey" if is_tukey_correction else "t-test"

    return metadata
