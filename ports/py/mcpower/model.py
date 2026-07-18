"""MCPower Python frontend over the Rust engine; deferred-apply chaining over ``mcpower._engine``.

The public ``MCPower`` class collects ``set_*`` configuration calls and
applies them lazily when ``find_power`` or ``find_sample_size`` is first
called, so callers can build up a model across multiple statements before
running any simulation.

Current caller-visible limitations:

  * ``family="ols"``, ``family="logit"``, and ``family="lme"`` are
    supported. LME covers random intercepts, random slopes (``(1 + x|g)``),
    crossed and nested groupings (``(1|a/b)``), and cluster-level
    (between-cluster) predictors — all configured through ``set_cluster``.
  * ``estimator=`` (alias ``solve_as=``) overrides the analysis estimator
    without changing the data-generating process. Accepted values: ``"ols"``,
    ``"glm"``, ``"mle"``, ``None`` (default: derive from ``family``).
  * ``set_parallel`` is removed (parallelism is automatic; use
    ``mcpower._engine.set_n_threads`` to override the thread count).
"""

from __future__ import annotations

import copy
import math
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from .spec.parsers import _parser
from .spec.scenario_config import get_default_scenario_config
from .spec.spec_builder import build_linear_spec, build_scenario_dict
from .spec.test_selector import overall_test_available, resolve_tests
from .spec.validators import (
    _validate_alpha,
    _validate_correction_method,
    _validate_correlation_matrix,
    _validate_estimator,
    _validate_family,
    _validate_power,
    _validate_sample_size,
    _validate_sample_size_for_model,
    _validate_sample_size_range,
    _validate_simulations,
    _validate_test_formula,
    _validate_upload_data,
)
from .spec.variables import VariableRegistry

import msgpack  # noqa: E402 — import after type stubs

# ── Wire encodings (Python → Rust LinearSpec) ─────────────────────────────
#
# The *scenario* sub-payload inside LinearSpec carries integer-coded
# distribution slots (engine-spec-builder/src/input.rs ::
# ScenarioInput.{new_distributions: Vec<i32>, residual_dists: Vec<i32>}). The
# name → code tables are single-sourced in the engine and read via
# `config.get_dist_codes()` / `config.get_residual_codes()` — no copy here.
#
# Residual public surface: the canonical five names normal/right_skewed/
# left_skewed/high_kurtosis/uniform (all valid for model and scenario pools).
# The engine returns only the canonical names from `residual_codes()`.


def _class_of(registry: "VariableRegistry", col_name: str) -> str:
    """Return the type class for a predictor in the registry.

    Maps the stored ``var_type`` to one of three class strings:
      ``"factor"``     — var_type == "factor"
      ``"binary"``     — var_type == "binary"
      ``"continuous"`` — everything else (normal, right_skewed, uploaded_data, …)

    Called from ``_apply`` to compare the modeled class against the detected
    class from uploaded data (upload type-lock guard).
    """
    pred = registry.get_predictor(col_name)
    if pred is None:
        return "continuous"
    vt = pred.var_type
    if vt == "factor":
        return "factor"
    if vt == "binary":
        return "binary"
    return "continuous"


def _reuse_fraction(U: int, N: int) -> float:
    """Expected % of uploaded rows reused within one strict-bootstrap dataset.

    A dataset of size N is drawn with replacement from U uploaded rows.
    Closed form: g = 100 * [1 - (1-1/U)^N - (N/U)*(1-1/U)^(N-1)].
    Guard: U <= 0 -> 0.0; U == 1 -> 100.0.
    """
    if U <= 0:
        return 0.0
    if U == 1:
        return 100.0
    p = 1.0 - 1.0 / U
    return 100.0 * (1.0 - p ** N - (N / U) * p ** (N - 1))


def _strict_reuse_warning(U: int, N: int, ratio: float) -> "Optional[str]":
    """Return a warning string when N > ratio*U (suggesting a lighter mode), else None."""
    if N > ratio * U:
        return (
            f"N={N} is more than {ratio:.4g}x the uploaded rows ({U}). "
            "Each strict-bootstrap dataset will reuse many rows; consider mode='partial' "
            "or mode='none' for a faster and more generalizable simulation."
        )
    return None


def _glmm_laplace_bias_warning(
    estimator_extras: "Dict[str, Any]",
    min_cluster_size: int,
    cfg: "Dict[str, Any]",
) -> "Optional[str]":
    """Return a Laplace-bias warning string for GLMM results, or None.

    Fires when both conditions hold:
      1. estimator_extras carries a GLM tau_squared_hat_mean above the
         configured threshold (report.thresholds.glmm_tau_sq_warn).
      2. min_cluster_size is strictly below limits.recommended_rows_per_cluster.
         (Not reliable_rows_per_cluster: the cluster-size validator already
         rejects configs below reliable, so the only reachable — and meaningful
         — warning band is [reliable, recommended): allowed but Laplace-risky.)

    The message is canonical (verbatim across all ports — mirrors the R port's
    .check_glmm_laplace_bias_warning; change all ports together).
    """
    tau = estimator_extras.get("tau_squared_hat_mean")
    if tau is None:
        return None
    thr: float = cfg["report"]["thresholds"]["glmm_tau_sq_warn"]
    # Use recommended (not reliable) so the warning covers the "risky but
    # allowed" zone (reliable ≤ obs/cluster < recommended); configurations
    # below reliable are already rejected by the validator.
    min_size: int = cfg["limits"]["recommended_rows_per_cluster"]
    if tau > thr and min_cluster_size < min_size:
        return (
            f"Laplace-approximation bias likely: estimated random-intercept "
            f"variance τ̂² = {tau:.2f} exceeds {thr:.2f} with small clusters "
            f"(min cluster size {min_cluster_size} < {min_size}). GLMM power "
            f"may be optimistic — interpret with caution or increase cluster size."
        )
    return None


def _surface_warnings(result: Dict[str, Any]) -> None:
    """Surface engine ``grid_warnings`` exactly once per distinct message.

    Handles both the single-scenario result dict and the multi-scenario
    envelope (deduped across scenarios). Shared by ``find_power`` and
    ``find_sample_size``; ``stacklevel=3`` keeps the warning pointing at the
    user's ``find_*`` call site.
    """
    if "scenarios" in result:
        seen_gw: set = set()
        for sc_dict in result["scenarios"].values():
            for w in sc_dict.get("grid_warnings", []) or []:
                if w not in seen_gw:
                    seen_gw.add(w)
                    warnings.warn(w, UserWarning, stacklevel=3)
    else:
        for w in result.get("grid_warnings", []) or []:
            warnings.warn(w, UserWarning, stacklevel=3)


# Latent-scale ICC→τ² conversion factor for binary (logistic) outcomes.
# For GLMM with logistic link the random-intercept variance lives on the
# latent (log-odds) scale; the residual variance of a standard logistic
# is π²/3. So τ²_latent = ICC/(1−ICC) · π²/3, while the Gaussian formula
# is τ² = ICC/(1−ICC) (σ²=1). Both collapse to 0 at ICC=0.
# Mirrors the R port's .encode_outcome_and_clusters binary branch — change together.
_PI_SQ_OVER_3: float = math.pi ** 2 / 3.0


class MCPower:
    """Monte Carlo Power Analysis — OLS, Logit, and LME (mixed-effects) designs.

  Two independent axes control every analysis:
    * ``family=`` — the data-generating process (outcome type + whether clusters
      exist). Maps to ``outcome.kind`` and the generation spec.
    * ``estimator=`` / ``solve_as=`` — the statistical model fitted to each
      simulated dataset. Defaults to the correctly-specified match for the DGP;
      override to study misspecification (e.g. ``estimator="ols"`` on a
      clustered DGP to quantify the cost of ignoring clustering).

    Public surface mirrors :class:`mcpower.MCPower` from v1; LME now covers
    random slopes, crossed/nested groupings, and cluster-level predictors
    (all via ``set_cluster``). See the module docstring for the few v1 entry
    points that remain dropped (e.g. ``set_parallel``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        formula: str,
        *,
        family: str = "ols",
        estimator: Optional[str] = None,
        solve_as: Optional[str] = None,
    ) -> None:
        # Family gate — OLS, logit, and lme are supported. The validator
        # rejects unknown strings / non-strings with a friendly message.
        family_result = _validate_family(family)
        family_result.raise_if_invalid()
        family_norm = family.lower()
        self.family: str = family_norm  # "ols"|"logit"|"probit"|"poisson"|"lme"

        # estimator= / solve_as= override.  solve_as is a synonym; estimator wins
        # when both are supplied.
        _estimator_raw: Optional[str] = estimator if estimator is not None else solve_as
        est_result = _validate_estimator(_estimator_raw)
        est_result.raise_if_invalid()

        # Derive outcome_kind + link from family.
        #   "logit"/"probit" → binary  (probit sets the non-canonical link)
        #   "poisson"        → count
        #   "ols"/"lme"      → continuous
        if family_norm in ("logit", "probit"):
            self.outcome_kind: str = "binary"
        elif family_norm == "poisson":
            self.outcome_kind = "count"
        else:
            self.outcome_kind = "continuous"
        # Non-canonical link override sent on the wire ("probit"), else canonical.
        self.link: str = "probit" if family_norm == "probit" else "canonical"

        # Default estimator coupling (matches spec-builder default coupling):
        #   binary/count outcome (logit/probit/poisson) → "glm" (GLMM when clustered)
        #   cluster present (lme) → "mle"
        #   else → "ols"
        # The user may override via estimator= / solve_as=.
        _default_estimator: str
        if family_norm in ("logit", "probit", "poisson"):
            _default_estimator = "glm"
        elif family_norm == "lme":
            _default_estimator = "mle"
        else:
            _default_estimator = "ols"

        # self.estimator is what is sent on the wire; it may differ from the DGP.
        self.estimator: str = _estimator_raw.lower() if _estimator_raw is not None else _default_estimator

        # Core configuration — defaults from configs/config.json.
        from .config import get_simulation_defaults
        _sim = get_simulation_defaults()
        self.seed: Optional[int] = _sim["seed"]
        self.power: float = _sim["target_power"] * 100.0
        self.alpha: float = _sim["alpha"]
        # Family-aware: lme fits are more expensive, so they default to the
        # lighter `mixed` budget; OLS and logit (GLM) use the `ols` budget.
        self.n_simulations: int = _sim["n_sims"]["mixed" if family_norm == "lme" else "ols"]
        # Post-batch gate — fraction of unconverged sims tolerated before raising.
        self.max_failed_simulations: float = _sim["max_failed_fraction"]

        # Variable registry, parsed from the formula.
        self._registry = VariableRegistry(formula)

        # Scenario configs start at the defaults; ``set_scenario_configs``
        # merges user overrides on top.
        self._scenario_configs: Dict[str, Dict[str, Any]] = get_default_scenario_config()

        # Residual distribution name; True when the user explicitly chose it
        # (incl. explicit "normal") — scenario residual swaps leave it alone.
        self._residual_dist_name: str = "normal"
        self._residual_pinned: bool = False

        # Heteroskedasticity spec. Mirrors ``HeteroskedasticityInput`` from the
        # engine spec-builder: driver_var_index = 0-based non-factor predictor
        # index (None → Xβ). λ is scenario-only — no model-level ratio key.
        self._heteroskedasticity: Dict[str, Any] = {"driver_var_index": None}

        # Pending settings — applied lazily before find_power / find_sample_size.
        # The three assignment-string setters ACCUMULATE: each call appends a
        # fragment to its list and _apply replays them in order (last-wins
        # per key in the registry). Overwriting would silently drop every
        # earlier declaration — so chained/separate calls (declaring one
        # predictor at a time, the natural incremental pattern) kept only the
        # last, demoting earlier factors to continuous N(0,1) columns. (A single
        # combined string also works — the parser is paren-aware — but separate
        # calls must be correct too.)
        self._pending_variable_types: List[str] = []
        self._pending_effects: List[str] = []
        # Ordered list whose elements are either a string/dict fragment or a
        # full matrix (list-of-lists); see set_correlations for the matrix-resets
        # rule and _apply step 5 for the per-element dispatch.
        self._pending_correlations: List[Any] = []
        self._pending_data: Optional[Dict[str, Any]] = None
        self._applied: bool = False
        self._effects_set: bool = False

        # Binary-outcome state. _pending_baseline_probability stays set after
        # _apply_baseline_probability runs (v1 semantics — used by the runtime
        # gate in _validate_logit_runtime). intercept is logit(p) (logit) or
        # Φ⁻¹(p) (probit) once set_baseline_probability has been called.
        self._pending_baseline_probability: Optional[float] = None
        # Poisson-outcome state. Baseline rate λ₀ > 0; intercept becomes ln(λ₀).
        self._pending_baseline_rate: Optional[float] = None
        self.intercept: float = 0.0

        # Empirical data (uploaded): stored in slots that match
        # engine-core spec fields.
        self._uploaded_data_n: int = 0
        self._uploaded_data_mode: Optional[str] = None

        # LME state. ``_pending_clusters`` holds {grouping_var: {n_clusters,
        # cluster_size, icc}} as stored by ``set_cluster``. The
        # ``_effective_n_clusters`` transient is populated by
        # ``_validate_lme_runtime`` (which derives missing n_clusters from
        # ``sample_size``) and consumed by ``to_simulation_spec`` to fill
        # the engine's ``ClusterSpec``.
        self._pending_clusters: Dict[str, Dict[str, Any]] = {}
        self._effective_n_clusters: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        """Loud, instructive errors for the removed ``set_*`` setters.

        ``target_test`` and ``correction`` are now per-call kwargs on
        ``find_power`` / ``find_sample_size`` (not model state), so the v2
        ``set_tests`` / ``set_correction_method`` setters were removed. Intercept
        only those two names here and point the caller at the new kwargs;
        every other missing attribute re-raises a normal ``AttributeError``.

        ``__getattr__`` only fires on lookups that ordinary resolution missed,
        so this never shadows a real attribute.
        """
        if name == "set_tests":
            raise AttributeError(
                "MCPower.set_tests was removed — pass target_test=... directly to "
                "find_power()/find_sample_size() instead, e.g. "
                "find_power(160, target_test='all, -x2')."
            )
        if name == "set_correction_method":
            raise AttributeError(
                "MCPower.set_correction_method was removed — pass correction=... "
                "directly to find_power()/find_sample_size() instead, e.g. "
                "find_power(160, correction='bh')."
            )
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def equation(self) -> str:
        """Original formula string passed to the constructor."""
        return self._registry.equation

    @property
    def correlation_matrix(self) -> Optional[List[List[float]]]:
        """Correlation matrix for non-factor predictors (read-only, list-of-lists)."""
        return self._registry.get_correlation_matrix()

    @property
    def predictor_vars_order(self) -> List[str]:
        """Ordered predictor variables (continuous + dummies)."""
        return self._registry.non_factor_names + self._registry.dummy_names

    # ------------------------------------------------------------------
    # Dropped / not-yet-implemented entry points
    # ------------------------------------------------------------------

    def set_parallel(self, *_args: Any, **_kwargs: Any) -> None:
        raise AttributeError(
            "mcpower has no set_parallel — parallelism is automatic and "
            "controlled by mcpower._engine.set_n_threads(n)."
        )

    def set_cluster(
        self,
        grouping_var: str,
        ICC: Optional[float] = None,
        n_clusters: Optional[int] = None,
        cluster_size: Optional[int] = None,
        *,
        tau_squared: Optional[float] = None,
        random_slopes: Optional[List[str]] = None,
        slope_variance: float = 0.0,
        slope_intercept_corr: float = 0.0,
        n_per_parent: Optional[int] = None,
        cluster_level_vars: Optional[List[str]] = None,
    ) -> "MCPower":
        """Configure one grouping factor of a clustered (mixed-effects) DGP.

        Used with ``family='lme'`` (default: MLE estimator) or with
        ``family='lme', estimator='ols'`` (fit OLS to clustered data — the
        study of what ignoring clustering costs).  Call once per grouping
        factor; random intercepts, random slopes, crossed and nested
        groupings, and cluster-level predictors are all supported.

        Args:
            grouping_var: Name of the grouping variable (must match a
                random-effect term in the formula).
            ICC: Intraclass correlation coefficient (0 <= ICC < 1). 0 is
                allowed (degenerate / no clustering); non-zero values must
                lie in [0.1, 0.9] for numerical stability. This is the
                *conditional* (residual) ICC ``tau^2 / (tau^2 + sigma^2)`` --
                the within-cluster correlation after the predictors are
                accounted for; the random intercept is sized as
                ``tau^2 = ICC / (1 - ICC)``. The raw outcome's *marginal* ICC is
                lower whenever the fixed effects explain variance (they add
                ``Var(X*beta)`` to the denominator), so the observed correlation
                of the generated ``y`` can sit below the value you set --
                expected, not a bug.
            tau_squared: Raw random-intercept variance τ² (>= 0), for
                ``family='poisson'`` only. A log-link count model has no
                standard latent-scale ICC, so Poisson mixed models are sized by
                τ² directly instead of ``ICC=``. Passing both, or passing
                ``tau_squared`` for a non-Poisson family, is an error.
            n_clusters: Number of clusters. Mutually exclusive with
                ``cluster_size``.
            cluster_size: Number of observations per cluster. Mutually
                exclusive with ``n_clusters``; n_clusters is derived from
                ``sample_size`` at ``find_power`` time.
            random_slopes: Predictor names whose slopes vary randomly across
                groups. Each name must match a predictor in the model formula.
                The variance and intercept-correlation of the slope distribution
                are controlled by slope_variance and slope_intercept_corr.
                Passed raw (unconverted) to the engine SlopeTerm.
            slope_variance: Variance of the random-slope distribution (raw,
                passed unconverted to SlopeTerm). 0.0 disables the slope term.
            slope_intercept_corr: Pearson correlation between the random
                intercept and each random slope (raw, passed unconverted to
                SlopeTerm).
            n_per_parent: For nested random effects ``(1|A/B)``: the number of
                B-level units per A-level unit (fixed, not derived from sample_size).
                Required when the grouping_var is of the form ``"A:B"`` (the parser
                encodes ``(1|A/B)`` as a composite ``"A:B"`` grouping).
            cluster_level_vars: Names of predictors in the model formula that
                are measured at the cluster level (i.e. constant within each
                cluster, varying across clusters). These are sent to the engine
                as ``cluster_level_vars`` in the ``LinearSpec`` payload so the
                DGP generator draws them once per cluster. Each name must be a
                predictor already in the formula via the registry.

                D3: Cross-type correlations (cluster-level vars with
                observation-level vars) are documented but not guaranteed —
                the engine applies the correlation spec on a best-effort basis
                for mixed-level designs.

                D4: Uploaded/resampled-bound predictor names cannot be used as
                cluster_level_vars (the engine re-uses the uploaded column
                directly and cannot reclassify its sampling level). Raises
                ValueError if a name resolves to an uploaded predictor.

        Returns:
            self: For method chaining.
        """
        # Nested-child detection via formula-parsed grouping vars: the parser
        # encodes `(1|A/B)` as a `parent_var` plus a composite "A:B" entry.
        parsed_re = self._registry._random_effects_parsed
        parsed_grouping_vars = [re["grouping_var"] for re in parsed_re]

        from .spec.validators import _validate_cluster_config

        # Poisson (count) mixed models size the random effect by a RAW τ² — no
        # standard latent-scale ICC exists for a log-link count model
        # (Decision 8). Every other family uses ICC. Gate the two so they can't
        # be mixed up.
        if self.family == "poisson":
            if ICC is not None:
                raise ValueError(
                    "family='poisson' sizes the random effect by tau_squared, "
                    "not ICC; pass tau_squared= (raw τ²) instead of ICC="
                )
            if tau_squared is None:
                tau_squared = 0.0
            if not (isinstance(tau_squared, (int, float)) and not isinstance(tau_squared, bool) and float(tau_squared) >= 0.0):
                raise ValueError(
                    f"tau_squared must be a non-negative number, got {tau_squared!r}"
                )
        elif tau_squared is not None:
            raise ValueError(
                f"tau_squared= is only for family='poisson'; family={self.family!r} "
                "sizes the random effect by ICC="
            )

        if ICC is None:
            ICC = 0.0
        _validate_cluster_config(
            grouping_var,
            ICC,
            n_clusters,
            cluster_size,
            parsed_grouping_vars,
        ).raise_if_invalid()

        if cluster_level_vars:
            _all_predictor_names = set(
                self._registry.non_factor_names + self._registry.factor_names
            )
            _uploaded_names: set = set()
            if getattr(self, "_pending_data", None) is not None:
                _uploaded_names = {
                    col_name
                    for col_name, *_ in self._pending_data["columns_typed"]
                }
            for _cv in cluster_level_vars:
                if _cv == grouping_var:
                    raise ValueError(
                        f"cluster_level_vars: {_cv!r} is the grouping variable; "
                        "a grouping variable cannot also be a cluster-level predictor"
                    )
                if _cv not in _all_predictor_names:
                    raise ValueError(
                        f"cluster_level_vars: {_cv!r} is not a predictor in the "
                        f"formula; valid predictors: {sorted(_all_predictor_names)}"
                    )
                if _cv in _uploaded_names:
                    raise ValueError(
                        f"cluster_level_vars: {_cv!r} is bound to uploaded/resampled "
                        "data — the engine cannot reclassify its sampling level. "
                        "Use a synthesized predictor instead (D4)."
                    )

        self._pending_clusters[grouping_var] = {
            "n_clusters": n_clusters,
            "cluster_size": cluster_size,
            "icc": float(ICC),
            # Poisson raw τ² (Decision 8); None for ICC-sized families.
            "tau_squared": float(tau_squared) if tau_squared is not None else None,
            "cluster_level_vars": list(cluster_level_vars) if cluster_level_vars else [],
            "n_per_parent": n_per_parent,
            "random_slopes": list(random_slopes) if random_slopes else [],
            "slope_variance": float(slope_variance),
            "slope_intercept_corr": float(slope_intercept_corr),
        }
        self._applied = False
        return self

    def set_baseline_probability(self, p: float) -> "MCPower":
        """Set the baseline probability for ``family="logit"`` / ``"probit"``.

        The baseline is the conditional probability of ``y=1`` when all
        predictors equal their reference value. It becomes the intercept on the
        link scale — ``log(p / (1 - p))`` for logit, ``Φ⁻¹(p)`` for probit —
        used as the constant term in every Monte Carlo iteration.

        Args:
            p: Probability in the open interval (0, 1).

        Returns:
            self: For method chaining.

        Raises:
            ValueError: ``family`` is not ``"logit"``/``"probit"``. Passing a
                probability for a Poisson model would silently overwrite
                whichever baseline was set last (rate or probability) with no
                warning; mirrors the ``set_cluster`` ICC/tau_squared gate.
        """
        if self.family not in ("logit", "probit"):
            raise ValueError(
                "set_baseline_probability is only for family='logit'/'probit'; "
                f"family={self.family!r} sizes the intercept by set_baseline_rate="
            )

        from .spec.validators import _validate_baseline_probability

        _validate_baseline_probability(p).raise_or_warn()
        self._pending_baseline_probability = float(p)
        self._applied = False
        return self

    def set_baseline_rate(self, rate: float) -> "MCPower":
        """Set the baseline event rate λ₀ for ``family="poisson"`` models.

        The baseline is the expected count when all predictors equal their
        reference value. It is converted to ``self.intercept = ln(λ₀)`` (the
        log-link intercept) and used as the constant term in every Monte Carlo
        iteration.

        Args:
            rate: Event rate λ₀ > 0.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: ``family`` is not ``"poisson"``. Passing a rate for a
                logit/probit model would silently overwrite whichever
                baseline was set last (rate or probability) with no warning;
                mirrors the ``set_cluster`` ICC/tau_squared gate.
        """
        if self.family != "poisson":
            raise ValueError(
                "set_baseline_rate is only for family='poisson'; "
                f"family={self.family!r} sizes the intercept by set_baseline_probability="
            )

        from .spec.validators import _validate_baseline_rate

        _validate_baseline_rate(rate).raise_or_warn()
        self._pending_baseline_rate = float(rate)
        self._applied = False
        return self

    # ------------------------------------------------------------------
    # Configuration setters
    # ------------------------------------------------------------------

    def set_seed(self, seed: Optional[int]) -> "MCPower":
        """Set the base RNG seed. ``None`` enables random seeding."""
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError("seed must be an integer or None")
            if seed < 0:
                raise ValueError("seed must be non-negative")
        self.seed = seed
        return self

    def set_power(self, power: float) -> "MCPower":
        """Set the target power (used by ``find_sample_size``)."""
        _validate_power(power).raise_if_invalid()
        self.power = float(power)
        return self

    def set_alpha(self, alpha: float) -> "MCPower":
        """Set the type-I error rate alpha. Any value in ``(0, 1)`` is accepted
        (the engine enforces that range); values above 0.25 emit a soft warning."""
        _validate_alpha(alpha).raise_or_warn()
        self.alpha = float(alpha)
        return self

    def set_simulations(self, n_simulations: int) -> "MCPower":
        """Set the per-call Monte Carlo simulation count."""
        n_sims, result = _validate_simulations(n_simulations)
        result.raise_if_invalid()
        self.n_simulations = n_sims
        return self

    def set_max_failed_simulations(self, fraction: float) -> "MCPower":
        """Maximum fraction of failed (rank-deficient) sims tolerated."""
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be between 0 and 1")
        self.max_failed_simulations = float(fraction)
        return self

    def set_effects(self, spec: Union[str, Dict[str, float]]) -> "MCPower":
        """Set standardised effect sizes.

        Accepts either a comma-separated ``"name=value"`` string (v1 form)
        or a dict ``{name: value}``.
        """
        if isinstance(spec, dict):
            self._pending_effects.append(", ".join(f"{k}={v}" for k, v in spec.items()))
        elif isinstance(spec, str):
            if not spec.strip():
                raise ValueError("effects spec cannot be empty")
            self._pending_effects.append(spec)
        else:
            raise TypeError("set_effects expects a string or dict")
        self._effects_set = True
        self._applied = False
        return self

    def set_correlations(
        self, spec: Union[str, List[List[float]], Dict[Tuple[str, str], float]]
    ) -> "MCPower":
        """Set correlations between non-factor predictors.

        Accepts a v1-style assignment string (``"corr(x1,x2)=0.3"``), a full
        correlation matrix (list-of-lists, or a numpy array — accepted by
        duck-typing, never importing numpy), or a dict keyed by ``(var1, var2)``.
        """
        from .data.upload import _is_module
        # Accumulation with matrix-resets semantics: a string/dict fragment is a
        # partial pairwise spec and APPENDS; a full matrix is a complete
        # specification and RESETS the accumulator (drops prior fragments). So a
        # string can only ever follow a matrix in the list, and _apply layers its
        # pairwise entries on top of that matrix (see _apply step 5).
        if isinstance(spec, dict):
            # Render dict back into the assignment-string form the parser uses.
            self._pending_correlations.append(
                ", ".join(f"corr({a},{b})={v}" for (a, b), v in spec.items())
            )
        elif _is_module(spec, "numpy"):
            self._pending_correlations = [spec.tolist()]  # ndarray → list-of-lists
        elif isinstance(spec, str):
            self._pending_correlations.append(spec)
        elif isinstance(spec, list):
            self._pending_correlations = [spec]  # full matrix resets the accumulator
        else:
            raise TypeError("set_correlations expects a string, matrix (list/ndarray), or dict")
        self._applied = False
        return self

    def set_variable_type(self, spec: str) -> "MCPower":
        """Set distribution types for predictor variables.

        Accepts the v1 assignment-string form (``"x1=binary, x2=factor(3)"``).
        """
        if not isinstance(spec, str):
            raise TypeError("set_variable_type expects a string")
        self._pending_variable_types.append(spec)
        self._applied = False
        return self

    def set_residual_distribution(self, name: str) -> "MCPower":
        """Set the residual error distribution and pin it against scenario swaps.

        Args:
            name: One of the canonical five: ``"normal"``, ``"right_skewed"``,
                ``"left_skewed"``, ``"high_kurtosis"``, ``"uniform"``.
                ``"high_kurtosis"`` uses the t distribution; its degrees of
                freedom come from the active scenario's ``residual_df`` knob.
                Calling this method — including with ``"normal"`` — pins the
                residual so scenario perturbations do not swap it.
        """
        from .config import get_residual_codes

        residual_codes = get_residual_codes()
        if name not in residual_codes:
            raise ValueError(
                f"unknown residual distribution {name!r}; must be one of "
                f"{sorted(residual_codes)}"
            )
        self._residual_dist_name = name
        self._residual_pinned = True
        return self

    def set_heteroskedasticity_driver(
        self,
        var: Optional[str] = None,
    ) -> "MCPower":
        """Set the predictor that drives heteroskedastic residual variance.

        Args:
            var: predictor name driving the variance, or None for the linear
                predictor Xβ. Must be a non-factor predictor.

        The variance ratio λ is scenario-only — set it via
        ``set_scenario_configs({"optimistic": {"heteroskedasticity_ratio": …}})``.
        """
        driver_var_index: Optional[int] = None
        if var is not None:
            non_factor = self._registry.non_factor_names
            if var not in non_factor:
                raise ValueError(
                    f"heteroskedasticity variable {var!r} must be a non-factor "
                    f"predictor; available: {non_factor}"
                )
            driver_var_index = non_factor.index(var)

        # Heteroskedasticity is an OLS-only DGP feature; warn for other families.
        if self.family not in ("ols", None):
            warnings.warn(
                f"set_heteroskedasticity_driver() has no effect for family={self.family!r}; "
                "heteroskedasticity is an OLS-only DGP parameter and will be ignored by the engine.",
                UserWarning,
                stacklevel=2,
            )

        self._heteroskedasticity = {"driver_var_index": driver_var_index}
        return self

    def _validate_correction_arg(self, correction: Optional[str]) -> None:
        """Validate a per-call ``correction=`` kwarg.

        Reuses :func:`_validate_correction_method` for the name check.
        Called from the ``find_power`` / ``find_sample_size`` entry points
        alongside the ``sample_size`` / ``test_formula`` validation.
        """
        _validate_correction_method(correction).raise_if_invalid()

    def _validate_wald_se_arg(self, wald_se: str) -> None:
        """Validate a per-call ``wald_se=`` kwarg.

        Accepts ``"hessian"`` and ``"rx"`` only.
        """
        key = (wald_se or "rx").lower().replace("-", "_").replace(" ", "_")
        if key not in {"hessian", "rx"}:
            raise ValueError(
                f"wald_se={wald_se!r} is not recognised. "
                "Valid values: 'rx' (default, fastmode) or 'hessian'."
            )

    def _agq_eligible(self) -> bool:
        """Whether the current design admits adaptive Gauss–Hermite quadrature
        (``agq > 1``). Mirrors the contract backstop (invariant 25) and glmm's
        ``assert_model_shape`` — change all three together: a Binary/Count GLMM
        with a single grouping factor and at most 3 random effects per group
        (intercept + slopes)."""
        if self.outcome_kind not in ("binary", "count"):
            return False
        # Clustered (GLMM) with exactly one grouping factor — crossed/nested
        # extra groupings (a second _pending_clusters entry) are ineligible.
        if len(self._pending_clusters) != 1:
            return False
        cfg = next(iter(self._pending_clusters.values()))
        n_re = 1 + len(cfg.get("random_slopes") or [])  # intercept + slopes
        return n_re <= 3

    def _resolve_estimation(self, wald_se, agq):
        """Resolve the ``wald_se`` / ``agq`` kwargs against the config defaults,
        validate them, and warn-and-strip an ineligible ``agq > 1`` to Laplace.

        Returns ``(wald_se_str, nagq_int)``. ``None`` inputs fall back to the
        ``configs/config.json`` ``estimation`` block (the cross-port home)."""
        from .config import get_estimation_defaults
        _est = get_estimation_defaults()
        if wald_se is None:
            wald_se = _est["wald_se"]
        self._validate_wald_se_arg(wald_se)
        nagq = int(_est["nagq"]) if agq is None else int(agq)
        if nagq < 1 or nagq > 25 or nagq % 2 == 0:
            raise ValueError(
                f"agq must be an odd integer in 1..=25, got {agq!r}"
            )
        if nagq > 1 and not self._agq_eligible():
            import warnings as _warnings
            _warnings.warn(
                f"agq={nagq} is not available for this design; running at "
                "agq=1 (Laplace). AGQ requires a clustered binary or count "
                "(logit/probit/poisson) model with a single grouping factor "
                "and at most 3 random effects per group.",
                stacklevel=3,
            )
            nagq = 1
        return wald_se, nagq

    def _resolve_tests(self, target_test: str) -> Dict[str, Any]:
        """Parse a ``target_test`` DSL string into the wire dict.

        Thin wrapper over :func:`mcpower.spec.test_selector.resolve_tests`,
        which holds the selection-DSL logic.
        """
        return resolve_tests(
            target_test,
            self._registry,
            overall_available=overall_test_available(
                self.estimator, self._registry
            ),
        )

    def upload_data(
        self,
        data: Any,
        columns: Optional[List[str]] = None,
        mode: str = "partial",
        verbose: bool = True,
    ) -> "MCPower":
        """Upload empirical data for predictor-driven simulation.

        ``mode`` controls how faithfully the uploaded predictor matrix X is
        reproduced in each simulated dataset:

        - ``"none"``    — match each predictor's empirical marginal; correlations
                          stay user-set (or zero if not set).
        - ``"partial"`` — marginals **plus** the measured predictor correlation matrix
                          installed as a default (overridable by ``set_correlations``).
        - ``"strict"``  — bootstrap actual rows (preserves the full empirical joint
                          distribution, including nonlinear dependencies).

        Column types are auto-detected and stored; the registry is updated for
        every predictor found in both the upload and the model formula.
        """
        from .config import get_config
        from .data.upload import detect_column_types, normalize_upload_input

        columns_data, cols = normalize_upload_input(data, columns)
        _validate_upload_data(columns_data, cols).raise_if_invalid()
        if mode not in {"none", "partial", "strict"}:
            raise ValueError(
                "mode must be one of 'none', 'partial', 'strict'"
            )
        cfg = get_config()["upload"]
        max_k = cfg["max_factor_k_soft"]
        max_ratio = cfg["max_factor_ratio"]
        types, labels_list = detect_column_types(columns_data, cols, max_k, max_ratio)
        n_uploaded_rows = len(columns_data[0]) if columns_data else 0

        # Store raw columns for ALL uploaded columns (D4 will need y + predictors).
        raw_columns: Dict[str, Any] = {}
        for i, col_name in enumerate(cols):
            raw_columns[col_name] = list(columns_data[i])

        # Identify columns that match modeled predictors (before _apply expansion).
        modeled_names = set(self._registry.predictor_names)
        columns_typed = []  # only matched predictors
        for i, col_name in enumerate(cols):
            if col_name not in modeled_names:
                continue
            col_type = types[i]
            col_labels = labels_list[i]
            raw_vals = list(columns_data[i])
            columns_typed.append((col_name, col_type, raw_vals, col_labels))

            # Update registry with detected type for matched predictors.
            if col_type == "factor":
                n_lvl = len(col_labels)
                self._registry.set_variable_type(
                    col_name,
                    "factor",
                    n_levels=n_lvl,
                    labels=col_labels,
                    reference=col_labels[0] if col_labels else None,
                )
            elif col_type == "binary":
                self._registry.set_variable_type(col_name, "binary")
            # continuous: leave as default "normal" — engine handles via upload path

        if verbose:
            print(f"Uploaded {n_uploaded_rows} rows, {len(cols)} columns.")
            for i, col_name in enumerate(cols):
                status = "matched" if col_name in modeled_names else "extra"
                print(f"  {col_name}: {types[i]} ({status})")

        self._pending_data = {
            "columns_typed": columns_typed,
            "raw_columns": raw_columns,
            "mode": mode,
            "uploaded_n": n_uploaded_rows,
        }
        self._uploaded_data_n = n_uploaded_rows
        self._uploaded_data_mode = mode
        self._applied = False
        return self

    def get_effects_from_data(self, y: str, *, verbose: bool = True) -> str:
        """Estimate standardized effect sizes from uploaded data (OLS).

        Fits the already-specified model against the uploaded predictor columns
        and the uploaded outcome column named ``y``, then returns a
        ``set_effects``-style string (``"x1=0.13, x2=0.41, ..."``) of the
        recovered standardized coefficients (intercept dropped). The string
        uses the canonical effect names so it parses back through
        :meth:`set_effects`.

        The returned values are an **approximation only** — standardization,
        the random-X assumption, and sampling error all bias them away from the
        population standardized coefficients. The method does **not** auto-apply
        the recovered effects; call ``set_effects`` yourself if you want them.

        Requirements:

          * uploaded data must be present (call :meth:`upload_data` first), and
          * every modeled main-effect predictor must be present as an upload
            column (a multivariable fit can't estimate an absent predictor).

        The fitted estimator follows the model family: OLS (linear) z-scores the
        outcome to recover the standardized β; GLM (logit) and MLE (mixed) fit
        the native outcome (raw 0/1 / raw response). Clustered recovery (linear
        mixed or logistic GLMM) is fixed-effects-only — the uploaded data must
        carry the grouping column.

        For a clustered model the verbose note also reports the **estimated
        ICC** recovered from the random-intercept variance (latent log-odds
        scale for logistic), with a copy-paste ``set_cluster(...)`` snippet for
        single-grouping models. Like the effects, the ICC is an approximation
        and is **not** auto-applied.

        Args:
            y: Name of the uploaded outcome column.
            verbose: When ``True`` (default), print the approximation note
                (and the estimated ICC for clustered models).

        Returns:
            A comma-separated ``"name=value"`` string of standardized effects.
        """
        pending = getattr(self, "_pending_data", None)
        if pending is None:
            raise RuntimeError(
                "no uploaded data; call upload_data(...) before "
                "get_effects_from_data()"
            )

        raw_columns: Dict[str, Any] = pending["raw_columns"]
        if y not in raw_columns:
            raise ValueError(
                f"outcome column {y!r} not found in uploaded data; "
                f"available columns: {sorted(raw_columns)}"
            )

        # Expand factors so dummy_names / effect_names match the canonical order.
        if not self._applied:
            self._apply()

        reg = self._registry
        nrow = int(pending["uploaded_n"])

        import json as _json

        from . import _engine

        # Validate that every modeled main-effect predictor is present in the
        # upload. columns_typed holds only predictors present in BOTH the upload
        # and the model formula; the design matrix itself is assembled engine-side
        # (build_recovery_design), single-sourced with the R / Tauri ports.
        present = {col_name for col_name, *_rest in pending["columns_typed"]}
        for name in reg.non_factor_names:
            if name not in present:
                raise ValueError(
                    f"predictor {name!r} is in the model but missing from the "
                    "uploaded data; get_effects_from_data needs every modeled "
                    "main-effect predictor as an upload column"
                )
        for factor_name in reg.factor_names:
            if factor_name not in present:
                raise ValueError(
                    f"factor {factor_name!r} is in the model but missing from "
                    "the uploaded data; get_effects_from_data needs every "
                    "modeled main-effect predictor as an upload column"
                )

        # set_effects is NOT required for the fit — the contract's coefficients
        # are unused by fit_uploaded_data (it fits the provided design directly).
        # Pass placeholder zero effects so the spec builder validates. The payload
        # carries the coded upload columns, which the engine reads to assemble the
        # recovery design [Intercept, non-factors, factor dummies, interactions].
        payload = self._to_linear_spec_dict(["optimistic"])
        for entry in payload["effects"]:
            entry["size"] = 0.0

        design_flat, semantic_names, ncol = _engine.build_recovery_design(
            _json.dumps(payload)
        )

        # Outcome scaling depends on the estimator: OLS z-scores it (recovers the
        # standardized β) via the shared engine helper; GLM (logit) and MLE
        # (mixed) fit it on its native scale (raw 0/1 / raw response).
        y_raw = [float(v) for v in raw_columns[y]]
        if self.estimator == "ols":
            outcome = _engine.standardize_continuous(y_raw)
        else:
            outcome = y_raw

        outcome_kind_wire, link_wire, estimator_wire, intercept_arg, clusters_json = (
            self._encode_outcome_and_clusters()
        )

        # Clustered recovery (linear mixed `mle` OR logistic GLMM `glm`) is
        # fixed-effects-only: the uploaded data must carry the grouping column.
        # Map its distinct values to contiguous 0-based cluster IDs and pin the
        # contract's cluster count to the data's distinct-group count so the
        # fitter's per-cluster buffers line up. Gated on the model being
        # clustered (not the estimator) so glm clusters reach the engine's GLMM
        # branch — otherwise cluster_ids=None and it never sees the groups.
        cluster_ids: Optional[List[int]] = None
        if self._pending_clusters:
            grouping_var = next(iter(self._pending_clusters), None)
            if grouping_var is None or grouping_var not in raw_columns:
                raise ValueError(
                    "clustered get_effects_from_data needs the grouping column "
                    f"{grouping_var!r} present in the uploaded data"
                )
            id_of: Dict[Any, int] = {}
            cluster_ids = []
            for v in raw_columns[grouping_var]:
                if v not in id_of:
                    id_of[v] = len(id_of)
                cluster_ids.append(id_of[v])
            clusters = _json.loads(clusters_json)
            if clusters:
                clusters[0]["sizing"] = {"FixedClusters": {"n_clusters": len(id_of)}}
                clusters_json = _json.dumps(clusters)

        _names, contracts_bytes, _skeleton_json = _engine.build_contract_from_spec(
            _json.dumps(payload),
            outcome_kind_wire,
            link_wire,
            estimator_wire,
            intercept_arg,
            clusters_json,
        )
        base_seed = int(self.seed) if self.seed is not None else 0
        fit = _engine.fit_uploaded_data(
            contracts_bytes,
            0,
            base_seed,
            design_flat,
            nrow,
            ncol,
            outcome,
            cluster_ids,
        )

        betas = fit["betas"]
        if len(betas) != len(semantic_names):
            raise RuntimeError(
                f"fit returned {len(betas)} betas but {len(semantic_names)} "
                "design columns were built; canonical column order mismatch"
            )

        # Map betas → semantic names, drop the intercept (index 0), format.
        parts = [
            f"{name}={round(float(beta), 4)}"
            for name, beta in zip(semantic_names[1:], betas[1:])
        ]
        effects_str = ", ".join(parts)

        if verbose:
            print(
                "Note: these effect sizes are an APPROXIMATION only — "
                "standardization, the random-X assumption, and sampling error "
                "all bias them away from the population values. They are NOT "
                "auto-applied; call set_effects(...) to use them."
            )

            # Clustered fits also recover the random-intercept variance: report
            # the estimated ICC so the user need not guess it for set_cluster.
            # variance_components[0] is the primary grouping's intercept variance
            # τ̂² — exactly the quantity set_cluster(ICC=...) reconstructs. A
            # degenerate (non-converged) fit yields an empty list; note that the
            # ICC is unavailable rather than crashing on the missing component.
            # Poisson has no residual-variance scale to form an ICC ratio
            # against (raw τ², not ICC-derived) — no meaningful ICC to report;
            # mirrors driver.rs's cluster_icc, which returns None for it.
            vc = fit["variance_components"]
            if self._pending_clusters and self.family != "poisson" and not vc:
                print(
                    "Estimated ICC: unavailable (the random-intercept fit did "
                    "not converge to a variance estimate)."
                )
            elif self._pending_clusters and self.family != "poisson":
                tau_sq = float(vc[0])
                if self.family == "probit":
                    # Probit's latent-variable residual variance is fixed at 1
                    # (Φ's scale), unlike logit's π²/3 — mirrors driver.rs.
                    icc = tau_sq / (tau_sq + 1.0)
                    scale_note = " (probit latent scale)"
                elif self.outcome_kind == "binary":
                    # Latent (log-odds) scale: residual variance is π²/3, not σ̂²
                    # (sigma_sq_hat is a 1.0 placeholder for the binomial fit).
                    # Inverse of the set_cluster latent conversion.
                    icc = tau_sq / (tau_sq + _PI_SQ_OVER_3)
                    scale_note = " (logit latent scale)"
                else:
                    icc = tau_sq / (tau_sq + float(fit["sigma_sq_hat"]))
                    scale_note = ""
                if len(self._pending_clusters) > 1:
                    # variance_components[0] is the primary grouping only; a
                    # single set_cluster snippet would misrepresent the others.
                    print(
                        f"Estimated ICC{scale_note} for the primary grouping: "
                        f"{icc:.4f} (APPROXIMATION; not auto-applied)."
                    )
                else:
                    grouping_var = next(iter(self._pending_clusters))
                    print(
                        f"Estimated ICC{scale_note}: {icc:.4f} (APPROXIMATION; "
                        f"not auto-applied). To use it: "
                        f"set_cluster({grouping_var!r}, ICC={round(icc, 4)})."
                    )

            # Binary fits also recover the baseline event probability: the
            # inverse link of the fitted intercept betas[0], undoing the forward
            # link (logit(p) or probit's Phi^-1(p)) the generator applies.
            # Reported for any binary outcome (clustered or not), unlike the
            # ICC, which is clustered-only.
            if self.outcome_kind == "binary":
                if self.family == "probit":
                    import statistics
                    p_hat = statistics.NormalDist().cdf(float(betas[0]))
                else:
                    p_hat = 1.0 / (1.0 + math.exp(-float(betas[0])))
                print(
                    f"Estimated baseline probability: {p_hat:.4f} (APPROXIMATION; "
                    f"not auto-applied). To use it: "
                    f"set_baseline_probability({round(p_hat, 4)})."
                )

        return effects_str

    def set_scenario_configs(self, configs: Dict[str, Dict[str, Any]]) -> "MCPower":
        """Merge custom scenario configurations on top of the defaults.

        Merge behaviour matches v1:

          * If a scenario name already exists in the defaults (``optimistic``,
            ``realistic``, ``doomer``), the user fields *update* the preset —
            unspecified keys keep their preset values.
          * If a scenario name is brand-new (custom), it inherits every key
            from ``optimistic`` and then applies the user overrides on top —
            EXCEPT ``truth_start``: it is a scenario assumption ("estimation
            is well-behaved"), not a generic knob, so a custom scenario stays
            cold-start (``False``) unless the user sets it explicitly.

        User-supplied keys are validated: an unknown key raises ``ValueError``
        (a typo would otherwise silently no-op). The mixed-model knobs
        (``icc_noise_sd``, ``random_effect_dist``, ``random_effect_df``) are
        accepted like any other key.
        """
        if not isinstance(configs, dict):
            raise TypeError("configs must be a dictionary")

        live_keys = sorted(set(get_default_scenario_config()["optimistic"]))
        merged: Dict[str, Dict[str, Any]] = {
            k: dict(v) for k, v in get_default_scenario_config().items()
        }
        for name, user_cfg in configs.items():
            if not isinstance(user_cfg, dict):
                raise TypeError(
                    f"scenario config for {name!r} must be a dict, got "
                    f"{type(user_cfg).__name__}"
                )
            for key in user_cfg:
                if key not in live_keys:
                    raise ValueError(
                        f"scenario {name!r}: unknown config key {key!r}; "
                        f"valid keys: {live_keys}"
                    )
            if name in merged:
                merged[name].update(user_cfg)
            else:
                custom_cfg = {**get_default_scenario_config()["optimistic"], **user_cfg}
                if "truth_start" not in user_cfg:
                    custom_cfg["truth_start"] = False
                merged[name] = custom_cfg

        self._scenario_configs = merged
        return self

    # ------------------------------------------------------------------
    # Deferred apply
    # ------------------------------------------------------------------

    def _apply_baseline_probability(self) -> None:
        """Translate the pending baseline into ``self.intercept`` on the link scale.

        Binary: ``log(p/(1-p))`` (logit) or ``Φ⁻¹(p)`` (probit). Count (Poisson):
        ``ln(λ₀)`` (log link). No-op when nothing is pending. The pending value
        is NOT cleared — it remains the canonical record that the user supplied a
        baseline, used by ``_validate_logit_runtime``.
        """
        if self._pending_baseline_probability is not None:
            p = self._pending_baseline_probability
            if self.family == "probit":
                import statistics
                self.intercept = statistics.NormalDist().inv_cdf(p)
            else:  # logit
                self.intercept = math.log(p / (1.0 - p))
        if self._pending_baseline_rate is not None:
            self.intercept = math.log(self._pending_baseline_rate)

    def _validate_logit_runtime(
        self, scenario_filter: Optional[List[str]]
    ) -> None:
        """Pre-flight checks that only apply when the DGP is family='logit'.

        Called from find_power / find_sample_size after scenario resolution
        and before the engine call.

        Args:
            scenario_filter: Resolved list of scenario names (e.g.
                ``["optimistic"]`` for the default) or ``None``. Unused —
                scenario analysis is supported for logit; kept so call sites
                stay symmetric with ``_validate_lme_runtime``.

        Raises:
            ValueError: Missing baseline (probability/rate) or intercept-only model.
        """
        if self.family not in ("logit", "probit", "poisson"):
            return

        # Missing baseline. Binary families need a probability; Poisson needs a rate.
        if self.family in ("logit", "probit"):
            if self._pending_baseline_probability is None:
                raise ValueError(
                    f"baseline probability required for family={self.family!r}; "
                    "call set_baseline_probability(p) before find_power"
                )
        else:  # poisson
            if self._pending_baseline_rate is None:
                raise ValueError(
                    "baseline rate required for family='poisson'; call "
                    "set_baseline_rate(rate) before find_power"
                )

        # Intercept-only model (no testable effect).
        effects = self._registry.effect_names
        if len(effects) == 0:
            raise ValueError(
                f"family={self.family!r} requires at least one predictor; "
                "intercept-only models have no testable effect"
            )

    def _validate_lme_runtime(
        self,
        sample_size: Optional[int],
        scenario_filter: Optional[List[str]],
    ) -> None:
        """Pre-flight checks for ``family == 'lme'``.

        Called from ``find_power`` / ``find_sample_size`` before the engine
        call. Also validates the clustered-OLS path (family != 'lme' but pending_clusters is non-empty due to an estimator override).
        Populates ``self._effective_n_clusters`` for downstream
        ``to_simulation_spec`` consumption. Mirrors ``_validate_logit_runtime``.

        Args:
            sample_size: Total observations for ``find_power``; ``None`` from
                ``find_sample_size`` (which sweeps a range — n_clusters must
                be user-specified there, otherwise raises).
            scenario_filter: Resolved scenario names (unused at present —
                LME scenarios land later).

        Raises:
            ValueError: Formula has random effects but no ``set_cluster``;
                ``set_cluster`` was called on a non-LME family whose formula
                has no random-effect terms; ICC out of range (defensive
                re-check); sample_size incompatible with cluster spec.
        """
        from .spec.validators import (
            _validate_cluster_config,
            _validate_cluster_sample_size,
        )

        parsed_re = self._registry._random_effects_parsed
        parsed_grouping_vars = [re["grouping_var"] for re in parsed_re]

        # Cross-family coherence.
        # A formula with random effects but no cluster spec is still an error
        # regardless of the estimator — the DGP needs the cluster configuration.
        # set_cluster() on a non-lme family is allowed *only* when the formula
        # contains random-effect terms (the user is explicitly building a
        # clustered DGP and overriding the estimator).
        if self.family != "lme":
            if self._pending_clusters and not parsed_re:
                # set_cluster called but formula has no random effects — still an error.
                raise ValueError(
                    "set_cluster(...) was called but the formula has no random-effect "
                    "terms and family is not 'lme'; add '(1|group)' to the formula or "
                    "use family='lme'"
                )
            if not self._pending_clusters:
                # No clusters anywhere; nothing to validate.
                return
            # Clusters present + formula has random effects + non-lme family:
            # this is the clustered-OLS path (estimator override). Fall through
            # to the per-cluster validation below so n_clusters / ICC are still
            # checked. The "family must be lme" guard is intentionally removed.

        # family == "lme" path (or clustered-OLS fall-through above).
        if not parsed_grouping_vars:
            raise ValueError(
                "family='lme' requires a random-effect term in the formula, "
                "e.g. 'y ~ x + (1|group)'"
            )
        if not self._pending_clusters:
            raise ValueError(
                f"family='lme' requires set_cluster(...) for each random "
                f"effect grouping variable; missing: {parsed_grouping_vars}"
            )

        # Each formula grouping_var must have a matching set_cluster call.
        for g in parsed_grouping_vars:
            if g not in self._pending_clusters:
                raise ValueError(
                    f"random effect '(1|{g})' has no set_cluster call; "
                    "every grouping variable in the formula must be "
                    "configured before find_power"
                )

        items = list(self._pending_clusters.items())
        primary_var, primary_cfg = items[0]
        icc_primary = float(primary_cfg["icc"])
        n_clusters_primary = primary_cfg.get("n_clusters")
        cluster_size_primary = primary_cfg.get("cluster_size")

        # Re-validate primary grouping.
        _validate_cluster_config(
            primary_var,
            icc_primary,
            n_clusters_primary,
            cluster_size_primary,
            parsed_grouping_vars,
        ).raise_if_invalid()

        # Validate extra groupings (must have n_clusters explicitly; no
        # sample_size derivation for secondary groupings).
        for extra_var, extra_cfg in items[1:]:
            _validate_cluster_config(
                extra_var,
                float(extra_cfg["icc"]),
                extra_cfg.get("n_clusters"),
                extra_cfg.get("cluster_size"),
                parsed_grouping_vars,
            ).raise_if_invalid()

        # Derive effective n_clusters from the primary grouping only.
        if sample_size is None:
            if n_clusters_primary is not None:
                self._effective_n_clusters = int(n_clusters_primary)
            return

        if n_clusters_primary is not None:
            ss_result = _validate_cluster_sample_size(
                int(sample_size), int(n_clusters_primary), cluster_size_primary
            )
            ss_result.raise_if_invalid()
            for w in ss_result.warnings:
                warnings.warn(w, UserWarning, stacklevel=2)
            self._effective_n_clusters = int(n_clusters_primary)
        else:
            ss_result = _validate_cluster_sample_size(
                int(sample_size), None, int(cluster_size_primary)
            )
            ss_result.raise_if_invalid()
            for w in ss_result.warnings:
                warnings.warn(w, UserWarning, stacklevel=2)
            derived = int(sample_size) // int(cluster_size_primary)
            if derived < 2:
                raise ValueError(
                    f"derived n_clusters={derived} is below 2 "
                    f"(sample_size={sample_size}, cluster_size={cluster_size_primary}); "
                    "increase sample_size"
                )
            self._effective_n_clusters = derived

    def _apply(self) -> None:
        """Flush pending settings into the registry in dependency order: variable
        types first (factors need their type set before dummy expansion), then
        factor level ordering, then effect sizes. Order matters — effects
        reference β-column indices that factor expansion determines."""
        # 1. Variable types. Fragments are parsed in call order; errors are
        #    aggregated across all of them (nothing is registered until every
        #    fragment parses clean), then registered in order so a predictor
        #    re-declared in a later fragment last-wins in the registry.
        if self._pending_variable_types:
            parsed_fragments: List[Dict[str, Any]] = []
            vt_errors: List[str] = []
            for fragment in self._pending_variable_types:
                parsed, errors = _parser._parse(
                    fragment,
                    "variable_type",
                    self._registry.predictor_names,
                )
                vt_errors.extend(errors)
                parsed_fragments.append(parsed)
            if vt_errors:
                raise ValueError(
                    "Variable type validation failed:\n"
                    + "\n".join(f"- {e}" for e in vt_errors)
                )
            for parsed in parsed_fragments:
                for name, info in parsed.items():
                    self._registry.set_variable_type(name, info["type"], **{k: v for k, v in info.items() if k != "type"})

        # 1b. Upload type-lock: for matched columns, the detected class is
        #     authoritative — a conflicting explicit declaration raises a
        #     clear error; matching or undeclared columns have data-wins
        #     re-apply (factor levels/count from data; continuous is no-op).
        if self._pending_data is not None:
            for (col_name, detected_type, _raw, col_labels) in self._pending_data["columns_typed"]:
                modeled_class = _class_of(self._registry, col_name)
                if modeled_class != detected_type:
                    raise ValueError(
                        f"Column {col_name!r} was detected as {detected_type} from your uploaded data; "
                        f"it can't be modeled as {modeled_class}. Uploaded columns take their type from the data."
                    )
                # Data wins on details: re-apply detected type/levels so
                # factor labels and n_levels always come from the upload,
                # not from a factor(k) declaration.
                # Continuous columns: deliberate no-op — leave the declared
                # distribution (e.g. right_skewed) untouched.
                if detected_type == "factor":
                    n_lvl = len(col_labels)
                    self._registry.set_variable_type(
                        col_name,
                        "factor",
                        n_levels=n_lvl,
                        labels=col_labels,
                        reference=col_labels[0] if col_labels else None,
                    )
                elif detected_type == "binary":
                    self._registry.set_variable_type(col_name, "binary")
                # detected_type == "continuous": no-op (keep declared distribution)

        # 2. Expand factors.
        if self._registry.factor_names:
            self._registry.expand_factors()

        # 3. Effects. Fragments replay in call order; an effect re-declared in a
        #    later fragment last-wins (the registry overwrites its size).
        if self._pending_effects:
            from . import _engine

            effect_names = self._registry.effect_names
            effect_errors: List[str] = []
            parsed_for_warn: Dict[str, float] = {}
            for fragment in self._pending_effects:
                for assignment in _engine.split_assignments(fragment):
                    if "=" not in assignment:
                        effect_errors.append(f"Invalid format '{assignment}'. Expected: 'name=value'")
                        continue
                    name, value_str = (s.strip() for s in assignment.split("=", 1))
                    try:
                        value = float(value_str)
                    except ValueError:
                        effect_errors.append(f"Invalid value '{value_str}' for '{name}'")
                        continue
                    if name not in effect_names:
                        effect_errors.append(
                            f"Effect '{name}' not found. Available: {', '.join(effect_names)}"
                        )
                        continue
                    self._registry.set_effect_size(name, value)
                    parsed_for_warn[name] = value
            if effect_errors:
                raise ValueError(
                    "Effect validation failed:\n" + "\n".join(f"- {e}" for e in effect_errors)
                )
            if self.family == "logit":
                from .spec.validators import _warn_logit_effect_scale

                for w in _warn_logit_effect_scale(parsed_for_warn, self._registry):
                    warnings.warn(w, UserWarning, stacklevel=2)

        # 4. Baseline probability → intercept (logit only; no-op for OLS).
        self._apply_baseline_probability()

        # 5. Correlations. The accumulator holds string fragments and/or a full
        #    matrix; set_correlations resets it on a matrix call, so any matrix is
        #    the first element and later string fragments layer their pairwise
        #    entries on top of it (set_correlation edits, not replaces).
        if self._pending_correlations:
            non_factor = self._registry.non_factor_names
            n = len(non_factor)
            if self._registry.get_correlation_matrix() is None:
                self._registry.set_correlation_matrix(
                    [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
                )
            for corr_input in self._pending_correlations:
                if isinstance(corr_input, str):
                    if corr_input.strip():
                        if n < 2:
                            raise ValueError(
                                "Need at least 2 non-factor variables for correlations"
                            )
                        parsed_corr, errors = _parser._parse(
                            corr_input, "correlation", non_factor
                        )
                        if errors:
                            raise ValueError(
                                "Error setting correlations:\n"
                                + "\n".join(f"- {e}" for e in errors)
                            )
                        for (v1, v2), val in parsed_corr.items():
                            self._registry.set_correlation(v1, v2, val)
                        # String-spec matrices are symmetric and unit-diagonal by
                        # construction; range + PSD are enforced by the engine.
                elif isinstance(corr_input, list):
                    n_in = len(corr_input)
                    if n_in != n or any(len(row) != n for row in corr_input):
                        raise ValueError(
                            f"Matrix shape ({n_in}x{n_in if corr_input else 0}) doesn't match "
                            f"{n} non-factor variables"
                        )
                    # Guard symmetry + unit diagonal, which the wire format (upper
                    # triangle only) cannot preserve; range + PSD are the engine's job.
                    _validate_correlation_matrix(corr_input).raise_if_invalid()
                    self._registry.set_correlation_matrix(corr_input)

        self._applied = True

    # ------------------------------------------------------------------
    # Spec construction
    # ------------------------------------------------------------------

    def _scenario_dict(self, scenario_name: str) -> Dict[str, Any]:
        """Build the ``ScenarioPerturbations`` dict for ``scenario_name``:
        residual-distribution overrides, random-effect distribution knobs, and
        ICC noise, sourced from ``self._scenario_configs``."""
        return build_scenario_dict(scenario_name, self._scenario_configs)

    def _to_linear_spec_dict(
        self,
        scenario_names: List[str],
        *,
        test_formula: Optional[str] = None,
        target_test: Optional[str] = None,
        correction: Optional[str] = None,
        wald_se: Optional[str] = None,
        nagq: int = 1,
    ) -> Dict[str, Any]:
        """Project current state into the Rust ``LinearSpec`` JSON contract.

        Ensures pending state is applied first, then assembles the contract
        dict: registry (predictors, betas, factor dummies), cluster groupings,
        scenario perturbations, correction method, and formula/target overrides.

        ``wald_se=None`` falls back to the config ``estimation.wald_se`` default
        (the cross-port home — no hardcoded per-port default). ``find_power`` /
        ``find_sample_size`` pass the already-resolved value; the direct callers
        (``to_simulation_spec``, golden fixtures) rely on this fallback.
        """
        if not self._applied:
            self._apply()
        if wald_se is None:
            from .config import get_estimation_defaults
            wald_se = get_estimation_defaults()["wald_se"]
        # Collect cluster_level_vars across all pending cluster specs.
        _clv: List[str] = []
        for _cfg in self._pending_clusters.values():
            _clv.extend(_cfg.get("cluster_level_vars", []))
        return build_linear_spec(
            self._registry,
            scenario_names,
            heteroskedasticity=self._heteroskedasticity,
            residual_dist_name=self._residual_dist_name,
            residual_pinned=self._residual_pinned,
            alpha=self.alpha,
            correction=correction,
            wald_se=wald_se,
            nagq=nagq,
            target_test=target_test,
            test_formula=test_formula,
            pending_data=getattr(self, "_pending_data", None),
            equation=self.equation,
            scenario_configs=self._scenario_configs,
            max_failed_simulations=self.max_failed_simulations,
            estimator=self.estimator,
            cluster_level_vars=_clv if _clv else None,
        )

    def to_simulation_spec(
        self,
        scenario_name: str = "optimistic",
        *,
        test_formula: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build the dict that round-trips through msgpack into the Rust engine.

        The builder is parametrised on outcome_kind, estimator, intercept, and
        clusters via ``_engine.build_contract_from_spec(...)``. Outcome /
        estimator / intercept / cluster all land in the contract at construction
        time; the engine kernel sees a single coherent ``SimulationContract``
        per scenario.

        Returns the decoded contract dict for the requested ``scenario_name``.
        Callers that want the wire payload (find_power / find_sample_size) go
        through ``_engine.build_contract_from_spec`` directly.
        """
        if not self._effects_set:
            raise RuntimeError(
                "no effect sizes configured; call set_effects() before "
                "to_simulation_spec()"
            )

        import json as _json

        from . import _engine

        payload = self._to_linear_spec_dict([scenario_name], test_formula=test_formula)
        outcome_kind_wire, link_wire, estimator_wire, intercept_arg, clusters_json = (
            self._encode_outcome_and_clusters()
        )
        names, contracts_bytes, _skeleton_json = _engine.build_contract_from_spec(
            _json.dumps(payload),
            outcome_kind_wire,
            link_wire,
            estimator_wire,
            intercept_arg,
            clusters_json,
        )
        contracts = msgpack.unpackb(contracts_bytes, raw=False)
        for name, contract in zip(names, contracts):
            if name == scenario_name:
                return contract
        raise RuntimeError(f"builder did not return scenario {scenario_name!r}")

    def _encode_outcome_and_clusters(self) -> Tuple[str, str, str, float, str]:
        """Encode ``(outcome_kind_wire, link_wire, estimator_wire, intercept,
        clusters_json)`` for the ``_engine.build_contract_from_spec`` signature.

        ``outcome_kind_wire``: "continuous", "binary", or "count" (matches
          OutcomeKind snake_case serde names).

        ``link_wire``: "canonical" or "probit" — the non-canonical link override
          for a binary outcome (probit); "canonical" for every other family.

        ``estimator_wire``: "ols", "glm", or "mle" (matches EstimatorSpec
          snake_case serde names).

        ``clusters_json``: non-empty JSON array of ClusterSpec dicts only when
          the DGP includes clustering (family="lme" OR estimator override on a
          formula with random effects).  Empty array otherwise.

        The clusters_json presence is driven by ``self._pending_clusters``, NOT
        by ``self.estimator`` — the estimator may be "ols" while clusters are
        still present in the DGP (the headline clustered-OLS study).
        """
        import json as _json

        # Cluster presence tracks the DGP, independent of the estimator axis.
        if self._pending_clusters:
            clusters_json = _json.dumps([self._build_cluster_spec_dict()])
        else:
            clusters_json = "[]"

        return (
            self.outcome_kind,          # "continuous" | "binary" | "count"
            self.link,                  # "canonical" | "probit"
            self.estimator,             # "ols" | "glm" | "mle"
            float(self.intercept),
            clusters_json,
        )

    def _slope_terms_for(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the SlopeTerm list for one grouping from its _pending_clusters config.

        Shape mirrors R debug seam (spec-builder.R .encode_outcome_and_clusters) and
        the engine's SlopeTerm serde:
          {column: <0-based gen col>, variance: f64, corr_with_intercept: f64,
           corr_with: Vec<f64>}  (corr_with = correlations with EARLIER slopes;
           empty list for the first slope, zero for subsequent ones because the
           scalar set_cluster API cannot express non-zero slope↔slope correlations).

        Returns [] when the config has no random_slopes — caller omits "slopes" key.
        """
        _random_slopes = cfg.get("random_slopes") or []
        if not _random_slopes:
            return []
        _nf_names = list(self._registry.non_factor_names)  # generation column order
        _sv = float(cfg.get("slope_variance", 0.0))
        _sc = float(cfg.get("slope_intercept_corr", 0.0))
        _terms: List[Dict[str, Any]] = []
        for _sname in _random_slopes:
            if _sname not in _nf_names:
                raise ValueError(
                    f"random_slopes: {_sname!r} is not a non-factor predictor; "
                    f"valid: {_nf_names}"
                )
            _terms.append({
                "column": _nf_names.index(_sname),
                "variance": _sv,
                "corr_with_intercept": _sc,
                "corr_with": [0.0] * len(_terms),
            })
        return _terms

    def _re_tau_squared(self, cfg: Dict[str, Any]) -> float:
        """Family/link-aware random-effect variance τ² for one grouping.

        - Poisson (count): the raw ``tau_squared`` the user supplied — no
          standard latent-scale ICC exists for a log-link count model
          (Decision 8).
        - Binary logit: ``icc/(1−icc) · π²/3`` (log-odds residual variance).
        - Binary probit: ``icc/(1−icc) · 1`` — the probit latent residual
          variance is 1, not π²/3 (Decision 9).
        - Gaussian: ``icc/(1−icc)`` (σ²=1).

        Mirrors R spec-builder.R:.encode_outcome_and_clusters — change together.
        """
        if self.family == "poisson":
            return float(cfg.get("tau_squared") or 0.0)
        icc = float(cfg["icc"])
        denom = 1.0 - icc
        tau = icc / denom if denom > 0 else 0.0
        if self.outcome_kind == "binary" and self.link != "probit":
            tau *= _PI_SQ_OVER_3  # logit log-odds residual variance
        # probit (latent variance 1) and Gaussian (σ²=1) take no extra factor.
        return tau

    def _build_cluster_spec_dict(self) -> Dict[str, Any]:
        """Project all pending LME cluster specs into a single engine ClusterSpec.

        The primary (first) grouping carries sizing + tau_squared. Additional
        groupings are folded into extra_groupings with a Crossed or Nested
        relation — Nested when n_per_parent is set (composite "A:B" grouping
        var from (1|A/B) formula syntax), Crossed otherwise.

        Per-grouping τ² is family/link-aware — see `_re_tau_squared`.
        """
        if not self._pending_clusters:
            raise RuntimeError(
                "a clustered DGP requires set_cluster(...) before "
                "to_simulation_spec(); no cluster configured"
            )
        items = list(self._pending_clusters.items())
        primary_var, primary_cfg = items[0]

        tau_squared = self._re_tau_squared(primary_cfg)

        n_clusters = self._effective_n_clusters
        if n_clusters is None:
            n_clusters = primary_cfg.get("n_clusters")
        cluster_size = primary_cfg.get("cluster_size")

        if n_clusters is not None:
            sizing = {"FixedClusters": {"n_clusters": int(n_clusters)}}
        elif cluster_size is not None:
            sizing = {"FixedSize": {"cluster_size": int(cluster_size)}}
        else:
            raise RuntimeError(
                "cluster spec has neither n_clusters nor cluster_size; "
                "call set_cluster with one of them"
            )

        spec: Dict[str, Any] = {
            "sizing": sizing,
            "tau_squared": float(tau_squared),
        }

        # Extra groupings: all groupings beyond the primary, in insertion order.
        if len(items) > 1:
            extra: List[Dict[str, Any]] = []
            for _gvar, _cfg in items[1:]:
                _tau = self._re_tau_squared(_cfg)
                _n_per_parent = _cfg.get("n_per_parent")
                _n_cls = _cfg.get("n_clusters")
                if _n_per_parent is not None:
                    # Nested: (1|A/B) composite grouping. n_per_parent is
                    # the B-units-per-A-unit count (fixed by the user, not
                    # derived from sample_size — matches engine GroupingRelation::NestedWithin).
                    _relation: Dict[str, Any] = {
                        "NestedWithin": {"n_per_parent": int(_n_per_parent)}
                    }
                else:
                    # Crossed: independent grouping factor.
                    _relation = {
                        "Crossed": {"n_clusters": int(_n_cls) if _n_cls is not None else 0}
                    }
                _extra_spec: Dict[str, Any] = {"relation": _relation, "tau_squared": float(_tau)}
                _extra_slopes = self._slope_terms_for(_cfg)
                if _extra_slopes:
                    _extra_spec["slopes"] = _extra_slopes
                extra.append(_extra_spec)
            spec["extra_groupings"] = extra

        # Random slopes for the primary grouping — reuses the same helper.
        _primary_slopes = self._slope_terms_for(primary_cfg)
        if _primary_slopes:
            spec["slopes"] = _primary_slopes

        return spec

    # ------------------------------------------------------------------
    # Analysis entry points
    # ------------------------------------------------------------------

    def _resolve_scenarios_arg(
        self, scenarios: Union[bool, List[str]]
    ) -> List[str]:
        """Translate ``scenarios`` argument into a concrete list of names.

        Rules:
          * ``False`` → ``["optimistic"]`` (always the zero-perturbation baseline).
          * ``True``  → every configured scenario key (with ``optimistic`` first
            if present).
          * list → validated subset; unknown names raise ``ValueError``.
        """
        configs = self._scenario_configs
        if scenarios is False:
            return ["optimistic"]
        if scenarios is True:
            names = list(configs.keys())
            # Pin optimistic first so the multi-scenario envelope is stable.
            if "optimistic" in names:
                names = ["optimistic"] + [n for n in names if n != "optimistic"]
            return names
        if not isinstance(scenarios, list):
            raise TypeError(
                "scenarios must be True, False, or a list of strings; got "
                f"{type(scenarios).__name__}"
            )
        available = set(configs.keys())
        invalid = [s for s in scenarios if s not in available]
        if invalid:
            raise ValueError(
                f"Unknown scenario(s): {invalid}; configured: "
                f"{sorted(available)}"
            )
        if not scenarios:
            raise ValueError("scenarios list cannot be empty")
        return list(scenarios)

    def find_power(
        self,
        sample_size: int,
        *,
        target_test: Optional[str] = None,
        correction: Optional[str] = None,
        wald_se: Optional[str] = None,
        agq: Optional[int] = None,
        test_formula: Optional[str] = None,
        n_sims: Optional[int] = None,
        seed: Optional[int] = None,
        scenarios: Union[bool, List[str]] = False,
        progress_callback: Any = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Estimate power at a single sample size.

        ``target_test`` selects which effect(s) the power refers to (the v1
        DSL — ``"all"``, ``"x1"``, ``"all, -x2"``, ``"f[a] vs f[b]"``, …);
        ``None`` uses the family default (omnibus + every β). The omnibus /
        overall test (the OLS F-test / GLM likelihood-ratio test) is reported
        only for OLS and unclustered GLM fits — mixed-effects fits (LME and
        clustered GLMM) have no exposed omnibus, so for them ``None`` reports
        every β without it, ``"all"`` drops it, and an explicit
        ``target_test="overall"`` raises. ``correction``
        is the multiple-comparison correction (``"none"`` /
        ``"bonferroni"`` / ``"holm"`` / ``"bh"`` aka ``"fdr"``,
        case-insensitive); ``None`` means no correction. ``wald_se`` selects
        the standard-error flavour for the clustered-binary/count GLMM
        estimator: ``"rx"`` (default, the Schur speed knob, lme4
        ``use.hessian = FALSE``) or ``"hessian"`` (per-fit FD-Hessian SE,
        lme4 ``use.hessian = TRUE`` — slower, the rigor opt-in). ``agq`` sets
        the adaptive Gauss-Hermite quadrature node count for a clustered
        binary/count GLMM fit; ``None``/``1`` uses Laplace (the default).
        An odd value in ``3..=25`` opts into AGQ, but only when the design is
        eligible (a Binary or Count GLMM with a single grouping factor and at
        most 3 random effects per group) — an ineligible or even/out-of-range
        value warns and runs at Laplace instead. No-op for OLS/LMM.

        Returns a result dict (single scenario) or a scenarios envelope
        (``{"scenarios": {...}, "comparison": {...}}``) when ``scenarios`` is
        truthy or a list.
        """
        if not self._applied:
            self._apply()

        _validate_sample_size(sample_size).raise_if_invalid()
        n_variables = len(self._registry.effect_names)
        _validate_sample_size_for_model(sample_size, n_variables).raise_if_invalid()
        self._validate_correction_arg(correction)
        wald_se, nagq = self._resolve_estimation(wald_se, agq)

        if test_formula is not None:
            available = self._registry.non_factor_names + self._registry.factor_names
            # Validator scans for *all* identifiers — strip the LHS so the
            # dependent-variable name doesn't trip it.
            rhs = test_formula
            for sep in ("~", "="):
                if sep in rhs:
                    rhs = rhs.split(sep, 1)[1]
                    break
            _validate_test_formula(rhs, available).raise_if_invalid()

        from . import _engine
        from .progress import managed_progress
        from .output.results import make_power_result, unwrap_scenario_result

        names = self._resolve_scenarios_arg(scenarios)
        self._validate_logit_runtime(names)
        # N is snapped to the cluster atom inside the engine (find_power); the
        # snapped value comes back on the result and is surfaced as a warning.
        self._validate_lme_runtime(int(sample_size), names)
        n = int(n_sims) if n_sims is not None else int(self.n_simulations)
        # base_seed: explicit `seed=` kwarg wins; else self.seed (default 2137);
        # else 0 as a final fallback.
        base_seed = (
            int(seed)
            if seed is not None
            else (int(self.seed) if self.seed is not None else 0)
        )
        # Build the Vec<SimulationContract> msgpack blob via the parametrised
        # builder; carries family + intercept + clusters in one shot.
        import json as _json
        payload = self._to_linear_spec_dict(
            names,
            test_formula=test_formula,
            target_test=target_test,
            correction=correction,
            wald_se=wald_se,
            nagq=nagq,
        )
        outcome_kind_wire, link_wire, estimator_wire, intercept_arg, clusters_json = (
            self._encode_outcome_and_clusters()
        )
        _names, scenarios_bytes, skeleton_json = _engine.build_contract_from_spec(
            _json.dumps(payload),
            outcome_kind_wire,
            link_wire,
            estimator_wire,
            intercept_arg,
            clusters_json,
        )
        effect_skeleton = _json.loads(skeleton_json)
        with managed_progress(progress_callback) as progress_cb:
            raw = _engine.find_power(
                scenarios_bytes,
                sample_size,
                n,
                base_seed,
                progress_cb,
            )
        result = unwrap_scenario_result(raw, names)
        # Post-batch: check that the failure rate (1 - convergence_rate) does
        # not exceed the configured threshold for any scenario / sample size.
        self._check_result_failure_threshold(result)
        # Surface grid_warnings exactly once per distinct message.
        _surface_warnings(result)
        # min_cluster_size: FixedClusters → sample_size // n_clusters; FixedSize →
        # the configured cluster_size. Primary grouping only (the smallest atomic
        # cluster at the requested N). None unless this is a clustered binary run
        # (GLMM) — feeds both the transient Laplace warn and the persistent
        # report line via _report_meta.
        _min_cs = None
        if self._pending_clusters and self.outcome_kind == "binary":
            from .config import get_config as _get_config
            _cfg = _get_config()
            _primary_cfg = next(iter(self._pending_clusters.values()))
            _cs = _primary_cfg.get("cluster_size")
            if _cs is not None:
                _min_cs = int(_cs)
            else:
                _nc = self._effective_n_clusters or _primary_cfg.get("n_clusters") or 1
                _min_cs = int(sample_size) // int(_nc)
            # Inspect every scenario's extras, warning at most once — any
            # breaching scenario triggers it. Scenarios can carry different
            # τ̂² (distinct RE perturbations), so checking only the first would
            # miss a high-τ̂² later scenario. The R helper fires once the same
            # way; both ports must agree on the firing rule, not just the copy.
            _scenario_extras = (
                [s.get("estimator_extras", {}) for s in result["scenarios"].values()]
                if "scenarios" in result
                else [result.get("estimator_extras", {})]
            )
            for _extras in _scenario_extras:
                _w = _glmm_laplace_bias_warning(_extras, _min_cs, _cfg)
                if _w is not None:
                    warnings.warn(_w, UserWarning, stacklevel=2)
                    break
        _baseline_req = (self._pending_baseline_probability
                         if self.outcome_kind == "binary" else None)
        wire_posthoc_factors = [r["factor"] for r in payload.get("posthoc_requests", [])]
        result = make_power_result(result, self._report_meta(
            correction=correction, posthoc_factors=wire_posthoc_factors,
            effect_skeleton=effect_skeleton,
            baseline_prob_requested=_baseline_req, min_cluster_size=_min_cs))
        if verbose:
            print(result)
        if self._uploaded_data_mode == "strict" and self._uploaded_data_n > 0:
            from .config import get_config as _get_config
            U = self._uploaded_data_n
            N = sample_size
            ratio = _get_config()["upload"]["strict_warning_ratio"]
            frac = _reuse_fraction(U, N)
            if verbose:
                print(
                    f"[strict bootstrap] N={N}, uploaded rows U={U}: "
                    f"~{frac:.0f}% of rows reused per simulated dataset."
                )
            w = _strict_reuse_warning(U, N, ratio)
            if w is not None:
                warnings.warn(w, UserWarning, stacklevel=2)
        return result

    def _report_meta(self, *, correction, posthoc_factors=None, effect_skeleton=None,
                     baseline_prob_requested=None, min_cluster_size=None):
        """Label + header context the result object needs to render. The port
        owns labels (joined to the engine's integer target_indices).

        effect_skeleton: the engine's index-only effect layout (parsed from
        ``build_contract_from_spec``), β-column aligned with ``target_indices``.
        Result names are rendered from it + each factor's ``levels`` store, so
        no port re-derives the factor-expansion layout.

        posthoc_factors: list of factor names (in request order) for which
        post-hoc contrasts were requested. Used to build posthoc_factors in the
        returned dict, which maps position → factor name + ordered level labels.

        baseline_prob_requested / min_cluster_size: meta-level diagnostics inputs
        (one per run, not per scenario). The requested GLM event probability
        (drives the live baseline-drift gate) and the smallest cluster size at
        the evaluated N (drives the persistent Laplace-bias line). Both None for
        OLS / non-binary / non-clustered runs. Consumed by diagnostic_warnings.
        """
        factors = {}
        for fname, info in getattr(self._registry, "_factors", {}).items():
            factors[fname] = {
                "baseline": info.get("reference_level"),
                # Full ordered label list the skeleton's `level` index resolves
                # against (reference included).
                "levels": self._registry.factor_levels(fname),
            }
        meta = {
            "effect_names": list(self._registry.effect_names),
            "effect_skeleton": effect_skeleton,
            "effect_sizes": list(self._registry.get_effect_sizes()),
            "factors": factors,
            "estimator": self.estimator,
            # "binary" for logit-link outcomes (logistic regression and binary
            # GLMM, whose estimator is "mle" not "glm"); gates the OR = exp(β)
            # readout in the report. "continuous" otherwise.
            "outcome_kind": self.outcome_kind,
            "alpha": self.alpha,
            "correction": correction or "none",
            "target_power": self.power / 100.0,  # stored as 80.0, render as 0.8
            "formula": self.equation,
            "residual": self._residual_dist_name,
            "baseline_prob_requested": baseline_prob_requested,
            "min_cluster_size": min_cluster_size,
        }
        if posthoc_factors:
            meta["posthoc_factors"] = [
                {"name": fname, "levels": self._registry.factor_levels(fname)}
                for fname in posthoc_factors
            ]
        else:
            meta["posthoc_factors"] = []
        return meta

    def find_sample_size(
        self,
        *,
        target_test: Optional[str] = None,
        correction: Optional[str] = None,
        wald_se: Optional[str] = None,
        agq: Optional[int] = None,
        test_formula: Optional[str] = None,
        target_power: Optional[float] = None,
        from_size: Optional[int] = None,
        to_size: Optional[int] = None,
        by: Union[int, str, None] = None,
        mode: str = "linear",
        n_sims: Optional[int] = None,
        seed: Optional[int] = None,
        scenarios: Union[bool, List[str]] = False,
        progress_callback: Any = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Search sample sizes and find the smallest N that hits target power.

        ``target_test`` / ``correction`` mirror :meth:`find_power` — they
        select the tested effect(s) and the multiple-comparison correction for
        this call (see that method's docstring). ``wald_se`` / ``agq`` mirror
        :meth:`find_power` — see that docstring for details.

        The search sweeps ``from_size .. to_size`` and reports the smallest N
        that reaches the target power for each tested effect. ``by`` controls
        the grid:

        * ``by="auto"`` (default) — the engine places ~12 points between the
          bounds. The point selection is a pure function of the inputs (never
          of observed power), so the search is reproducible and mergeable.
        * ``by=<int>`` — a fixed step (``mode="linear"``) or point count
          (``mode="log"``).

        For clustered (LME) models the engine snaps every requested N to the
        cluster *atom* so each grid point is a valid balanced design:
        ``set_cluster(..., n_clusters=k)`` makes the atom ``k`` (the cluster
        count is fixed; cluster size grows with N), while
        ``set_cluster(..., cluster_size=s)`` makes the atom ``s`` and each grid
        point a whole number of complete clusters (now supported for
        ``find_sample_size``). Bounds not already on the atom are raised/lowered
        to valid multiples and the adjustment is surfaced as a warning.

        The power-vs-N curve uses common random numbers across sample sizes, so
        it is smooth and the reported N is stable from run to run.
        """
        if not self._applied:
            self._apply()

        from .config import get_simulation_defaults
        _ssb = get_simulation_defaults()["sample_size_bounds"]
        if from_size is None:
            from_size = _ssb["from"]
        if to_size is None:
            to_size = _ssb["to"]
        if by is None:
            by = _ssb["by"]  # "auto"

        self._validate_correction_arg(correction)
        wald_se, nagq = self._resolve_estimation(wald_se, agq)

        _validate_sample_size_range(from_size, to_size, by).raise_if_invalid()

        if test_formula is not None:
            available = self._registry.non_factor_names + self._registry.factor_names
            # Validator scans for *all* identifiers — strip the LHS so the
            # dependent-variable name doesn't trip it.
            rhs = test_formula
            for sep in ("~", "="):
                if sep in rhs:
                    rhs = rhs.split(sep, 1)[1]
                    break
            _validate_test_formula(rhs, available).raise_if_invalid()

        from . import _engine
        from .progress import managed_progress
        from .output.results import make_sample_size_result, unwrap_scenario_result

        names = self._resolve_scenarios_arg(scenarios)
        self._validate_logit_runtime(names)
        self._validate_lme_runtime(None, names)

        n = int(n_sims) if n_sims is not None else int(self.n_simulations)
        base_seed = (
            int(seed)
            if seed is not None
            else (int(self.seed) if self.seed is not None else 0)
        )
        import json as _json
        payload = self._to_linear_spec_dict(
            names,
            test_formula=test_formula,
            target_test=target_test,
            correction=correction,
            wald_se=wald_se,
            nagq=nagq,
        )
        outcome_kind_wire, link_wire, estimator_wire, intercept_arg, clusters_json = (
            self._encode_outcome_and_clusters()
        )
        _names, scenarios_bytes, skeleton_json = _engine.build_contract_from_spec(
            _json.dumps(payload),
            outcome_kind_wire,
            link_wire,
            estimator_wire,
            intercept_arg,
            clusters_json,
        )
        effect_skeleton = _json.loads(skeleton_json)
        tp = target_power if target_power is not None else self.power
        if by == "auto":
            from .config import get_simulation_defaults
            by_value = int(get_simulation_defaults()["cluster_auto_count"])
            by_kind = "auto"
        else:
            by_value = int(by)
            by_kind = "fixed"
        with managed_progress(progress_callback) as progress_cb:
            raw = _engine.find_sample_size(
                scenarios_bytes,
                float(tp),
                from_size,
                to_size,
                n,
                base_seed,
                "grid",
                by_value,
                by_kind,
                str(mode),
                None,
                progress_cb,
            )
        result = unwrap_scenario_result(raw, names)
        # Post-batch: check failure threshold across all sample sizes / scenarios.
        self._check_result_failure_threshold(result)
        # Surface grid_warnings exactly once per distinct message.
        _surface_warnings(result)
        # min_cluster_size: FixedClusters → from_size // n_clusters (lower bound of
        # the search range); FixedSize → the configured cluster_size. None unless
        # this is a clustered binary run (GLMM) — feeds both the transient Laplace
        # warn and the persistent report line via _report_meta.
        _min_cs = None
        if self._pending_clusters and self.outcome_kind == "binary":
            from .config import get_config as _get_config
            _cfg = _get_config()
            _primary_cfg = next(iter(self._pending_clusters.values()))
            _cs = _primary_cfg.get("cluster_size")
            if _cs is not None:
                _min_cs = int(_cs)
            else:
                _nc = self._effective_n_clusters or _primary_cfg.get("n_clusters") or 1
                _min_cs = int(from_size) // int(_nc)
            _scenario_extras = (
                [s.get("estimator_extras", {}) for s in result["scenarios"].values()]
                if "scenarios" in result
                else [result.get("estimator_extras", {})]
            )
            for _extras in _scenario_extras:
                _w = _glmm_laplace_bias_warning(_extras, _min_cs, _cfg)
                if _w is not None:
                    warnings.warn(_w, UserWarning, stacklevel=2)
                    break
        _baseline_req = (self._pending_baseline_probability
                         if self.outcome_kind == "binary" else None)
        wire_posthoc_factors = [r["factor"] for r in payload.get("posthoc_requests", [])]
        result = make_sample_size_result(result, self._report_meta(
            correction=correction, posthoc_factors=wire_posthoc_factors,
            effect_skeleton=effect_skeleton,
            baseline_prob_requested=_baseline_req, min_cluster_size=_min_cs))
        if verbose:
            print(result)
        if self._uploaded_data_mode == "strict" and self._uploaded_data_n > 0:
            from .config import get_config as _get_config
            U = self._uploaded_data_n
            ratio = _get_config()["upload"]["strict_warning_ratio"]
            # Collect first_achieved N per target across all scenarios.
            if "scenarios" in result and isinstance(result["scenarios"], dict):
                inner_list = list(result["scenarios"].values())
            else:
                inner_list = [result]
            for inner in inner_list:
                fa = inner.get("first_achieved", {}) or {}
                for pos, achieved_n in fa.items():
                    if achieved_n is None:
                        continue
                    frac = _reuse_fraction(U, achieved_n)
                    if verbose:
                        print(
                            f"[strict bootstrap] target {pos}: first N={achieved_n}, "
                            f"uploaded rows U={U}: ~{frac:.0f}% of rows reused per "
                            "simulated dataset."
                        )
                    w = _strict_reuse_warning(U, achieved_n, ratio)
                    if w is not None:
                        warnings.warn(w, UserWarning, stacklevel=2)
        return result

    def _check_result_failure_threshold(self, result: Dict[str, Any]) -> None:
        """Raise RuntimeError if any scenario's failure rate exceeds the threshold.

        Works for both single-result dicts and multi-scenario envelopes.
        For multi-scenario envelopes the *worst* scenario across all scenarios
        triggers the error — so each scenario is checked independently.
        """
        from .output.results import _check_failure_threshold

        if "scenarios" in result:
            for scenario_dict in result["scenarios"].values():
                self._check_result_failure_threshold(scenario_dict)
            return

        cr = result.get("convergence_rate")
        if cr is None:
            # Non-LME result or empty run: no failure tracking.
            return
        _check_failure_threshold(
            convergence_rate=list(cr),
            boundary_hit_rate_tau_zero=result.get("boundary_hit_rate_tau_zero", [0.0] * len(cr)),
            boundary_hit_rate_high_tau=result.get("boundary_hit_rate_high_tau", [0.0] * len(cr)),
            threshold=self.max_failed_simulations,
        )

    def summary(self) -> Dict[str, Any]:
        """Return a structured snapshot of the current model state."""
        if not self._applied:
            self._apply()
        return {
            "formula": self.equation,
            "family": self.family,           # kept for back-compat
            "outcome_kind": self.outcome_kind,
            "estimator": self.estimator,
            "effects": dict(zip(
                self._registry.effect_names,
                [float(b) for b in self._registry.get_effect_sizes()],
                strict=False,
            )),
            "n_simulations": self.n_simulations,
            "alpha": self.alpha,
            "power_target": self.power,
            "residual_distribution": self._residual_dist_name,
            "residual_pinned": self._residual_pinned,
            "scenarios": sorted(self._scenario_configs),
        }

    def __repr__(self) -> str:  # pragma: no cover — cosmetic
        return (
            f"MCPower(formula={self.equation!r}, family={self.family!r}, "
            f"estimator={self.estimator!r})"
        )


__all__ = ["MCPower"]
