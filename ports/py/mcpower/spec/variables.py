"""Variable and effect registry: single source of truth for predictors, factors, and effect sizes within one MCPower model instance."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class PredictorVar:
    """A single predictor variable in the design matrix.

    Attributes:
        name: Variable name as it appears in the formula.
        var_type: Distribution type used for data generation (e.g.
            ``"normal"``, ``"binary"``, ``"factor_dummy"``,
            ``"uploaded_data"``, ``"cluster_effect"``).
        proportion: Probability of success for binary variables.
        n_levels: Number of categorical levels (factors only).
        proportions: Per-level probabilities (factors only).
        is_factor: Whether this variable is a factor (before expansion).
        is_dummy: Whether this variable is a dummy created by factor expansion.
        factor_source: Original factor name (dummies only).
        factor_level: Categorical level this dummy represents (dummies only).
        column_index: Position in the raw design matrix ``X``.
        level_labels: Original data labels for factor levels (factors/dummies only).
    """

    name: str
    var_type: str = "normal"
    proportion: float = 0.5
    n_levels: Optional[int] = None
    proportions: Optional[List[float]] = None
    is_factor: bool = False
    is_dummy: bool = False
    factor_source: Optional[str] = None
    factor_level: Optional[Union[int, str]] = None
    column_index: Optional[int] = None
    level_labels: Optional[List[str]] = None
    # True when the user explicitly chose this continuous distribution (incl.
    # explicit "normal") — scenario distribution swaps leave this column alone.
    pinned: bool = False


@dataclass
class Effect:
    """A main effect or interaction term in the regression model.

    Attributes:
        name: Effect name (e.g. ``"x1"`` or ``"x1:x2"``).
        effect_type: ``"main"`` or ``"interaction"``.
        effect_size: Standardised regression coefficient (beta weight).
        var_names: Predictor variable names involved in this effect.
        column_index: Column in the design matrix (main effects only).
        column_indices: Columns for each component (interactions only).
        factor_source: Original factor name (factor-derived effects only).
        factor_level: Categorical level (factor-derived effects only).
    """

    name: str
    effect_type: str  # "main" or "interaction"
    effect_size: float = 0.0
    var_names: List[str] = field(default_factory=list)
    column_index: Optional[int] = None
    column_indices: List[int] = field(default_factory=list)
    factor_source: Optional[str] = None
    factor_level: Optional[Union[int, str]] = None


def _resolve_reference(reference, labels):
    """Resolve a by-VALUE factor reference to its canonical label string.

    - ``None``           → first label (or ``1`` when there are no labels, the
      parser-driven 1..k path).
    - ``str``            → must equal one of ``labels`` exactly.
    - ``int`` / ``float``→ matched via :func:`value_to_label` (so ``6`` selects
      ``"6"`` for labels ``["4","6","8"]``).

    Raises ``ValueError`` when a supplied reference matches no label.
    """
    if labels is None:
        # Parser path: integer labels 1..k; keep the raw reference (str()'d
        # downstream). A bare default becomes level 1.
        return reference if reference is not None else 1
    if reference is None:
        return labels[0]
    from ..data.upload import value_to_label
    resolved = reference if isinstance(reference, str) else value_to_label(reference)
    if resolved not in labels:
        raise ValueError(
            f"reference {reference!r} (resolved to {resolved!r}) is not among "
            f"the factor labels {labels}"
        )
    return resolved


class VariableRegistry:
    """Single source of truth for all variable and effect state.

    Parses an R-style formula and maintains:

    - The dependent variable name.
    - An ordered dictionary of ``PredictorVar`` instances (continuous,
      binary, factor dummies, cluster-effect columns).
    - An ordered dictionary of ``Effect`` instances (main effects and
      interactions).
    - Factor expansion metadata and correlation matrices.

    Column ordering in the raw design matrix ``X`` is:
    non-factor predictors | cluster-effect columns | factor dummies.
    """

    def __init__(self, equation: str):
        """Parse an R-style formula and initialise the registry.

        Args:
            equation: Formula string. Supports ``=`` or ``~`` as
                separators, ``+`` for additive terms, ``:`` for
                interactions, and ``(1|group)`` for random intercepts.

        Raises:
            ValueError: If the formula is empty, has no fixed effects, or
                contains only random effects.
        """
        from .parsers import _parse_equation, _parse_independent_variables

        self.equation = equation.strip()
        self._dependent: str = ""
        self._predictors: Dict[str, PredictorVar] = {}
        self._effects: Dict[str, Effect] = {}
        self._factors: Dict[str, Dict[str, Any]] = {}
        self._factor_dummies: Dict[str, Dict[str, Any]] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._random_effects_parsed: List[Dict] = []

        # Parse equation
        dep_var, formula_part, random_effects = _parse_equation(self.equation)
        self._dependent = dep_var
        self._random_effects_parsed = random_effects

        if not formula_part.strip():
            if random_effects:
                grouping_vars = ", ".join(re["grouping_var"] for re in random_effects)
                raise ValueError(
                    f"Model has random effects (1|{grouping_vars}) but no fixed effects. "
                    "Power analysis requires at least one fixed effect to test. "
                    "Example: 'y ~ treatment + (1|cluster)'"
                )
            raise ValueError("Equation cannot be empty. Expected format: 'y = x1 + x2'")

        # Parse variables and effects
        variables, effects = _parse_independent_variables(formula_part)

        if not effects:
            raise ValueError("No predictor variables found in equation")

        # Initialize predictors
        for _var_key, var_info in variables.items():
            name = var_info["name"]
            self._predictors[name] = PredictorVar(name=name, column_index=len(self._predictors))

        # Initialize effects
        for _effect_key, effect_info in effects.items():
            name = effect_info["name"]
            eff = Effect(
                name=name,
                effect_type=effect_info["type"],
                var_names=effect_info.get("var_names", [name]),
            )
            if effect_info["type"] == "main":
                eff.column_index = effect_info.get("column_index")
            else:
                eff.column_indices = effect_info.get("column_indices", [])
            self._effects[name] = eff

    @property
    def dependent(self) -> str:
        """Name of dependent variable."""
        return self._dependent

    @property
    def predictor_names(self) -> List[str]:
        """List of predictor variable names (in order)."""
        return sorted(self._predictors.keys(), key=lambda x: self._predictors[x].column_index or 0)

    @property
    def effect_names(self) -> List[str]:
        """List of effect names."""
        return list(self._effects.keys())

    @property
    def factor_names(self) -> List[str]:
        """List of factor variable names."""
        return list(self._factors.keys())

    @property
    def non_factor_names(self) -> List[str]:
        """List of non-factor predictor names (excludes cluster effects)."""
        return [
            name
            for name, pred in self._predictors.items()
            if not pred.is_factor and not pred.is_dummy and pred.var_type != "cluster_effect"
        ]

    @property
    def cluster_effect_names(self) -> List[str]:
        """List of cluster random effect column names."""
        return [name for name, pred in self._predictors.items() if pred.var_type == "cluster_effect"]

    @property
    def dummy_names(self) -> List[str]:
        """List of factor dummy variable names."""
        return [name for name, pred in self._predictors.items() if pred.is_dummy]

    def factor_levels(self, name: str) -> List[str]:
        """Full ordered level-label list for factor ``name`` (reference
        included), as strings. This is the port's ``labels[factor]`` store that
        the engine's ``EffectSkeleton`` ``FactorLevel.level`` index resolves
        against, and the same list sent to the contract as ``levels``."""
        info = self._factors[name]
        level_labels = info.get("level_labels")
        if level_labels is None:
            return [str(i) for i in range(1, int(info["n_levels"]) + 1)]
        return [str(lb) for lb in level_labels]

    def get_predictor(self, name: str) -> Optional[PredictorVar]:
        """Get predictor by name."""
        return self._predictors.get(name)

    def get_effect(self, name: str) -> Optional[Effect]:
        """Get effect by name."""
        return self._effects.get(name)

    def set_variable_type(self, name: str, var_type: str, **kwargs) -> None:
        """
        Set the distribution type for a predictor variable.

        Args:
            name: Variable name
            var_type: Distribution type ('normal', 'binary', 'factor', etc.)
            **kwargs: Additional parameters (proportion, n_levels, proportions,
                sampled_proportions). ``sampled_proportions`` (factors only):
                per-factor override for proportion sampling — None/omitted
                inherits the scenario ``sampled_factor_proportions``; True samples
                per row (Multinomial jitter); False holds exact counts.
        """
        if name not in self._predictors:
            raise ValueError(f"Unknown variable: {name}")

        pred = self._predictors[name]
        pred.var_type = var_type

        # Explicit continuous-distribution assignment pins the column so
        # scenario distribution swaps leave it alone. All five canonical
        # residual-space names count as "explicit" (including explicit "normal").
        _PINNABLE = {"normal", "right_skewed", "left_skewed", "high_kurtosis", "uniform"}
        if var_type in _PINNABLE:
            pred.pinned = True

        if var_type == "binary":
            pred.proportion = kwargs.get("proportion", 0.5)
        elif var_type == "factor":
            pred.is_factor = True
            pred.n_levels = kwargs.get("n_levels", 2)
            pred.proportions = kwargs.get("proportions")
            labels = kwargs.get("labels")
            reference = kwargs.get("reference")
            sampled_proportions = kwargs.get("sampled_proportions")  # None | True | False

            if labels is not None:
                pred.level_labels = labels

            # `reference` is by VALUE, not position: resolve it to a canonical
            # label string here (port-side), then store it like before so the
            # contract builder, factor expansion, and result naming are unchanged.
            # (Internal `_factors` keys stay `level_labels`/`reference_level`.)
            self._factors[name] = {
                "n_levels": pred.n_levels,
                "proportions": pred.proportions,
                "level_labels": labels,
                "reference_level": _resolve_reference(reference, labels),
                "sampled_proportions": sampled_proportions,
            }

    def set_effect_size(self, name: str, value: float) -> None:
        """
        Set effect size for an effect.

        Args:
            name: Effect name
            value: Effect size (standardized coefficient)
        """
        if name not in self._effects:
            raise ValueError(f"Unknown effect: {name}")

        self._effects[name].effect_size = value

    def expand_factors(self) -> None:
        """Expand factor variables into dummy-coded predictors and effects.

        Modifies the registry in place:

        1. Removes original factor predictor entries.
        2. Adds ``n_levels - 1`` dummy ``PredictorVar`` instances per factor
           (reference-coded, level 1 omitted).
        3. Creates main ``Effect`` entries for each dummy.
        4. Expands interactions involving factors into per-level interactions.
        5. Re-indexes all column indices.
        """
        if not self._factors:
            return

        original_effects = dict(self._effects)

        # Track new predictors and effects
        new_predictors: Dict[str, PredictorVar] = {}
        new_effects: Dict[str, Effect] = {}

        # Keep non-factor predictors
        col_idx = 0
        for name, pred in self._predictors.items():
            if not pred.is_factor:
                pred.column_index = col_idx
                new_predictors[name] = pred
                col_idx += 1

        # Keep non-factor effects
        for name, eff in self._effects.items():
            has_factor = any(vn in self._factors for vn in eff.var_names)
            if not has_factor:
                new_effects[name] = eff

        # Create dummy variables and effects for each factor
        for factor_name, factor_info in self._factors.items():
            n_levels = factor_info["n_levels"]
            level_labels = factor_info.get("level_labels")
            reference_level = factor_info.get("reference_level", 1)

            # Compute non-reference levels once
            if level_labels is not None:
                non_ref = [lb for lb in level_labels if lb != str(reference_level)]
            else:
                non_ref = list(range(2, n_levels + 1))

            for level in non_ref:
                dummy_name = f"{factor_name}[{level}]"

                dummy_pred = PredictorVar(
                    name=dummy_name,
                    var_type="factor_dummy",
                    is_dummy=True,
                    factor_source=factor_name,
                    factor_level=level,
                    column_index=col_idx,
                    level_labels=level_labels if level_labels is not None else None,
                )
                new_predictors[dummy_name] = dummy_pred

                dummy_eff = Effect(
                    name=dummy_name,
                    effect_type="main",
                    var_names=[dummy_name],
                    column_index=col_idx,
                    factor_source=factor_name,
                    factor_level=level,
                )
                new_effects[dummy_name] = dummy_eff

                self._factor_dummies[dummy_name] = {
                    "factor_name": factor_name,
                    "level": level,
                }

                col_idx += 1

        # Handle interactions involving factors — Cartesian product of
        # non-reference dummy levels across all factor components.
        from itertools import product as cartesian_product

        for _name, eff in original_effects.items():
            if eff.effect_type == "interaction":
                factor_vars = [vn for vn in eff.var_names if vn in self._factors]

                if factor_vars:
                    # Build per-component level options
                    level_options: list[list[str]] = []
                    for vn in eff.var_names:
                        if vn in self._factors:
                            factor_info = self._factors[vn]
                            n_levels = factor_info["n_levels"]
                            level_labels = factor_info.get("level_labels")
                            reference_level = factor_info.get("reference_level", 1)

                            if level_labels is not None:
                                non_ref = [f"{vn}[{lb}]" for lb in level_labels if lb != str(reference_level)]
                            else:
                                non_ref = [f"{vn}[{lvl}]" for lvl in range(2, n_levels + 1)]
                            level_options.append(non_ref)
                        else:
                            level_options.append([vn])

                    for combo in cartesian_product(*level_options):
                        new_var_names = list(combo)
                        new_interaction_name = ":".join(new_var_names)
                        new_eff = Effect(
                            name=new_interaction_name,
                            effect_type="interaction",
                            var_names=new_var_names,
                            column_indices=[],  # Updated later
                        )
                        new_effects[new_interaction_name] = new_eff

        # Update predictors and effects
        self._predictors = new_predictors
        self._effects = new_effects

        # Update column indices for all effects
        self._update_effect_indices()

    def _update_effect_indices(self) -> None:
        """Update column indices for all effects based on current predictor order."""
        predictor_order = self.predictor_names

        for _name, eff in self._effects.items():
            if eff.effect_type == "main":
                if eff.name in predictor_order:
                    eff.column_index = predictor_order.index(eff.name)
            else:  # interaction
                eff.column_indices = [predictor_order.index(vn) for vn in eff.var_names if vn in predictor_order]

    def get_effect_sizes(self) -> np.ndarray:
        """Get effect sizes as numpy array in effect order."""
        return np.array([eff.effect_size for eff in self._effects.values()])

    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get correlation matrix for non-factor variables."""
        return self._correlation_matrix

    def set_correlation_matrix(self, matrix: np.ndarray) -> None:
        """Set correlation matrix for non-factor variables."""
        self._correlation_matrix = matrix.copy()

    def set_correlation(self, var1: str, var2: str, value: float) -> None:
        """Set a single correlation value."""
        non_factor = self.non_factor_names

        if var1 not in non_factor or var2 not in non_factor:
            raise ValueError("Can only set correlations between non-factor variables")

        if self._correlation_matrix is None:
            n = len(non_factor)
            self._correlation_matrix = np.eye(n)

        idx1 = non_factor.index(var1)
        idx2 = non_factor.index(var2)
        self._correlation_matrix[idx1, idx2] = value
        self._correlation_matrix[idx2, idx1] = value

