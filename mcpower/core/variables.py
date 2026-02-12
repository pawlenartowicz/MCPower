"""
Variable and Effect Registry for MCPower framework.

This module provides unified management of variables, effects, and factors.
It replaces the scattered state in the original base.py with a single
source of truth for all variable-related information.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
    """

    name: str
    var_type: str = "normal"
    proportion: float = 0.5
    n_levels: Optional[int] = None
    proportions: Optional[List[float]] = None
    is_factor: bool = False
    is_dummy: bool = False
    factor_source: Optional[str] = None
    factor_level: Optional[int] = None
    column_index: Optional[int] = None


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
    factor_level: Optional[int] = None


@dataclass
class ClusterSpec:
    """Specification for a cluster/grouping variable with a random intercept.

    Attributes:
        grouping_var: Name of the grouping variable from the formula.
        n_clusters: Number of clusters (derived at analysis time if not set).
        cluster_size: Observations per cluster (derived if not set).
        icc: Intraclass correlation coefficient.
        tau_squared: Between-cluster variance, computed as
            ``ICC / (1 - ICC)`` (adjusted for fixed-effect variance in
            ``prepare_metadata``).
        id_effect_name: Name of the synthetic predictor that carries the
            random intercept values (e.g. ``"school_id_effect"``).
    """

    grouping_var: str
    n_clusters: Optional[int] = None
    cluster_size: Optional[int] = None
    icc: float = 0.0
    tau_squared: float = 0.0
    id_effect_name: str = ""


class VariableRegistry:
    """Single source of truth for all variable and effect state.

    Parses an R-style formula and maintains:

    - The dependent variable name.
    - An ordered dictionary of ``PredictorVar`` instances (continuous,
      binary, factor dummies, cluster-effect columns).
    - An ordered dictionary of ``Effect`` instances (main effects and
      interactions).
    - Factor expansion metadata and correlation matrices.
    - ``ClusterSpec`` instances for mixed-model grouping variables.

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
        from ..utils.parsers import _parse_equation, _parse_independent_variables

        self.equation = equation.strip()
        self._dependent: str = ""
        self._predictors: Dict[str, PredictorVar] = {}
        self._effects: Dict[str, Effect] = {}
        self._factors: Dict[str, Dict[str, Any]] = {}
        self._factor_dummies: Dict[str, Dict[str, Any]] = {}
        self._correlation_matrix: Optional[np.ndarray] = None
        self._cluster_specs: Dict[str, ClusterSpec] = {}
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
    def n_predictors(self) -> int:
        """Number of predictor variables."""
        return len(self._predictors)

    @property
    def n_effects(self) -> int:
        """Number of effects."""
        return len(self._effects)

    @property
    def factor_names(self) -> List[str]:
        """List of factor variable names."""
        return list(self._factors.keys())

    @property
    def cluster_names(self) -> List[str]:
        """List of cluster grouping variable names."""
        return list(self._cluster_specs.keys())

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
            **kwargs: Additional parameters (proportion, n_levels, proportions)
        """
        if name not in self._predictors:
            raise ValueError(f"Unknown variable: {name}")

        pred = self._predictors[name]
        pred.var_type = var_type

        if var_type == "binary":
            pred.proportion = kwargs.get("proportion", 0.5)
        elif var_type == "factor":
            pred.is_factor = True
            pred.n_levels = kwargs.get("n_levels", 2)
            pred.proportions = kwargs.get("proportions")

            # Store factor info
            self._factors[name] = {
                "n_levels": pred.n_levels,
                "proportions": pred.proportions,
                "reference_level": 1,
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

            for level in range(2, n_levels + 1):
                dummy_name = f"{factor_name}[{level}]"

                # Create dummy predictor
                dummy_pred = PredictorVar(
                    name=dummy_name,
                    var_type="factor_dummy",
                    is_dummy=True,
                    factor_source=factor_name,
                    factor_level=level,
                    column_index=col_idx,
                )
                new_predictors[dummy_name] = dummy_pred

                # Create main effect for dummy
                dummy_eff = Effect(
                    name=dummy_name,
                    effect_type="main",
                    var_names=[dummy_name],
                    column_index=col_idx,
                    factor_source=factor_name,
                    factor_level=level,
                )
                new_effects[dummy_name] = dummy_eff

                # Store dummy mapping
                self._factor_dummies[dummy_name] = {
                    "factor_name": factor_name,
                    "level": level,
                }

                col_idx += 1

        # Handle interactions involving factors
        for _name, eff in original_effects.items():
            if eff.effect_type == "interaction":
                factor_vars = [vn for vn in eff.var_names if vn in self._factors]

                if factor_vars:
                    for factor_var in factor_vars:
                        n_levels = self._factors[factor_var]["n_levels"]

                        for level in range(2, n_levels + 1):
                            dummy_name = f"{factor_var}[{level}]"

                            # Replace factor name with dummy name
                            new_var_names = [dummy_name if vn == factor_var else vn for vn in eff.var_names]
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

    def get_column_order(self) -> List[str]:
        """Get the column order for design matrix construction."""
        return self.non_factor_names + self.cluster_effect_names + self.dummy_names

    def get_effect_sizes(self) -> np.ndarray:
        """Get effect sizes as numpy array in effect order."""
        return np.array([eff.effect_size for eff in self._effects.values()])

    def get_var_types(self) -> np.ndarray:
        """Get variable types as numpy array (for data generation)."""
        type_mapping = {
            "normal": 0,
            "binary": 1,
            "right_skewed": 2,
            "left_skewed": 3,
            "high_kurtosis": 4,
            "uniform": 5,
            "uploaded_factor": 97,
            "uploaded_binary": 98,
            "uploaded_data": 99,
        }

        non_factor = [self._predictors[name] for name in self.non_factor_names]
        return np.array([type_mapping.get(p.var_type, 0) for p in non_factor], dtype=np.int64)

    def get_var_params(self) -> np.ndarray:
        """Get variable parameters (proportions for binary) as numpy array."""
        non_factor = [self._predictors[name] for name in self.non_factor_names]
        return np.array([p.proportion for p in non_factor], dtype=np.float64)

    def get_factor_specs(self) -> List[Dict[str, Any]]:
        """Get factor specifications for data generation."""
        return [{"n_levels": info["n_levels"], "proportions": info["proportions"]} for info in self._factors.values()]

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

    def is_dummy_variable(self, name: str) -> bool:
        """Check if a variable is a factor dummy."""
        return name in self._factor_dummies

    def get_target_indices(self, target_tests: List[str]) -> np.ndarray:
        """Map target test names to indices in the effect-size array.

        Args:
            target_tests: Effect names to test. ``"overall"`` expands to
                all non-cluster fixed effects.

        Returns:
            Integer array of indices into the effect order.
        """
        effect_order = list(self._effects.keys())
        indices = []

        # Check if "overall" is requested
        if "overall" in target_tests:
            # Include ALL non-cluster fixed effects
            for idx, effect_name in enumerate(effect_order):
                # Skip cluster effects (they're random effects, not to be tested)
                if effect_name not in self.cluster_effect_names:
                    indices.append(idx)
        else:
            # Include only specified effects (excluding cluster effects)
            for test in target_tests:
                if test in effect_order and test not in self.cluster_effect_names:
                    indices.append(effect_order.index(test))

        return np.array(indices, dtype=np.int64)

    def register_cluster(
        self,
        grouping_var: str,
        n_clusters: Optional[int],
        cluster_size: Optional[int],
        icc: float,
    ) -> None:
        """Register a cluster/grouping variable with a random intercept.

        Creates a ``ClusterSpec``, adds a synthetic ``cluster_effect``
        predictor and a fixed-at-1.0 ``Effect`` entry, then re-indexes
        all predictor columns.

        Args:
            grouping_var: Name of the grouping variable.
            n_clusters: Number of clusters (one of *n_clusters* or
                *cluster_size* must be provided).
            cluster_size: Observations per cluster.
            icc: Intraclass correlation coefficient (0 <= ICC < 1).
        """
        # Calculate tau_squared from ICC
        # tau² = ICC / (1 - ICC), where sigma² = 1
        if icc == 0:
            tau_squared = 0.0
        else:
            tau_squared = icc / (1.0 - icc)

        # Create id_effect name
        id_effect_name = f"{grouping_var}_id_effect"

        # Create ClusterSpec
        spec = ClusterSpec(
            grouping_var=grouping_var,
            n_clusters=n_clusters,
            cluster_size=cluster_size,
            icc=icc,
            tau_squared=tau_squared,
            id_effect_name=id_effect_name,
        )
        self._cluster_specs[grouping_var] = spec

        # Add id_effect as a predictor
        # Column index will be assigned after non_factor vars but before dummies
        n_non_factor = len([p for p in self._predictors.values() if not p.is_factor and not p.is_dummy and p.var_type != "cluster_effect"])
        n_existing_cluster = len([p for p in self._predictors.values() if p.var_type == "cluster_effect"])

        id_effect_pred = PredictorVar(
            name=id_effect_name,
            var_type="cluster_effect",
            column_index=n_non_factor + n_existing_cluster,
        )
        self._predictors[id_effect_name] = id_effect_pred

        # Add effect with beta=1.0
        id_effect_eff = Effect(
            name=id_effect_name,
            effect_type="main",
            effect_size=1.0,
            var_names=[id_effect_name],
            column_index=n_non_factor + n_existing_cluster,
        )
        self._effects[id_effect_name] = id_effect_eff

        # Update all column indices to maintain correct order
        self._reindex_predictors()

    def _reindex_predictors(self) -> None:
        """Reindex all predictors to maintain order: non_factor | cluster_effect | dummies."""
        col_idx = 0

        # Non-factor predictors first
        for name in sorted(self._predictors.keys(), key=lambda x: self._predictors[x].column_index or 0):
            pred = self._predictors[name]
            if not pred.is_factor and not pred.is_dummy and pred.var_type != "cluster_effect":
                pred.column_index = col_idx
                col_idx += 1

        # Cluster effect predictors second
        for name in sorted(self._predictors.keys(), key=lambda x: self._predictors[x].column_index or 0):
            pred = self._predictors[name]
            if pred.var_type == "cluster_effect":
                pred.column_index = col_idx
                col_idx += 1

        # Factor dummies last
        for name in sorted(self._predictors.keys(), key=lambda x: self._predictors[x].column_index or 0):
            pred = self._predictors[name]
            if pred.is_dummy:
                pred.column_index = col_idx
                col_idx += 1

        # Update effect indices
        self._update_effect_indices()
