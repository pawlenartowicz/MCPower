"""
Parsing utilities for Monte Carlo Power Analysis.

This module provides parsing functions for equations, formulas, assignments,
and various model-specific configurations.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = []

# Unicode-aware identifier pattern: letter or underscore, then word characters
_IDENT = r"[^\W\d]\w*"


class _AssignmentParser:
    """Parses comma-separated ``name=value`` assignment strings.

    Supports three parse types — ``"variable_type"``, ``"correlation"``,
    and ``"effect"`` — each with a specialised value handler. Correlation
    assignments use the syntax ``corr(var1, var2)=value``.

    A module-level singleton ``_parser`` is used throughout the codebase.
    """

    def __init__(self):
        self.handlers = {
            "variable_type": self._parse_variable_type_value,
            "correlation": self._parse_correlation_value,
            "effect": self._parse_effect_value,
        }

    def _parse(self, input_string: str, parse_type: str, available_items: List[str]) -> Tuple[Dict, List[str]]:
        """Parse a comma-separated assignment string.

        Args:
            input_string: Raw user input (e.g. ``"x1=0.5, x2=binary"``).
            parse_type: One of ``"variable_type"``, ``"correlation"``,
                or ``"effect"``.
            available_items: Valid names that may appear on the left-hand
                side of assignments.

        Returns:
            Tuple of ``(parsed_dict, error_list)``. For correlations the
            dict is keyed by sorted variable-name tuples; otherwise by
            variable name.
        """
        if parse_type not in self.handlers:
            return {}, [f"Unknown parse type: {parse_type}"]

        assignments = self._split_assignments(input_string)
        parsed_items = {}
        errors = []

        for assignment in assignments:
            try:
                name, value = self._parse_assignment(assignment, parse_type)

                # Validate name
                if parse_type == "correlation":
                    # Special validation for correlation pairs
                    valid, error = self._validate_correlation_pair(name, available_items)
                    if not valid:
                        if error is not None:
                            errors.append(error)
                        continue
                else:
                    if name not in available_items:
                        errors.append(f"'{name}' not found. Available: {', '.join(available_items)}")
                        continue

                # Parse value using type-specific handler
                parsed_value, error = self.handlers[parse_type](value)
                if error:
                    errors.append(f"{name}: {error}")
                    continue

                # Store result
                if parse_type == "correlation":
                    var1, var2 = name  # name is tuple for correlations
                    key = tuple(sorted([var1, var2]))
                    parsed_items[key] = parsed_value
                else:
                    parsed_items[name] = parsed_value

            except ValueError as e:
                errors.append(str(e))

        return parsed_items, errors

    def _split_assignments(self, input_string: str) -> List[str]:
        """Split assignments respecting parentheses."""
        assignments = []
        current: List[str] = []
        paren_count = 0

        for char in input_string:
            if char == "," and paren_count == 0:
                if current:
                    assignments.append("".join(current).strip())
                    current = []
            else:
                if char == "(":
                    paren_count += 1
                elif char == ")":
                    paren_count -= 1
                current.append(char)

        if current:
            assignments.append("".join(current).strip())

        return assignments

    def _parse_assignment(self, assignment: str, parse_type: str) -> Tuple[Any, str]:
        """Parse single assignment into name and value parts."""
        if "=" not in assignment:
            raise ValueError(f"Invalid format: '{assignment}'. Expected 'name=value'")

        if parse_type == "correlation":
            # Special parsing for correlation format: corr(x1,x2)=0.5
            return self._parse_correlation_assignment(assignment)
        else:
            # Standard name=value format
            name, value = assignment.split("=", 1)
            return name.strip(), value.strip()

    def _parse_correlation_assignment(self, assignment: str) -> Tuple[Tuple[str, str], str]:
        """Parse correlation assignment like 'corr(x1,x2)=0.5'."""
        left, right = assignment.split("=", 1)

        # Match correlation pattern
        pattern = r"(?:corr?)?(?:\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*\))"
        match = re.match(pattern, left.strip())

        if not match:
            raise ValueError(f"Invalid correlation format: '{left}'. Expected 'corr(var1, var2)' or '(var1, var2)'")

        var1, var2 = match.groups()
        return (var1.strip(), var2.strip()), right.strip()

    def _validate_correlation_pair(self, pair: Tuple[str, str], available_vars: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate correlation variable pair."""
        var1, var2 = pair

        if var1 not in available_vars:
            return False, f"Variable '{var1}' not found"
        if var2 not in available_vars:
            return False, f"Variable '{var2}' not found"
        if var1 == var2:
            return False, f"Cannot correlate variable with itself: '{var1}'"

        return True, None

    def _parse_variable_type_value(self, value: str) -> Tuple[Dict[str, Any], Optional[str]]:
        """Parse variable type value including factor support."""
        supported_types = [
            "normal",
            "binary",
            "right_skewed",
            "left_skewed",
            "high_kurtosis",
            "uniform",
            "factor",
        ]

        if value.startswith("(") and value.endswith(")"):
            # Tuple format: (type, param) or (type, param1, param2, ...)
            content = value[1:-1]
            if "," not in content:
                return (
                    {},
                    "Invalid tuple format. Expected '(type,value)' or '(type,val1,val2,...)'",
                )

            parts = [p.strip() for p in content.split(",")]
            if len(parts) < 2:
                return {}, "Expected at least 2 values in tuple"

            var_type = parts[0]

            if var_type not in supported_types:
                return {}, f"Unsupported type '{var_type}'"

            if var_type == "binary":
                if len(parts) != 2:
                    return (
                        {},
                        "Binary type expects exactly 2 values: (binary, proportion)",
                    )
                try:
                    proportion = float(parts[1])
                    if not 0 <= proportion <= 1:
                        return {}, "Proportion must be between 0 and 1"
                    return {"type": var_type, "proportion": proportion}, None
                except ValueError:
                    return {}, f"Invalid proportion value '{parts[1]}'"

            elif var_type == "factor":
                if len(parts) == 2:
                    # Format: (factor, n_levels) - equal proportions
                    try:
                        n_levels = int(parts[1])
                        if n_levels < 2:
                            return {}, "Factor must have at least 2 levels"
                        if n_levels > 20:
                            return {}, "Factor cannot have more than 20 levels"

                        # Equal proportions for all levels
                        proportions = [1.0 / n_levels] * n_levels
                        return {
                            "type": var_type,
                            "n_levels": n_levels,
                            "proportions": proportions,
                        }, None
                    except ValueError:
                        return (
                            {},
                            f"Invalid number of levels '{parts[1]}'. Must be integer",
                        )

                elif len(parts) >= 3:
                    # Format: (factor, prop1, prop2, ...) - custom proportions
                    try:
                        proportions = [float(p) for p in parts[1:]]
                        n_levels = len(proportions)

                        if n_levels < 2:
                            return {}, "Factor must have at least 2 levels"
                        if n_levels > 20:
                            return {}, "Factor cannot have more than 20 levels"

                        # Check for zero/negative proportions
                        if any(p <= 0 for p in proportions):
                            return (
                                {},
                                "All proportions must be positive (greater than 0)",
                            )

                        # Normalize proportions to sum to 1
                        total = sum(proportions)
                        proportions = [p / total for p in proportions]

                        return {
                            "type": var_type,
                            "n_levels": n_levels,
                            "proportions": proportions,
                        }, None
                    except ValueError:
                        return {}, "Invalid proportions. All values must be numeric"
                else:
                    return (
                        {},
                        "Factor format: (factor,n_levels) or (factor,prop1,prop2,...)",
                    )
            else:
                return {}, "Tuple format only supported for binary and factor variables"
        else:
            # Simple type
            if value not in supported_types:
                return (
                    {},
                    f"Unsupported type '{value}'. Valid: {', '.join(supported_types)}",
                )

            result: Dict[str, Any] = {"type": value}
            if value == "binary":
                result["proportion"] = 0.5
            elif value == "factor":
                # Default factor: 3 levels with equal proportions
                result["n_levels"] = 3
                result["proportions"] = [1 / 3, 1 / 3, 1 / 3]
            return result, None

    def _parse_correlation_value(self, value: str) -> Tuple[float, Optional[str]]:
        """Parse correlation value."""
        try:
            corr = float(value)
            if not -1 <= corr <= 1:
                return 0.0, "Correlation must be between -1 and 1"
            return corr, None
        except ValueError:
            return 0.0, f"Invalid correlation value '{value}'"

    def _parse_effect_value(self, value: str) -> Tuple[float, Optional[str]]:
        """Parse effect size value."""
        try:
            return float(value), None
        except ValueError:
            return 0.0, f"Invalid effect size '{value}'. Must be a number"


_parser = _AssignmentParser()


def _parse_equation(equation: str) -> Tuple[str, str, List[Dict]]:
    """Parse an R-style formula into its components.

    Splits the equation at ``~`` or ``=``, extracts random-effect
    terms, and returns the cleaned fixed-effect formula.

    Supported random-effect syntax:
    - ``(1|group)`` — random intercept
    - ``(1 + x|group)`` — random intercept and slope
    - ``(1 + x1 + x2|group)`` — random intercept and multiple slopes
    - ``(1|A/B)`` — nested random intercepts (expands to ``(1|A) + (1|A:B)``)

    Args:
        equation: Formula string (e.g. ``"y ~ x1 + x2 + (1|cluster)"``).

    Returns:
        Tuple of ``(dependent_var, fixed_formula, random_effects)`` where
        *random_effects* is a list of dicts with keys:
        - ``"type"``: ``"random_intercept"`` or ``"random_slope"``
        - ``"grouping_var"``: grouping variable name
        - ``"slope_vars"``: list of slope variable names (slopes only)
        - ``"parent_var"``: parent grouping var name (nested only)

    Raises:
        ValueError: If a grouping variable appears more than once.
    """
    equation = equation.replace(" ", "")

    if "~" in equation:
        left_side, right_side = equation.split("~", 1)
        dep_var = left_side.strip()
        formula_part = right_side
    elif "=" in equation:
        left_side, right_side = equation.split("=", 1)
        dep_var = left_side.strip()
        formula_part = right_side
    else:
        dep_var = "explained_variable"
        formula_part = equation

    # Extract random effects from formula
    random_effects: List[Dict] = []
    seen_grouping_vars: set = set()

    # 1. Extract nested random intercepts: (1|A/B)
    nested_pattern = rf"\(\s*1\s*\|\s*({_IDENT})\s*/\s*({_IDENT})\s*\)"
    for match in re.finditer(nested_pattern, formula_part):
        parent_var = match.group(1)
        child_var = match.group(2)
        nested_var = f"{parent_var}:{child_var}"

        if parent_var in seen_grouping_vars:
            raise ValueError(f"Duplicate random effect grouping variable: '{parent_var}'")
        if nested_var in seen_grouping_vars:
            raise ValueError(f"Duplicate random effect grouping variable: '{nested_var}'")

        seen_grouping_vars.add(parent_var)
        seen_grouping_vars.add(nested_var)

        random_effects.append({"type": "random_intercept", "grouping_var": parent_var})
        random_effects.append(
            {
                "type": "random_intercept",
                "grouping_var": nested_var,
                "parent_var": parent_var,
            }
        )

    formula_part = re.sub(nested_pattern, "", formula_part)

    # 2. Extract random slopes: (1 + var1 + var2 | group)
    slope_pattern = rf"\(\s*1\s*\+\s*([^|]+?)\s*\|\s*({_IDENT})\s*\)"
    for match in re.finditer(slope_pattern, formula_part):
        slope_vars_str = match.group(1)
        grouping_var = match.group(2)

        if grouping_var in seen_grouping_vars:
            raise ValueError(f"Duplicate random effect grouping variable: '{grouping_var}'")
        seen_grouping_vars.add(grouping_var)

        slope_vars = [v.strip() for v in slope_vars_str.split("+") if v.strip()]
        if not slope_vars:
            raise ValueError(f"No slope variables found in random effect term for '{grouping_var}'")

        random_effects.append(
            {
                "type": "random_slope",
                "grouping_var": grouping_var,
                "slope_vars": slope_vars,
            }
        )

    formula_part = re.sub(slope_pattern, "", formula_part)

    # 3. Extract random intercepts: (1|var)
    intercept_pattern = rf"\(\s*1\s*\|\s*({_IDENT})\s*\)"
    for match in re.finditer(intercept_pattern, formula_part):
        grouping_var = match.group(1)
        if grouping_var in seen_grouping_vars:
            raise ValueError(f"Duplicate random effect grouping variable: '{grouping_var}'")
        seen_grouping_vars.add(grouping_var)
        random_effects.append({"type": "random_intercept", "grouping_var": grouping_var})

    formula_part = re.sub(intercept_pattern, "", formula_part)

    # Clean up extra + signs and whitespace
    formula_part = re.sub(r"\+\s*\+", "+", formula_part)  # ++ -> +
    formula_part = re.sub(r"^\+", "", formula_part)  # leading +
    formula_part = re.sub(r"\+$", "", formula_part)  # trailing +
    formula_part = formula_part.strip()

    return dep_var, formula_part, random_effects


def _parse_independent_variables(formula: str) -> Tuple[Dict, Dict]:
    """Extract predictor variables and effects from the fixed-effect formula.

    Handles ``+`` for additive terms, ``:`` for specific interactions, and
    ``*`` for full factorial expansion (all main effects plus all two-way
    through n-way interactions).

    Args:
        formula: Right-hand side of the equation (fixed effects only).

    Returns:
        Tuple of ``(variables_dict, effects_dict)`` where each dict is
        keyed by auto-generated identifiers (``variable_1``, ``effect_1``,
        etc.) with value dicts describing the variable or effect.
    """
    from itertools import combinations

    terms = re.split(r"[+\-]", formula)

    variables = {}
    effects = {}
    variable_counter = 1
    effect_counter = 1
    seen_variables = set()
    seen_effects = set()

    for term in terms:
        term = term.strip()
        if not term:
            continue

        if "*" in term or ":" in term:
            interaction_vars = re.findall(_IDENT, term)

            # Add individual variables
            for var in interaction_vars:
                if var not in seen_variables:
                    variables[f"variable_{variable_counter}"] = {"name": var}
                    seen_variables.add(var)
                    variable_counter += 1

            if "*" in term:
                # For x1*x2*x3: add main effects + all possible interactions

                # Add main effects first
                for var in interaction_vars:
                    if var not in seen_effects:
                        effects[f"effect_{effect_counter}"] = {
                            "name": var,
                            "type": "main",
                        }
                        seen_effects.add(var)
                        effect_counter += 1

                # Add all possible interactions (2-way, 3-way, ..., n-way)
                for r in range(2, len(interaction_vars) + 1):
                    for combo in combinations(interaction_vars, r):
                        interaction_name = ":".join(combo)
                        if interaction_name not in seen_effects:
                            effects[f"effect_{effect_counter}"] = {
                                "name": interaction_name,
                                "type": "interaction",
                                "var_names": list(combo),
                            }
                            seen_effects.add(interaction_name)
                            effect_counter += 1
            else:
                # For x1:x2:x3: add only the specific interaction
                interaction_name = ":".join(interaction_vars)
                if interaction_name not in seen_effects:
                    effects[f"effect_{effect_counter}"] = {
                        "name": interaction_name,
                        "type": "interaction",
                        "var_names": interaction_vars,
                    }
                    seen_effects.add(interaction_name)
                    effect_counter += 1
        else:
            # Main effect term
            variables_in_term = re.findall(_IDENT, term)

            for var in variables_in_term:
                if var not in seen_variables:
                    variables[f"variable_{variable_counter}"] = {"name": var}
                    seen_variables.add(var)
                    variable_counter += 1

                if var not in seen_effects:
                    effects[f"effect_{effect_counter}"] = {"name": var, "type": "main"}
                    seen_effects.add(var)
                    effect_counter += 1

    # Add column indices after parsing
    predictor_vars = [info["name"] for key, info in variables.items() if key != "variable_0"]

    for effect_info in effects.values():
        if effect_info["type"] == "main":
            var_name = effect_info["name"]
            if var_name in predictor_vars:
                effect_info["column_index"] = predictor_vars.index(var_name)
        else:  # interaction
            var_names = effect_info["var_names"]
            effect_info["column_indices"] = [predictor_vars.index(var) for var in var_names]

    return variables, effects
