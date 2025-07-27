"""
Parsing utilities for Monte Carlo Power Analysis.

This module provides parsing functions for equations, formulas, assignments,
and various model-specific configurations.
"""

import re
from typing import Dict, List, Tuple, Any, Optional, Callable

__all__ = []


class _AssignmentParser:
    """Generic parser for assignment strings with specialized handlers."""

    def __init__(self):
        self.handlers = {
            "variable_type": self._parse_variable_type_value,
            "correlation": self._parse_correlation_value,
            "effect": self._parse_effect_value,
        }

    def _parse(
        self, input_string: str, parse_type: str, available_items: List[str]
    ) -> Tuple[Dict, List[str]]:
        """
        Generic parser for assignment strings.

        Args:
            input_string: String with assignments
            parse_type: Type of parsing ('variable_type', 'correlation', 'effect')
            available_items: List of valid item names

        Returns:
            tuple: (parsed_dict, error_list)
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
                    valid, error = self._validate_correlation_pair(
                        name, available_items
                    )
                    if not valid:
                        errors.append(error)
                        continue
                else:
                    if name not in available_items:
                        errors.append(
                            f"'{name}' not found. Available: {', '.join(available_items)}"
                        )
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
        current = []
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

    def _parse_correlation_assignment(
        self, assignment: str
    ) -> Tuple[Tuple[str, str], str]:
        """Parse correlation assignment like 'corr(x1,x2)=0.5'."""
        left, right = assignment.split("=", 1)

        # Match correlation pattern
        pattern = r"(?:corr?)?(?:\s*\(\s*([^,]+?)\s*,\s*([^,]+?)\s*\))"
        match = re.match(pattern, left.strip())

        if not match:
            raise ValueError(
                f"Invalid correlation format: '{left}'. "
                "Expected 'corr(var1, var2)' or '(var1, var2)'"
            )

        var1, var2 = match.groups()
        return (var1.strip(), var2.strip()), right.strip()

    def _validate_correlation_pair(
        self, pair: Tuple[str, str], available_vars: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """Validate correlation variable pair."""
        var1, var2 = pair

        if var1 not in available_vars:
            return False, f"Variable '{var1}' not found"
        if var2 not in available_vars:
            return False, f"Variable '{var2}' not found"
        if var1 == var2:
            return False, f"Cannot correlate variable with itself: '{var1}'"

        return True, None

    def _parse_variable_type_value(
        self, value: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
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
                        return {}, f"Proportion must be between 0 and 1"
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

            result = {"type": value}
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


def _parse_equation(equation: str) -> Tuple[str, str]:
    """
    Parse R-style equation into components.

    Args:
        equation: R-style equation string (e.g., "y ~ x1 + x2" or "y = x1 + x2")

    Returns:
        tuple: (dependent_variable_name, formula_part)
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

    return dep_var, formula_part


def _parse_independent_variables(formula: str) -> Tuple[Dict, Dict]:
    """
    Extract independent variables and effects from formula string.

    Args:
        formula: Formula string (right side of equation)

    Returns:
        tuple: (variables_dict, effects_dict)
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
            interaction_vars = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", term)

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
            variables_in_term = re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", term)

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
    predictor_vars = [
        info["name"] for key, info in variables.items() if key != "variable_0"
    ]

    for effect_info in effects.values():
        if effect_info["type"] == "main":
            var_name = effect_info["name"]
            if var_name in predictor_vars:
                effect_info["column_index"] = predictor_vars.index(var_name)
        else:  # interaction
            var_names = effect_info["var_names"]
            effect_info["column_indices"] = [
                predictor_vars.index(var) for var in var_names
            ]

    return variables, effects


def _validate_and_parse_effects(
    input_data: Any,
    available_items: Any,
    item_type: str = "item",
    equation: Optional[str] = None,
) -> Tuple[List[Dict], Callable]:
    """
    Parse and validate names/assignments against available items.

    Simplified version that no longer handles factor-wide effects.
    Only supports explicit effect names (including bracket notation like 'treatment[2]').

    Args:
        input_data: String assignments, dict, or list of names
        available_items: Dict or list of available items to validate against
        item_type: Type description for error messages
        equation: Optional equation context for error messages

    Returns:
        tuple: (valid_items, find_by_name_function)
    """
    # Handle different input formats
    if isinstance(input_data, str):
        # Use the parser's assignment splitting to handle parentheses correctly
        from .parsers import _parser

        assignments = _parser._split_assignments(input_data)
        parsed_items = []
        parsing_errors = []

        for assignment in assignments:
            if "=" not in assignment:
                parsing_errors.append(
                    f"Invalid format '{assignment}'. Expected: 'name=value'"
                )
                continue

            name, value_str = assignment.split("=", 1)
            name, value_str = name.strip(), value_str.strip()

            try:
                value = float(value_str)
                parsed_items.append({"name": name, "value": value})
            except ValueError:
                parsing_errors.append(
                    f"Invalid value '{value_str}' for '{name}'. Must be a number."
                )

        names_to_check = [item["name"] for item in parsed_items]

    elif isinstance(input_data, dict):
        names_to_check = list(input_data.keys())
        parsed_items = [
            {"name": name, "value": value} for name, value in input_data.items()
        ]
        parsing_errors = []

    else:
        names_to_check = list(input_data)
        parsed_items = [{"name": name} for name in names_to_check]
        parsing_errors = []

    # Get available names
    if isinstance(available_items, dict):
        available_names = [
            item["name"] for item in available_items.values() if "name" in item
        ]
    else:
        available_names = list(available_items)

    # Validate names - simple exact matching only
    valid_names = [name for name in names_to_check if name in available_names]
    invalid_names = [name for name in names_to_check if name not in available_names]

    # Collect validation errors
    validation_errors = []
    validation_errors.extend(parsing_errors)

    if invalid_names:
        context = f" in equation '{equation}'" if equation else ""
        validation_errors.append(
            f"The following {item_type}(s) were not found: {', '.join(invalid_names)}. "
            f"Available {item_type}s{context}: {', '.join(available_names)}"
        )

    if validation_errors:
        error_msg = f"Validation failed:\n" + "\n".join(
            f"• {err}" for err in validation_errors
        )
        raise ValueError(error_msg)

    # Return valid items and find function
    valid_items = [item for item in parsed_items if item["name"] in valid_names]

    def find_by_name(name):
        for key, item in available_items.items():
            if isinstance(item, dict) and item.get("name") == name:
                return key, item
        return None, None

    return valid_items, find_by_name


def _parse_lr_variable_types(
    assignments: List[str], available_vars: List[str]
) -> Tuple[Dict, List[str]]:
    """Parse variable type assignments for LinearRegression."""
    input_string = ", ".join(assignments)
    return _parser._parse(input_string, "variable_type", available_vars)


def _parse_lr_correlations(
    assignments: List[str], available_vars: List[str]
) -> Tuple[Dict, List[str]]:
    """Parse correlation assignments with function syntax."""
    input_string = ", ".join(assignments)
    return _parser._parse(input_string, "correlation", available_vars)


def _parse_lr_assignments_with_parentheses(input_string: str) -> List[str]:
    """Parse comma-separated assignments that respects parentheses."""
    return _parser._split_assignments(input_string)
