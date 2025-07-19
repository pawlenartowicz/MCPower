"""
Result formatting and display utilities for Monte Carlo Power Analysis.

This module provides functions for printing and formatting analysis results
in a clear, professional manner.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

__all__ = []

MEDIUM_VULNERABILITY_TRESHOLD = 10  # drop in p.p.
HIGH_VULNERABILITY_TRESHOLD = 20  # drop in p.p.
INFLATED_ERROR_TRESHOLD = -2


class _TableFormatter:
    """Generic table formatting utilities."""

    @staticmethod
    def _create_table(
        headers: List[str],
        rows: List[List[Any]],
        col_widths: Optional[List[int]] = None,
    ) -> str:
        """Create formatted table with headers and rows."""

        if not col_widths:
            col_widths = [
                max(len(str(h)), max(len(str(row[i])) + 2 for row in rows))
                for i, h in enumerate(headers)
            ]

        lines = []

        # Header
        header_line = " ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        lines.append(header_line)
        lines.append("-" * len(header_line))

        # Rows
        for row in rows:
            row_line = " ".join(f"{str(val):<{w}}" for val, w in zip(row, col_widths))
            lines.append(row_line)

        return "\n".join(lines)

    @staticmethod
    def _format_value(value: Any, format_spec: str = "") -> str:
        """Format value with appropriate precision."""

        if isinstance(value, float):
            if format_spec:
                return f"{value:{format_spec}}"
            elif abs(value) < 0.0001:
                return f"{value:.6f}"
            else:
                return f"{value:.4f}"
        return str(value)


class _ResultFormatter:
    """Main formatter for different result types."""

    def __init__(self):
        self._table = _TableFormatter()

    def format(self, result_type: str, data: Dict, summary_type: str = "short") -> str:
        """Main formatting dispatcher."""

        formatters = {
            "power": self._format_power,
            "sample_size": self._format_sample_size,
            "scenario_power": self._format_scenario_power,
            "scenario_sample_size": self._format_scenario_sample_size,
            "regression": self._format_regression,
        }

        if result_type not in formatters:
            raise ValueError(f"Unknown result type: {result_type}")

        return formatters[result_type](data, summary_type)

    def _format_power(self, data: Dict, summary_type: str) -> str:
        """Format power analysis results."""

        if summary_type == "short":
            return self._format_short_power(data)
        else:
            return self._format_long_power(data)

    def _format_short_power(self, data: Dict) -> str:
        """Short power summary."""

        # Check if this is scenario data
        if "scenarios" in data and "model" not in data:
            return self._format_scenario_power(data, "short")

        model = data["model"]
        results = data["results"]

        lines = [f"\nPower Analysis Results (N={model['sample_size']}):"]

        # Create table data
        headers = ["Test", "Power", "Target", "Status"]
        rows = []
        target = model.get("target_power", 80.0)

        for test in model["target_tests"]:
            power = results["individual_powers"][test]
            status = "✓" if power >= target else "✗"
            rows.append([test, f"{power:.1f}", f"{target:.0f}", status])

        lines.append(self._table._create_table(headers, rows, [40, 8, 8, 8]))

        # Correction results if applicable
        if model.get("correction") and results.get("individual_powers_corrected"):
            lines.append(f"\nWith {model['correction']} correction:")
            rows_corrected = []

            for test in model["target_tests"]:
                power_corr = results["individual_powers_corrected"][test]
                status = "✓" if power_corr >= target else "✗"
                rows_corrected.append(
                    [test, f"{power_corr:.1f}", f"{target:.0f}", status]
                )

            lines.append(
                self._table._create_table(headers, rows_corrected, [40, 8, 8, 8])
            )

        # Summary
        achieved = sum(1 for row in rows if row[3] == "✓")
        lines.append(f"\nResult: {achieved}/{len(rows)} tests achieved target power")

        return "\n".join(lines)

    def _format_long_power(self, data: Dict) -> str:
        """Detailed power summary."""

        lines = []
        model = data["model"]
        results = data["results"]

        # Individual test powers
        lines.append(f"\n{'Individual Test Powers:':<30}")

        if model.get("correction"):
            headers = ["Test", "Power (%)", "Corrected (%)", "Target (%)", "Achieved"]
            rows = []
            for test in model["target_tests"]:
                power = results["individual_powers"][test]
                power_corr = results.get("individual_powers_corrected", {}).get(
                    test, power
                )
                target = model.get("target_power", 80.0)
                achieved = "✓" if power_corr >= target else "✗"
                rows.append(
                    [
                        test,
                        f"{power:.2f}",
                        f"{power_corr:.2f}",
                        f"{target:.1f}",
                        achieved,
                    ]
                )

            lines.append(self._table._create_table(headers, rows))
        else:
            headers = ["Test", "Power (%)", "Target (%)", "Achieved"]
            rows = []
            for test in model["target_tests"]:
                power = results["individual_powers"][test]
                target = model.get("target_power", 80.0)
                achieved = "✓" if power >= target else "✗"
                rows.append([test, f"{power:.2f}", f"{target:.1f}", achieved])

            lines.append(self._table._create_table(headers, rows))

        # Cumulative probabilities
        lines.append(f"\n{'Cumulative Probabilities:'}")
        self._add_cumulative_table(lines, results, model.get("correction"))

        lines.append("=" * 80)

        return "\n".join(lines)

    def _format_sample_size(self, data: Dict, summary_type: str) -> str:
        """Format sample size analysis results."""
        if summary_type == "short":
            return self._format_short_sample_size(data)
        else:
            return self._format_long_sample_size(data)

    def _format_short_sample_size(self, data: Dict) -> str:
        """Short sample size summary."""

        model = data["model"]
        results = data["results"]
        correction = model.get("correction")

        lines = ["\nSample Size Requirements:"]

        if correction and results.get("first_achieved_corrected"):
            # Show both uncorrected and corrected Required N
            headers = ["Test", "Uncorrected N", "Corrected N"]
            rows = []

            max_n = 0
            max_n_corrected = 0
            achieved_count = 0
            achieved_count_corrected = 0
            to_size = model["sample_size_range"]["to_size"]

            for test in model["target_tests"]:
                n_required = results["first_achieved"][test]
                n_required_corrected = results["first_achieved_corrected"][test]

                uncorr_str = str(n_required) if n_required > 0 else f">{to_size}"
                corr_str = (
                    str(n_required_corrected)
                    if n_required_corrected > 0
                    else f">{to_size}"
                )

                rows.append([test, uncorr_str, corr_str])

                if n_required > 0:
                    max_n = max(max_n, n_required)
                    achieved_count += 1
                if n_required_corrected > 0:
                    max_n_corrected = max(max_n_corrected, n_required_corrected)
                    achieved_count_corrected += 1

            lines.append(self._table._create_table(headers, rows, [40, 14, 14]))

        else:
            # Original format for uncorrected
            headers = ["Test", "Required N"]
            rows = []

            max_n = 0
            achieved_count = 0
            to_size = model["sample_size_range"]["to_size"]

            for test in model["target_tests"]:
                n_required = results["first_achieved"][test]
                if n_required > 0:
                    rows.append([test, str(n_required)])
                    max_n = max(max_n, n_required)
                    achieved_count += 1
                else:
                    rows.append([test, f">{to_size}"])

            lines.append(self._table._create_table(headers, rows, [40, 12]))

        return "\n".join(lines)

    def _format_long_sample_size(self, data: Dict) -> str:
        """Detailed sample size analysis with power probability table."""

        lines = []
        lines.append(self._format_short_sample_size(data))

        # Add power probability table
        model = data["model"]
        results = data["results"]
        correction = model.get("correction")

        sample_sizes = results["sample_sizes_tested"]
        target_tests = model["target_tests"]
        powers_by_test = results["powers_by_test"]
        powers_by_test_corrected = results.get("powers_by_test_corrected")

        if correction and powers_by_test_corrected:
            # Two separate tables for uncorrected and corrected

            # Uncorrected cumulative probability table
            lines.append(f"\n\nUncorrected Cumulative Significance Probability (%):")
            self._add_cumulative_sample_size_table(
                lines, sample_sizes, target_tests, powers_by_test
            )

            # Corrected cumulative probability table
            lines.append(
                f"\nCorrected Cumulative Significance Probability (%) - {correction}:"
            )
            self._add_cumulative_sample_size_table(
                lines, sample_sizes, target_tests, powers_by_test_corrected
            )
        else:
            # Uncorrected only table
            lines.append(f"\n\nCumulative Significance Probability (%):")
            self._add_cumulative_sample_size_table(
                lines, sample_sizes, target_tests, powers_by_test
            )

        # Add cumulative probability analysis
        cumulative_lines = self._format_cumulative_recommendations(
            data, is_scenario=False
        )
        lines.extend(cumulative_lines)

        return "\n".join(lines)

    def _format_scenario_power(self, data: Dict, summary_type: str) -> str:
        """Format scenario power analysis."""

        scenarios = data.get("scenarios", {})

        # Get target tests and correction info from first scenario
        target_tests = None
        correction = None
        for scenario_data in scenarios.values():
            if "model" in scenario_data:
                target_tests = scenario_data["model"]["target_tests"]
                correction = scenario_data["model"].get("correction")
                break

        if not target_tests:
            return "No scenario data available"

        if summary_type == "short":
            return self._format_scenario_power_short(
                scenarios, target_tests, correction
            )
        else:
            return self._format_scenario_power_long(
                data, scenarios, target_tests, correction
            )

    def _format_scenario_power_short(
        self, scenarios: Dict, target_tests: List[str], correction: Optional[str]
    ) -> str:
        """Short scenario power summary - comparison table only."""

        lines = [f"\n{'='*80}", "SCENARIO SUMMARY", f"{'='*80}"]

        # Uncorrected table
        headers = ["Test", "Optimistic", "Realistic", "Doomer"]
        rows = []

        for test in target_tests:
            row = [test]
            for scenario in ["optimistic", "realistic", "doomer"]:
                if scenario in scenarios and "results" in scenarios[scenario]:
                    power = scenarios[scenario]["results"]["individual_powers"][test]
                    row.append(f"{power:.1f}")
                else:
                    row.append("N/A")
            rows.append(row)

        lines.append("\nUncorrected Power:")
        lines.append(self._table._create_table(headers, rows, [40, 12, 12, 12]))

        # Corrected table if applicable
        if correction:
            rows_corr = []
            for test in target_tests:
                row = [test]
                for scenario in ["optimistic", "realistic", "doomer"]:
                    if scenario in scenarios and "results" in scenarios[scenario]:
                        power_corr = scenarios[scenario]["results"][
                            "individual_powers_corrected"
                        ][test]
                        row.append(f"{power_corr:.1f}")
                    else:
                        row.append("N/A")
                rows_corr.append(row)

            lines.append(f"\nCorrected Power ({correction}):")
            lines.append(
                self._table._create_table(headers, rows_corr, [40, 12, 12, 12])
            )

        lines.append(f"{'='*80}")

        return "\n".join(lines)

    def _format_scenario_power_long(
        self,
        data: Dict,
        scenarios: Dict,
        target_tests: List[str],
        correction: Optional[str],
    ) -> str:
        """Long scenario power summary - detailed results for each scenario."""

        # Define thresholds
        HIGH_VULNERABILITY_THRESHOLD = 30
        MEDIUM_VULNERABILITY_THRESHOLD = 15
        INFLATED_ERROR_THRESHOLD = -10

        lines = []

        # 1. Overall summary (same as short)
        lines.append(
            self._format_scenario_power_short(scenarios, target_tests, correction)
        )

        # 2. Individual scenario details
        lines.append(f"\n{'='*80}")
        lines.append("DETAILED SCENARIO RESULTS")
        lines.append(f"{'='*80}")

        for scenario_name in ["optimistic", "realistic", "doomer"]:
            if scenario_name in scenarios:
                lines.append(f"\n{'-'*80}")
                lines.append(f"{scenario_name.upper()} SCENARIO")
                lines.append(f"{'-'*80}")

                # Use regular power formatter for each scenario
                scenario_data = {
                    "model": scenarios[scenario_name]["model"],
                    "results": scenarios[scenario_name]["results"],
                }
                lines.append(self._format_long_power(scenario_data))

        # 3. Comparison analysis
        lines.append(f"\n{'='*80}")
        lines.append("ROBUSTNESS ANALYSIS")
        lines.append(f"{'='*80}")

        # Power reduction table
        headers = ["Test", "Opt→Real Drop", "Opt→Doom Drop", "Vulnerability"]
        rows = []
        vulnerable_tests = []
        inflated_tests = []

        for test in target_tests:
            opt_power = scenarios["optimistic"]["results"]["individual_powers"][test]
            real_power = (
                scenarios.get("realistic", {})
                .get("results", {})
                .get("individual_powers", {})
                .get(test, opt_power)
            )
            doom_power = (
                scenarios.get("doomer", {})
                .get("results", {})
                .get("individual_powers", {})
                .get(test, opt_power)
            )

            real_drop = opt_power - real_power
            doom_drop = opt_power - doom_power

            # Format drops with proper signs
            real_drop_str = (
                f"+{abs(real_drop):.1f}%" if real_drop < 0 else f"-{real_drop:.1f}%"
            )
            doom_drop_str = (
                f"+{abs(doom_drop):.1f}%" if doom_drop < 0 else f"-{doom_drop:.1f}%"
            )

            # Vulnerability assessment and categorization
            if doom_drop > HIGH_VULNERABILITY_THRESHOLD:
                vulnerability = "HIGH"
                vulnerable_tests.append(test)
            elif doom_drop > MEDIUM_VULNERABILITY_THRESHOLD:
                vulnerability = "MEDIUM"
            elif doom_drop < INFLATED_ERROR_THRESHOLD:
                vulnerability = "INFLATED FALSE POSITIVES"
                inflated_tests.append(test)
            else:
                vulnerability = "LOW"

            rows.append([test, real_drop_str, doom_drop_str, vulnerability])

        lines.append(self._table._create_table(headers, rows))

        # 4. Recommendations
        lines.append(f"\n{'='*80}")
        lines.append("RECOMMENDATIONS")
        lines.append(f"{'='*80}")

        if vulnerable_tests:
            lines.append(f"• High vulnerability tests: {', '.join(vulnerable_tests)}")
            lines.append(
                "• Consider increasing sample size to maintain power under adverse conditions"
            )

        if inflated_tests:
            lines.append(f"• Inflated false positive risk: {', '.join(inflated_tests)}")
            lines.append("• Be careful about interpretation")

        if not vulnerable_tests and not inflated_tests:
            lines.append("• Power analysis appears robust to assumption violations")
            lines.append("• Original sample size should be sufficient")

    def _format_scenario_sample_size(self, data: Dict, summary_type: str) -> str:
        """Format scenario sample size analysis."""

        scenarios = data.get("scenarios", {})

        # Get target tests and correction info
        target_tests = None
        correction = None
        for scenario_data in scenarios.values():
            if "model" in scenario_data:
                target_tests = scenario_data["model"]["target_tests"]
                correction = scenario_data["model"].get("correction")
                break

        if not target_tests:
            return "No scenario data available"

        if summary_type == "short":
            return self._format_scenario_sample_size_short(
                scenarios, target_tests, correction
            )
        else:
            return self._format_scenario_sample_size_long(
                data, scenarios, target_tests, correction
            )

    def _format_scenario_sample_size_short(
        self, scenarios: Dict, target_tests: List[str], correction: Optional[str]
    ) -> str:
        """Short scenario sample size summary."""

        lines = [f"\n{'='*80}", "SCENARIO SUMMARY", f"{'='*80}"]

        if correction:
            # Combined table with uncorrected and corrected
            lines.append("\nSample Size Requirements:")
            headers = [
                "Test",
                "Opt(U)",
                "Opt(C)",
                "Real(U)",
                "Real(C)",
                "Doom(U)",
                "Doom(C)",
            ]

            rows = []
            for test in target_tests:
                row = [test[:40]]  # Truncate to 40 chars

                for scenario in ["optimistic", "realistic", "doomer"]:
                    if scenario in scenarios and "results" in scenarios[scenario]:
                        n_uncorr = scenarios[scenario]["results"]["first_achieved"][
                            test
                        ]
                        n_corr = scenarios[scenario]["results"][
                            "first_achieved_corrected"
                        ][test]
                        max_tested = scenarios[scenario]["model"]["sample_size_range"][
                            "to_size"
                        ]

                        uncorr_str = str(n_uncorr) if n_uncorr > 0 else f">{max_tested}"
                        corr_str = str(n_corr) if n_corr > 0 else f">{max_tested}"
                        row.extend([uncorr_str, corr_str])
                    else:
                        row.extend(["N/A", "N/A"])
                rows.append(row)

            lines.append(
                self._table._create_table(headers, rows, [40, 8, 8, 8, 8, 8, 8])
            )
            lines.append("Note: (U) = Uncorrected, (C) = Corrected")
        else:
            # Uncorrected only
            headers = ["Test", "Optimistic", "Realistic", "Doomer"]

            rows = []
            for test in target_tests:
                row = [test[:40]]  # Truncate to 40 chars
                for scenario in ["optimistic", "realistic", "doomer"]:
                    if scenario in scenarios and "results" in scenarios[scenario]:
                        n_required = scenarios[scenario]["results"]["first_achieved"][
                            test
                        ]
                        if n_required > 0:
                            row.append(str(n_required))
                        else:
                            max_tested = scenarios[scenario]["model"][
                                "sample_size_range"
                            ]["to_size"]
                            row.append(f">{max_tested}")
                    else:
                        row.append("N/A")
                rows.append(row)

            lines.append("\nUncorrected Sample Sizes:")
            lines.append(self._table._create_table(headers, rows, [40, 12, 12, 12]))

        lines.append(f"{'='*80}")

        return "\n".join(lines)

    def _format_scenario_sample_size_long(
        self,
        data: Dict,
        scenarios: Dict,
        target_tests: List[str],
        correction: Optional[str],
    ) -> str:
        """Long scenario sample size summary."""

        lines = []

        # 1. Overall summary
        lines.append(
            self._format_scenario_sample_size_short(scenarios, target_tests, correction)
        )

        # 2. Recommendations
        lines.append(f"\n{'='*80}")
        lines.append("RECOMMENDATIONS")
        lines.append(f"{'='*80}")

        # Calculate max required N across scenarios
        max_n_realistic = max(
            (
                scenarios.get("realistic", {})
                .get("results", {})
                .get("first_achieved", {})
                .get(test, 0)
                for test in target_tests
            ),
            default=0,
        )
        max_n_doomer = max(
            (
                scenarios.get("doomer", {})
                .get("results", {})
                .get("first_achieved", {})
                .get(test, 0)
                for test in target_tests
            ),
            default=0,
        )

        max_tested = (
            scenarios.get("realistic", {})
            .get("model", {})
            .get("sample_size_range", {})
            .get("to_size", 200)
        )

        if max_n_realistic > 0 and max_n_realistic <= max_tested:
            lines.append(
                f"• For robust power under realistic conditions: N = {max_n_realistic}"
            )
        elif max_n_realistic <= 0:
            lines.append(
                f"• For robust power under realistic conditions: N > {max_tested}"
            )

        if max_n_doomer > 0 and max_n_doomer <= max_tested:
            lines.append(f"• For power under worst-case conditions: N = {max_n_doomer}")
        elif max_n_doomer <= 0:
            lines.append(f"• For power under worst-case conditions: N > {max_tested}")

        # Check if any tests couldn't achieve power
        unachievable = [
            test
            for test in target_tests
            if scenarios.get("doomer", {})
            .get("results", {})
            .get("first_achieved", {})
            .get(test, -1)
            <= 0
        ]
        if unachievable:
            lines.append(
                f"• Warning: These tests may not achieve target power under adverse conditions: "
                f"{', '.join(unachievable)}"
            )

        # Add cumulative probability analysis
        cumulative_lines = self._format_cumulative_recommendations(
            data, is_scenario=True
        )
        lines.extend(cumulative_lines)

        return "\n".join(lines)

    def _format_regression(self, data: Dict, summary_type: str) -> str:
        """Format regression results."""

        results = data["results"]
        effect_names = data["effect_names"]
        dep_var = data.get("dep_var", "y")
        correction = data.get("correction")

        stats = results["statistics"]

        lines = [
            f"\n{'='*80}",
            "REGRESSION RESULTS",
            f"{'='*80}",
            f"Dependent Variable: {dep_var}",
            f"R-squared: {stats['r_squared']:.4f}",
            f"F-statistic: {stats['f_statistic']:.4f}, p-value: {stats['f_p_value']:.4f}",
            f"Overall significant: {results['overall_significant']}",
        ]

        if correction:
            lines.append(f"Multiple Comparison Correction: {correction}")

        lines.append("-" * 80)

        # Coefficient table
        if correction:
            headers = [
                "Variable",
                "Coeff",
                "SE",
                "t-stat",
                "p-value",
                "Signif",
                "Corrected",
            ]
            rows = []

            for name in effect_names:
                coef = stats["coefficients"][name]
                std_err = stats["standard_errors"][name]
                t_stat = stats["t_statistics"][name]
                p_val = stats["p_values"][name]
                signif = _get_significance_code(p_val)
                corrected = (
                    "***" if results.get(f"{name}_significant_corrected", False) else ""
                )

                if np.isinf(std_err):
                    rows.append([name, f"{coef:.3f}", "inf", "0.00", "1.00", "", ""])
                else:
                    rows.append(
                        [
                            name,
                            f"{coef:.3f}",
                            f"{std_err:.3f}",
                            f"{t_stat:.3f}",
                            f"{p_val:.4f}",
                            signif,
                            corrected,
                        ]
                    )
        else:
            headers = ["Variable", "Coeff", "Std Err", "t-stat", "p-value", "Signif"]
            rows = []

            for name in effect_names:
                coef = stats["coefficients"][name]
                std_err = stats["standard_errors"][name]
                t_stat = stats["t_statistics"][name]
                p_val = stats["p_values"][name]
                signif = _get_significance_code(p_val)

                if np.isinf(std_err):
                    rows.append([name, f"{coef:.4f}", "inf", "0.0000", "1.0000", ""])
                else:
                    rows.append(
                        [
                            name,
                            f"{coef:.4f}",
                            f"{std_err:.4f}",
                            f"{t_stat:.4f}",
                            f"{p_val:.4f}",
                            signif,
                        ]
                    )

        lines.append(self._table._create_table(headers, rows))
        lines.append("-" * 80)
        lines.append("Significance codes: *** p<0.001, ** p<0.01, * p<0.05")
        lines.append("=" * 80)

        return "\n".join(lines)

    def _add_cumulative_sample_size_table(
        self,
        lines: List[str],
        sample_sizes: List[int],
        target_tests: List[str],
        powers_by_test: Dict[str, List[float]],
    ):
        """Add cumulative probability table with sample sizes as rows."""

        n_tests = len(target_tests)

        # Create headers: Sample Size | ≥1 | ≥2 | ≥3 | ... | All
        headers = ["Sample Size"]
        for k in range(1, n_tests):
            headers.append(f"≥{k}")
        headers.append("All")

        # Calculate column widths
        col_widths = [12] + [8] * len(headers[1:])

        rows = []
        for i, size in enumerate(sample_sizes):
            row = [f"N={size}"]

            # Get individual probabilities for this sample size
            individual_probs = [
                powers_by_test[test][i] / 100.0 for test in target_tests
            ]

            # Calculate cumulative probabilities using binomial approach
            # P(at least k significant) = sum of combinations
            for k in range(1, n_tests + 1):
                if k == n_tests:  # "All" case
                    prob_all = 1.0
                    for prob in individual_probs:
                        prob_all *= prob
                    row.append(f"{prob_all * 100:.1f}%")
                else:  # ≥k cases
                    # Approximate using independence assumption
                    prob_at_least_k = 0.0
                    from itertools import combinations

                    # Sum over all ways to choose at least k tests
                    for num_sig in range(k, n_tests + 1):
                        for combo in combinations(range(n_tests), num_sig):
                            prob_combo = 1.0
                            for j in range(n_tests):
                                if j in combo:
                                    prob_combo *= individual_probs[j]
                                else:
                                    prob_combo *= 1 - individual_probs[j]
                            prob_at_least_k += prob_combo

                    row.append(f"{prob_at_least_k * 100:.1f}%")

            rows.append(row)

        lines.append(self._table._create_table(headers, rows, col_widths))

    def _add_cumulative_table(self, lines: List[str], results: Dict, correction: bool):
        """Add cumulative probability table to output."""

        cumulative = results["cumulative_probabilities"]

        if correction and results.get("cumulative_probabilities_corrected"):
            headers = ["Outcome", "Uncorrected (%)", "Corrected (%)"]
            rows = []
            cumulative_corr = results["cumulative_probabilities_corrected"]

            for key in cumulative.keys():
                outcome = key.replace("_", " ").title()
                prob = cumulative[key]
                prob_corr = cumulative_corr[key]
                rows.append([outcome, f"{prob:.2f}", f"{prob_corr:.2f}"])

            lines.append(self._table._create_table(headers, rows))
        else:
            headers = ["Outcome", "Probability (%)"]
            rows = []

            for key, prob in cumulative.items():
                outcome = key.replace("_", " ").title()
                rows.append([outcome, f"{prob:.2f}"])

            lines.append(self._table._create_table(headers, rows))

    def _format_cumulative_recommendations(
        self, results: Dict, is_scenario: bool = False
    ) -> List[str]:
        """Format cumulative probability recommendations."""

        lines = []

        if is_scenario:
            # Handle scenario results
            scenarios = results.get("scenarios", {})
            if not scenarios:
                return lines

            # Use optimistic scenario as baseline
            opt_results = scenarios.get("optimistic", {})
            if "results" not in opt_results:
                return lines

            sample_sizes = opt_results["results"]["sample_sizes_tested"]
            target_tests = opt_results["model"]["target_tests"]
            target_power = opt_results["model"].get("target_power", 80.0)

            # Find sample size for "all tests significant with target% probability"
            for scenario_name in ["optimistic", "realistic", "doomer"]:
                if scenario_name in scenarios:
                    scenario_data = scenarios[scenario_name]
                    powers_by_test = scenario_data["results"]["powers_by_test"]

                    # Calculate probability of all tests significant at each sample size
                    all_sig_probs = []
                    for i, n in enumerate(sample_sizes):
                        prob_all = 1.0
                        for test in target_tests:
                            prob_all *= powers_by_test[test][i] / 100.0
                        all_sig_probs.append(prob_all * 100)

                    # Find minimum N for target% probability of all significant
                    min_n_target = None
                    for i, prob in enumerate(all_sig_probs):
                        if prob >= target_power:
                            min_n_target = sample_sizes[i]
                            break

                    if min_n_target:
                        lines.append(
                            f"• {scenario_name.title()}: N={min_n_target} for {target_power:.0f}% chance all tests significant"
                        )
                    else:
                        max_tested = sample_sizes[-1]
                        lines.append(
                            f"• {scenario_name.title()}: >{max_tested} needed for {target_power:.0f}% chance all tests significant"
                        )
        else:
            # Handle regular (non-scenario) results
            if "results" not in results:
                return lines

            sample_sizes = results["results"]["sample_sizes_tested"]
            powers_by_test = results["results"]["powers_by_test"]
            target_tests = results["model"]["target_tests"]
            target_power = results["model"].get("target_power", 80.0)

            # Calculate probability of all tests significant
            all_sig_probs = []
            for i, n in enumerate(sample_sizes):
                prob_all = 1.0
                for test in target_tests:
                    prob_all *= powers_by_test[test][i] / 100.0
                all_sig_probs.append(prob_all * 100)

            # Find minimum N for target% probability of all significant
            min_n_target = None
            for i, prob in enumerate(all_sig_probs):
                if prob >= target_power:
                    min_n_target = sample_sizes[i]
                    break
        return lines


def _get_significance_code(p_value: float) -> str:
    """Get significance stars based on p-value."""

    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


def _format_results(result_type: str, data: Dict, summary_type: str = "short") -> str:
    """
    Format analysis results for display.

    Args:
        result_type: Type of results ('power', 'sample_size', 'scenario_power',
                    'scenario_sample_size', 'regression')
        data: Result data dictionary
        summary_type: 'short' or 'long' format

    Returns:
        Formatted string output
    """

    return _formatter.format(result_type, data, summary_type)


_formatter = _ResultFormatter()
