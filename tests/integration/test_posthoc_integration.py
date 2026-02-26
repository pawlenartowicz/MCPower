"""
Integration tests for ANOVA post-hoc pairwise comparisons in MCPower.
"""

import numpy as np
import pytest


class TestPostHocParsing:
    """Test parsing of post-hoc target_test syntax."""

    def test_parse_vs_syntax(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("group[1] vs group[2]")
        assert "group[1] vs group[2]" in tests
        assert len(model._posthoc_specs) == 1

    def test_parse_multiple_vs(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("group[1] vs group[2], group[2] vs group[3]")
        assert "group[1] vs group[2]" in tests
        assert "group[2] vs group[3]" in tests
        assert len(model._posthoc_specs) == 2

    def test_parse_mixed_regular_and_posthoc(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("overall, group[1] vs group[2]")
        assert "overall" in tests
        assert "group[1] vs group[2]" in tests

    def test_all_does_not_include_posthoc(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("all")
        # "all" should NOT include any post-hoc comparisons
        for t in tests:
            assert "vs" not in t
        assert len(model._posthoc_specs) == 0

    def test_invalid_factor_name(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        with pytest.raises(ValueError, match="Factor.*not found"):
            model._parse_target_tests("notafactor[1] vs notafactor[2]")

    def test_invalid_level(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        with pytest.raises(ValueError, match="out of range"):
            model._parse_target_tests("group[0] vs group[5]")

    def test_same_level_comparison_rejected(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        with pytest.raises(ValueError, match="Cannot compare a level to itself"):
            model._parse_target_tests("group[2] vs group[2]")

    def test_cross_factor_comparison_rejected(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = a + b")
        model.set_variable_type("a=(factor,3), b=(factor,2)")
        model.set_effects("a[2]=0.3, a[3]=0.2, b[2]=0.1")
        model._apply()

        with pytest.raises(ValueError, match="same factor"):
            model._parse_target_tests("a[1] vs b[1]")


class TestPostHocFindPower:
    """Integration tests for find_power with post-hoc comparisons."""

    def test_posthoc_runs_without_error(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        result = model.find_power(
            sample_size=100,
            target_test="group[1] vs group[2]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        assert "group[1] vs group[2]" in result["model"]["target_tests"]
        power = result["results"]["individual_powers"]["group[1] vs group[2]"]
        assert 0 <= power <= 100

    def test_posthoc_with_regular_tests(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        result = model.find_power(
            sample_size=100,
            target_test="overall, x1, group[1] vs group[2]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        tests = result["model"]["target_tests"]
        assert "overall" in tests
        assert "x1" in tests
        assert "group[1] vs group[2]" in tests

    def test_posthoc_tukey_runs(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        result = model.find_power(
            sample_size=100,
            target_test="group[1] vs group[2]",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        power = result["results"]["individual_powers"]["group[1] vs group[2]"]
        assert 0 <= power <= 100

    def test_posthoc_with_correction(self, suppress_output):
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        result = model.find_power(
            sample_size=100,
            target_test="overall, group[1] vs group[2]",
            correction="bonferroni",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        # Corrected power should be <= uncorrected
        uncorr = result["results"]["individual_powers"]["group[1] vs group[2]"]
        corr = result["results"]["individual_powers_corrected"]["group[1] vs group[2]"]
        assert corr <= uncorr + 1  # +1 for Monte Carlo noise

    def test_ref_vs_nonref_matches_ttest(self, suppress_output):
        """Post-hoc group[1] vs group[2] with t-test method should match the
        regular t-test for group[2] (the dummy for level 2)."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(800)
        model.set_seed(42)

        # Regular t-test on group[2] (which IS the ref-vs-level2 comparison)
        result_regular = model.find_power(
            sample_size=100,
            target_test="group[2]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        # Post-hoc: group[1] vs group[2] (same comparison, ref vs level 2)
        model.set_seed(42)
        result_posthoc = model.find_power(
            sample_size=100,
            target_test="group[1] vs group[2]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        power_regular = result_regular["results"]["individual_powers"]["group[2]"]
        power_posthoc = result_posthoc["results"]["individual_powers"]["group[1] vs group[2]"]

        # Should be very close (both test the same contrast)
        assert abs(power_regular - power_posthoc) < 5.0, (
            f"Power mismatch: regular={power_regular:.1f}, posthoc={power_posthoc:.1f}"
        )

    def test_nonref_vs_nonref_comparison(self, suppress_output):
        """Non-ref vs non-ref (group[2] vs group[3]) should produce valid power."""
        from mcpower import MCPower

        model = MCPower("y = group")
        model.set_variable_type("group=(factor,3)")
        # Large effect difference between levels 2 and 3
        model.set_effects("group[2]=0.0, group[3]=1.0")
        model.set_simulations(400)

        result = model.find_power(
            sample_size=150,
            target_test="group[2] vs group[3]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        power = result["results"]["individual_powers"]["group[2] vs group[3]"]
        # With a 1.0 effect difference and n=150, should have decent power
        assert power > 30

    def test_tukey_more_conservative_than_ttest(self, suppress_output):
        """Tukey should generally produce lower power than uncorrected t-test."""
        from mcpower import MCPower

        model = MCPower("y = group")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.3, group[3]=0.5")
        model.set_simulations(400)
        model.set_seed(42)

        # t-test (uncorrected)
        result_ttest = model.find_power(
            sample_size=100,
            target_test="group[1] vs group[2]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        # Tukey
        model.set_seed(42)
        result_tukey = model.find_power(
            sample_size=100,
            target_test="group[1] vs group[2]",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        power_ttest = result_ttest["results"]["individual_powers"]["group[1] vs group[2]"]
        power_tukey = result_tukey["results"]["individual_powers"]["group[1] vs group[2]"]

        # Tukey should be more conservative (lower or equal power)
        assert power_tukey <= power_ttest + 3  # +3 for MC noise

    def test_posthoc_in_results_table(self, suppress_output, capsys):
        """Post-hoc labels should appear naturally in the printed output."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(100)

        model.find_power(
            sample_size=100,
            target_test="overall, group[1] vs group[2]",
            print_results=True,
            progress_callback=False,
        )

        captured = capsys.readouterr()
        assert "group[1] vs group[2]" in captured.out

    def test_tukey_non_contrast_nan(self, suppress_output):
        """Non-contrast tests should have NaN corrected power under Tukey."""
        import math

        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        result = model.find_power(
            sample_size=100,
            target_test="overall, x1, group[1] vs group[2]",
            correction="tukey",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        corrected = result["results"]["individual_powers_corrected"]
        # Non-contrast tests should be NaN
        assert math.isnan(corrected["overall"])
        assert math.isnan(corrected["x1"])
        # Contrast test should have a valid number
        assert not math.isnan(corrected["group[1] vs group[2]"])
        assert 0 <= corrected["group[1] vs group[2]"] <= 100

    def test_tukey_requires_posthoc(self, suppress_output):
        """Tukey correction without any post-hoc comparison should raise ValueError."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        with pytest.raises(ValueError, match="Tukey correction requires"):
            model.find_power(
                sample_size=100,
                target_test="overall",
                correction="tukey",
                print_results=False,
                progress_callback=False,
            )

    def test_equal_effects_contrast_power_at_alpha(self, suppress_output):
        """When two factor levels have the same effect size, comparing them
        should yield power ≈ alpha (the null is true for that contrast)."""
        from mcpower import MCPower

        model = MCPower("y = group")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.5")
        model.set_simulations(2000)
        model.set_seed(123)
        model.set_alpha(0.05)

        result = model.find_power(
            sample_size=200,
            target_test="group[2] vs group[3]",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        power = result["results"]["individual_powers"]["group[2] vs group[3]"]
        # Under H0, power = alpha. With 2000 sims, expect ~5% ± noise.
        # Allow [1%, 2*alpha*100% = 10%].
        assert 1 <= power <= 10, (
            f"Expected power near alpha=5%, got {power:.1f}%"
        )


class TestKeywordExpansion:
    """Tests for keyword expansion, exclusions, and uniqueness validation."""

    def test_all_posthoc_keyword(self, suppress_output):
        """'all-posthoc' generates all C(n,2) pairs for each factor, no regular effects."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("all-posthoc")
        # 3-level factor → C(3,2) = 3 pairs
        assert len(tests) == 3
        assert "group[1] vs group[2]" in tests
        assert "group[1] vs group[3]" in tests
        assert "group[2] vs group[3]" in tests
        # No regular effects
        assert "overall" not in tests
        assert "x1" not in tests

    def test_all_plus_all_posthoc(self, suppress_output):
        """'all, all-posthoc' produces the union of both expansions."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("all, all-posthoc")
        # "all" → overall + group[2] + group[3] + x1 = 4
        # "all-posthoc" → 3 pairs
        assert "overall" in tests
        assert "x1" in tests
        assert "group[1] vs group[2]" in tests
        assert "group[1] vs group[3]" in tests
        assert "group[2] vs group[3]" in tests
        assert len(tests) == 7  # 4 regular + 3 posthoc

    def test_all_posthoc_multiple_factors(self, suppress_output):
        """Two factors generate correct total pairs."""
        from mcpower import MCPower

        model = MCPower("y = a + b")
        model.set_variable_type("a=(factor,3), b=(factor,2)")
        model.set_effects("a[2]=0.3, a[3]=0.2, b[2]=0.1")
        model._apply()

        tests = model._parse_target_tests("all-posthoc")
        # a: C(3,2)=3, b: C(2,2)=1 → 4 total
        assert len(tests) == 4
        assert "a[1] vs a[2]" in tests
        assert "b[1] vs b[2]" in tests

    def test_all_posthoc_no_factors_with_all(self, suppress_output):
        """'all, all-posthoc' on a factorless model → same as 'all' (no error)."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        tests = model._parse_target_tests("all, all-posthoc")
        assert "overall" in tests
        assert "x1" in tests
        assert "x2" in tests

    def test_all_posthoc_alone_no_factors_raises(self, suppress_output):
        """'all-posthoc' alone with no factors → ValueError."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        with pytest.raises(ValueError, match="no factor variables"):
            model._parse_target_tests("all-posthoc")

    def test_exclusion_removes_test(self, suppress_output):
        """'all, -overall' removes overall from the result."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        tests = model._parse_target_tests("all, -overall")
        assert "overall" not in tests
        assert "x1" in tests
        assert "x2" in tests

    def test_exclusion_posthoc(self, suppress_output):
        """Excluding a specific posthoc pair removes it."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.4, group[3]=0.3, x1=0.2")
        model._apply()

        tests = model._parse_target_tests("all-posthoc, -group[1] vs group[2]")
        assert "group[1] vs group[2]" not in tests
        assert "group[1] vs group[3]" in tests
        assert "group[2] vs group[3]" in tests
        assert len(tests) == 2

    def test_exclusion_invalid_raises(self, suppress_output):
        """Excluding a nonexistent test raises ValueError."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        with pytest.raises(ValueError, match="does not match"):
            model._parse_target_tests("all, -nonexistent")

    def test_exclusion_all_raises(self, suppress_output):
        """Excluding everything raises ValueError."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        with pytest.raises(ValueError, match="nothing left"):
            model._parse_target_tests("all, -overall, -x1, -x2")

    def test_duplicate_raises(self, suppress_output):
        """'all, x1' raises because x1 is already in 'all' expansion."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model._apply()

        with pytest.raises(ValueError, match="Duplicate"):
            model._parse_target_tests("all, x1")

    def test_default_is_all(self, suppress_output):
        """find_power() without target_test uses 'all' (overall + all fixed effects)."""
        from mcpower import MCPower

        model = MCPower("y = x1 + x2")
        model.set_effects("x1=0.5, x2=0.3")
        model.set_simulations(100)

        result = model.find_power(
            sample_size=100,
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        tests = result["model"]["target_tests"]
        assert "overall" in tests
        assert "x1" in tests
        assert "x2" in tests

    def test_all_posthoc_integration_runs(self, suppress_output):
        """Full integration: find_power with 'all, all-posthoc' runs and returns valid powers."""
        from mcpower import MCPower

        model = MCPower("y = group + x1")
        model.set_variable_type("group=(factor,3)")
        model.set_effects("group[2]=0.5, group[3]=0.3, x1=0.2")
        model.set_simulations(200)

        result = model.find_power(
            sample_size=100,
            target_test="all, all-posthoc",
            print_results=False,
            return_results=True,
            progress_callback=False,
        )

        assert result is not None
        powers = result["results"]["individual_powers"]
        assert "overall" in powers
        assert "x1" in powers
        assert "group[1] vs group[2]" in powers
        assert "group[1] vs group[3]" in powers
        assert "group[2] vs group[3]" in powers
        for power in powers.values():
            assert 0 <= power <= 100
