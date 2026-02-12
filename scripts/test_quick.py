#!/usr/bin/env python3
"""
Quick functionality test script for MCPower.

Usage:
    python scripts/test_quick.py
    python scripts/test_quick.py -v  # verbose mode
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv


def log(msg):
    if VERBOSE:
        print(f"  {msg}")


def test_basic():
    """Test basic power analysis."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2")
    model.set_effects("x1=0.3, x2=0.2")
    result = model.find_power(100, print_results=False, return_results=True)

    power = result["results"]["individual_powers"]["overall"]
    assert 70 < power < 100, f"Unexpected power: {power}"
    log(f"Power: {power:.1f}%")
    return True


def test_properties():
    """Test model properties."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2")

    assert model.generation_formula == "linear_regression"
    assert model.test_formula_type == "linear_regression"
    assert model.model_type == "Linear Regression"
    assert model.equation == "y = x1 + x2"

    log(f"generation_formula: {model.generation_formula}")
    log(f"test_formula_type: {model.test_formula_type}")
    return True


def test_interaction():
    """Test model with interaction term."""
    from mcpower import MCPower

    model = MCPower("y = a + b + a:b")
    model.set_effects("a=0.4, b=0.3, a:b=0.2")
    result = model.find_power(120, print_results=False, return_results=True)

    power = result["results"]["individual_powers"]["overall"]
    assert power > 80, f"Expected high power, got: {power}"
    log(f"Power with interaction: {power:.1f}%")
    return True


def test_correlations():
    """Test correlated predictors."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2")
    model.set_correlations("(x1,x2)=0.5")
    model.set_effects("x1=0.3, x2=0.2")
    result = model.find_power(100, print_results=False, return_results=True)

    power = result["results"]["individual_powers"]["overall"]
    assert 50 < power < 100, f"Unexpected power: {power}"
    log(f"Power with correlation: {power:.1f}%")
    return True


def test_factor():
    """Test factor variable."""
    from mcpower import MCPower

    model = MCPower("y = group + covariate")
    model.set_variable_type("group=(factor,3)")
    model.set_effects("group[2]=0.4, group[3]=0.3, covariate=0.2")
    result = model.find_power(150, print_results=False, return_results=True)

    power = result["results"]["individual_powers"]["overall"]
    assert 50 < power < 100, f"Unexpected power: {power}"
    log(f"Power with factor: {power:.1f}%")
    return True


def test_sample_size():
    """Test sample size analysis."""
    from mcpower import MCPower

    model = MCPower("y = x1")
    model.set_effects("x1=0.3")
    result = model.find_sample_size(
        from_size=50, to_size=150, by=25,
        print_results=False, return_results=True
    )

    assert "results" in result
    assert "first_achieved" in result["results"]
    log(f"Sample sizes tested: {result['results']['sample_sizes_tested']}")
    return True


def test_binary_variable():
    """Test binary predictor."""
    from mcpower import MCPower

    model = MCPower("y = treatment + age")
    model.set_variable_type("treatment=(binary,0.3)")
    model.set_effects("treatment=0.5, age=0.2")
    result = model.find_power(100, print_results=False, return_results=True)

    power = result["results"]["individual_powers"]["overall"]
    assert 50 < power < 100, f"Unexpected power: {power}"
    log(f"Power with binary: {power:.1f}%")
    return True


def test_multiple_targets():
    """Test targeting specific effects."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2 + x3")
    model.set_effects("x1=0.5, x2=0.3, x3=0.1")
    result = model.find_power(
        100, target_test="x1,x2",
        print_results=False, return_results=True
    )

    powers = result["results"]["individual_powers"]
    assert "x1" in powers
    assert "x2" in powers
    log(f"x1 power: {powers['x1']:.1f}%, x2 power: {powers['x2']:.1f}%")
    return True


def test_correction():
    """Test multiple comparison correction."""
    from mcpower import MCPower

    model = MCPower("y = x1 + x2 + x3")
    model.set_effects("x1=0.3, x2=0.3, x3=0.3")
    result = model.find_power(
        150, correction="bonferroni",
        print_results=False, return_results=True
    )

    corrected = result["results"]["individual_powers_corrected"]
    assert "overall" in corrected
    log(f"Corrected power: {corrected['overall']:.1f}%")
    return True


def run_all():
    """Run all tests."""
    tests = [
        ("Basic power", test_basic),
        ("Properties", test_properties),
        ("Interaction", test_interaction),
        ("Correlations", test_correlations),
        ("Factor variable", test_factor),
        ("Sample size", test_sample_size),
        ("Binary variable", test_binary_variable),
        ("Multiple targets", test_multiple_targets),
        ("Correction", test_correction),
    ]

    passed = 0
    failed = 0

    print("=" * 50)
    print("MCPower Quick Test")
    print("=" * 50)

    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)

    return failed == 0


if __name__ == "__main__":
    # Suppress MCPower output
    import io
    import contextlib

    if not VERBOSE:
        with contextlib.redirect_stdout(io.StringIO()):
            success = run_all()
    else:
        success = run_all()

    sys.exit(0 if success else 1)
