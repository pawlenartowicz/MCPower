"""
Multiple Testing Correction Example
===================================

This example demonstrates how to control Type I error when testing multiple
hypotheses simultaneously. Essential for studies with many variables.
"""

import mcpower

# Example: Medical study testing multiple biomarkers
# Research question: Which biomarkers predict treatment response?

print("=" * 60)
print("MULTIPLE TESTING CORRECTION EXAMPLE")
print("=" * 60)

# 1. Define model with multiple predictors to test
model = mcpower.LinearRegression(
    "treatment_response = biomarker1 + biomarker2 + biomarker3 + biomarker4 + age"
)

# 2. Set effect sizes - some biomarkers have no effect (null hypotheses)
model.set_effects(
    "biomarker1=0.4, biomarker2=0.0, biomarker3=0.3, biomarker4=0.0, age=0.2"
)

print("Multiple testing scenario:")
print(f"Formula: {model.equation}")
print(
    "True effects: biomarker1=0.4, biomarker2=0.0, biomarker3=0.3, biomarker4=0.0, age=0.2"
)
print("Testing 5 hypotheses simultaneously")

# 3. Power analysis without correction (liberal)
print("\n" + "=" * 60)
print("UNCORRECTED ANALYSIS")
print("=" * 60)

print("\n1. NO CORRECTION (Higher false positive risk):")
uncorrected = model.find_power(
    sample_size=200,
    target_test="all",
    correction=None,
    scenarios=False,
    summary="short",
)

# 4. Bonferroni correction (conservative)
print("\n2. BONFERRONI CORRECTION (Conservative):")
bonferroni = model.find_power(
    sample_size=200,
    target_test="all",
    correction="Bonferroni",
    scenarios=False,
    summary="short",
)

# 5. Benjamini-Hochberg correction (balanced)
print("\n3. BENJAMINI-HOCHBERG CORRECTION (Balanced):")
bh_correction = model.find_power(
    sample_size=200,
    target_test="all",
    correction="Benjamini-Hochberg",
    scenarios=False,
    summary="short",
)

# 6. Holm correction (step-down)
print("\n4. HOLM CORRECTION (Step-down):")
holm = model.find_power(
    sample_size=200,
    target_test="all",
    correction="Holm",
    scenarios=False,
    summary="short",
)

# 7. Sample size requirements with corrections
print("\n" + "=" * 60)
print("SAMPLE SIZE WITH CORRECTIONS")
print("=" * 60)

print("\n1. BONFERRONI SAMPLE SIZE:")
bonf_n = model.find_sample_size(
    target_test="all",
    from_size=50,
    to_size=250,
    by=10,
    correction="Bonferroni",
    scenarios=True,
    summary="short",
)

print("\n2. BENJAMINI-HOCHBERG SAMPLE SIZE:")
bh_n = model.find_sample_size(
    target_test="all",
    from_size=75,
    to_size=400,
    by=25,
    correction="BH",
    scenarios=True,
    summary="short",
)

# 8. Comprehensive comparison
print("\n" + "=" * 60)
print("COMPREHENSIVE COMPARISON")
print("=" * 60)

print("\n1. DETAILED ANALYSIS WITH CORRECTIONS:")
detailed = model.find_power(
    sample_size=300,
    target_test="all",
    correction="Benjamini-Hochberg",
    scenarios=True,
    summary="long",  # Full output with plots
)

# 9. Focused testing example
print("\n" + "=" * 60)
print("FOCUSED TESTING STRATEGY")
print("=" * 60)

# Test only primary hypotheses
print("\n1. PRIMARY HYPOTHESES ONLY:")
primary = model.find_power(
    sample_size=200,
    target_test="biomarker1, biomarker3",  # Only test promising biomarkers
    correction="Bonferroni",
    scenarios=True,
    summary="short",
)

# 10. Exploratory vs confirmatory
print("\n2. EXPLORATORY PHASE (Liberal):")
exploratory = model.find_power(
    sample_size=150,
    target_test="all",
    correction=None,  # No correction for exploration
    scenarios=False,
    summary="short",
)

print("\n3. CONFIRMATORY PHASE (Conservative):")
confirmatory = model.find_power(
    sample_size=400,
    target_test="biomarker1, biomarker3",  # Only confirmed hypotheses
    correction="Bonferroni",
    scenarios=True,
    summary="short",
)

print("\n" + "=" * 60)
print("MULTIPLE TESTING GUIDELINES")
print("=" * 60)
print(
    """
Correction method selection:

1. BONFERRONI:
   - Most conservative (lowest false positives)
   - Use when: Few tests, need strict control
   - Drawback: May miss true effects (low power)

2. BENJAMINI-HOCHBERG (FDR):
   - Balanced approach (controls false discovery rate)
   - Use when: Many tests, exploratory research
   - Good compromise between power and control

3. HOLM:
   - Step-down method, less conservative than Bonferroni
   - Use when: Need strong control but more power
   - Alternative to Bonferroni

4. NO CORRECTION:
   - Highest power, but increased false positives
   - Use when: Single planned comparison, exploratory

Strategy recommendations:
- Plan primary hypotheses in advance
- Use scenarios=True for realistic estimates
- Consider two-stage design: exploratory → confirmatory
- FDR correction for most situations
- Bonferroni for critical decisions
"""
)
