"""
Own Data Upload Example
=======================

This example shows how to use your own pilot data for realistic power analysis.
Using real data distributions and correlations improves power estimates.
"""

import mcpower
import pandas as pd

# Example: Using pilot study data for power analysis
# Research question: Scale up pilot study to full trial

print("=" * 60)
print("OWN DATA UPLOAD EXAMPLE")
print("=" * 60)

# 1. Load your real pilot data
print("1. LOADING PILOT DATA:")
try:
    pilot_df = pd.read_csv('examples/cars.csv')  # If you run this file from project root
except:
    pilot_df = pd.read_csv('cars.csv')           # If you run this file from  examples/ dir

print(f"✓ Loaded pilot data with {len(pilot_df)} observations")
print(f"Available columns: {list(pilot_df.columns)}")
print("\n📌 IMPORTANT: Use these exact column names in your model formula!")
print("   Example: 'mpg = hp + wt + am' (mpg as outcome, hp/wt/am as predictors)")

# 2. Define model using pilot variables  
print("\n" + "=" * 60)
print("MODEL WITH PILOT DATA") 
print("=" * 60)

# Updated model to use actual car data columns
model = mcpower.LinearRegression("mpg = hp + wt + am")

# Upload pilot data - automatically preserves distributions and correlations
model.upload_own_data(pilot_df, preserve_correlation=True)

# Set expected effect sizes based on pilot results or literature
model.set_effects("hp=-0.3, wt=-0.5, am=0.4")  # hp/wt decrease mpg, manual transmission increases mpg

print("Model setup with pilot data:")
print(f"Formula: {model.equation}")
print("Pilot data uploaded - using empirical distributions and correlations")

# 3. Power analysis with realistic data
print("\n" + "=" * 60)
print("POWER ANALYSIS WITH REAL DATA")
print("=" * 60)

print("\n1. BASIC POWER WITH PILOT DATA:")
pilot_power = model.find_power(
    sample_size=100,
    target_test="hp",
    scenarios=False,
    summary='short'
)

print("\n2. ROBUST ANALYSIS WITH PILOT DATA:")
robust_pilot = model.find_power(
    sample_size=100,
    target_test="all",
    scenarios=True,                # Still important with real data
    summary='short'
)

# 4. Sample size calculation with pilot data
print("\n3. SAMPLE SIZE WITH PILOT DATA:")
pilot_n = model.find_sample_size(
    target_test="wt",
    from_size=50,
    to_size=150,
    scenarios=True,
    summary='short'
)

# 5. Detailed analysis with pilot data
print("\n" + "=" * 60)
print("COMPREHENSIVE PILOT DATA ANALYSIS")
print("=" * 60)

comprehensive = model.find_sample_size(
    target_test="all",
    from_size=60,
    to_size=160,
    scenarios=True,
    summary='long'                 # Full output with power curves
)

# 7. Mixed approach - some variables from pilot, others synthetic
print("\n" + "=" * 60)
print("MIXED DATA APPROACH")
print("=" * 60)

# Create dataset with only some variables from pilot data
partial_data = pilot_df[['hp', 'wt']].copy()  # Only horsepower and weight from pilot

mixed_model = mcpower.LinearRegression("mpg = hp + wt + am + cyl")
mixed_model.upload_own_data(partial_data)  # Uses pilot data for available variables
mixed_model.set_effects("hp=-0.3, wt=-0.5, am=0.4, cyl=-0.2")
mixed_model.set_variable_type("am=binary")  # Synthetic binary transmission

print("Mixed approach:")
print("- hp, wt: From pilot data")
print("- am: Synthetic binary variable (transmission)")
print("- cyl: Synthetic normal variable (cylinders)")

mixed_power = mixed_model.find_power(
    sample_size=250,
    target_test="all",
    scenarios=True,
    summary='short'
)

print("\n" + "=" * 60)
print("OWN DATA GUIDELINES")
print("=" * 60)
print("""
Best practices for using pilot data:

1. DATA QUALITY:
   - Use complete cases (no missing values)
   - If you want to perform some transformation of variables, do it before uploading

2. SAMPLE SIZE:
   - Pilot data should have n≥30 for reliable distributions
   - Larger pilots (n≥50) give more stable correlations
   - Small pilots may not capture full variability

3. VARIABLE MATCHING:
   - Variable names must match model formula exactly
   - Upload data uses empirical distributions automatically
   - Not matched variables are generated synthetically

4. CORRELATION PRESERVATION:
   - Set preserve_correlation=True (default)
   - Maintains realistic predictor relationships

5. SCENARIO ANALYSIS:
   - Pilot data may not represent full population variability
   - Some features for scenarios are disabled for scenario analysis

6. MIXED APPROACHES:
   - Use pilot data for key variables
   - Generate additional variables synthetically
   - Flexible for complex study designs
""")