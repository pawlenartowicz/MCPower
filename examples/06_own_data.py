"""
Own Data Upload Example
=======================

This example shows how to use your own pilot data for realistic power analysis.
Using real data distributions and correlations improves power estimates.
"""

import csv
from pathlib import Path

from mcpower import MCPower

# Example: Using pilot study data for power analysis
# Research question: Scale up pilot study to full trial

print("=" * 60)
print("OWN DATA UPLOAD EXAMPLE")
print("=" * 60)


def load_csv(path) -> dict[str, list]:
    """Load a CSV file into a dict of column lists, auto-converting numeric values."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    result: dict[str, list] = {}
    for col in rows[0]:
        if col == "":
            continue
        raw = [r[col] for r in rows]
        try:
            result[col] = [float(v) for v in raw]
        except (ValueError, TypeError):
            result[col] = raw
    return result


# 1. Load your real pilot data
print("1. LOADING PILOT DATA:")
# Try project root first, then examples/ dir
csv_path = Path("examples/cars.csv")
if not csv_path.exists():
    csv_path = Path("cars.csv")

pilot_data = load_csv(csv_path)
n_rows = len(next(iter(pilot_data.values())))

print(f"Loaded pilot data with {n_rows} observations")
print(f"Available columns: {list(pilot_data.keys())}")
print("\nIMPORTANT: Use these exact column names in your model formula!")
print("   Example: 'mpg = hp + wt + am' (mpg as outcome, hp/wt/am as predictors)")

# 2. Define model using pilot variables
print("\n" + "=" * 60)
print("MODEL WITH PILOT DATA")
print("=" * 60)

model = MCPower("mpg = hp + wt + am")

# Upload pilot data — automatically preserves distributions and correlations
model.upload_data(pilot_data)

# Set expected effect sizes based on pilot results or literature
model.set_effects(
    "hp=-0.3, wt=-0.5, am=0.4"
)  # hp/wt decrease mpg, manual transmission increases mpg

print("Model setup with pilot data:")
print(f"Formula: {model.equation}")
print("Pilot data uploaded — using empirical distributions and correlations")

# 3. Power analysis with realistic data
print("\n" + "=" * 60)
print("POWER ANALYSIS WITH REAL DATA")
print("=" * 60)

print("\n1. BASIC POWER WITH PILOT DATA:")
model.find_power(sample_size=100, target_test="hp")

print("\n2. ROBUST ANALYSIS WITH PILOT DATA:")
model.find_power(sample_size=100, target_test="all", scenarios=True)

# 4. Sample size calculation with pilot data
print("\n3. SAMPLE SIZE WITH PILOT DATA:")
model.find_sample_size(target_test="wt", from_size=50, to_size=150, scenarios=True)

# 5. Mixed approach — some variables from pilot, others synthetic
print("\n" + "=" * 60)
print("MIXED DATA APPROACH")
print("=" * 60)

# Create dataset with only some variables from pilot data
partial_data = {k: pilot_data[k] for k in ["hp", "wt"]}

mixed_model = MCPower("mpg = hp + wt + am + cyl")
mixed_model.upload_data(partial_data)
mixed_model.set_effects("hp=-0.3, wt=-0.5, am=0.4, cyl=-0.2")
mixed_model.set_variable_type("am=binary")  # Synthetic binary transmission

print("Mixed approach:")
print("- hp, wt: From pilot data")
print("- am: Synthetic binary variable (transmission)")
print("- cyl: Synthetic normal variable (cylinders)")

mixed_model.find_power(sample_size=250, target_test="all", scenarios=True)

print("\n" + "=" * 60)
print("OWN DATA GUIDELINES")
print("=" * 60)
print(
    """
Best practices for using pilot data:

1. DATA QUALITY:
   - Use complete cases (no missing values)
   - If you want to perform some transformation of variables, do it before uploading

2. SAMPLE SIZE:
   - Pilot data should have n>=30 for reliable distributions
   - Larger pilots (n>=50) give more stable correlations
   - Small pilots may not capture full variability

3. VARIABLE MATCHING:
   - Variable names must match model formula exactly
   - Upload data uses empirical distributions automatically
   - Not matched variables are generated synthetically

4. CORRELATION PRESERVATION:
   - Default mode is 'strict' (bootstrap whole rows)
   - Use preserve_correlation='partial' for data + manual overrides
   - Use preserve_correlation='no' for full manual control

5. MIXED APPROACHES:
   - Use pilot data for key variables
   - Generate additional variables synthetically
   - Flexible for complex study designs
"""
)
