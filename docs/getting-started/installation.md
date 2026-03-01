---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Installation

This page covers how to install MCPower, its optional dependency groups, and how to verify that everything is working.

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Basic Installation

```bash
pip install mcpower
```

This installs the core package with all required dependencies. It is sufficient for standard power analyses using linear regression models with continuous, binary, and factor variables.

---

## Updating

```bash
pip install --upgrade mcpower
```

To check your current version:

```bash
pip show mcpower
```

---

## Optional Dependency Groups

MCPower provides optional dependency groups for features that require additional packages:

| Group | Command | What It Adds |
|---|---|---|
| Core (no extras) | `pip install mcpower` | Standard power analysis (OLS, factors, scenarios, mixed models) |
| `[pandas]` | `pip install mcpower[pandas]` | pandas DataFrame support for `upload_data()` |
| `[lme]` | `pip install mcpower[lme]` | statsmodels (legacy LME fallback) |
| `[all]` | `pip install mcpower[all]` | pandas (DataFrame support) |

### All Optional Dependencies

```bash
pip install mcpower[all]
```

Installs pandas for DataFrame support.

---

## Requirements

**Python:** >= 3.10

**Core dependencies** (installed automatically):

| Package | Purpose |
|---|---|
| NumPy | Array operations, random number generation |
| matplotlib | Power curve plots and visualizations |
| joblib | Parallel execution for mixed-effects models |
| tqdm | Progress bars for simulation runs |

**Optional dependencies:**

| Package | Install With | Purpose |
|---|---|---|
| pandas | `mcpower[pandas]` | DataFrame input for `upload_data()` |
| statsmodels | `mcpower[lme]` | Legacy LME solver fallback |

---

## C++ Backend

MCPower includes a native C++ backend built with pybind11 and Eigen for high-performance computation. During installation, `pip install mcpower` automatically compiles this backend using scikit-build-core and CMake.

The C++ backend is **required** -- it provides ~200x speedups for mixed-effects models and handles all distribution functions via Boost.Math. If compilation fails (e.g., no C++ compiler available), MCPower will not work.

For more details, see [Performance & Backends](../concepts/performance.md).

---

## Verifying Your Installation

Run a quick power analysis to confirm everything is working:

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")
result = model.find_power(sample_size=100, return_results=True, progress_callback=False)

print(f"MCPower is working. Power for x1: {result['results']['individual_powers']['x1']:.1f}%")
```

---

## Desktop GUI Application

If you prefer a graphical interface, **[MCPower GUI](https://github.com/pawlenartowicz/mcpower-gui)** is a standalone desktop application for Windows, Linux, and macOS. Download ready-to-run executables from the [releases page](https://github.com/pawlenartowicz/mcpower-gui/releases/latest). No Python installation required.
