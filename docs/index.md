---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

```{image} _static/mcpower_asci.png
:alt: MCPower
:class: hero-logo
```

# MCPower

Monte Carlo power analysis for complex statistical models. Instead of relying on closed-form formulas, MCPower **simulates thousands of datasets** and counts how often your analysis detects real effects.

Two core functions drive every analysis:

- **`find_power()`** — Given a sample size, estimate statistical power
- **`find_sample_size()`** — Find the minimum sample size for a target power level

```{note}
This is the documentation for the **MCPower Python package**. If you're looking for the desktop application, **[MCPower GUI](https://github.com/pawlenartowicz/mcpower-gui)** is a standalone app for Windows, Linux, and macOS with built-in documentation — download it from the [releases page](https://github.com/pawlenartowicz/mcpower-gui/releases/latest). These docs are still useful for understanding how power analysis settings work under the hood.
```

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import numpy as np
np.random.seed(42)
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
```

## Install

```bash
pip install mcpower
```

## Quick Example

```{code-cell} ipython3
:tags: [remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2 + x1:x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3, x1:x2=0.2")
model.find_power(sample_size=100)
```

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
getting-started/faq
```

```{toctree}
:maxdepth: 2

tutorials/index
concepts/index
api/index
```

```{toctree}
:maxdepth: 2
:caption: Other

model-specification
limitations
validation
lme-validation
```
