---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# MCPower Documentation

Monte Carlo power analysis for complex statistical models. Instead of relying on closed-form formulas, MCPower **simulates thousands of datasets** and counts how often your analysis detects real effects.

Two core functions drive every analysis:

- **`find_power()`** — Given a sample size, estimate statistical power
- **`find_sample_size()`** — Find the minimum sample size for a target power level

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

## About

MCPower is developed by [Paweł Lenartowicz](https://freestylerscientist.pl/about-me/), an independent researcher working on Monte Carlo methods, metascience, and open science tools.

- [Project homepage](https://freestylerscientist.pl/projects/mcpower/) — screenshots, GUI walkthrough, citation info
- [Support this project](https://freestylerscientist.pl/support_me/) — MCPower is free and open source
- [GitHub](https://github.com/pawlenartowicz/MCPower)

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/quickstart
getting-started/faq
```

```{toctree}
:maxdepth: 2
:caption: Tutorials

tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: Concepts

concepts/index
```

```{toctree}
:maxdepth: 2
:caption: API Reference

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
