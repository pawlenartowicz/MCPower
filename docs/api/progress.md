---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
---

# Progress Reporting

```{code-cell} ipython3
:tags: [remove-input, remove-output]
import warnings
warnings.filterwarnings("ignore", message="Low simulation")
import numpy as np
np.random.seed(42)
```

Both `find_power()` and `find_sample_size()` accept a `progress_callback` parameter that controls how progress is reported during the simulation.

---

## Resolution Logic

The behavior of `progress_callback` depends on both its value and `print_results`:

| `progress_callback` | `print_results` | Behavior |
|---|---|---|
| `None` (default) | `True` (default) | Auto `PrintReporter()` -- prints progress to stderr |
| `None` | `False` | Silent -- no progress output |
| `callable` | any | Uses the provided callback `(current: int, total: int)` |
| `False` | any | Explicitly disables progress |

## Built-in Reporters

### PrintReporter

```{eval-rst}
.. autoclass:: mcpower.PrintReporter
   :members:
```

Console progress reporter. Writes text-based progress to stderr. This is the default when `print_results=True`.

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower, PrintReporter

model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")

# Default behavior (PrintReporter auto-selected)
model.find_power(sample_size=100)
# Output on stderr: "Progress: 45.2% (723/1600)"
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Explicit PrintReporter
model.find_power(sample_size=100, progress_callback=PrintReporter())
```

### TqdmReporter

```{eval-rst}
.. autoclass:: mcpower.TqdmReporter
   :members:
```

Optional tqdm-based progress reporter. Requires the `tqdm` package.

```python
from mcpower import MCPower, TqdmReporter

model = MCPower("y = x1 + x2")
model.set_effects("x1=0.5, x2=0.3")

model.find_power(sample_size=100, progress_callback=TqdmReporter())
# Displays: [=========>          ] 45% 723/1600
```

### ProgressReporter

```{eval-rst}
.. autoclass:: mcpower.ProgressReporter
   :members:
```

Low-level wrapper that tracks completed simulation steps and throttles callback invocations. Typically you do not need to use this directly -- it is used internally by MCPower.

### SimulationCancelled

```{eval-rst}
.. autoclass:: mcpower.SimulationCancelled
   :members:
```

Exception raised when a simulation is cancelled via the `cancel_check` callback.

## Custom Callbacks

Any callable that accepts `(current: int, total: int)` can be used as a progress callback:

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
from mcpower import MCPower

model = MCPower("y = x1 + x2")
model.set_simulations(400)
model.set_effects("x1=0.5, x2=0.3")

# Simple print callback
model.find_power(
    sample_size=100,
    progress_callback=lambda cur, tot: print(f"{cur}/{tot}"),
)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Percentage callback
def show_percent(current, total):
    pct = current / total * 100
    print(f"\r{pct:.1f}% complete", end="", flush=True)

model.find_power(sample_size=100, progress_callback=show_percent)
```

### GUI Integration

For GUI applications, the callback bridges simulation progress to UI updates:

```python
# PySide6 / PyQt example
def on_progress(current, total):
    progress_signal.emit(current, total)  # Emit Qt signal to update progress bar

model.find_power(
    sample_size=100,
    print_results=False,
    progress_callback=on_progress,
)
```

## Disabling Progress

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Explicitly disable (even when print_results=True)
model.find_power(sample_size=100, progress_callback=False)
```

```{code-cell} ipython3
:tags: [remove-output, remove-stderr]
# Implicitly disabled (no print_results, no explicit callback)
model.find_power(sample_size=100, print_results=False)
```

## Total Simulation Count

The `total` value passed to callbacks reflects the total number of simulation iterations across all scenarios and sample sizes:

| Analysis | Total |
|---|---|
| `find_power(N)` | `n_simulations` |
| `find_power(N, scenarios=True)` | `n_simulations * 3` (optimistic + realistic + doomer) |
| `find_sample_size(from, to, by)` | `n_simulations * n_sample_sizes` |
| `find_sample_size(..., scenarios=True)` | `n_simulations * n_sample_sizes * 3` |

### Throttling

The built-in `ProgressReporter` wrapper throttles updates to approximately 200 calls maximum (`update_every = max(1, total // 200)`), avoiding excessive I/O overhead on fast simulations.

## See Also

- [Return Values](return-values.md) -- Accessing results programmatically
- {ref}`find_power() <find-power>` -- Analysis method that accepts progress callbacks
