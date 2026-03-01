# API Reference

Complete reference for the `MCPower` class and its public API.

---

## Quick Reference

| Method / Class | Description |
|---|---|
| [`MCPower(formula)`](constructor.md) | Create a power analysis model from an R-style formula |
| {ref}`find_power() <find-power>` | Estimate statistical power at a given sample size |
| {ref}`find_sample_size() <find-sample-size>` | Find the minimum sample size for target power |
| {ref}`set_effects() <set-effects>` | Set standardized effect sizes (beta weights) |
| {ref}`set_variable_type() <set-variable-type>` | Declare binary, factor, or non-normal variables |
| {ref}`set_correlations() <set-correlations>` | Specify predictor correlations |
| {ref}`set_alpha() <set-alpha>` | Set the significance level |
| {ref}`set_power() <set-power>` | Set the target power level (for sample size search) |
| {ref}`set_seed() <set-seed>` | Set random seed for reproducibility |
| {ref}`set_simulations() <set-simulations>` | Set number of Monte Carlo simulations |
| {ref}`set_parallel() <set-parallel>` | Configure parallel execution |
| {ref}`upload_data() <upload-data>` | Upload empirical data for realistic simulation |
| {ref}`set_factor_levels() <set-factor-levels>` | Define named factor levels |
| {ref}`set_cluster() <set-cluster>` | Configure clustering for mixed-effects models |
| {ref}`set_scenario_configs() <set-scenario-configs>` | Customize scenario analysis parameters |
| {ref}`set_max_failed_simulations() <set-max-failed-simulations>` | Set convergence failure tolerance |
| [Correction methods](corrections.md) | Multiple testing correction options |
| [Progress reporting](progress.md) | Progress callbacks and built-in reporters |
| [Return values](return-values.md) | Result dictionary structures |

---

## Pages

```{toctree}
:maxdepth: 2

constructor
power-analysis
configuration
data
scenarios
corrections
progress
return-values
```
