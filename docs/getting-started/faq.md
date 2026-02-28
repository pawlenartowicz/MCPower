# FAQ

## General

**Q: What models does MCPower support?**

A: MCPower currently supports:
- OLS linear regression (fully supported)
- Factor variables / ANOVA (fully supported)
- Post-hoc pairwise comparisons (fully supported)
- Mixed-effects models: random intercepts, random slopes, nested effects (fully supported)
- Logistic regression is planned for a future release.

**Q: How many simulations should I use?**

A: The default (1,600 for OLS) is sufficient for most cases. For mixed models, the default is 800 (compared to 1,600 for OLS) because each simulation is more expensive. Consider increasing for final calculations. See [Performance & Backends](../concepts/performance.md).

**Q: Can I use my own data?**

A: Yes, use `upload_data()` to preserve real-world distributions and correlations. MCPower auto-detects variable types and offers three correlation preservation modes. See [Uploading Data](../tutorials/own-data.md).

**Q: What effect size should I use?**

A: Effect sizes are standardized coefficients. Guidelines: small (0.1 continuous / 0.2 binary), medium (0.25 / 0.5), large (0.4 / 0.8). Base them on prior research or the smallest practically meaningful effect. See [Effect Sizes](../concepts/effect-sizes.md).

**Q: What's the difference between find_power and find_sample_size?**

A: `find_power(sample_size=N)` tells you the power at a given N. `find_sample_size()` searches a range of sample sizes to find the minimum N that achieves your target power (default 80%).

---

## Mixed Models

**Q: How many clusters do I need?**

A: For random intercept models, 10--20 clusters are sufficient for stable estimation. For random slope models, 30+ clusters are recommended (50+ for large slope variance). Each cluster must have at least 5 observations (warning below 10). Note: for individual-level treatment (MCPower's default), power depends primarily on total N, not on the number of clusters.

**Q: What ICC should I use?**

A: Use an ICC that reflects the expected clustering in your data. MCPower accepts ICC values of 0 (no clustering) or 0.10--0.90. Note that ICC has minimal impact on fixed-effect power in MCPower because treatment is assigned at the individual level within clusters -- power depends primarily on total N.

**Q: What about random slopes vs random intercepts?**

A: Random intercepts `(1|group)` model shared baseline differences between clusters. Random slopes `(1 + x|group)` allow the effect of a predictor to vary across clusters. Random slopes need larger samples and are more prone to convergence issues. Start with random intercepts if unsure.

---

## Troubleshooting

**Q: Power seems too low or too high**

A: Check your effect sizes. A "medium" effect is ~0.25 for continuous, ~0.5 for binary. Also check: Are predictors correlated? Correlations reduce power for individual effects. Is a correction applied? Corrections reduce power.

**Q: "Invalid target test" error**

A: Make sure the test name matches your formula exactly. For factor variables, use the bracket notation (e.g., `"group[2]"`). Use `target_test="all"` to see all available tests.

**Q: "Too many failed simulations"**

A: Mixed model convergence failures. Solutions:
```python
model.set_max_failed_simulations(0.10)  # Allow 10% failures
model.set_max_failed_simulations(0.30)  # For complex models
```
Also consider increasing sample size per cluster.

**Q: "Insufficient observations per cluster"**

A: Cluster size < 5 (the enforced minimum). Increase `sample_size` or decrease `n_clusters`:
```python
# 150/30 = 5 per cluster
model.find_power(sample_size=150)
```

**Q: C++ backend not available**

A: The C++ extension failed to compile during installation. This is usually due to a missing C++ compiler. The C++ backend is required for MCPower to function. Make sure you have a C++ compiler installed:
- **Linux:** `sudo apt install build-essential cmake` (Ubuntu/Debian) or `sudo dnf install gcc-c++ cmake` (Fedora)
- **macOS:** `xcode-select --install`
- **Windows:** Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

Then reinstall: `pip install --force-reinstall mcpower`

---

## Performance

**Q: How can I speed up the analysis?**

A: Several options:
1. The C++ backend is compiled automatically during install and provides major speedups.
2. Enable parallelization: `model.set_parallel(True)`
3. Reduce simulations for exploratory work: `model.set_simulations(500)`

See [Performance & Backends](../concepts/performance.md) for details.

**Q: Should I enable parallelization?**

A: The default (`"mixedmodels"` mode) is usually best -- it enables parallelization only for mixed models where the overhead is worthwhile. For OLS with default simulations it's already fast, also the overhead of spawning processes often exceeds the computation time.
