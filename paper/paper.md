---
title: 'MCPower: Monte Carlo Power Analysis for Complex Statistical Models'
tags:
  - Python
  - statistics
  - power analysis
  - Monte Carlo simulation
  - experimental design
  - linear regression
  - sample size calculation
authors:
  - name: Paweł Lenartowicz
    orcid: 0000-0002-6906-7217
    corresponding: true
    affiliation: 1
affiliations:
 - name: [Stowarzyszenie na rzecz Otwartej Nauki (Society for Open Science)], [Poland]
   index: 1
date: [Current Date]
bibliography: paper.bib
---

# Summary

Statistical power analysis is a frequently overlooked parameter that determines the sample size needed to detect meaningful effects in research studies. Traditional power analysis methods rely on mathematical formulas that work well for simple designs, yet fail for complex models involving interactions, correlated predictors, or non-normal distributions. MCPower overcomes these limitations by using Monte Carlo simulation to estimate statistical power for any model that can be expressed as an R formula. This makes power analysis accessible for complex research designs.

MCPower generates thousands of synthetic datasets with user-specified effect sizes and data characteristics. It then analyzes each dataset to determine how often the analysis successfully detects true effects. Importantly, the package enables proper robustness checks by systematically varying assumptions across simulations, testing how power estimates change under realistic violations of homoscedasticity, effect size consistency, and distributional assumptions. This simulation-based approach reveals the sensitivity of power calculations to model assumptions, providing researchers with more realistic sample size recommendations.

# Statement of need

Power analysis is fundamental to research design. For complex models, several critical considerations emerge:

* Interactions demand significantly more observations than main effects alone
* Correlated predictors can dramatically improve or reduce statistical power [@montoya2019moderation]
* Effect heterogeneity across participants substantially impacts detectability [@kenny2019unappreciated]
* Assumption violations affect power unpredictably, with consequences varying by model complexity

However, existing tools impose severe limitations. For example, analytical software like G*Power [@faul2009gpower] only handles simple models without interactions or correlated predictors. Specialized tools, such as InteractionPowerR, require extensive programming knowledge and lack flexibility for complex designs. General-purpose packages like R, SPSS, and SAS demand custom simulation programming, which creates barriers for most researchers. Most critically, no existing software provides systematic robustness testing for assumption violations.

MCPower addresses these limitations by combining the flexibility of Monte Carlo simulation with an intuitive interface modeled after regression packages. Users can specify models using familiar R-style formulas (e.g., outcome = treatment + covariate + treatment*covariate), define effect sizes and correlation structures, and obtain power estimates through simple method calls. The package automatically handles complex scenarios and performs systematic robustness testing under realistic assumption violations.

Key distinguishing features include: (1) support for complex interactions without programming, (2) built-in scenario analysis to test optimistic, realistic, and pessimistic conditions, (3) handling of non-normal distributions with correlated predictors, and (4) integration of empirical data for realistic simulations. These capabilities address critical gaps in experimental psychology, the social sciences, and biomedical research, where complex designs are common, yet adequate power analysis tools are unavailable.

MCPower serves researchers planning studies and students learning experimental design. Its intuitive API allows users to focus on their research questions rather than on programming simulations, and scenario analysis reveals how assumption violations affect statistical conclusions.

# Key Features

MCPower implements several innovations in power analysis methodology:

**Formula-Based Interface**: Users can specify models using R-style formulas, which are familiar to social scientists. This eliminates the need for strong programming skils or complex simulation setup.

**Scenario Analysis**: MCPower goes beyond traditional power calculations by providing "optimistic," "realistic," and "doomer" scenarios that test robustness under varying degrees of assumption violations. This feature addresses the common issue of real-world studies failing to achieve the expected power due to unmet assumptions.

**Distribution Flexibility**: The package supports multiple variable distributions, including normal, binary, skewed, heavy-tailed, and uniform, and automatically handles correlations between predictors, including rank correlations for mixed distribution types.

**Empirical Data Integration**: Researchers can upload pilot data to preserve realistic correlation structures and distributions in their power simulations.

**Performance Optimization**: Critical simulation components use Numba's ahead-of-time compilation and fall back to just-in-time compilation. This provides performance comparable to that of compiled languages while maintaining Python's accessibility. The package also offers parallel computing to allow for the most accurate analysis with many tested sample sizes and simulations.

# Example Usage

```python
import mcpower

# Define model with interaction
model = mcpower.LinearRegression("satisfaction = treatment + age + treatment*age")

# Set effect sizes
model.set_effects("treatment=0.6, age=0.2, treatment:age=0.3")
model.set_variable_type("treatment=binary")

# Find required sample size with scenario analysis and power plot
model.find_sample_size(target_test="treatment", scenarios=True, summary = "long")
```

This example demonstrates the package's core workflow: specify a model, define expected effects, and obtain power estimates that account for realistic complications.

# References