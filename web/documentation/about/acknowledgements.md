---
title: "Acknowledgements - MCPower"
description: "The prior work MCPower's engine builds on — lme4, Julia's MixedModels.jl, and the statistical methods and open-source libraries behind them, with full references."
---
# Acknowledgements

MCPower's engine is an independent, performance-tuned implementation written in
Rust for simulation throughput. It does not wrap or embed existing statistics
packages; instead it implements **optimized code based on established
statistical methods and the work of the projects below**, and this page records
what it builds on and credits the authors. Where MCPower reproduces the
behaviour of another tool, it does so by following the published methods, and
the original work is credited here.

## Mixed-effects models

MCPower's linear and generalized linear mixed-model fitting follows the
computational approach of **R's lme4** (Bates, Mächler, Bolker & Walker) and
**Julia's MixedModels.jl** (Douglas Bates and contributors). Both descend from
the same line of work led by Douglas Bates, and MCPower's fitter follows the
same recipe: the profiled REML/ML deviance, the Cholesky-factor (θ) "spherical"
reparameterization of the random-effects covariance, sparse penalized least
squares, and — for GLMMs — the PIRLS inner loop with a Laplace approximation at
`nAGQ = 1`. The per-fit finite-difference Hessian standard errors reproduce
lme4's `use.hessian = TRUE` default.

Those packages are themselves implementations of methods that predate them, and
MCPower credits the methods directly. The fitter **uses algorithms** such as
**restricted maximum likelihood** (REML — Patterson & Thompson, 1971) for the
variance components, and **BOBYQA** (Powell, 2009), a bound-constrained
derivative-free optimizer, to minimize the deviance — the same optimizer lme4
uses by default.

MCPower's mixed-model results are validated by refitting the same data with
lme4 and comparing the numbers; see [[validation/index|Validation]] for how
that is done.

## Open-source libraries

The native engine is built on the Rust numerical ecosystem — **faer** for dense
and sparse linear algebra and **rayon** for data-parallel execution — and draws
its simulation randomness from the **Philox4x32-10** counter-based generator of
the Random123 family (Salmon, Moraes, Dror & Shaw, 2011), which gives
reproducible, independent streams across cores. The ports are bound to their
host languages with PyO3 (Python), extendr (R), Tauri (desktop), and
wasm-bindgen (browser).

## References

- Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). Fitting Linear
  Mixed-Effects Models Using lme4. *Journal of Statistical Software*, 67(1),
  1–48. <https://doi.org/10.18637/jss.v067.i01>
- Bates, D., Alday, P., Kleinschmidt, D., et al. *MixedModels.jl* — mixed-effects
  models in Julia. JuliaStats. <https://github.com/JuliaStats/MixedModels.jl>
- Patterson, H. D., & Thompson, R. (1971). Recovery of inter-block information
  when block sizes are unequal. *Biometrika*, 58(3), 545–554.
  <https://doi.org/10.1093/biomet/58.3.545>
- Powell, M. J. D. (2009). The BOBYQA algorithm for bound constrained
  optimization without derivatives. Technical Report DAMTP 2009/NA06,
  Department of Applied Mathematics and Theoretical Physics, University of
  Cambridge.
- Salmon, J. K., Moraes, M. A., Dror, R. O., & Shaw, D. E. (2011). Parallel
  random numbers: as easy as 1, 2, 3. *Proceedings of the International
  Conference for High Performance Computing, Networking, Storage and Analysis
  (SC '11)*. <https://doi.org/10.1145/2063384.2063405>
