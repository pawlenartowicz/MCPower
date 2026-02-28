# Mixed-Effects Models

## What It Is

Mixed-effects models (also called multilevel or hierarchical models) handle data where observations are grouped into clusters. Students nested in schools, patients nested in hospitals, repeated measures within participants -- whenever observations within a group are more similar to each other than to observations in other groups, you have clustered data.

The "mixed" refers to two types of effects. **Fixed effects** are the predictors you care about (treatment, age, motivation) -- the relationships you want to test. **Random effects** capture the shared variation within clusters. A random intercept means each cluster has its own baseline level; a random slope means the effect of a predictor varies across clusters.

Ignoring clustering inflates Type I error rates and produces misleadingly precise estimates. MCPower uses mixed-effects models to produce correct power estimates for clustered designs. Power is always tested for **fixed effects** (whether beta differs from zero), not random effects.

---

## How It Works in MCPower

MCPower supports three random effect structures, configured via the formula and [`set_cluster()`](../api/data.md):

**Random intercepts** -- `(1|school)`:

```python
model = MCPower("satisfaction ~ treatment + motivation + (1|school)")
model.set_cluster("school", ICC=0.2, n_clusters=20)
```

**Random slopes** -- `(1 + x|school)`:

```python
model = MCPower("y ~ x1 + (1 + x1|school)")
model.set_cluster("school", ICC=0.2, n_clusters=20,
                   random_slopes=["x1"], slope_variance=0.1, slope_intercept_corr=0.3)
```

**Nested effects** -- `(1|school/classroom)`:

```python
model = MCPower("y ~ treatment + (1|school/classroom)")
model.set_cluster("school", ICC=0.15, n_clusters=10)
model.set_cluster("classroom", ICC=0.10, n_per_parent=3)
```

---

## Guidelines

### Statistical Models

**Random intercepts:**

$$y_{ij} = \mathbf{x}_{ij}'\boldsymbol{\beta} + b_j + e_{ij}, \quad b_j \sim \mathcal{N}(0, \tau^2), \quad e_{ij} \sim \mathcal{N}(0, \sigma^2)$$

**Random slopes** (adds slope variation per cluster):

$$y_{ij} = \mathbf{x}_{ij}'\boldsymbol{\beta} + b_{0j} + b_{1j} x_{ij} + e_{ij}$$

$$[b_{0j}, b_{1j}] \sim \mathcal{N}(\mathbf{0}, \mathbf{G}), \quad \mathbf{G} = \begin{bmatrix} \tau^2_{\text{int}} & \rho \cdot \tau_{\text{int}} \cdot \tau_{\text{slope}} \\ \rho \cdot \tau_{\text{int}} \cdot \tau_{\text{slope}} & \tau^2_{\text{slope}} \end{bmatrix}$$

**Nested** (two levels of clustering):

$$y_{ijk} = \mathbf{x}_{ijk}'\boldsymbol{\beta} + b_{\text{school},j} + b_{\text{classroom},jk} + e_{ijk}$$

### Design Recommendations

| Recommendation | Rationale |
|---|---|
| 10--20 clusters minimum | Below 10, random effect estimation becomes unstable |
| 30+ clusters for random slopes | Slope variance estimation needs more cluster-level data |
| Minimum 5 obs/cluster | Enforced by MCPower; warning below 10 |
| Slope variance has large impact | Even slope_var=0.05 can require 2--3x more observations |

MCPower assigns treatment at the individual level within clusters. ICC and cluster allocation have minimal impact on fixed-effect power -- power depends primarily on total N. See the [Random Slopes tutorial](../tutorials/mixed-slopes.md#design-recommendations) for the factor that does matter: `slope_variance`.

### Constraints

- No crossed random effects (only nested and independent clusters)
- No random slopes without intercept (`(0 + x|group)` not supported)
- ICC valid range: 0 (no clustering) or 0.1 to 0.9. Values between 0 and 0.1 (exclusive) are rejected for numerical stability.

---

## Common Patterns

### Typical ICC Values by Research Domain

| Domain | Grouping | Typical ICC |
|---|---|---|
| Education | Schools | 0.10--0.25 |
| Healthcare | Hospitals / clinics | 0.05--0.20 |
| Psychology | Therapists / sites | 0.10--0.30 |
| Organizational | Companies / teams | 0.10--0.25 |

### Which Random Effect Structure?

| Scenario | Formula | Rationale |
|---|---|---|
| Observations clustered, effect same everywhere | `(1\|cluster)` | Simplest; accounts for shared baseline |
| Effect of a predictor varies across clusters | `(1 + x\|cluster)` | Captures slope heterogeneity |
| Two-level hierarchy (e.g., classrooms in schools) | `(1\|school/classroom)` | Captures clustering at both levels |

### Recommended Convergence Thresholds

| Model Type | `set_max_failed_simulations()` |
|---|---|
| Random intercept | 0.03--0.10 |
| Random slopes (30+ clusters) | 0.10--0.20 |
| Nested models | 0.20--0.30 |

---

## Learn More

- **[LME Validation (vs lme4)](../lme-validation.md)** -- validation framework, thresholds, and latest results
- **[API Reference: set_cluster](../api/data.md)** -- complete parameter documentation
- **[Concept: Performance](performance.md)** -- C++ solver performance for mixed models
