"""Power for a clustered binary outcome — a logistic GLMM. Combine family='logit' with a (1|group) term and set_cluster(ICC, n_clusters): the engine fits a random-intercept logistic model (Laplace approximation) so the within-cluster correlation is accounted for on the log-odds scale."""

from mcpower import MCPower

# Example: Multi-clinic trial with a binary outcome.
# Patients are nested in clinics, and the outcome is whether each patient
# recovered (yes/no). Research question: does the treatment raise the recovery
# probability, accounting for the fact that patients in the same clinic are
# correlated (some clinics simply have healthier or sicker caseloads)?

# 1. Declare a clustered logistic model. family="logit" makes the outcome
#    binary; the (1|clinic) term adds a random intercept per clinic. Together
#    they are a logistic GLMM — fitted with a per-iteration GLM that carries the
#    random intercept (Laplace approximation).
model = MCPower(
    "recovered = treatment + baseline_severity + (1|clinic)", family="logit"
)

# 2. treatment is a binary predictor (0/1, control vs treated);
#    baseline_severity is continuous (default).
model.set_variable_type("treatment=binary")

# 3. Baseline probability — the recovery rate for an untreated patient of
#    average severity at a typical clinic. Required for logit; sets the model
#    intercept via log(p / (1 - p)). Here 30% of reference patients recover.
model.set_baseline_probability(0.3)

# 4. Effects are standardised log-odds shifts. treatment=0.5 (moderate) and
#    baseline_severity=0.25 (medium) — note the GLMM needs more N than a plain
#    logistic for the same effect, because the clinic clustering eats degrees
#    of freedom.
model.set_effects("treatment=0.5, baseline_severity=0.25")

# 5. Describe the clustering: ICC=0.10 (10% of the latent-scale variance is
#    between-clinic) across 40 clinics. The ICC is read on the logit scale and
#    converted to the random-intercept variance tau^2 = ICC/(1-ICC) * pi^2/3.
#    At N=600 that is 15 patients per clinic.
model.set_cluster("clinic", ICC=0.10, n_clusters=40)

# 6. Short form (printed automatically) — power per fixed effect at N=600. Both
#    real effects land mid-band: treatment ~82%, baseline_severity ~80%.
print(">>> model.find_power(sample_size=600, target_test='treatment, baseline_severity')")
model.find_power(sample_size=600, target_test="treatment, baseline_severity")

# 7. Long form via .summary() — adds the joint-significance distribution and the
#    GLMM diagnostics: the latent-scale tau-hat^2, the singular-fit rate, and the
#    realised baseline probability.
print("\n>>> result = model.find_power(sample_size=600, target_test='treatment, baseline_severity', verbose=False)")
print(">>> print(result.summary())")
result = model.find_power(
    sample_size=600, target_test="treatment, baseline_severity", verbose=False
)
print(result.summary())

# 8. Sample-size search for the primary effect. With a fixed number of clinics
#    (n_clusters=40) the search adds patients per clinic; the engine snaps each
#    grid point to a whole number of observations per cluster, so every reported
#    N is a balanced design.
print("\n>>> model.find_sample_size(target_test='treatment', from_size=240, to_size=720, by=40)")
model.find_sample_size(
    target_test="treatment", from_size=240, to_size=720, by=40
)
