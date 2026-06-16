"""Model misspecification — the model you TEST, not just the data you generate, decides what you can detect. One data-generating truth, fit three ways via test_formula: correct, confounded (omit a real cause), and over-specified (add a null covariate)."""

from mcpower import MCPower

# Story: students who study more also drink more coffee (corr 0.6). Studying
# genuinely raises the exam score; caffeine does not — its effect is 0, it only
# rides along with studying. So the data-generating formula carries caffeine at
# effect 0, but the genuinely correct model is score = study.
model = MCPower("score = study + caffeine")
model.set_effects("study=0.3, caffeine=0")
model.set_correlations("corr(study, caffeine)=0.6")

# test_formula fits a different model than the one that generated the data; every
# test term must already exist in the generation formula (both study and caffeine
# are in it). find_power does not mutate the model, so one model is fit three ways.

# 1. Correct model — matches the truth. study is well-powered (~85%).
print(">>> correct: test_formula='score = study'")
print(model.find_power(sample_size=100, target_test="study",
                       test_formula="score = study", verbose=False).summary())

# 2. Mis-specified — omit the real cause, keep its correlated proxy. Direction A:
#    dropping a correlated true cause manufactures a spurious, significant effect
#    on the innocent proxy (omitted-variable confounding).
print("\n>>> confounded: test_formula='score = caffeine'")
print(model.find_power(sample_size=100, target_test="caffeine",
                       test_formula="score = caffeine", verbose=False).summary())

# 3. Over-specified — keep the real cause but pad the model with the null
#    covariate. Direction B: a correlated null predictor steals unique variance
#    from study, so study's power drops below the correct-model level while
#    caffeine sits at ~alpha.
print("\n>>> over-specified: test_formula='score = study + caffeine'")
print(model.find_power(sample_size=100, target_test="study, caffeine",
                       test_formula="score = study + caffeine", verbose=False).summary())
