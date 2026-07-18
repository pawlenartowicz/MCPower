// Poisson family specs share every generation block with logit (formula, var
// types, effects, correlations, outcome options, run-config, find-call) except
// the constructor family token and the baseline setter — see generateGlmScript
// in script-generator-logit.ts, which this delegates to.
export { generatePoissonScript } from './script-generator-logit';
