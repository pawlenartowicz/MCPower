// TypeScript mirror of the scenario contract from `mcpower/configs/scenarios.json`; keep this shape aligned with the serde types `engine_contract::ScenarioPerturbations` / `LmeScenarioPerturbations`, which the bundled JSON deserializes into.
// The GUI maintains an editable app copy (three named sets); the bundled JSON seeds it on first run.

export type RandomEffectDist = 'normal' | 'heavy_tailed';
// Pool of predictor distributions a scenario may swap into; all non-normal DIST_CODE variants.
export type NewDistribution = 'right_skewed' | 'left_skewed' | 'high_kurtosis' | 'uniform';

export interface LmeScenario {
  random_effect_dist: RandomEffectDist;
  random_effect_df: number;
  icc_noise_sd: number;
}

export interface ScenarioConfig {
  name: string;
  heterogeneity: number;
  heteroskedasticity_ratio: number;
  correlation_noise_sd: number;
  distribution_change_prob: number;
  new_distributions: NewDistribution[];
  sampled_factor_proportions: boolean;
  /** Truth-seeded fitter start (mixed/clustered designs) vs. cold (blind) start. */
  truth_start: boolean;
  lme: LmeScenario;
}

export type ScenarioName = 'optimistic' | 'realistic' | 'doomer';
export const SCENARIO_NAMES: readonly ScenarioName[] = ['optimistic', 'realistic', 'doomer'];
