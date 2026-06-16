// TypeScript mirror of the scenario contract from `mcpower/configs/scenarios.json`; keep this shape aligned with `engine_contract::validate_scenarios`.
// The GUI maintains an editable app copy (three named sets); the bundled JSON seeds it on first run.

// Canonical residual-distribution names matching RESIDUAL_CODES in engine_contract::project_contract.
// The v1 aliases heavy_tailed/skewed/t are dropped (pre-1.0 break).
export type ResidualDist = 'normal' | 'right_skewed' | 'left_skewed' | 'high_kurtosis' | 'uniform';
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
  residual_change_prob: number;
  residual_dists: ResidualDist[];
  residual_df: number;
  sampled_factor_proportions: boolean;
  lme: LmeScenario;
}

export type ScenarioName = 'optimistic' | 'realistic' | 'doomer';
export const SCENARIO_NAMES: readonly ScenarioName[] = ['optimistic', 'realistic', 'doomer'];
