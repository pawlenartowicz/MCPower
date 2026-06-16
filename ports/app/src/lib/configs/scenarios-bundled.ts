import bundled from '$configs/scenarios.json';
import type { ScenarioConfig } from './scenarios';

/** Build-time snapshot of `mcpower/configs/scenarios.json` (object-keyed, alias, flat).
 *  Adapt to the app's array + nested-`lme` shape; values are unchanged. */
type FlatScenario = Omit<ScenarioConfig, 'name' | 'lme'> & {
  random_effect_dist: ScenarioConfig['lme']['random_effect_dist'];
  random_effect_df: number;
  icc_noise_sd: number;
};

export const BUNDLED_SCENARIOS: ScenarioConfig[] = Object.entries(
  bundled as Record<string, FlatScenario>,
).map(([name, v]) => {
  const { random_effect_dist, random_effect_df, icc_noise_sd, ...rest } = v;
  return { name, ...rest, lme: { random_effect_dist, random_effect_df, icc_noise_sd } };
});
