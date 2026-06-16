// Projects an editable `ScenarioConfig` (the GUI's app copy of the three named
// sets) to the wire `ScenarioWire` (= `engine_spec_builder::input::ScenarioInput`).
// This is the TS port of Python's `_scenario_dict` (model.py): the same fields,
// the same name → integer-code encoding for the two distribution lists, and the
// same rejection of residual distributions the scenario wire cannot carry.
// The `lme` sub-object is flattened onto the wire: its fields are hoisted
// directly onto `ScenarioWire`.
import type { ScenarioConfig } from '$lib/configs/scenarios';
import type { ScenarioWire } from './app-spec';

// Distribution name → integer code. Mirrors Python's `_DIST_CODE` (model.py);
// the parity test pins these against the Python constants so they cannot drift.
export const DIST_CODE: Readonly<Record<string, number>> = {
  normal: 0,
  binary: 1,
  right_skewed: 2,
  left_skewed: 3,
  high_kurtosis: 4,
  uniform: 5,
  uploaded_factor: 97,
  uploaded_binary: 98,
  uploaded_data: 99,
};

// Residual-distribution name → integer code. Canonical five shapes, matching
// RESIDUAL_CODES in project_contract.rs. The v1 aliases (heavy_tailed/skewed/t)
// are dropped in line with the spec-builder pre-1.0 break.
export const RESIDUAL_CODE: Readonly<Record<string, number>> = {
  normal: 0,
  right_skewed: 2,
  left_skewed: 3,
  high_kurtosis: 4,
  uniform: 5,
};

// Random-effect distribution name → integer code. Reuses the RESIDUAL_CODE
// integer space (normal=0, heavy_tailed=1); mirrors Python's RE-dist encoding
// and the engine's `residual_dist_from_code`. The UI only exposes normal /
// heavy_tailed for random effects.
export const RE_DIST_CODE: Readonly<Record<string, number>> = {
  normal: 0,
  heavy_tailed: 1,
};

// Empty set; named export kept so membership-checking callers need no change.
export const UNSUPPORTED_RESIDUAL_IN_SCENARIOS: ReadonlySet<string> = new Set();

function encodeDist(name: string): number {
  const code = DIST_CODE[name];
  if (code === undefined) {
    throw new Error(
      `unknown distribution name '${name}'; valid: ${Object.keys(DIST_CODE).sort().join(', ')}`,
    );
  }
  return code;
}

function encodeResidual(name: string): number {
  if (UNSUPPORTED_RESIDUAL_IN_SCENARIOS.has(name)) {
    throw new Error(
      `residual distribution '${name}' is not yet supported in scenario perturbations; ` +
        `the scenario wire format supports normal/right_skewed/left_skewed/high_kurtosis/uniform`,
    );
  }
  const code = RESIDUAL_CODE[name];
  if (code === undefined) {
    throw new Error(
      `unknown residual distribution '${name}'; valid: ${Object.keys(RESIDUAL_CODE).sort().join(', ')}`,
    );
  }
  return code;
}

/**
 * Project one `ScenarioConfig` to the wire `ScenarioWire`, mirroring Python's
 * `_scenario_dict`. Throws on a residual distribution the wire cannot carry. The
 * `lme` sub-object is flattened: its three fields are hoisted onto the wire
 * (random_effect_dist name-encoded via RE_DIST_CODE; df and icc_noise_sd raw),
 * mirroring the Python port. The `lme` sub-object itself is not on the wire.
 */
export function projectScenario(cfg: ScenarioConfig): ScenarioWire {
  return {
    name: cfg.name,
    heterogeneity: cfg.heterogeneity,
    heteroskedasticity_ratio: cfg.heteroskedasticity_ratio,
    correlation_noise_sd: cfg.correlation_noise_sd,
    distribution_change_prob: cfg.distribution_change_prob,
    new_distributions: cfg.new_distributions.map(encodeDist),
    residual_change_prob: cfg.residual_change_prob,
    residual_dists: cfg.residual_dists.map(encodeResidual),
    residual_df: cfg.residual_df,
    // Persisted scenario snapshots from before the knob lack the field;
    // default to the exact-allocation engine default.
    sampled_factor_proportions: cfg.sampled_factor_proportions ?? false,
    // Pre-lme snapshots lack the sub-object; default to a neutral (no-op) RE
    // perturbation so the wire is always complete.
    random_effect_dist: RE_DIST_CODE[cfg.lme?.random_effect_dist ?? 'normal'] ?? 0,
    random_effect_df: cfg.lme?.random_effect_df ?? 0,
    icc_noise_sd: cfg.lme?.icc_noise_sd ?? 0,
  };
}
