// Typed wrapper over `mcpower/configs/config.json` — exposes SIMULATION, BENCHMARKS, LIMITS, UPLOAD.
// Field shapes must stay in sync with mcpower/configs/config.json.
// The App reads config.json directly via Vite's $configs alias; no Rust bridge involved.
import config from '$configs/config.json';

export const SIMULATION = config.simulation as {
  seed: number; alpha: number; target_power: number;
  n_sims: { ols: number; mixed: number; anova: number };
  max_failed_fraction: number;
  sample_size_bounds: { from: number; to: number; by: number | 'auto' };
  cluster_auto_count: number;
};
export const BENCHMARKS = config.benchmarks as {
  continuous: number[];
  binary_factor: number[];
  // Logit-outcome (beta) presets: β = log(OR) for OR = 1.5 / 2.5 / 4.0 (Chen et al. 2010).
  odds: number[];
};
export const LIMITS = config.limits as unknown as {
  max_alpha: number; icc_stability: [number, number]; baseline_p_warn: [number, number];
  factor_levels: [number, number];
  min_clusters: number; min_rows_per_cluster: number;
  reliable_rows_per_cluster: number; recommended_rows_per_cluster: number;
};
export const UPLOAD = config.upload as {
  max_rows: number;
  max_rows_wasm: number;
  min_rows: number;
  max_factor_k_soft: number;
  max_factor_ratio: number;
  strict_warning_ratio: number;
};
