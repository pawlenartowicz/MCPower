use engine_contract::{validate_config, ByStep, ContractError};
use std::path::Path;

fn workspace_config() -> Vec<u8> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../configs/config.json");
    std::fs::read(p).expect("read configs/config.json")
}

#[test]
fn workspace_config_json_parses() {
    let cfg = validate_config(&workspace_config()).expect("valid config");
    assert_eq!(cfg.simulation.seed, 2137);
    assert_eq!(cfg.simulation.alpha, 0.05);
    assert_eq!(cfg.simulation.target_power, 0.8);
    assert_eq!(cfg.simulation.n_sims.ols, 1600);
    assert_eq!(cfg.simulation.n_sims.mixed, 800);
    assert_eq!(cfg.simulation.n_sims.anova, 1000);
    assert_eq!(cfg.simulation.max_failed_fraction, 0.1);
    assert_eq!(cfg.simulation.cluster_auto_count, 12);
    assert_eq!(cfg.benchmarks.continuous, [0.1, 0.25, 0.4]);
    assert_eq!(cfg.benchmarks.binary_factor, [0.2, 0.5, 0.8]);
    assert_eq!(cfg.benchmarks.odds, [0.405, 0.916, 1.386]);
    assert_eq!(cfg.limits.max_alpha, 0.25);
    assert_eq!(cfg.limits.icc_stability, [0.05, 0.95]);
    assert_eq!(cfg.limits.baseline_p_warn, [0.05, 0.95]);
    assert_eq!(cfg.limits.factor_levels, [2, 20]);
    assert_eq!(cfg.limits.min_rows_per_cluster, 2);
    assert_eq!(cfg.limits.min_clusters, 5);
    assert_eq!(cfg.limits.reliable_rows_per_cluster, 5);
    assert_eq!(cfg.limits.recommended_rows_per_cluster, 10);
    assert_eq!(cfg.report.format.power_decimals_long, 1);
    assert_eq!(
        cfg.report.overall_label_by_estimator.get("ols").unwrap(),
        "Overall F"
    );
    assert_eq!(cfg.upload.max_rows, 1_000_000);
    assert_eq!(cfg.upload.min_rows, 20);
    assert_eq!(cfg.upload.max_factor_ratio, 15);
}

#[test]
fn sample_size_bounds_by_is_auto() {
    let cfg = validate_config(&workspace_config()).unwrap();
    assert_eq!(cfg.simulation.sample_size_bounds.from, 30);
    assert_eq!(cfg.simulation.sample_size_bounds.to, 200);
    assert!(matches!(
        cfg.simulation.sample_size_bounds.by,
        ByStep::Auto(_)
    ));
}

#[test]
fn upload_config_exposes_strict_warning_ratio() {
    let cfg = validate_config(&workspace_config()).unwrap();
    assert_eq!(cfg.upload.strict_warning_ratio, 2.0);
}

#[test]
fn rejects_malformed_json() {
    assert!(matches!(
        validate_config(b"{"),
        Err(ContractError::InvalidConfig(_))
    ));
    assert!(matches!(
        validate_config(b"{\"simulation\":{}}"),
        Err(ContractError::InvalidConfig(_))
    ));
}
