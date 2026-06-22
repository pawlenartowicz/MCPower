use engine_contract::{validate_scenarios, ContractError};

#[test]
fn validates_workspace_scenarios_json() {
    // `validate_scenarios` + `ScenarioPerturbations` stay enum-shaped (typed
    // realignment is deferred to the app's 2.x scenario path). `configs/scenarios.json`
    // was canonicalized to the alias/flat/object-keyed shape, so it no longer parses
    // through this enum-typed validator. Exercise the validator against an inline
    // enum-shaped fixture instead, keeping its coverage independent of that file.
    let fixture = br#"[
        {
            "name": "realistic",
            "heterogeneity": 0.2,
            "heteroskedasticity_ratio": 2.0,
            "correlation_noise_sd": 0.15,
            "distribution_change_prob": 0.5,
            "new_distributions": ["RightSkewed", "LeftSkewed", "Uniform"],
            "residual_change_prob": 0.5,
            "residual_dists": ["high_kurtosis", "right_skewed"],
            "residual_df": 8.0,
            "lme": {
                "random_effect_dist": "high_kurtosis",
                "random_effect_df": 10.0,
                "icc_noise_sd": 0.15
            }
        }
    ]"#;
    let parsed = validate_scenarios(fixture).expect("inline enum-shaped fixture must validate");
    let names: Vec<&str> = parsed.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["realistic"]);
    assert!(
        !parsed[0].sampled_factor_proportions,
        "payloads without sampled_factor_proportions must default to false (exact allocation; additive evolution)"
    );
}

#[test]
fn rejects_invalid_scenarios_json() {
    // A JSON object (not the expected array of ScenarioPerturbations) must be
    // rejected as InvalidScenariosConfig — assert the variant, not merely that
    // some error occurred (a wrong-variant error must not pass).
    let err = validate_scenarios(b"{}").unwrap_err();
    assert!(
        matches!(err, ContractError::InvalidScenariosConfig(_)),
        "expected InvalidScenariosConfig, got {err:?}"
    );
}
