use engine_contract::ScenarioPerturbations;

#[test]
fn enum_shaped_scenario_payload_deserializes_with_defaults() {
    // `ScenarioPerturbations` is enum-shaped on the wire; `configs/scenarios.json`
    // uses the alias/flat/object-keyed shape and does not parse as this type.
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
    let parsed: Vec<ScenarioPerturbations> =
        serde_json::from_slice(fixture).expect("inline enum-shaped fixture must deserialize");
    let names: Vec<&str> = parsed.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, vec!["realistic"]);
    assert!(
        !parsed[0].sampled_factor_proportions,
        "payloads without sampled_factor_proportions must default to false (exact allocation; additive evolution)"
    );
}
