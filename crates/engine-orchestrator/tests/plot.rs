//! Snapshot tests + schema sanity for engine_orchestrator::plot.

use engine_orchestrator::{
    plot::{
        exactly_k_curve_spec, joint_detection_curve_spec, power_at_n_spec, power_plot_set,
        sample_size_curve_spec, sample_size_plot_set, PlotOptions, PlotPoint, PlotScenario,
    },
    Ci,
};
use serde_json::Value;

const SCHEMA: &str = "https://vega.github.io/schema/vega-lite/v5.json";

// ── Fixtures ───────────────────────────────────────────────────────────────

fn point(n: usize, targets: &[usize], p: &[f64], ci: &[(f64, f64)]) -> PlotPoint {
    assert_eq!(targets.len(), p.len());
    assert_eq!(targets.len(), ci.len());
    PlotPoint {
        n,
        target_indices: targets.to_vec(),
        contrast_pairs: vec![],
        power: p.to_vec(),
        ci: ci.iter().map(|&(lo, hi)| Ci { lo, hi }).collect(),
        histogram: vec![],
        overall_power: None,
        overall_ci: None,
    }
}

fn single_scenario() -> Vec<PlotScenario> {
    vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![point(
            120,
            &[1, 2],
            &[0.83, 0.71],
            &[(0.80, 0.86), (0.67, 0.74)],
        )],
    }]
}

fn multi_scenario() -> Vec<PlotScenario> {
    vec![
        PlotScenario {
            label: "optimistic".into(),
            points: vec![point(
                120,
                &[1, 2],
                &[0.83, 0.71],
                &[(0.80, 0.86), (0.67, 0.74)],
            )],
        },
        PlotScenario {
            label: "realistic".into(),
            points: vec![point(
                120,
                &[1, 2],
                &[0.72, 0.55],
                &[(0.69, 0.75), (0.51, 0.59)],
            )],
        },
        PlotScenario {
            label: "doomer".into(),
            points: vec![point(
                120,
                &[1, 2],
                &[0.51, 0.34],
                &[(0.47, 0.55), (0.30, 0.38)],
            )],
        },
    ]
}

fn ssr_grid() -> Vec<PlotScenario> {
    vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![
            point(50, &[1, 2], &[0.51, 0.32], &[(0.47, 0.55), (0.28, 0.36)]),
            point(100, &[1, 2], &[0.74, 0.55], &[(0.70, 0.78), (0.51, 0.59)]),
            point(150, &[1, 2], &[0.86, 0.71], &[(0.83, 0.89), (0.67, 0.75)]),
            point(200, &[1, 2], &[0.93, 0.82], &[(0.91, 0.95), (0.79, 0.85)]),
        ],
    }]
}

/// Like [`ssr_grid`] but with power that never reaches the 0.80 target.
fn ssr_grid_below_target() -> Vec<PlotScenario> {
    vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![
            point(50, &[1, 2], &[0.41, 0.22], &[(0.37, 0.45), (0.18, 0.26)]),
            point(100, &[1, 2], &[0.54, 0.35], &[(0.50, 0.58), (0.31, 0.39)]),
            point(150, &[1, 2], &[0.66, 0.51], &[(0.63, 0.69), (0.47, 0.55)]),
            point(200, &[1, 2], &[0.73, 0.62], &[(0.71, 0.75), (0.59, 0.65)]),
        ],
    }]
}

/// Grid point carrying a joint-significance histogram (len == n_targets + 1) for
/// the joint-detection curve, which reads `histogram` rather than `power`/`ci`.
fn point_h(n: usize, targets: &[usize], hist: &[u64]) -> PlotPoint {
    assert_eq!(targets.len() + 1, hist.len());
    PlotPoint {
        n,
        target_indices: targets.to_vec(),
        contrast_pairs: vec![],
        power: vec![],
        ci: vec![],
        histogram: hist.to_vec(),
        overall_power: None,
        overall_ci: None,
    }
}

/// Multi-scenario, multi-N fixture: 3 scenarios, each a 4-point grid over the
/// same two targets, with joint histograms. Drives the faceted curve / joint /
/// report snapshots.
fn multi_scenario_grid() -> Vec<PlotScenario> {
    let mk = |label: &str, p: &[[f64; 2]], ci: &[[(f64, f64); 2]], hist: &[[u64; 3]]| {
        let ns = [50usize, 100, 150, 200];
        let mut points = Vec::new();
        for i in 0..ns.len() {
            let mut pt = point(ns[i], &[1, 2], &p[i], &ci[i]);
            pt.histogram = hist[i].to_vec();
            points.push(pt);
        }
        PlotScenario {
            label: label.into(),
            points,
        }
    };
    vec![
        mk(
            "optimistic",
            &[[0.51, 0.32], [0.74, 0.55], [0.86, 0.71], [0.93, 0.82]],
            &[
                [(0.47, 0.55), (0.28, 0.36)],
                [(0.70, 0.78), (0.51, 0.59)],
                [(0.83, 0.89), (0.67, 0.75)],
                [(0.91, 0.95), (0.79, 0.85)],
            ],
            &[[40, 35, 25], [20, 35, 45], [8, 27, 65], [3, 17, 80]],
        ),
        mk(
            "realistic",
            &[[0.41, 0.22], [0.62, 0.43], [0.75, 0.58], [0.84, 0.69]],
            &[
                [(0.37, 0.45), (0.18, 0.26)],
                [(0.58, 0.66), (0.39, 0.47)],
                [(0.72, 0.78), (0.54, 0.62)],
                [(0.81, 0.87), (0.65, 0.73)],
            ],
            &[[55, 30, 15], [30, 38, 32], [15, 35, 50], [7, 30, 63]],
        ),
        mk(
            "doomer",
            &[[0.31, 0.15], [0.48, 0.28], [0.61, 0.41], [0.71, 0.52]],
            &[
                [(0.27, 0.35), (0.12, 0.18)],
                [(0.44, 0.52), (0.24, 0.32)],
                [(0.57, 0.65), (0.37, 0.45)],
                [(0.68, 0.74), (0.48, 0.56)],
            ],
            &[[65, 27, 8], [42, 36, 22], [28, 38, 34], [18, 40, 42]],
        ),
    ]
}

fn parse(s: &str) -> Value {
    serde_json::from_str(s).expect("emitter must produce valid JSON")
}

// ── power_at_n_spec ────────────────────────────────────────────────────────

#[test]
fn power_at_n_single_bare() {
    insta::assert_json_snapshot!(parse(&power_at_n_spec(
        &single_scenario(),
        &PlotOptions::default()
    )));
}

#[test]
fn power_at_n_multi_bare() {
    insta::assert_json_snapshot!(parse(&power_at_n_spec(
        &multi_scenario(),
        &PlotOptions::default()
    )));
}

#[test]
fn power_at_n_single_with_ci() {
    let opts = PlotOptions {
        show_ci: true,
        ..Default::default()
    };
    insta::assert_json_snapshot!(parse(&power_at_n_spec(&single_scenario(), &opts)));
}

#[test]
fn power_at_n_single_with_ci_and_target_line_and_title() {
    let opts = PlotOptions {
        title: Some("Power at N=120".into()),
        show_ci: true,
        target_power_line: Some(0.80),
    };
    insta::assert_json_snapshot!(parse(&power_at_n_spec(&single_scenario(), &opts)));
}

// ── sample_size_curve_spec ─────────────────────────────────────────────────

#[test]
fn curve_grid_with_ci_and_target_line() {
    let opts = PlotOptions {
        title: Some("Power vs sample size".into()),
        show_ci: true,
        target_power_line: Some(0.80),
    };
    insta::assert_json_snapshot!(parse(&sample_size_curve_spec(&ssr_grid(), &opts)));
}

#[test]
fn curve_grid_bare() {
    insta::assert_json_snapshot!(parse(&sample_size_curve_spec(
        &ssr_grid_below_target(),
        &PlotOptions::default()
    )));
}

#[test]
fn curve_grid_ci_band_only() {
    // CI band on, target line off — proves None really means no rule layer.
    let opts = PlotOptions {
        show_ci: true,
        ..Default::default()
    };
    insta::assert_json_snapshot!(parse(&sample_size_curve_spec(&ssr_grid(), &opts)));
}

// ── sample_size_report_spec (composite) — deleted; blocks now via plot-set ──

#[test]
fn curve_multi_scenario_faceted() {
    let opts = PlotOptions {
        title: Some("Power vs sample size".into()),
        show_ci: true,
        target_power_line: Some(0.80),
    };
    insta::assert_json_snapshot!(parse(&sample_size_curve_spec(
        &multi_scenario_grid(),
        &opts
    )));
}

// ── joint_detection_curve_spec ─────────────────────────────────────────────

fn joint_single() -> Vec<PlotScenario> {
    vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![
            point_h(50, &[1, 2], &[80, 15, 5]),
            point_h(100, &[1, 2], &[40, 40, 20]),
            point_h(200, &[1, 2], &[5, 35, 60]),
        ],
    }]
}

#[test]
fn joint_single_scenario() {
    let opts = PlotOptions {
        target_power_line: Some(0.80),
        ..Default::default()
    };
    insta::assert_json_snapshot!(parse(&joint_detection_curve_spec(&joint_single(), &opts)));
}

#[test]
fn joint_multi_scenario_faceted() {
    let opts = PlotOptions {
        target_power_line: Some(0.80),
        ..Default::default()
    };
    insta::assert_json_snapshot!(parse(&joint_detection_curve_spec(
        &multi_scenario_grid(),
        &opts
    )));
}

// ── Load-bearing structure (too fine-grained / arithmetic for snapshots) ────

#[test]
fn power_at_n_is_horizontal_with_power_sort() {
    // Single: x maps to power (the axis flip), bars sized 360 wide.
    let single = parse(&power_at_n_spec(
        &single_scenario(),
        &PlotOptions::default(),
    ));
    let bar = &single["layer"][0]["encoding"];
    assert_eq!(bar["x"]["field"], "power");
    assert_eq!(bar["y"]["field"], "target");
    assert_eq!(single["width"], 360.0);
    // S=1 ⇒ y.scale.paddingInner == 2/5 == 0.4 (⅔-bar gap with one bar/group).
    assert_eq!(bar["y"]["scale"]["paddingInner"], 0.4);
    assert_eq!(bar["y"]["scale"]["paddingOuter"], 0);

    // Multi: grouping moves to yOffset + fillOpacity, both ordered by HOST order
    // (the order scenarios are passed in) so the layered spec renders in the correct
    // order (Vega-Lite ignores aggregate {field,op,order} sort on yOffset/color
    // in a layered spec — an explicit array IS honoured).
    //
    // multi_scenario() passes: ["optimistic", "realistic", "doomer"] — host order.
    let expected_sort = serde_json::json!(["optimistic", "realistic", "doomer"]);
    let multi = parse(&power_at_n_spec(&multi_scenario(), &PlotOptions::default()));
    let mbar = &multi["layer"][0]["encoding"];
    assert_eq!(mbar["yOffset"]["field"], "scenario");
    assert!(
        mbar["yOffset"]["sort"].is_array(),
        "yOffset.sort must be an explicit array, got: {}",
        mbar["yOffset"]["sort"]
    );
    assert_eq!(mbar["yOffset"]["sort"], expected_sort);
    // Colour keys on the effect (target), not the scenario, with NO pinned domain:
    // hosts relabel the `target` data values, so a token domain would null out the
    // marks. Vega derives the domain from the (relabelled) data.
    assert_eq!(mbar["color"]["field"], "target");
    assert!(
        mbar["color"]["scale"].is_null(),
        "color must carry no explicit scale, got: {}",
        mbar["color"]["scale"]
    );
    // Errorbar layer (layer[1] when show_ci=true) must carry the same colour binding.
    let multi_ci = parse(&power_at_n_spec(
        &multi_scenario(),
        &PlotOptions {
            show_ci: true,
            ..Default::default()
        },
    ));
    let ci_enc = &multi_ci["layer"][1]["encoding"];
    assert_eq!(ci_enc["yOffset"]["sort"], expected_sort);
    assert_eq!(ci_enc["color"]["field"], "target");
    assert_eq!(mbar["yOffset"]["scale"]["paddingInner"], 0);
    assert_eq!(mbar["yOffset"]["scale"]["paddingOuter"], 0);
    // S=3 ⇒ y.scale.paddingInner == 2/11 (compare with tolerance: the JSON
    // round-trip shifts the last ULP off the Rust literal).
    let pad = mbar["y"]["scale"]["paddingInner"].as_f64().unwrap();
    assert!((pad - 2.0 / 11.0).abs() < 1e-12, "paddingInner was {pad}");
    assert!(
        mbar["x"].get("xOffset").is_none(),
        "no leftover horizontal offset"
    );

    // CI errorbar layer uses horizontal encoding (x = ci_lo, x2 = ci_hi, y = target).
    let single_ci = parse(&power_at_n_spec(
        &single_scenario(),
        &PlotOptions {
            show_ci: true,
            ..Default::default()
        },
    ));
    let ci_enc = &single_ci["layer"][1]["encoding"];
    assert_eq!(ci_enc["x"]["field"], "ci_lo");
    assert_eq!(ci_enc["x2"]["field"], "ci_hi");
    assert_eq!(ci_enc["y"]["field"], "target");
}

#[test]
fn power_at_n_height_formula() {
    // G=2 targets, S=3 scenarios: G*S + (G-1)*⅔ = 6.667 → floored to 7 units
    // → round(7*16) == 112.
    let multi = parse(&power_at_n_spec(&multi_scenario(), &PlotOptions::default()));
    assert_eq!(multi["height"], 112.0);

    // G=2, S=1 (single scenario, two targets): 2 + ⅔ = 2.667 < 7 ⇒ also floored
    // to 112. Force a case above the floor with more targets.
    let many = vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![point(
            120,
            &[1, 2, 3, 4, 5, 6, 7, 8],
            &[0.8; 8],
            &[(0.7, 0.9); 8],
        )],
    }];
    // G=8, S=1: 8 + 7*⅔ = 12.667 (> 7) ⇒ round(12.667*16) == round(202.67) == 203.
    let v = parse(&power_at_n_spec(&many, &PlotOptions::default()));
    assert_eq!(v["height"], 203.0);
}

#[test]
fn curve_multi_facets_and_rekeys_to_target() {
    let v = parse(&sample_size_curve_spec(
        &multi_scenario_grid(),
        &PlotOptions::default(),
    ));
    assert_eq!(v["facet"]["field"], "scenario");
    assert_eq!(v["facet"]["columns"], 3);
    // Panels follow the host's scenario order, not Vega-Lite's alphabetical
    // default (which would render doomer, optimistic, realistic).
    assert_eq!(
        v["facet"]["sort"],
        serde_json::json!(["optimistic", "realistic", "doomer"])
    );
    assert_eq!(v["spec"]["width"], 200.0);
    assert_eq!(v["spec"]["height"], 240.0);
    assert!(
        v.get("layer").is_none(),
        "layers move under spec when faceted"
    );
    // Inside a panel, targets are keyed by colour (shared domain, stable palette
    // slot) and also by strokeDash (redundant for colourblind/print). The engine
    // emits separate specs for the curve and joint panels, so there is no
    // shared-scale conflict at the engine level (a host vconcat would need
    // `resolve.scale.color = independent`). `series` drops the scenario label so
    // each panel is multi-series.
    let line_enc = &v["spec"]["layer"][0]["encoding"];
    assert_eq!(line_enc["color"]["field"], "target");
    assert_eq!(line_enc["strokeDash"]["field"], "target");
    let rows = v["data"]["values"].as_array().unwrap();
    assert!(rows
        .iter()
        .all(|r| !r["series"].as_str().unwrap().contains("optimistic")));

    // Single scenario stays unfaceted with a fixed 360x240 panel.
    let s = parse(&sample_size_curve_spec(
        &ssr_grid(),
        &PlotOptions::default(),
    ));
    assert!(s.get("facet").is_none());
    assert_eq!(s["width"], 360.0);
    assert_eq!(s["height"], 240.0);
}

// ── Overall (omnibus) curve series ──────────────────────────────────────────

/// One-target grid with an overall-test power per grid point. The overall test
/// rides the power curve as a second `"overall"` series with its own CI band.
fn ssr_grid_with_overall() -> Vec<PlotScenario> {
    let mut points = vec![
        point(50, &[1], &[0.41], &[(0.37, 0.45)]),
        point(100, &[1], &[0.66], &[(0.63, 0.69)]),
        point(150, &[1], &[0.86], &[(0.83, 0.89)]),
    ];
    let overall = [(0.45, 0.41, 0.49), (0.70, 0.67, 0.73), (0.90, 0.88, 0.92)];
    for (p, &(op, lo, hi)) in points.iter_mut().zip(overall.iter()) {
        p.overall_power = Some(op);
        p.overall_ci = Some(Ci { lo, hi });
    }
    vec![PlotScenario {
        label: "optimistic".into(),
        points,
    }]
}

#[test]
fn overall_series_is_emitted_and_strokedashed_on_single_target() {
    let curve = parse(&sample_size_curve_spec(
        &ssr_grid_with_overall(),
        &PlotOptions {
            show_ci: true,
            target_power_line: Some(0.80),
            ..Default::default()
        },
    ));
    let rows = curve["data"]["values"].as_array().unwrap();
    // One marginal target + one overall series, three grid points each.
    let overall_rows: Vec<&Value> = rows.iter().filter(|r| r["target"] == "overall").collect();
    assert_eq!(overall_rows.len(), 3, "one overall row per grid point");
    // Overall carries its own CI band columns, like every other effect.
    assert_eq!(overall_rows[0]["ci_lo"], 0.41);
    assert_eq!(overall_rows[0]["ci_hi"], 0.49);
    // A single marginal target + overall = 2 rendered series → strokeDash must
    // fire so the two near-overlapping lines (F = t²) are told apart.
    let line_enc = &curve["layer"][0]["encoding"];
    assert_eq!(line_enc["strokeDash"]["field"], "target");
}

#[test]
fn overall_series_absent_when_no_overall_power() {
    // The default ssr_grid carries no overall_power: no "overall" rows appear,
    // and a single-target version stays a solid line (no strokeDash).
    let one_target = vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![
            point(50, &[1], &[0.41], &[(0.37, 0.45)]),
            point(100, &[1], &[0.86], &[(0.83, 0.89)]),
        ],
    }];
    let curve = parse(&sample_size_curve_spec(
        &one_target,
        &PlotOptions::default(),
    ));
    let rows = curve["data"]["values"].as_array().unwrap();
    assert!(rows.iter().all(|r| r["target"] != "overall"));
    assert!(curve["layer"][0]["encoding"].get("strokeDash").is_none());
}

// ── Axis titles (no concatenation across co-scaled layers) ──────────────────

#[test]
fn axis_titles_do_not_concatenate() {
    // power_at_n: x is "Power" (not "Power, ci_lo"), y is "Effect" (not "Target").
    // The errorbar layer must repeat the same titles on its shared x/y scales.
    let bars = parse(&power_at_n_spec(
        &single_scenario(),
        &PlotOptions {
            show_ci: true,
            ..Default::default()
        },
    ));
    let bar_enc = &bars["layer"][0]["encoding"];
    assert_eq!(bar_enc["x"]["title"], "Power");
    assert_eq!(bar_enc["y"]["title"], "Effect");
    let err_enc = &bars["layer"][1]["encoding"];
    assert_eq!(
        err_enc["x"]["title"], "Power",
        "errorbar x must repeat \"Power\""
    );
    assert_eq!(
        err_enc["y"]["title"], "Effect",
        "errorbar y must repeat \"Effect\""
    );

    // sample_size_curve: line y "Power" / x "Sample size (N)"; the errorband
    // layer repeats them so the shared scale renders a single title.
    let curve = parse(&sample_size_curve_spec(
        &ssr_grid(),
        &PlotOptions {
            show_ci: true,
            target_power_line: Some(0.80),
            ..Default::default()
        },
    ));
    let layers = curve["layer"].as_array().unwrap();
    for layer in layers {
        let enc = &layer["encoding"];
        // Only check layers that bind the shared n/power fields (skip the
        // datum-based target rule, which carries no field title).
        if enc["x"]["field"] == "n" {
            assert_eq!(enc["x"]["title"], "Sample size (N)");
        }
        if enc["y"]["field"] == "power" || enc["y"]["field"] == "ci_lo" {
            assert_eq!(enc["y"]["title"], "Power");
        }
    }
}

// ── Cross-emitter schema sanity ────────────────────────────────────────────

#[test]
fn all_emitters_pin_v5_schema_and_emit_no_config() {
    for s in [
        power_at_n_spec(&single_scenario(), &PlotOptions::default()),
        sample_size_curve_spec(&ssr_grid(), &PlotOptions::default()),
        joint_detection_curve_spec(&joint_single(), &PlotOptions::default()),
        exactly_k_curve_spec(&joint_single(), &PlotOptions::default()),
    ] {
        let v = parse(&s);
        assert_eq!(v["$schema"].as_str(), Some(SCHEMA));
        assert!(v.get("config").is_none(), "emitter must be theme-naked");
    }
}

// ── plot-set block keys ────────────────────────────────────────────────────

/// Build a single-scenario, single-target fixture for plot-set m=1 tests.
fn plot_set_s1_m1() -> Vec<PlotScenario> {
    vec![PlotScenario {
        label: "optimistic".into(),
        points: vec![
            point(50, &[1], &[0.51], &[(0.47, 0.55)]),
            point(100, &[1], &[0.74], &[(0.70, 0.78)]),
            point(150, &[1], &[0.86], &[(0.83, 0.89)]),
        ],
    }]
}

/// Build a single-scenario, two-target fixture with histograms for plot-set m=2 tests.
fn plot_set_s1_m2() -> Vec<PlotScenario> {
    let mut points = vec![
        point(50, &[1, 2], &[0.51, 0.32], &[(0.47, 0.55), (0.28, 0.36)]),
        point(100, &[1, 2], &[0.74, 0.55], &[(0.70, 0.78), (0.51, 0.59)]),
    ];
    points[0].histogram = vec![40, 35, 25];
    points[1].histogram = vec![20, 35, 45];
    vec![PlotScenario {
        label: "optimistic".into(),
        points,
    }]
}

/// Build a two-scenario, two-target fixture with histograms for plot-set S≥2 m≥2 tests.
fn plot_set_s2_m2() -> Vec<PlotScenario> {
    let mk = |label: &str| {
        let mut points = vec![
            point(50, &[1, 2], &[0.41, 0.22], &[(0.37, 0.45), (0.18, 0.26)]),
            point(100, &[1, 2], &[0.62, 0.43], &[(0.58, 0.66), (0.39, 0.47)]),
        ];
        points[0].histogram = vec![55, 30, 15];
        points[1].histogram = vec![30, 38, 32];
        PlotScenario {
            label: label.into(),
            points,
        }
    };
    vec![mk("a"), mk("b")]
}

/// Build a two-scenario, single-target fixture (no histograms needed) for S≥2 m=1 tests.
fn plot_set_s2_m1() -> Vec<PlotScenario> {
    let mk = |label: &str| PlotScenario {
        label: label.into(),
        points: vec![
            point(50, &[1], &[0.41], &[(0.37, 0.45)]),
            point(100, &[1], &[0.62], &[(0.58, 0.66)]),
        ],
    };
    vec![mk("a"), mk("b")]
}

#[test]
fn power_plot_set_always_single_power_block() {
    // power_plot_set always emits exactly ["power"].
    for scenarios in [&single_scenario(), &multi_scenario(), &plot_set_s1_m2()] {
        let blocks = power_plot_set(scenarios, &PlotOptions::default());
        let keys: Vec<&str> = blocks.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["power"], "scenarios count {}", scenarios.len());
        let v = parse(&blocks[0].1);
        assert_eq!(v["$schema"].as_str(), Some(SCHEMA));
        assert!(v.get("config").is_none());
    }
}

#[test]
fn sample_size_plot_set_s1_m1_single_curve_block() {
    // Row 1: S=1, m=1 → ["curve"]
    let blocks = sample_size_plot_set(&plot_set_s1_m1(), &PlotOptions::default());
    let keys: Vec<&str> = blocks.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["curve"]);
    let v = parse(&blocks[0].1);
    assert_eq!(v["$schema"].as_str(), Some(SCHEMA));
    assert!(v.get("config").is_none());
}

#[test]
fn sample_size_plot_set_s1_m2_curve_at_least_exactly() {
    // Row 2: S=1, m≥2 → ["curve", "at_least_k", "exactly_k"]
    let blocks = sample_size_plot_set(&plot_set_s1_m2(), &PlotOptions::default());
    let keys: Vec<&str> = blocks.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["curve", "at_least_k", "exactly_k"]);
    for (_, spec) in &blocks {
        let v = parse(spec);
        assert!(v["$schema"].as_str().unwrap().contains("v5"));
        assert!(v.get("config").is_none());
    }
}

#[test]
fn sample_size_plot_set_s2_m1_scenario_blocks_then_overlay_no_aux() {
    // Row 3: S≥2, m=1 → ["scenario:a", "scenario:b", "overlay"] — no at_least_k/exactly_k.
    let blocks = sample_size_plot_set(&plot_set_s2_m1(), &PlotOptions::default());
    let keys: Vec<&str> = blocks.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, vec!["scenario:a", "scenario:b", "overlay"]);
    for (_, spec) in &blocks {
        let v = parse(spec);
        assert!(v["$schema"].as_str().unwrap().contains("v5"));
        assert!(v.get("config").is_none());
    }
}

#[test]
fn sample_size_plot_set_s2_m2_all_blocks() {
    // Row 4: S≥2, m≥2 → ["scenario:a", "scenario:b", "overlay", "at_least_k", "exactly_k"]
    let blocks = sample_size_plot_set(&plot_set_s2_m2(), &PlotOptions::default());
    let keys: Vec<&str> = blocks.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "scenario:a",
            "scenario:b",
            "overlay",
            "at_least_k",
            "exactly_k"
        ]
    );
    for (_, spec) in &blocks {
        let v = parse(spec);
        assert!(v["$schema"].as_str().unwrap().contains("v5"));
        assert!(v.get("config").is_none());
    }
}

#[test]
fn m1_emits_no_aux_blocks() {
    // Explicit: single-target scenarios never get at_least_k / exactly_k.
    for scenarios in [&plot_set_s1_m1(), &plot_set_s2_m1()] {
        let blocks = sample_size_plot_set(scenarios, &PlotOptions::default());
        let keys: Vec<&str> = blocks.iter().map(|(k, _)| k.as_str()).collect();
        assert!(
            !keys.contains(&"at_least_k"),
            "m=1 must not emit at_least_k; got {keys:?}"
        );
        assert!(
            !keys.contains(&"exactly_k"),
            "m=1 must not emit exactly_k; got {keys:?}"
        );
    }
}
