import json, pathlib
from cases import load_cases, Case

ROOT = pathlib.Path(__file__).parent


def test_load_cases_merges_family_defaults():
    cases = load_cases(ROOT / "benchmark_cases.json")
    by_id = {c.id: c for c in cases}
    assert len(cases) == 33
    ols = by_id["ols_simple"]
    assert ols.family == "ols"
    assert ols.n_grid == list(range(20, 201, 20))          # from family defaults (10 points)
    assert ols.n_sims["mcpower"] == 10000
    lme = by_id["lme_multi"]
    assert lme.n_grid == list(range(100, 1001, 100))
    assert lme.n_sims["naive"] == 30
    assert lme.cluster["n_clusters"] == 20
    glm = by_id["glm_simple"]
    assert glm.baseline_p == 0.3                            # from logit family default
    assert glm.max_failed_frac == 0.25


def test_load_cases_new_fields():
    cases = load_cases(ROOT / "benchmark_cases.json")
    by_id = {c.id: c for c in cases}
    assert by_id["anova_2x2"].n_grid == list(range(40, 401, 40))       # per-case n grid (tuned)
    assert by_id["anova_2x2"].n_sims["tool"] == 1000                   # per-case key-level override
    assert by_id["anova_2x2"].n_sims["mcpower"] == 10000               # family default preserved
    assert by_id["ols_large_n"].n_grid == list(range(500, 3001, 250))
    assert by_id["ols_correlated"].correlations == "corr(x1,x2)=0.4"
    assert by_id["ols_simple"].correlations is None
    assert by_id["glm_rare"].baseline_p == 0.05                        # per-case override of family 0.3
    assert by_id["glm_simple"].baseline_p == 0.3
    assert by_id["ols_simple"].tool == "simglm"
    assert by_id["anova_2x2"].tool == "superpower"
    assert by_id["lme_simple"].tool == "simr"
    assert by_id["ols_correlated"].tool is None
    assert by_id["ols_simple"].n_sims["tool"] == 500                   # family-default tool count
    # GLMM is a 4th family (logit outcome + clusters); tool is cliff (simr
    # GLMM adapter not yet wired), baseline_p comes from the glmm family default.
    assert by_id["glmm_simple"].family == "glmm"
    assert by_id["glmm_simple"].baseline_p == 0.3
    assert by_id["glmm_simple"].cluster["n_clusters"] == 20
    assert by_id["glmm_simple"].tool is None
    # GLMM random-slope cases (LOSF 24/25): slope config rides inside the
    # cluster dict, consumed by build_model + the R glmer loop.
    assert by_id["glmm_slope"].cluster["random_slopes"] == ["x1"]
    assert by_id["glmm_slope"].cluster["slope_variance"] == 0.1
    assert by_id["glmm_multislope"].cluster["random_slopes"] == ["x1", "x2"]


def test_build_model_runs_every_case_at_nsim4():
    from harness import build_model
    failures = []
    for c in load_cases(ROOT / "benchmark_cases.json"):
        try:
            m = build_model(c)
            r = m.find_power(sample_size=c.n_grid[-1], n_sims=4, seed=2137,
                             progress_callback=False, verbose=False)
            pwr = r.get("power_uncorrected", r.get("power"))
            assert pwr is not None and len(list(pwr)) >= 1
        except Exception as e:
            failures.append(f"{c.id}: {type(e).__name__}: {e}")
    assert not failures, "engine rejected:\n" + "\n".join(failures)


def test_loop_best_returns_finite_power():
    import numpy as np
    from loops_py import LOOPS
    cases = {c.id: c for c in load_cases(ROOT / "benchmark_cases.json")}
    for cid in ("ols_multi", "glm_multi", "lme_multi"):
        c = cases[cid]
        n = c.n_grid[len(c.n_grid)//2]
        out = LOOPS[c.family]["best"](c, n, 50, 2137)
        tpwr = np.atleast_1d(np.asarray(out[0], dtype=float))
        assert np.all(np.isfinite(tpwr)) and tpwr.size >= 1


def _case_by_id(cid):
    return {c.id: c for c in load_cases(ROOT / "benchmark_cases.json")}[cid]


def test_loop_design_factor_factor_product():
    import numpy as np
    from loops_py import _design, _draw_X
    spec = _design(_case_by_id("anova_2x2"))             # y = f1*f2
    assert spec["beta"].tolist() == [0.3, 0.3, 0.3]      # f1, f2[2], f1:f2[2] (tuned from 0.5)
    assert spec["targets"].tolist() == [0]               # target "f1" = bare 2-level factor
    X = _draw_X(np.random.default_rng(7), 500, spec)
    assert X.shape == (500, 3)
    assert np.array_equal(X[:, 2], X[:, 0] * X[:, 1])    # ff product column


def test_loop_design_cont_factor_product():
    import numpy as np
    from loops_py import _design, _draw_X
    spec = _design(_case_by_id("ols_factor_inter"))      # y = x1 + f + x1:f, cols: x1, f[2], x1*f[2]
    assert spec["beta"].tolist() == [0.40, 0.5, 0.3]   # x1 tuned from 0.2
    X = _draw_X(np.random.default_rng(7), 500, spec)
    assert np.array_equal(X[:, 2], X[:, 0] * X[:, 1])


def test_loop_design_anova_2x3_names():
    from loops_py import _design
    spec = _design(_case_by_id("anova_2x3"))
    # cols: f1, f2[2], f2[3], f1:f2[2], f1:f2[3]
    assert spec["beta"].tolist() == [0.5, 0.5, 0.5, 0.5, 0.5]
    assert spec["targets"].tolist() == [0]


def test_loop_design_oneway4_dummy_block():
    from loops_py import _design
    spec = _design(_case_by_id("anova_oneway4"))         # 4-level factor falls out of existing machinery
    assert spec["beta"].tolist() == [0.5, 0.5, 0.5]
    assert spec["targets"].tolist() == [0]               # target "f[2]" -> first dummy


def test_correlated_draw():
    import numpy as np
    from loops_py import _design, _draw_X
    spec = _design(_case_by_id("ols_correlated"))
    X = _draw_X(np.random.default_rng(7), 20000, spec)
    assert abs(np.corrcoef(X[:, 0], X[:, 1])[0, 1] - 0.4) < 0.03
    assert abs(np.corrcoef(X[:, 0], X[:, 2])[0, 1]) < 0.03


def test_loop_best_vs_naive_power_band():
    # parallel best kernel vs serial naive kernel — different kernels, same DGP
    import numpy as np
    from loops_py import LOOPS
    c = _case_by_id("ols_simple")
    best = float(np.atleast_1d(LOOPS["ols"]["best"](c, 200, 2000, 2137)[0])[0])
    naive = float(np.atleast_1d(LOOPS["ols"]["naive"](c, 200, 500, 2137)[0])[0])
    assert abs(best - naive) < 0.06        # ~5 sigma at these sim counts


def _mk_results(tmp_path, name, lang, rows):
    import json as _json
    meta = {"lang": lang, "timestamp_utc": "2026-06-03T00:00:00", "os": "linux",
            "cpu_model": "test", "cores_physical": 4, "cores_logical": 8,
            "threads_mode": "auto", "lang_version": "x", "packages": {}}
    p = tmp_path / name
    p.write_text(_json.dumps({"meta": meta, "records": rows}))
    return p


def _row(case_id, lang, method, n, n_sims, time_s):
    return {"case_id": case_id, "family": "ols", "lang": lang, "method": method,
            "n": n, "n_sims": n_sims, "time_s": time_s,
            "per_sim_s": time_s / n_sims, "power": [0.8]}


def test_combine_reads_meta_records(tmp_path):
    from combine import combine
    py = _mk_results(tmp_path, "py.json", "py", [_row("ols_multi", "py", "mcpower_find_power", 100, 10000, 0.1)])
    rr = _mk_results(tmp_path, "r.json", "r", [_row("ols_multi", "r", "mcpower_find_power", 100, 10000, 0.12)])
    py_meta, r_meta, series, tool_names, fss = combine(py, rr)
    assert py_meta["lang"] == "py" and r_meta["lang"] == "r"
    assert series[("ols_multi", 100)]["mcpower:py"] == 1e-5
    assert series[("ols_multi", 100)]["mcpower:r"] == 1.2e-5
    assert fss == {}


def test_aggregation_goldens(tmp_path):
    import json as _json
    from cases import load_cases
    from combine import combine, aggregate
    cases_doc = {
        "defaults": {"ols": {"n": {"from": 100, "to": 200, "by": 100},
                             "n_sims": {"mcpower": 10000, "best": 1000, "naive": 100, "tool": 500},
                             "target_power": 0.80}},
        "cases": [
            {"id": "c1", "family": "ols", "formula": "y = x1", "effects": "x1=0.2", "targets": ["x1"], "tool": "simglm"},
            {"id": "c2", "family": "ols", "formula": "y = x1", "effects": "x1=0.2", "targets": ["x1"], "tool": None},
        ],
    }
    cj = tmp_path / "cases.json"
    cj.write_text(_json.dumps(cases_doc))
    fixture_cases = load_cases(cj)
    py = _mk_results(tmp_path, "py.json", "py", [
        _row("c1", "py", "mcpower_find_power", 100, 10000, 0.1),    # 1e-5  (fastest @100)
        _row("c1", "py", "mcpower_find_power", 200, 10000, 0.2),    # 2e-5  (fastest @200)
        _row("c1", "py", "loop_naive", 100, 100, 0.1),              # 1e-3 -> ratio 100
        _row("c1", "py", "loop_naive", 200, 100, 0.1),              # 1e-3 -> ratio 50
        _row("c2", "py", "mcpower_find_power", 100, 10000, 0.1),    # cliff case: in table, not bars
        _row("c2", "py", "mcpower_find_power", 200, 10000, 0.2),
    ])
    rr = _mk_results(tmp_path, "r.json", "r", [
        _row("c1", "r", "mcpower_find_power", 100, 10000, 0.15),    # ratio 1.5
        _row("c1", "r", "mcpower_find_power", 200, 10000, 0.3),     # ratio 1.5
        _row("c1", "r", "tool_simglm", 100, 500, 0.25),             # 5e-4 -> ratio 50
        _row("c1", "r", "tool_simglm", 200, 500, 0.25),             # 5e-4 -> ratio 25
    ])
    _, _, series, tool_names, _ = combine(py, rr)
    assert tool_names["c1"] == "simglm"
    agg, coverage = aggregate(series, fixture_cases)
    assert round(agg["ols"]["mcpower:py"], 6) == 1.0
    assert round(agg["ols"]["mcpower:r"], 6) == 1.5
    assert round(agg["ols"]["loop_naive:py"], 4) == 70.7107         # sqrt(100*50)
    assert round(agg["ols"]["simglm:r"], 4) == 35.3553              # sqrt(50*25), own covered cases
    assert "1/2" in coverage["ols"] and "simglm 1" in coverage["ols"]


def test_method_id_consistency():
    from harness import RECORDED_METHODS
    from combine import KNOWN_METHODS, TOOL_METHODS, FSS_METHOD
    # the `tool` selector expands to the resolved tool ids; everything recorded
    # must be known to combine and vice versa. find_sample_size is recorded but
    # is a different unit (one row per case), routed to `fss`, not `series`.
    assert FSS_METHOD in RECORDED_METHODS
    expanded = (set(RECORDED_METHODS) - {"tool", FSS_METHOD}) | set(TOOL_METHODS)
    assert expanded == set(KNOWN_METHODS)


def test_fss_summary_grid_vs_grid(tmp_path):
    import json as _json
    from cases import load_cases
    from combine import combine, fss_summary
    cases_doc = {
        "defaults": {"ols": {"n": {"from": 100, "to": 200, "by": 100},
                             "n_sims": {"mcpower": 10000, "best": 1000, "naive": 100, "tool": 500},
                             "target_power": 0.80}},
        "cases": [{"id": "c1", "family": "ols", "formula": "y = x1", "effects": "x1=0.2",
                   "targets": ["x1"], "tool": "simglm"}],
    }
    cj = tmp_path / "cases.json"
    cj.write_text(_json.dumps(cases_doc))
    fss_row = {"case_id": "c1", "family": "ols", "lang": "py",
               "method": "mcpower_find_sample_size", "n": 100, "n_sims": 10000,
               "time_s": 0.15, "per_sim_s": 1.5e-5, "power": [[0.81], [0.95]]}
    py = _mk_results(tmp_path, "py.json", "py", [
        _row("c1", "py", "mcpower_find_power", 100, 10000, 0.1),    # per_sim 1e-5
        _row("c1", "py", "mcpower_find_power", 200, 10000, 0.2),    # per_sim 2e-5
        fss_row,
    ])
    rr = _mk_results(tmp_path, "r.json", "r", [])
    _, _, series, _, fss = combine(py, rr)
    assert fss[("c1", "py")]["power"] == [[0.81], [0.95]]           # full curve saved
    rows = fss_summary(fss, series, load_cases(cj))
    assert len(rows) == 1 and rows[0]["n_star"] == 100
    assert rows[0]["fss"]["py"] == 0.15
    assert round(rows[0]["grid"]["py"], 6) == 0.3                   # 10000 * (1e-5 + 2e-5)
    assert "r" not in rows[0]["fss"]
