"""Merge per-language {meta, records} results into a coverage-aware per-(case,n)
table, per-family geometric-mean aggregates, a summary footer, a
find_sample_size grid-search comparison, and two grouped bar charts. All
cross-method comparisons are per-sim normalized. Run with no arguments: the
result/plot paths are the canonical results/ files (see run_all.sh). Only
results/py.json is required — a missing results/r.json (or the *_1t.json) just
drops those series, so the charts render python-only with no R bars."""
from __future__ import annotations
import json, math, pathlib, statistics, sys

from cases import load_cases

BENCH_DIR = pathlib.Path(__file__).parent
CASES_PATH = BENCH_DIR / "benchmark_cases.json"
# Hardcoded canonical paths — combine.py takes no arguments. py.json is the only
# required input; the rest are optional (skipped when absent).
RESULTS_DIR = BENCH_DIR / "results"
PY_JSON = RESULTS_DIR / "py.json"
R_JSON = RESULTS_DIR / "r.json"
PY_1T_JSON = RESULTS_DIR / "py_1t.json"
R_1T_JSON = RESULTS_DIR / "r_1t.json"
JL_JSON = RESULTS_DIR / "jl.json"          # competitor-only loops (no engine/fss/tool, no _1t twin)
PLOT_FP = RESULTS_DIR / "summary_fp.png"
PLOT_FSS = RESULTS_DIR / "summary_fss.png"

TOOL_METHODS = ("tool_simr", "tool_superpower", "tool_simglm")
KNOWN_METHODS = ("mcpower_find_power", "loop_naive", "loop_best") + TOOL_METHODS
TOOL_SHORT = tuple(m.removeprefix("tool_") for m in TOOL_METHODS)  # split series — one bar per tool
# One row per case (n = recommended n, n_sims = the one call's sim count —
# the call evaluates the full grid from shared draws); different unit from the
# per-(case,n) grid rows, so kept out of `series`.
FSS_METHOD = "mcpower_find_sample_size"

# method -> color, identical across every chart this benchmark emits;
# py/R variants of a method share the hue (hatch marks the R bar)
METHOD_COLORS = {
    "mcpower_fss": "#08519c",   # the find_sample_size hero — darkest of the mcpower blues
    "mcpower":    "#1f77b4",
    "mcpower-1t": "#6baed6",
    "simr":       "#17becf",
    "superpower": "#9467bd",
    "simglm":     "#2ca02c",
    "loop_best":  "#ff7f0e",
    "loop_naive": "#d62728",
}

FAMILY_LABEL = {"ols": "OLS", "logit": "GLM", "lme": "LME", "glmm": "GLMM"}

# Bar order for the two charts (_draw_bars packs left-to-right, skipping any
# series a family lacks). The fss chart leads with the find_sample_size pair so
# the hero sits at the 1x baseline, then the competition — the dedicated tools
# and DIY loops. It deliberately omits mcpower's own find_power grid: that
# within-engine "specially adapted" win (≈2-7x) lives on a different scale and
# is reported per-case in print_fss_table, so it would only squash this chart.
SUMMARY_ORDER = ["mcpower:py", "mcpower:r",
                 "simr:r", "superpower:r", "simglm:r",
                 "loop_best:py", "loop_best:r", "loop_best:jl",
                 "loop_naive:py", "loop_naive:r", "loop_naive:jl"]
FSS_ORDER = ["mcpower_fss:py", "mcpower_fss:r",
             "simr:r", "superpower:r", "simglm:r",
             "loop_best:py", "loop_best:r", "loop_best:jl",
             "loop_naive:py", "loop_naive:r", "loop_naive:jl"]


def _short(method):
    if method in TOOL_METHODS:
        return method.removeprefix("tool_")
    return "mcpower" if method == "mcpower_find_power" else method


def load_results(path):
    doc = json.loads(open(path).read())
    return doc["meta"], doc["records"]


def combine(py_path, r_path=None, py_1t_path=None, r_1t_path=None, jl_path=None):
    """Returns (py_meta, r_meta, series, tool_names, fss).

    series: (case_id, n) -> {"<short>:<lang>": per_sim_s}
    tool_names: case_id -> resolved tool name (from recorded rows)
    fss: (case_id, lang) -> find_sample_size record (one per case)

    r_path None => python-only: r_meta is None and no `:r` series are produced
    (every R column renders as `--`/no bar downstream). The optional *_1t paths
    are results from a `--threads 1` harness run; only their mcpower_find_power
    rows are used, as the `mcpower-1t:<lang>` series (single-core engine — the
    same-cores comparison against the serial tools).

    jl_path None => no `:jl` series (absent jl.json renders identically to today).
    Julia is competitor-only: it records only the loop tiers, so jl rides the main
    series-build loop but NOT the `*_1t` loop (no engine, hence no jl_1t.json).
    """
    py_meta, py_rows = load_results(py_path)
    r_meta, r_rows = load_results(r_path) if r_path is not None else (None, [])
    jl_meta, jl_rows = load_results(jl_path) if jl_path is not None else (None, [])
    scale = py_meta.get("n_sims_scale") or 1.0
    if r_meta is not None and (r_meta.get("n_sims_scale") or 1.0) != scale:
        print(f"WARNING: n_sims_scale differs between runs (py={scale:g}, "
              f"r={(r_meta.get('n_sims_scale') or 1.0):g}) — per-sim times are "
              f"not comparable across scales", file=sys.stderr)
    if jl_meta is not None and (jl_meta.get("n_sims_scale") or 1.0) != scale:
        print(f"WARNING: n_sims_scale differs between runs (py={scale:g}, "
              f"jl={(jl_meta.get('n_sims_scale') or 1.0):g}) — per-sim times are "
              f"not comparable across scales", file=sys.stderr)
    series, tool_names, fss = {}, {}, {}
    for lang, rows in (("py", py_rows), ("r", r_rows), ("jl", jl_rows)):
        for row in rows:
            if row["method"] == FSS_METHOD:
                fss[(row["case_id"], lang)] = row
                continue
            key = (row["case_id"], row["n"])
            series.setdefault(key, {})[f"{_short(row['method'])}:{lang}"] = row["per_sim_s"]
            if row["method"] in TOOL_METHODS:
                tool_names[row["case_id"]] = row["method"].removeprefix("tool_")
    for lang, path in (("py", py_1t_path), ("r", r_1t_path)):
        if path is None:
            continue
        meta_1t, rows = load_results(path)
        if meta_1t.get("threads_mode") != "1":
            print(f"WARNING: {path}: threads_mode is "
                  f"{meta_1t.get('threads_mode')!r}, expected '1'", file=sys.stderr)
        if (meta_1t.get("n_sims_scale") or 1.0) != scale:
            print(f"WARNING: {path}: n_sims_scale "
                  f"{(meta_1t.get('n_sims_scale') or 1.0):g} differs from the "
                  f"main py run ({scale:g})", file=sys.stderr)
        for row in rows:
            if row["method"] == "mcpower_find_power":
                series.setdefault((row["case_id"], row["n"]), {})[f"mcpower-1t:{lang}"] = row["per_sim_s"]
    return py_meta, r_meta, series, tool_names, fss


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def aggregate(series, cases):
    """(family -> {series_key: geomean ratio}, family -> coverage annotation).

    Pinned: every method is normalized to mcpower:py = 1 at the same (case,n)
    BEFORE any averaging -> geomean over the case's n grid -> geomean over the
    family's tool-covered subset. Tools are split series: each tool's bar uses
    only the cases that tool covers (each compared to mcpower on its own cases);
    every other bar uses the whole tool-covered subset. Points without an
    mcpower:py measurement cannot be normalized and are skipped.
    """
    by_case = {}
    for (cid, _n), times in series.items():
        base = times.get("mcpower:py")
        if not base:
            continue
        for k, t in times.items():
            by_case.setdefault(cid, {}).setdefault(k, []).append(t / base)
    case_ratio = {cid: {k: _geomean(v) for k, v in d.items()} for cid, d in by_case.items()}
    tool_of = {c.id: c.tool for c in cases}

    agg, coverage = {}, {}
    for fam in ("ols", "logit", "lme", "glmm"):
        fam_cases = [c for c in cases if c.family == fam]
        if not fam_cases:
            continue
        subset = [c.id for c in fam_cases if c.tool and c.id in case_ratio]
        keys = sorted({k for cid in subset for k in case_ratio[cid]})
        agg[fam] = {}
        for k in keys:
            base = k.split(":")[0]
            sub_k = ([cid for cid in subset if tool_of.get(cid) == base]
                     if base in TOOL_SHORT else subset)
            missing = [cid for cid in sub_k if k not in case_ratio[cid]]
            if missing:
                print(f"WARNING: {fam}: series {k!r} missing from covered case(s) "
                      f"{', '.join(missing)} — its bar uses "
                      f"{len(sub_k) - len(missing)}/{len(sub_k)} cases", file=sys.stderr)
            vals = [case_ratio[cid][k] for cid in sub_k if k in case_ratio[cid]]
            if vals:
                agg[fam][k] = _geomean(vals)
        counts = {}
        for c in fam_cases:
            if c.tool:
                counts[c.tool] = counts.get(c.tool, 0) + 1
        per_tool = ", ".join(f"{t} {n}" for t, n in sorted(counts.items()))
        coverage[fam] = (f"tool coverage: {len(subset)}/{len(fam_cases)} "
                         f"{FAMILY_LABEL[fam]} cases ({per_tool})")
    return agg, coverage


def print_meta(meta):
    pk = ", ".join(f"{k} {v}" for k, v in meta["packages"].items())
    scale = meta.get("n_sims_scale") or 1.0
    preview = f"  PREVIEW sims×{scale:g}" if scale != 1.0 else ""
    print(f"[{meta['lang']}] {meta['timestamp_utc']}  {meta['os']}  {meta['cpu_model']}  "
          f"{meta['cores_physical']}c/{meta['cores_logical']}t  "
          f"threads={meta['threads_mode']}  v{meta.get('lang_version', '?')}{preview}  |  {pk}")


def print_table(series, tool_names):
    hdr = (f"{'case':>18} {'n':>5} | {'mc-py':>10} {'mc-r':>10} {'py/r':>6} | "
           f"{'tool':>11} {'tool×r':>7} | {'naive×py':>9} {'naive×r':>9} {'naive×jl':>9} "
           f"{'best×py':>8} {'best×r':>8} {'best×jl':>8}")
    print(hdr)
    print("-" * len(hdr))
    for (c, n) in sorted(series):
        t = series[(c, n)]
        def ratio(a, b):
            return (a / b) if (a and b) else None
        def e(x, w=10):
            return f"{x:>{w}.2e}" if x is not None else f"{'--':>{w}}"
        def f(x, w):
            return f"{x:>{w}.1f}" if x is not None else f"{'--':>{w}}"
        mc_py, mc_r = t.get("mcpower:py"), t.get("mcpower:r")
        tool = t.get(f"{tool_names.get(c)}:r")
        tool_lbl = tool_names.get(c, "none") if tool is not None else "none"
        # jl loops are normalized to the py engine (no jl engine; py/r per-sim ≈ equal).
        print(f"{c:>18} {n:>5} | {e(mc_py)} {e(mc_r)} {f(ratio(mc_r, mc_py), 6)} | "
              f"{tool_lbl:>11} {f(ratio(tool, mc_r), 7)} | "
              f"{f(ratio(t.get('loop_naive:py'), mc_py), 9)} "
              f"{f(ratio(t.get('loop_naive:r'), mc_r), 9)} "
              f"{f(ratio(t.get('loop_naive:jl'), mc_py), 9)} "
              f"{f(ratio(t.get('loop_best:py'), mc_py), 8)} "
              f"{f(ratio(t.get('loop_best:r'), mc_r), 8)} "
              f"{f(ratio(t.get('loop_best:jl'), mc_py), 8)}")


def print_aggregates(agg, coverage):
    print("\nPer-family aggregate (per-sim time ratio, mcpower:py = 1, geometric means"
          " over the tool-covered subset; each tool over its own covered cases):")
    for fam, d in agg.items():
        print(f"\n  {FAMILY_LABEL[fam]} — {coverage[fam]}")
        for k in sorted(d, key=d.get):
            print(f"    {k:>14}: {d[k]:>10.2f}")


def print_footer(series, tool_names, cases):
    print("\nPer-family summary (full power curve projected at 10,000 sims/point,"
          " from per-sim times):")
    for fam in ("ols", "logit", "lme", "glmm"):
        fam_ids = {c.id for c in cases if c.family == fam}
        proj, speed = {}, {}
        for (cid, _n), t in series.items():
            if cid not in fam_ids:
                continue
            for k, v in t.items():
                if k.split(":")[0] in TOOL_SHORT:
                    k = f"tool:{k.split(':')[1]}"     # bucket split tools for the medians
                proj.setdefault(k, {}).setdefault(cid, 0.0)
                proj[k][cid] += v * 10000.0
                base = t.get("mcpower:py")            # mcpower:py is the reference
                if base and not k.startswith("mcpower"):
                    speed.setdefault(k, []).append(v / base)
        if not proj:
            continue
        def med_curve(k):
            d = proj.get(k)
            return statistics.median(d.values()) if d else None
        def med_speed(k):
            return statistics.median(speed[k]) if k in speed else None
        def s(x):
            return f"{x:,.0f} s" if x is not None else "--"
        tools = sorted({tool_names[cid] for cid in fam_ids if cid in tool_names})
        tool_lbl = "/".join(tools) if tools else "none"
        print(f"  {FAMILY_LABEL[fam]}: MCPower {med_curve('mcpower:py') or 0:.1f} s (py) / "
              f"{med_curve('mcpower:r') or 0:.1f} s (r) — best loop ~{s(med_curve('loop_best:r'))}, "
              f"naive loop ~{s(med_curve('loop_naive:r'))}, tool ({tool_lbl}) ~{s(med_curve('tool:r'))}, "
              f"jl best ~{s(med_curve('loop_best:jl'))}, jl naive ~{s(med_curve('loop_naive:jl'))}")
        sp = ", ".join(f"{k} {med_speed(k):.0f}×" for k in
                       ("loop_best:r", "loop_naive:r", "tool:r", "loop_best:jl", "loop_naive:jl")
                       if med_speed(k))
        print(f"       median per-sim slowdown vs MCPower (py): {sp}")


def fss_summary(fss, series, cases):
    """Per-case full-grid-vs-full-grid rows: the measured one find_sample_size
    call vs the summed find_power grid of the same language — same engine, same
    grid, same sims/point, so the ratio isolates the one-call advantage
    (shared draws across the grid + single dispatch)."""
    by_id = {c.id: c for c in cases}
    rows = []
    for cid in sorted({cid for cid, _ in fss}):
        case = by_id.get(cid)
        if case is None:
            continue
        row = {"case_id": cid, "n_star": None, "fss": {}, "grid": {}}
        for lang in ("py", "r"):
            rec = fss.get((cid, lang))
            if rec is None:
                continue
            row["fss"][lang] = rec["time_s"]
            if row["n_star"] is None and rec["n"]:
                row["n_star"] = rec["n"]
            # grid seconds = sims/point x sum of mcpower per-sim times — exact,
            # both tiers run the same scaled mcpower sim count per point.
            s = [series[(cid, n)][f"mcpower:{lang}"] for n in case.n_grid
                 if f"mcpower:{lang}" in series.get((cid, n), {})]
            if s:
                row["grid"][lang] = rec["n_sims"] * sum(s)
        rows.append(row)
    return rows


def print_fss_table(rows):
    if not rows:
        return
    print("\nfind_sample_size vs find_power grid — one call evaluating the full"
          " grid vs summed\nper-point calls (same engine, grid, sims/point;"
          " n* = recommended n, 0 = not achieved):")
    hdr = (f"{'case':>18} {'n*':>5} | {'fss-py(s)':>10} {'grid-py(s)':>11} {'×':>6} | "
           f"{'fss-r(s)':>10} {'grid-r(s)':>11} {'×':>6}")
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        def cell(lang):
            f, g = row["fss"].get(lang), row["grid"].get(lang)
            fs = f"{f:>10.3f}" if f is not None else f"{'--':>10}"
            gs = f"{g:>11.3f}" if g is not None else f"{'--':>11}"
            xs = f"{g / f:>6.1f}" if f and g else f"{'--':>6}"
            return f"{fs} {gs} {xs}"
        print(f"{row['case_id']:>18} {row['n_star'] or 0:>5} | {cell('py')} | {cell('r')}")


# --- find_sample_size grid search (framing a) -------------------------------
# Real-world task: locate the n that hits target power by searching a grid.
# mcpower answers the whole grid in ONE find_sample_size call (budget S sims
# TOTAL, draws shared across the grid); every alternative must rerun a full
# power simulation at each of the G grid points (S per point, S*G total). S =
# the case's full mcpower sims/point, so the alternatives are projected to
# mcpower-grade precision per point — the same per-sim basis as print_footer.

# short keys whose per-point series sum into a full-grid cost (mcpower's own
# naive find_power grid, the DIY loops, and each dedicated tool)
ALT_SHORTS = ("mcpower", "loop_best", "loop_naive") + TOOL_SHORT


def fss_gridsearch(fss, series, cases):
    """Per-case projected grid-search seconds. Returns rows:
    {case_id, family, S, G, grid_s: {"<short>:<lang>": secs},
     total_sims: {...}, n_pts: {"<short>:<lang>": (measured, G)}}.
    The find_sample_size baseline is keyed "mcpower_fss:<lang>".

    Alternative grids sum only the grid points actually measured (a tool that
    crashed at some n contributes fewer points) — conservative: missing points
    can only understate the alternative's full-grid cost, never inflate it.
    n_pts records the coverage so print_fss_gridsearch can flag a short grid."""
    by_id = {c.id: c for c in cases}
    rows = []
    for cid in sorted({cid for cid, _ in fss}):
        case = by_id.get(cid)
        if case is None:
            continue
        S, G = case.n_sims["mcpower"], len(case.n_grid)
        row = {"case_id": cid, "family": case.family, "S": S, "G": G,
               "grid_s": {}, "total_sims": {}, "n_pts": {}}
        # jl included so loop_*:jl grid_s entries are built; the mcpower_fss:jl
        # branch simply finds no jl fss record (Julia has no engine) and is skipped.
        for lang in ("py", "r", "jl"):
            rec = fss.get((cid, lang))
            if rec is not None and rec["n_sims"]:
                # find_sample_size budget is S sims TOTAL (shared draws), not S*G
                row["grid_s"][f"mcpower_fss:{lang}"] = S * rec["time_s"] / rec["n_sims"]
                row["total_sims"][f"mcpower_fss:{lang}"] = S
            for short in ALT_SHORTS:
                pts = [series[(cid, n)][f"{short}:{lang}"] for n in case.n_grid
                       if f"{short}:{lang}" in series.get((cid, n), {})]
                if not pts:
                    continue
                # per_sim already; * S projects each point to S sims, sum = grid
                row["grid_s"][f"{short}:{lang}"] = S * sum(pts)
                row["total_sims"][f"{short}:{lang}"] = S * len(pts)
                row["n_pts"][f"{short}:{lang}"] = (len(pts), G)
        rows.append(row)
    return rows


def aggregate_fss(rows, cases):
    """Per-family geomean of each tier's grid-search time / find_sample_size:py
    at the same case (baseline mcpower_fss:py = 1). Tools over their own covered
    cases, every other tier over the family's tool-covered subset — mirrors
    aggregate(). Returns (agg, coverage)."""
    tool_of = {c.id: c.tool for c in cases}
    fam_of = {c.id: c.family for c in cases}
    by_case = {}
    for row in rows:
        base = row["grid_s"].get("mcpower_fss:py")
        if not base:
            continue
        by_case[row["case_id"]] = {k: t / base for k, t in row["grid_s"].items()}

    agg, coverage = {}, {}
    for fam in ("ols", "logit", "lme", "glmm"):
        fam_ids = [cid for cid in by_case if fam_of.get(cid) == fam]
        subset = [cid for cid in fam_ids if tool_of.get(cid)]
        if not subset:
            continue
        keys = sorted({k for cid in subset for k in by_case[cid]})
        agg[fam] = {}
        for k in keys:
            base = k.split(":")[0]
            sub = ([cid for cid in subset if tool_of.get(cid) == base]
                   if base in TOOL_SHORT else subset)
            vals = [by_case[cid][k] for cid in sub if k in by_case[cid]]
            if vals:
                agg[fam][k] = _geomean(vals)
        counts = {}
        for cid in subset:
            counts[tool_of[cid]] = counts.get(tool_of[cid], 0) + 1
        per_tool = ", ".join(f"{t} {n}" for t, n in sorted(counts.items()))
        coverage[fam] = (f"tool coverage: {len(subset)}/{len(fam_ids)} "
                         f"{FAMILY_LABEL[fam]} cases ({per_tool})")
    return agg, coverage


def print_fss_gridsearch(rows, agg, cases):
    """Per-family projected wall-clock to locate n* over the full grid, plus the
    slowdown vs find_sample_size. Seconds are the median across the family's
    cases; x is the geomean ratio from aggregate_fss."""
    if not rows or not agg:
        return
    print("\nFind-sample-size grid search — projected wall-clock to locate n* over"
          "\nthe full grid (framing a: find_sample_size spends S sims TOTAL via shared"
          "\ndraws; every alternative reruns a full power sim at each of the G grid"
          "\npoints, S*G total). Seconds = median across the family's tool-covered"
          "\ncases; x = geomean ratio to find_sample_size:py.")
    # competition only — find_sample_size vs the dedicated tools + DIY loops;
    # the within-MCPower find_power-grid win is reported in print_fss_table
    DISPLAY = ["mcpower_fss:py", "simglm:r", "superpower:r",
               "simr:r", "loop_best:r", "loop_naive:r",
               "loop_best:jl", "loop_naive:jl"]
    LABEL = {"mcpower_fss": "find_sample_size"}
    for fam in ("ols", "logit", "lme", "glmm"):
        if fam not in agg:
            continue
        fam_rows = [r for r in rows if r["family"] == fam]
        Ss = sorted({r["S"] for r in fam_rows})
        Gs = sorted({r["G"] for r in fam_rows})
        S_lbl = f"{Ss[0]:,}" if len(Ss) == 1 else f"{Ss[0]:,}-{Ss[-1]:,}"
        G_lbl = f"{Gs[0]}" if len(Gs) == 1 else f"{Gs[0]}-{Gs[-1]}"
        print(f"\n  {FAMILY_LABEL[fam]}  (S={S_lbl} sims/point, G={G_lbl} grid points)")
        for k in DISPLAY:
            if k not in agg[fam]:
                continue
            short, lang = k.split(":")
            secs = [r["grid_s"][k] for r in fam_rows if k in r["grid_s"]]
            if not secs:
                continue
            lbl = LABEL.get(short, f"{short} grid")
            sims = "S total" if short == "mcpower_fss" else "S*G"
            # tools can miss grid points (simglm at small n) — surface, don't hide
            pts = [r["n_pts"][k] for r in fam_rows if k in r["n_pts"]]
            cov = ""
            if pts and any(m < g for m, g in pts):
                cov = f"  [partial: {min(m for m, _ in pts)}/{max(g for _, g in pts)} pts min]"
            print(f"    {lbl:<18} ({lang})  {statistics.median(secs):>12,.1f} s  "
                  f"{agg[fam][k]:>8.1f}x  [{sims}]{cov}")


def write_plot(panel, out_path):
    """Standalone grouped-bar chart: grouped per family, log y, fastest = 1
    baseline. `panel` is (agg, coverage, series_order, ylabel, title);
    series_order fixes the left-to-right bar order and ylabel/title caption it.
    Colors from METHOD_COLORS; hatch marks the R variant of a method. Each family
    packs only the series it covers, left-to-right with no empty slots — a tool
    that doesn't apply to a family (simr only on LME, superpower/simglm only on
    their cases) just leaves no bar, not a gap. The single-core mcpower-1t series
    is collected but deliberately not plotted: useful for the same-cores
    analysis, not the headline. Each bar is annotated with its ratio (Nx
    slower)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg, coverage, series_order, ylabel, title = panel
    fams = [f for f in ("ols", "logit", "lme", "glmm") if agg.get(f)]
    keys = [k for k in series_order if any(k in agg[f] for f in fams)]
    width = 0.92 / len(keys)  # fixed bar width so bars stay comparable across families
    fig, ax = plt.subplots(figsize=(14, 6))

    def _mult(v):
        return f"{v:.1f}×" if v < 10 else f"{v:,.0f}×"

    labeled, max_v = set(), 1.0
    for j, f in enumerate(fams):
        present = [k for k in keys if k in agg[f]]  # packed, no gaps; keeps series_order
        for slot, k in enumerate(present):
            method, lang = k.split(":")
            x, y = j + slot * width, agg[f][k]
            max_v = max(max_v, y)
            ax.bar(x, y, width=width, color=METHOD_COLORS[method],
                   hatch={"r": "//", "jl": ".."}.get(lang), edgecolor="black", linewidth=0.4,
                   label=k if k not in labeled else None)
            labeled.add(k)
            ax.annotate(_mult(y), (x, y), textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=8)

    ax.set_yscale("log")
    ax.set_ylim(bottom=0.8, top=max_v * 3)  # headroom for the value labels
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.set_xticks([j + (len([k for k in keys if k in agg[f]]) - 1) * width / 2
                   for j, f in enumerate(fams)])
    ax.set_xticklabels([f"{FAMILY_LABEL[f]}\n{coverage[f]}" for f in fams], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # bars are added per family, so legend handles arrive in first-appearance order
    # (simr is LME-only -> last); reorder to series_order so tools stay grouped
    handles, labels = ax.get_legend_handles_labels()
    rank = {k: i for i, k in enumerate(keys)}
    ordered = sorted(zip(handles, labels), key=lambda hl: rank.get(hl[1], len(keys)))
    ax.legend([h for h, _ in ordered], [l for _, l in ordered], fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cases = load_cases(CASES_PATH)
    # optional inputs: absent file => that language/pass drops out (python-only)
    r_json = R_JSON if R_JSON.exists() else None
    py_1t = PY_1T_JSON if PY_1T_JSON.exists() else None
    r_1t = R_1T_JSON if R_1T_JSON.exists() else None
    jl_json = JL_JSON if JL_JSON.exists() else None
    py_meta, r_meta, series, tool_names, fss = combine(PY_JSON, r_json, py_1t, r_1t, jl_json)
    print_meta(py_meta)
    if r_meta is not None:
        print_meta(r_meta)
    print()
    print_table(series, tool_names)
    agg, coverage = aggregate(series, cases)
    print_aggregates(agg, coverage)
    print_footer(series, tool_names, cases)
    print_fss_table(fss_summary(fss, series, cases))
    fss_rows = fss_gridsearch(fss, series, cases)
    agg_fss, coverage_fss = aggregate_fss(fss_rows, cases)
    print_fss_gridsearch(fss_rows, agg_fss, cases)
    write_plot((agg, coverage, SUMMARY_ORDER,
                "per-sim time, mcpower:py = 1 (log scale)",
                "find_power: per-sim time vs dedicated tools & DIY loops"), PLOT_FP)
    print(f"\nWrote find_power plot to {PLOT_FP}")
    write_plot((agg_fss, coverage_fss, FSS_ORDER,
                "projected grid-search time, find_sample_size:py = 1 (log scale)",
                "find_sample_size grid search: one call vs a full power run at "
                "every grid point"), PLOT_FSS)
    print(f"\nWrote find_sample_size plot to {PLOT_FSS}")
