"""Power-agreement check: does MCPower's power agree with the reference loops and the dedicated tools?

Input is a `{meta, records}` JSON written by harness.R or by the example_*.R
scripts. Each record carries `case_id, family, method, n, n_sims, power,
targets`; `power[i]` belongs to `targets[i]`. A record may hold fewer power
values than targets (simr reports `targets[0]` only); the leading values are
paired with the leading targets. MCPower and its comparator are paired by
target name, never by index.

Statistic, per (case, target, n, comparator):
    z = (x_mc/n_mc - x_ref/n_ref) / sqrt(pt_mc(1-pt_mc)/n_mc + pt_ref(1-pt_ref)/n_ref)
with pt = (x + 1) / (n + 2) in the variances only, and x = round(power * n_sims).
A point is flagged when |z| > qnorm(0.95). Flags are leads to investigate, not a
pass/fail verdict.

Output, beside the input file (<stem> = input file name without extension):
    <stem>_points.csv, <stem>_summary.csv,
    <stem>_scatter.png, <stem>_curves_<k>.png
"""
import argparse
import csv
import json
import math
import os
import statistics
from collections import defaultdict

Z_FLAG = statistics.NormalDist().inv_cdf(0.95)
ALPHA = 0.1                                 # per-test flag rate, matches Z_FLAG

COMPARATORS = {"loop_best": "loop", "tool_simglm": "simglm",
               "tool_superpower": "superpower", "tool_simr": "simr"}

POINT_COLS = ["case", "family", "target", "n", "comparator", "mc_power", "ref_power",
              "mc_n_sims", "ref_n_sims", "z", "flagged", "mcpower_version"]
SUMMARY_COLS = ["case", "family", "comparator", "n_points", "n_flagged",
                "expected_flagged", "mean_z", "se_mean_z", "max_abs_diff"]


def agree_z(x_mc, n_mc, x_ref, n_ref):
    p_mc, p_ref = x_mc / n_mc, x_ref / n_ref
    t_mc, t_ref = (x_mc + 1) / (n_mc + 2), (x_ref + 1) / (n_ref + 2)
    se = math.sqrt(t_mc * (1 - t_mc) / n_mc + t_ref * (1 - t_ref) / n_ref)
    return (p_mc - p_ref) / se


def build_points(doc):
    """One point per (case, target, n, comparator), paired by target name."""
    version = doc.get("meta", {}).get("packages", {}).get("mcpower", "")
    recs = {}
    for r in doc["records"]:
        recs[(r["method"], r["case_id"], r["n"])] = r
    points = []
    for (method, cid, n), ref in recs.items():
        if method not in COMPARATORS:
            continue
        mc = recs.get(("mcpower_find_power", cid, n))
        if mc is None:
            continue
        mc_by_name = dict(zip(mc["targets"], mc["power"]))
        for target, ref_power in zip(ref["targets"], ref["power"]):
            if target not in mc_by_name:
                raise KeyError(f"{cid} n={n}: MCPower record has no target '{target}'")
            mc_power = mc_by_name[target]
            z = agree_z(round(mc_power * mc["n_sims"]), mc["n_sims"],
                        round(ref_power * ref["n_sims"]), ref["n_sims"])
            points.append({"case": cid, "family": ref["family"], "target": target, "n": n,
                           "comparator": COMPARATORS[method], "mc_power": mc_power,
                           "ref_power": ref_power, "mc_n_sims": mc["n_sims"],
                           "ref_n_sims": ref["n_sims"], "z": z, "flagged": abs(z) > Z_FLAG,
                           "mcpower_version": version})
    points.sort(key=lambda p: (p["case"], p["comparator"], p["target"], p["n"]))
    return points


def summarise(points):
    groups = defaultdict(list)
    for p in points:
        groups[(p["case"], p["family"], p["comparator"])].append(p)
    rows = []
    for (case, family, comparator), pts in groups.items():
        zs = [p["z"] for p in pts]
        diffs = [abs(p["mc_power"] - p["ref_power"]) for p in pts if "mc_power" in p and "ref_power" in p]
        rows.append({"case": case, "family": family, "comparator": comparator,
                     "n_points": len(pts),
                     "n_flagged": sum(abs(z) > Z_FLAG for z in zs),
                     "expected_flagged": ALPHA * len(pts),
                     "mean_z": statistics.mean(zs),
                     "se_mean_z": 1 / math.sqrt(len(pts)),
                     "max_abs_diff": max(diffs) if diffs else float("nan")})
    return rows


def write_csv(path, cols, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def plot(points, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"loop": "#9467bd", "simglm": "#1f77b4", "superpower": "#d62728", "simr": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(7, 6.5))
    for comp, col in colors.items():
        pts = [p for p in points if p["comparator"] == comp]
        if pts:
            ax.scatter([p["mc_power"] for p in pts], [p["ref_power"] for p in pts], s=35, alpha=0.7,
                       color=col, edgecolor="black", linewidth=0.3, label=f"{comp} (n={len(pts)})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("MCPower power"); ax.set_ylabel("Comparator power")
    ax.set_title("MCPower vs comparators, one point per (case, target, n)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}_scatter.png", dpi=130)
    plt.close(fig)

    # Power vs n for each case's first target, MCPower against every comparator.
    cases = []
    for p in points:
        if p["case"] not in cases:
            cases.append(p["case"])
    for part, start in enumerate(range(0, len(cases), 6), start=1):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        for ax, cid in zip(axes.flat, cases[start:start + 6]):
            cpts = [p for p in points if p["case"] == cid]
            focal = cpts[0]["target"]
            mc = sorted({(p["n"], p["mc_power"]) for p in cpts if p["target"] == focal})
            ax.plot([m[0] for m in mc], [m[1] for m in mc], "o-", color="#ff7f0e", lw=2, label="MCPower")
            for comp in sorted({p["comparator"] for p in cpts}):
                rp = sorted((p["n"], p["ref_power"]) for p in cpts
                            if p["comparator"] == comp and p["target"] == focal)
                if rp:
                    ax.plot([r[0] for r in rp], [r[1] for r in rp], "s--", color=colors[comp],
                            lw=1.3, label=comp)
            ax.set_title(f"{cid}  (target: {focal})", fontsize=10)
            ax.set_xlabel("sample size n"); ax.set_ylabel("power")
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{out_prefix}_curves_{part}.png", dpi=130)
        plt.close(fig)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", default=f"{here}/results/r.json", help="harness results JSON")
    args = ap.parse_args()

    out_prefix = os.path.join(os.path.dirname(os.path.abspath(args.infile)),
                              os.path.splitext(os.path.basename(args.infile))[0])
    with open(args.infile) as f:
        doc = json.load(f)
    points = build_points(doc)
    summary = summarise(points)
    write_csv(f"{out_prefix}_points.csv", POINT_COLS, points)
    write_csv(f"{out_prefix}_summary.csv", SUMMARY_COLS, summary)
    plot(points, out_prefix)

    print(f"{'case':18s} {'comparator':11s} {'points':>6s} {'flagged':>7s} {'expected':>8s} {'mean z':>7s} {'se':>5s}")
    for s in summary:
        print(f"{s['case']:18s} {s['comparator']:11s} {s['n_points']:6d} {s['n_flagged']:7d} "
              f"{s['expected_flagged']:8.1f} {s['mean_z']:7.2f} {s['se_mean_z']:5.2f}")
    print(f"wrote {out_prefix}_points.csv, {out_prefix}_summary.csv")


if __name__ == "__main__":
    main()
