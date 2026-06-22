"""Power-agreement cross-check: does MCPower's power agree with the dedicated tools?

Run as the final step of run_all.sh (right after combine.py's timing tables), so
a benchmark run reports both speed and this statistical sanity check; also
runnable standalone once results/r.json exists.

The speed benchmark records a `power` array per (case, n, method) as a sanity
signal. This script aligns MCPower's power against each dedicated tool's power
(R run, which is the only one that carries the tool tiers) and the independent
DIY loops, then plots agreement. Tool alignment follows tools_r.R:
  - simglm     -> one value per focal target (MCPower's leading predictors)
  - simr       -> single value for targets[0]   (LME)
The ANOVA cases (superpower tool) are excluded everywhere: superpower reports an
omnibus F-test, not comparable to MCPower's per-coefficient t-tests. The DIY loop
(loop_best) covers every remaining case, so it is the universal comparator and is
drawn on every curve panel — including the cases (ols_correlated, glmm_*) that
have no dedicated tool at all.

Output: results/power_agreement_scatter.png,
        results/power_agreement_curves_{1,2}.png
"""
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = __file__.rsplit("/", 1)[0]
RES = f"{HERE}/results"

cases = {c["id"]: c for c in json.load(open(f"{HERE}/benchmark_cases.json"))["cases"]}
r = json.load(open(f"{RES}/r.json"))["records"]

# Index every method's power by (case, n).
P = defaultdict(dict)          # P[method][(case,n)] = power list
nsims = defaultdict(dict)
for rec in r:
    P[rec["method"]][(rec["case_id"], rec["n"])] = rec["power"]
    nsims[rec["method"]][(rec["case_id"], rec["n"])] = rec["n_sims"]

TOOL_METHODS = {"tool_simglm": "simglm", "tool_simr": "simr"}

# ANOVA / superpower excluded everywhere — omnibus F, not comparable to per-coef t.
ANOVA_CASES = {"anova_2x2", "anova_oneway4"}

# How a tool's power vector maps onto MCPower's full power vector (by index).
# Returns list of (mcpower_index, tool_index, label) pairs for a case.
def alignment(case_id, tool_power):
    case = cases[case_id]
    if tool_for(case_id) == "simr":
        return [(0, 0, case["targets"][0])]  # simr tests targets[0] only
    # simglm: one tool value per focal target, focal targets lead the formula
    return [(i, i, case["targets"][i]) for i in range(len(tool_power))]

def tool_for(case_id):
    return cases[case_id].get("tool")

# ---- Build aligned pairs: (mcpower_power, tool_power, tool, case, n, label) ----
pairs = []
for meth, toolname in TOOL_METHODS.items():
    for (cid, n), tp in P[meth].items():
        mc = P["mcpower_find_power"].get((cid, n))
        if mc is None:
            continue
        for mi, ti, lab in alignment(cid, tp):
            if mi < len(mc) and ti < len(tp):
                pairs.append((mc[mi], tp[ti], toolname, cid, n, lab))

# Independent DIY-loop agreement (loop_best, same focal alignment as simglm-style
# i.e. all targets), for an extra column of evidence.
loop_pairs = []
for (cid, n), lp in P["loop_best"].items():
    if cid in ANOVA_CASES:
        continue
    mc = P["mcpower_find_power"].get((cid, n))
    if mc is None:
        continue
    # loops report power for the focal targets only, in target order.
    k = min(len(lp), len(mc))
    for i in range(k):
        loop_pairs.append((mc[i], lp[i], cid, n))

# ----------------------------- Scatter plot -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

ax = axes[0]
colors = {"simglm": "#1f77b4", "simr": "#2ca02c"}
for tool in colors:
    xs = [p[0] for p in pairs if p[2] == tool]
    ys = [p[1] for p in pairs if p[2] == tool]
    ax.scatter(xs, ys, s=55, alpha=0.8, label=f"{tool} (n={len(xs)})",
               color=colors[tool], edgecolor="black", linewidth=0.4)
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect agreement")
# ±0.05 Monte-Carlo band (tools run far fewer sims -> noisier)
ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="grey", alpha=0.12,
                label="±0.05 MC band")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("MCPower power (10,000 sims)")
ax.set_ylabel("Dedicated tool power")
ax.set_title("MCPower vs dedicated tools\n(per focal effect, R port)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
xs = [p[0] for p in loop_pairs]
ys = [p[1] for p in loop_pairs]
ax.scatter(xs, ys, s=30, alpha=0.5, color="#9467bd", edgecolor="black",
           linewidth=0.3, label=f"loop_best (n={len(xs)})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect agreement")
ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="grey", alpha=0.12,
                label="±0.05 MC band")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("MCPower power (10,000 sims)")
ax.set_ylabel("DIY loop_best power")
ax.set_title("MCPower vs independent DIY simulation loop\n(all focal effects, R port)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(f"{RES}/power_agreement_scatter.png", dpi=130)
print("wrote results/power_agreement_scatter.png")

# --------------------- Per-case power-vs-n curves -----------------------------
# Every non-ANOVA case except glmm_simple, in case-file order (ols -> glm -> lme ->
# glmm). The 12 cases are split across two 2x3 figures for readability. Each panel
# plots the focal target[0]; the dedicated tool is drawn only where one exists, but
# loop_best is the universal comparator and appears on every panel.
CURVE_EXCLUDE = ANOVA_CASES | {"glmm_simple"}
curve_cases = [cid for cid in cases if cid not in CURVE_EXCLUDE]
for part, start in enumerate(range(0, len(curve_cases), 6), start=1):
    chunk = curve_cases[start:start + 6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, cid in zip(axes.flat, chunk):
        case = cases[cid]
        ns = sorted({n for (c, n) in P["mcpower_find_power"] if c == cid})
        mc = [P["mcpower_find_power"][(cid, n)][0] for n in ns]
        ax.plot(ns, mc, "o-", color="#ff7f0e", label="MCPower (10k)", lw=2)
        # dedicated tool, only where one is defined for this case
        tool = tool_for(cid)
        if tool:
            tmeth = next(m for m, t in TOOL_METHODS.items() if t.lower() == tool.lower())
            tn = sorted({n for (c, n) in P[tmeth] if c == cid})
            if tn:
                tv = [P[tmeth][(cid, n)][0] for n in tn]
                nsim_t = nsims[tmeth][(cid, tn[0])]
                ax.plot(tn, tv, "s--", color=colors.get(tool, "#1f77b4"),
                        label=f"{tool} ({nsim_t} sims)", lw=1.5)
        # loop_best independent check — drawn for every case
        ln = sorted({n for (c, n) in P["loop_best"] if c == cid})
        lv = [P["loop_best"][(cid, n)][0] for n in ln]
        ax.plot(ln, lv, "^:", color="#9467bd", alpha=0.7, label="loop_best", lw=1.2)
        ax.set_title(f"{cid}  (focal: {case['targets'][0]})", fontsize=10)
        ax.set_xlabel("sample size n"); ax.set_ylabel("power")
        ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Power vs sample size: MCPower vs dedicated tool vs DIY loop "
                 f"(R port, {part}/2)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f"{RES}/power_agreement_curves_{part}.png", dpi=130)
    print(f"wrote results/power_agreement_curves_{part}.png")

# ------------------------------ Numeric summary -------------------------------
import statistics
print("\n=== Agreement summary (|MCPower - tool|) ===")
for tool in colors:
    diffs = [abs(p[0] - p[1]) for p in pairs if p[2] == tool]
    if diffs:
        print(f"{tool:11s}  n={len(diffs):2d}  mean|Δ|={statistics.mean(diffs):.3f}  "
              f"max|Δ|={max(diffs):.3f}  median|Δ|={statistics.median(diffs):.3f}")
ld = [abs(p[0] - p[1]) for p in loop_pairs]
if ld:
    print(f"{'loop_best':11s}  n={len(ld):2d}  mean|Δ|={statistics.mean(ld):.3f}  "
          f"max|Δ|={max(ld):.3f}  median|Δ|={statistics.median(ld):.3f}")

print("\n=== Largest disagreements vs tools ===")
for p in sorted(pairs, key=lambda x: -abs(x[0] - x[1]))[:8]:
    print(f"  {p[3]:18s} n={p[4]:<4} {p[5]:8s} MCPower={p[0]:.3f} {p[2]}={p[1]:.3f} "
          f"Δ={p[0]-p[1]:+.3f}")
