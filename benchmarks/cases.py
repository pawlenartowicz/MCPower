"""Load benchmark_cases.json and resolve each case against its family defaults."""
from __future__ import annotations
import json
from dataclasses import dataclass, field


@dataclass
class Case:
    id: str
    family: str            # "ols" | "logit" | "lme"
    formula: str
    effects: str
    targets: list[str]
    n_grid: list[int]
    n_sims: dict           # {"mcpower":int,"best":int,"naive":int,"tool":int}
    target_power: float
    variable_types: dict = field(default_factory=dict)
    cluster: dict | None = None
    tool: str | None = None            # "simr" | "superpower" | "simglm" | None (cliff)
    correlations: str | None = None    # passed verbatim to set_correlations()
    baseline_p: float | None = None
    max_failed_frac: float | None = None


def load_cases(path) -> list[Case]:
    doc = json.loads(open(path).read())
    defaults = doc["defaults"]
    out: list[Case] = []
    for c in doc["cases"]:
        d = defaults[c["family"]]
        n = c.get("n", d["n"])                          # per-case grid override
        out.append(
            Case(
                id=c["id"],
                family=c["family"],
                formula=c["formula"],
                effects=c["effects"],
                targets=c["targets"],
                n_grid=list(range(n["from"], n["to"] + 1, n["by"])),
                n_sims={**d["n_sims"], **c.get("n_sims", {})},   # key-level merge
                target_power=d["target_power"],
                variable_types=c.get("variable_types", {}),
                cluster=c.get("cluster"),
                tool=c["tool"],                          # explicit on every case; KeyError = bad JSON
                correlations=c.get("correlations"),
                baseline_p=c.get("baseline_p", d.get("baseline_p")),
                max_failed_frac=c.get("max_failed_frac", d.get("max_failed_frac")),
            )
        )
    return out
