import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from power_agreement import agree_z, summarise  # noqa: E402


def test_agree_z_known_value():
    assert agree_z(5500, 10000, 5000, 10000) == pytest.approx(7.0888, abs=1e-4)


def test_agree_z_equal_powers_is_zero():
    assert agree_z(500, 1000, 500, 1000) == 0


def test_summarise_counts_flags_and_mean_z():
    pts = [{"case": "c", "family": "ols", "comparator": "loop", "z": z,
            "mc_power": 0.5, "ref_power": 0.5} for z in (0, 2, -2, 0.5)]
    (row,) = summarise(pts)
    assert row["n_flagged"] == 2
    assert row["expected_flagged"] == pytest.approx(0.4)
    assert row["mean_z"] == pytest.approx(0.125)
    assert row["se_mean_z"] == pytest.approx(0.5)


def _rec(method, power, n_sims, targets):
    return {"case_id": "c", "family": "ols", "method": method, "n": 100, "n_sims": n_sims,
            "power": power, "targets": targets}


def test_output_names_carry_input_stem(tmp_path):
    doc = {"meta": {"packages": {"mcpower": "1.4.1"}},
           "records": [_rec("mcpower_find_power", [0.5, 0.8], 1000, ["a", "b"]),
                       _rec("loop_best", [0.8, 0.5], 1000, ["b", "a"])]}
    (tmp_path / "r.json").write_text(json.dumps(doc))
    other = tmp_path / "agreement_r_points.csv"
    other.write_text("keep")
    subprocess.run([sys.executable, str(HERE / "power_agreement.py"), "--in", str(tmp_path / "r.json")],
                   check=True, capture_output=True)
    assert (tmp_path / "r_points.csv").exists() and (tmp_path / "r_summary.csv").exists()
    assert other.read_text() == "keep"
    lines = (tmp_path / "r_points.csv").read_text().splitlines()
    assert len(lines) == 3 and all(l.endswith(",1.4.1") for l in lines[1:])


def test_pairs_by_target_name_and_handles_short_power(tmp_path):
    doc = {"meta": {"packages": {"mcpower": "1.4.1"}},
           "records": [_rec("mcpower_find_power", [0.5, 0.8], 1000, ["a", "b"]),
                       _rec("tool_simr", [0.5], 200, ["a", "b"])]}
    (tmp_path / "s.json").write_text(json.dumps(doc))
    subprocess.run([sys.executable, str(HERE / "power_agreement.py"), "--in", str(tmp_path / "s.json")],
                   check=True, capture_output=True)
    rows = (tmp_path / "s_points.csv").read_text().splitlines()
    assert len(rows) == 2 and rows[1].startswith("c,ols,a,100,simr,0.5,0.5")
