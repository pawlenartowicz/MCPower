"""D2 — host-side type detection: detect_column_types against shared golden fixture."""
import json
from pathlib import Path

import numpy as np
import pytest

from mcpower.data.upload import detect_column_types

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "upload_type_detection.json"


def build_array_from_fixture(fx: dict) -> tuple:
    """Materialise (arr_60_rows, names) from the fixture spec."""
    n_rows = fx["n_rows"]
    rng = np.random.default_rng(42)
    arrays = []
    names = []

    for col in fx["columns"]:
        names.append(col["name"])

        sample_key = col.get("sample")
        if sample_key == "continuous_60_distinct":
            # 60 fully distinct floats (e.g. from a normal draw with seed)
            col_data = rng.normal(size=n_rows).astype(object)
        elif sample_key == "20_distinct_over_60_rows_ratio_3":
            # 20 distinct numeric values spread over 60 rows → ratio = 3 < 15 → continuous
            # (also 20 > max_factor_k_soft=7, so doubly continuous)
            vals = np.arange(20, dtype=float)
            col_data = np.tile(vals, 3).astype(object)  # 60 rows total
        else:
            # Use the explicit values list, repeated/tiled to fill n_rows
            raw = col["values"]
            if len(raw) < n_rows:
                repeats = -(-n_rows // len(raw))  # ceiling div
                raw = (raw * repeats)[:n_rows]
            col_data = np.array(raw, dtype=object)[:n_rows]

        arrays.append(col_data)

    arr = np.column_stack(arrays)  # shape (n_rows, n_cols)
    return arr, names


def test_golden_fixture_exists():
    assert GOLDEN_PATH.exists(), f"Golden fixture not found: {GOLDEN_PATH}"


def test_all_columns_match_fixture():
    fx = json.loads(GOLDEN_PATH.read_text())
    arr, names = build_array_from_fixture(fx)
    max_k = fx["max_factor_k_soft"]
    max_ratio = fx["max_factor_ratio"]

    types, labels_list = detect_column_types(arr, names, max_k, max_ratio)

    assert len(types) == len(fx["columns"])
    assert len(labels_list) == len(fx["columns"])

    for i, col_spec in enumerate(fx["columns"]):
        expect_type = col_spec["expect"]
        assert types[i] == expect_type, (
            f"Column {col_spec['name']!r}: expected {expect_type!r}, got {types[i]!r}"
        )
        if "expect_levels" in col_spec:
            assert labels_list[i] == col_spec["expect_levels"], (
                f"Column {col_spec['name']!r}: expected levels {col_spec['expect_levels']!r}, "
                f"got {labels_list[i]!r}"
            )


def test_binary_detection():
    """Exactly 2 distinct numeric values → binary."""
    arr = np.array([[0], [1], [0], [1], [1]], dtype=object)
    types, labels = detect_column_types(arr, ["x"], 7, 15)
    assert types[0] == "binary"
    assert labels[0] == []


def test_string_column_is_factor():
    """Non-numeric string column → factor regardless of distinct count."""
    vals = np.array([["cat"], ["dog"], ["cat"], ["bird"], ["dog"]], dtype=object)
    types, labels = detect_column_types(vals, ["pet"], 7, 15)
    assert types[0] == "factor"
    assert labels[0] == ["bird", "cat", "dog"]  # sorted


def test_numeric_many_distinct_is_continuous():
    """8 distinct numeric values (> max_k=7) → continuous."""
    arr = np.arange(1, 9, dtype=float).reshape(-1, 1).astype(object)
    types, labels = detect_column_types(arr, ["z"], 7, 15)
    assert types[0] == "continuous"


def test_numeric_few_distinct_enough_rows_is_factor():
    """3 distinct values over 60 rows → ratio 20 ≥ 15 → factor.
    Integer-valued floats (1.0, 2.0, 3.0) are rendered without decimals."""
    vals = np.tile([1.0, 2.0, 3.0], 20)  # 60 rows, 3 distinct
    arr = vals.reshape(-1, 1).astype(object)
    types, labels = detect_column_types(arr, ["x"], 7, 15)
    assert types[0] == "factor"
    assert labels[0] == ["1", "2", "3"]


def test_numeric_few_distinct_too_sparse_is_continuous():
    """3 distinct values over 6 rows → ratio 2 < 15 → continuous.
    (below max_k_soft=7 but ratio guard fails)."""
    vals = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0], dtype=object)
    arr = vals.reshape(-1, 1)
    types, labels = detect_column_types(arr, ["x"], 7, 15)
    assert types[0] == "continuous"


def test_factor_labels_sorted_and_deduped():
    """Repeated string values produce sorted, deduplicated labels."""
    raw = np.array(["Japan", "USA", "Europe", "USA", "Japan", "Europe"] * 10, dtype=object)
    arr = raw.reshape(-1, 1)
    types, labels = detect_column_types(arr, ["origin"], 7, 15)
    assert types[0] == "factor"
    assert labels[0] == ["Europe", "Japan", "USA"]
