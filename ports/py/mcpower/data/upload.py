"""Normalises user-supplied data (list, numpy array, DataFrame, dict, file path) into column-major lists with column names for engine ingestion."""

import csv
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple


def value_to_label(v: Any) -> str:
    """Render a raw factor level value as its canonical string label.

    Single source of the factor-label format used across the upload path
    (``detect_column_types`` and the spec-build upload block) and consumed
    verbatim into dummy names like ``cyl[6]`` / ``origin[Japan]``. Integer-valued
    numerics render without a decimal (``4.0`` -> ``"4"``); non-numeric values
    pass through as ``str``.
    """
    try:
        fv = float(v)
    except (ValueError, TypeError):
        return str(v)
    return str(int(fv)) if fv == int(fv) else str(fv)


def _read_file_to_dict(path: Path) -> Tuple[dict, List[str]]:
    """Read a CSV/TSV file into a column dict using stdlib csv.

    Returns (col_dict, col_names) where values are strings or floats.
    """
    suffix = path.suffix.lower()
    # splitlines() normalises all line endings, so no newline= arg is needed
    # (and Path.read_text(newline=) is Python 3.13+ only — it broke on 3.12 CI).
    text = path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()

    if not lines:
        return {}, []

    # Detect delimiter
    if suffix == ".tsv":
        delimiter = "\t"
    elif suffix == ".csv":
        delimiter = ","
    else:
        # Sniff from the first non-empty line
        sample = "\n".join(lines[:5])
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","

    reader = csv.reader(lines, delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return {}, []

    header = rows[0]
    data_rows = rows[1:]

    col_dict: dict = {name: [] for name in header}
    for row in data_rows:
        for i, name in enumerate(header):
            raw = row[i] if i < len(row) else ""
            try:
                col_dict[name].append(float(raw))
            except (ValueError, TypeError):
                col_dict[name].append(raw)

    return col_dict, header


def _is_module(obj: Any, root: str) -> bool:
    """True when *obj*'s defining module's top package is *root* — lazy duck-typing
    so neither numpy nor pandas is ever imported by this package."""
    return type(obj).__module__.split(".")[0] == root


def normalize_upload_input(
    data,
    columns: Optional[List[str]] = None,
) -> Tuple[List[List[Any]], List[str]]:
    """Convert user-supplied data into column-major lists with column names.

    Accepted inputs:
        - str or os.PathLike (.csv / .tsv): read with stdlib csv, header row → names
        - pandas DataFrame: detected by module name (never forces pandas import)
        - numpy ndarray: detected by module name, converted with ``.tolist()`` (never imports numpy)
        - dict of {name: values}: keys become column names
        - list (1-D → single column; 2-D list-of-rows → transposed to columns)

    Returns ``(columns_data, names)`` where ``columns_data[i]`` is column ``i``'s
    values as a plain list and ``len(columns_data) == len(names)``.

    Raises:
        TypeError: unsupported *data* type.
        ValueError: ragged columns, or *columns* length mismatch.
    """
    # --- file path (str or os.PathLike) ------------------------------------
    if isinstance(data, (str, os.PathLike)):
        col_dict, col_names = _read_file_to_dict(Path(data))
        if columns is not None:
            if len(columns) != len(col_names):
                raise ValueError(
                    f"columns length ({len(columns)}) must match file columns ({len(col_names)})"
                )
            col_dict = {new: col_dict[old] for new, old in zip(columns, col_names)}
        return normalize_upload_input(col_dict, None)

    # --- pandas DataFrame (lazy, by module name) ---------------------------
    if _is_module(data, "pandas"):
        names = list(data.columns)
        rows = data.values.tolist()  # ndarray method → row-major python lists, no numpy import
        columns_data = [list(c) for c in zip(*rows)] if rows else [[] for _ in names]
        return columns_data, names

    # --- dict ---------------------------------------------------------------
    if isinstance(data, dict):
        names = list(data.keys())
        columns_data = [list(data[name]) for name in names]
        lengths = {len(c) for c in columns_data}
        if len(lengths) > 1:
            raise ValueError("all columns must have the same length (ragged input rejected)")
        return columns_data, names

    # --- numpy ndarray (lazy, by module name) or python list ---------------
    if _is_module(data, "numpy") or isinstance(data, list):
        rows = data.tolist() if _is_module(data, "numpy") else data

        is_2d = len(rows) > 0 and isinstance(rows[0], (list, tuple))
        if is_2d:
            row_lengths = {len(r) for r in rows}
            if len(row_lengths) > 1:
                raise ValueError("all rows must have the same length (ragged input rejected)")
            columns_data = [list(c) for c in zip(*rows)]
        else:
            columns_data = [list(rows)]

        n_cols = len(columns_data)
        if columns is None:
            columns = [f"column_{i + 1}" for i in range(n_cols)]
        if len(columns) != n_cols:
            raise ValueError(f"columns length ({len(columns)}) must match data columns ({n_cols})")
        return columns_data, list(columns)

    # --- unsupported --------------------------------------------------------
    raise TypeError("data must be a numpy array, list, pandas DataFrame, dict, or file path")


def detect_column_types(
    columns_data: List[List[Any]],
    names: List[str],
    max_k: int,
    max_ratio: float,
) -> Tuple[List[str], List[List[str]]]:
    """Detect the type of each column (column-major input).

    Detection rules:
    - **binary**: exactly 2 distinct values.
    - **factor**: non-numeric (string) column, OR ≤ max_k distinct numeric levels
      AND n_rows / n_distinct >= max_ratio.
    - **continuous**: otherwise.

    Returns ``(types, labels_list)`` where ``types[i]`` in
    {"binary","factor","continuous"} and ``labels_list[i]`` is a sorted list of
    string level labels for factors, [] otherwise.
    """
    n_rows = len(columns_data[0]) if columns_data else 0
    types: List[str] = []
    labels_list: List[List[str]] = []

    for col in columns_data:
        # Numeric probe: if any value fails float(), it's a string column.
        float_col: List[float] = []
        is_numeric = True
        for v in col:
            try:
                float_col.append(float(v))
            except (ValueError, TypeError):
                is_numeric = False
                break

        if not is_numeric:
            distinct = sorted(set(str(v) for v in col))
            types.append("factor")
            labels_list.append(distinct)
            continue

        n_distinct = len(set(float_col))

        if n_distinct == 2:
            types.append("binary")
            labels_list.append([])
            continue

        if n_distinct <= max_k and (n_rows / n_distinct) >= max_ratio:
            distinct_sorted = sorted(value_to_label(v) for v in set(float_col))
            types.append("factor")
            labels_list.append(distinct_sorted)
        else:
            types.append("continuous")
            labels_list.append([])

    return types, labels_list
