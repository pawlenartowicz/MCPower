"""Normalises user-supplied data (list, numpy array, DataFrame, dict, file path) into a 2D object array with column names for engine ingestion."""

import csv
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


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
    text = path.read_text(newline="", encoding="utf-8-sig")
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


def normalize_upload_input(
    data,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert user-supplied data into a 2D numpy array with column names.

    Accepted inputs:
        - str or os.PathLike (file path to .csv / .tsv): read with stdlib csv, header row → names
        - pandas DataFrame: detected by module name (lazy — never forces pandas import)
        - dict of {name: array}: keys become column names
        - list or 1D numpy array: treated as single column
        - 2D numpy array: used directly

    When *columns* is not provided and cannot be inferred (plain array / list),
    columns are auto-named ``column_1``, ``column_2``, ...

    Args:
        data: Raw data in any supported format.
        columns: Optional explicit column names (only used for numpy/list input).

    Returns:
        (array_2d, column_names) – a 2D array (may be object dtype when string columns
        are present) and matching names.

    Raises:
        TypeError: If *data* is an unsupported type.
        ValueError: If *columns* length doesn't match array width.
    """
    # --- file path (str or os.PathLike) ------------------------------------
    if isinstance(data, (str, os.PathLike)):
        col_dict, col_names = _read_file_to_dict(Path(data))
        if columns is not None:
            # rename: user-supplied columns override header names
            if len(columns) != len(col_names):
                raise ValueError(
                    f"columns length ({len(columns)}) must match file columns ({len(col_names)})"
                )
            col_dict = {new: col_dict[old] for new, old in zip(columns, col_names)}
            col_names = list(columns)
        return normalize_upload_input(col_dict, None)

    # --- lazy pandas DataFrame detection -----------------------------------
    # Detect by module name so users without pandas are never forced to import it.
    if type(data).__module__.split(".")[0] == "pandas":
        return data.values, list(data.columns)

    # --- dict ---------------------------------------------------------------
    if isinstance(data, dict):
        col_names = list(data.keys())
        arrays = [np.asarray(data[col]) for col in col_names]
        # Cast to object dtype when any column contains strings, so that
        # downstream code (which checks for object dtype) can detect and
        # encode string columns.  This matches pandas DataFrame.values behavior.
        if any(arr.dtype.kind in ("U", "S", "O") for arr in arrays):
            arrays = [arr.astype(object) for arr in arrays]
        return np.column_stack(arrays), col_names

    # --- list / numpy array -------------------------------------------------
    if isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data)

        # 1-D → single column
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        # Auto-generate column names when not supplied
        if columns is None:
            columns = [f"column_{i + 1}" for i in range(arr.shape[1])]

        if len(columns) != arr.shape[1]:
            raise ValueError(f"columns length ({len(columns)}) must match data columns ({arr.shape[1]})")

        return arr, columns

    # --- unsupported --------------------------------------------------------
    raise TypeError("data must be a numpy array, list, pandas DataFrame, dict, or file path")


def detect_column_types(
    arr: np.ndarray,
    names: List[str],
    max_k: int,
    max_ratio: float,
) -> Tuple[List[str], List[List[str]]]:
    """Detect the type of each column in a 2-D array (object dtype expected).

    Detection rules:
    - **binary**: exactly 2 distinct values.
    - **factor**: non-numeric (string) column, OR ≤ max_k distinct numeric levels
      AND n_rows / n_distinct >= max_ratio (enough rows per level).
    - **continuous**: otherwise (many distinct numerics, or few distinct but too sparse).

    Factor level names are sorted and de-duplicated.  Level labels are
    returned as strings so they can be used directly as ``cyl[6]`` / ``origin[Japan]``
    display names.

    Args:
        arr: 2-D array, shape (n_rows, n_cols). May be object dtype.
        names: Column names, length == arr.shape[1].
        max_k: Soft upper bound on factor levels (``max_factor_k_soft`` from config).
        max_ratio: Minimum rows-per-level ratio for a numeric column to be a factor
                   (``max_factor_ratio`` from config).

    Returns:
        (types, labels_list) where types[i] in {"binary","factor","continuous"} and
        labels_list[i] is a sorted list of string level labels for factors, [] otherwise.
    """
    n_rows = arr.shape[0]
    types: List[str] = []
    labels_list: List[List[str]] = []

    for col_idx in range(arr.shape[1]):
        col = arr[:, col_idx]

        # Attempt numeric cast to distinguish string vs numeric columns.
        is_numeric = True
        try:
            # If even one value can't convert, it's a string column.
            float_col = col.astype(np.float64)
        except (ValueError, TypeError):
            is_numeric = False

        if not is_numeric:
            # String / mixed column → always factor; gather distinct string labels
            distinct = sorted(set(str(v) for v in col))
            types.append("factor")
            labels_list.append(distinct)
            continue

        # Numeric column: count distinct values
        n_distinct = len(set(float_col.tolist()))

        if n_distinct == 2:
            types.append("binary")
            labels_list.append([])
            continue

        # Factor guard: few distinct AND enough rows per level
        if n_distinct <= max_k and (n_rows / n_distinct) >= max_ratio:
            # Labels via the single-sourced renderer (integer-valued floats → "4").
            distinct_sorted = sorted(value_to_label(v) for v in set(float_col.tolist()))
            types.append("factor")
            labels_list.append(distinct_sorted)
        else:
            types.append("continuous")
            labels_list.append([])

    return types, labels_list
