"""
Data input normalization for upload_data.

Converts various input formats (list, 1D/2D numpy array, pandas DataFrame, dict)
into a standardized (2D numpy array, column names list) pair.
"""

from typing import List, Optional, Tuple

import numpy as np


def normalize_upload_input(
    data,
    columns: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert user-supplied data into a 2D numpy array with column names.

    Accepted inputs:
        - pandas DataFrame: column names taken from df.columns
        - dict of {name: array}: keys become column names
        - list or 1D numpy array: treated as single column
        - 2D numpy array: used directly

    When *columns* is not provided and cannot be inferred (plain array / list),
    columns are auto-named ``column_1``, ``column_2``, ...

    Args:
        data: Raw data in any supported format.
        columns: Optional explicit column names (only used for numpy/list input).

    Returns:
        (array_2d, column_names) – a float-compatible 2D array and matching names.

    Raises:
        TypeError: If *data* is an unsupported type.
        ValueError: If *columns* length doesn't match array width.
    """
    try:
        import pandas as pd

        has_pandas = True
    except ImportError:
        has_pandas = False

    # --- pandas DataFrame ---------------------------------------------------
    if has_pandas and isinstance(data, pd.DataFrame):
        return data.values, list(data.columns)

    # --- dict ---------------------------------------------------------------
    if isinstance(data, dict):
        col_names = list(data.keys())
        arrays = [np.asarray(data[col]) for col in col_names]
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
    raise TypeError("data must be a numpy array, list, pandas DataFrame, or dict")
