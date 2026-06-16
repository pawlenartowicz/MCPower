"""D1 — polymorphic normalize_upload_input: file, lazy-pandas, dict, array."""
import numpy as np
import pytest

from mcpower.data.upload import normalize_upload_input


def test_csv_path_parses_with_header(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    arr, cols = normalize_upload_input(str(p), None)
    assert cols == ["a", "b"]
    assert arr.shape == (2, 2)
    assert float(arr[0, 0]) == 1.0
    assert float(arr[1, 1]) == 4.0


def test_tsv_sniffs_tab(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("a\tb\n1\t2\n")
    arr, cols = normalize_upload_input(str(p), None)
    assert cols == ["a", "b"]
    assert arr.shape == (1, 2)


def test_csv_pathlike(tmp_path):
    """os.PathLike objects (Path) should also be accepted."""
    from pathlib import Path

    p = tmp_path / "d.csv"
    p.write_text("x,y\n5,6\n7,8\n9,10\n")
    arr, cols = normalize_upload_input(p, None)
    assert cols == ["x", "y"]
    assert arr.shape == (3, 2)


def test_csv_with_string_column(tmp_path):
    """String columns should survive as object dtype."""
    p = tmp_path / "d.csv"
    p.write_text("cyl,origin\n4,USA\n6,Japan\n")
    arr, cols = normalize_upload_input(str(p), None)
    assert cols == ["cyl", "origin"]
    assert arr.shape == (2, 2)
    # string column preserved
    assert arr[0, 1] == "USA"


def test_lazy_pandas_detection_no_import():
    """Pandas-like objects detected by module name, not isinstance."""

    class _FakeDF:
        __module__ = "pandas.core.frame"
        values = np.array([[1.0, 2.0]])
        columns = ["a", "b"]

    arr, cols = normalize_upload_input(_FakeDF(), None)
    assert cols == ["a", "b"]
    assert arr.shape == (1, 2)


def test_dict_branch_still_works():
    arr, cols = normalize_upload_input({"x": [1, 2, 3], "y": [4, 5, 6]}, None)
    assert cols == ["x", "y"]
    assert arr.shape == (3, 2)


def test_array_branch_still_works():
    data = np.ones((5, 3))
    arr, cols = normalize_upload_input(data, ["a", "b", "c"])
    assert cols == ["a", "b", "c"]
    assert arr.shape == (5, 3)


# ---------------------------------------------------------------------------
# Reject-path tests (lines ~107-112 and ~143-144)
# ---------------------------------------------------------------------------


def test_file_columns_override_length_mismatch_raises(tmp_path):
    """columns override with wrong length must raise ValueError with count details."""
    p = tmp_path / "d.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match=r"columns length \(2\) must match file columns \(3\)"):
        normalize_upload_input(str(p), columns=["x", "y"])


def test_array_columns_length_mismatch_raises():
    """2-D numpy array with wrong columns length must raise ValueError."""
    data = np.ones((4, 3))
    with pytest.raises(ValueError, match=r"columns length \(2\) must match data columns \(3\)"):
        normalize_upload_input(data, columns=["a", "b"])


def test_unsupported_type_raises_typeerror():
    """A plain int (unsupported type) must raise TypeError with the right message."""
    with pytest.raises(TypeError, match="numpy array, list, pandas DataFrame, dict, or file path"):
        normalize_upload_input(42, None)
