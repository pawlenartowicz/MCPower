"""D1 — polymorphic normalize_upload_input: file, lazy-pandas, dict, array.

Returns column-major lists ``(columns_data, names)``: ``len(data)`` is the column
count, ``len(data[0])`` the row count, ``data[c][r]`` indexes column c, row r.
"""
import numpy as np
import pytest

from mcpower.data.upload import normalize_upload_input


def test_csv_path_parses_with_header(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,b\n1,2\n3,4\n")
    data, cols = normalize_upload_input(str(p), None)
    assert cols == ["a", "b"]
    assert len(data) == 2 and len(data[0]) == 2
    assert float(data[0][0]) == 1.0   # column "a", row 0
    assert float(data[1][1]) == 4.0   # column "b", row 1


def test_tsv_sniffs_tab(tmp_path):
    p = tmp_path / "d.tsv"
    p.write_text("a\tb\n1\t2\n")
    data, cols = normalize_upload_input(str(p), None)
    assert cols == ["a", "b"]
    assert len(data) == 2 and len(data[0]) == 1


def test_csv_pathlike(tmp_path):
    """os.PathLike objects (Path) should also be accepted."""
    p = tmp_path / "d.csv"
    p.write_text("x,y\n5,6\n7,8\n9,10\n")
    data, cols = normalize_upload_input(p, None)
    assert cols == ["x", "y"]
    assert len(data) == 2 and len(data[0]) == 3


def test_csv_with_string_column(tmp_path):
    """String columns should survive as plain lists."""
    p = tmp_path / "d.csv"
    p.write_text("cyl,origin\n4,USA\n6,Japan\n")
    data, cols = normalize_upload_input(str(p), None)
    assert cols == ["cyl", "origin"]
    assert len(data) == 2 and len(data[0]) == 2
    # string column preserved (column "origin" = index 1, row 0)
    assert data[1][0] == "USA"


def test_lazy_pandas_detection_no_import():
    """Pandas-like objects detected by module name, not isinstance."""

    class _FakeDF:
        __module__ = "pandas.core.frame"
        values = np.array([[1.0, 2.0]])
        columns = ["a", "b"]

    data, cols = normalize_upload_input(_FakeDF(), None)
    assert cols == ["a", "b"]
    assert len(data) == 2 and len(data[0]) == 1


def test_dict_branch_still_works():
    data, cols = normalize_upload_input({"x": [1, 2, 3], "y": [4, 5, 6]}, None)
    assert cols == ["x", "y"]
    assert data == [[1, 2, 3], [4, 5, 6]]


def test_array_branch_still_works():
    src = np.ones((5, 3))
    data, cols = normalize_upload_input(src, ["a", "b", "c"])
    assert cols == ["a", "b", "c"]
    assert len(data) == 3 and len(data[0]) == 5


# ---------------------------------------------------------------------------
# Reject-path tests
# ---------------------------------------------------------------------------


def test_file_columns_override_length_mismatch_raises(tmp_path):
    """columns override with wrong length must raise ValueError with count details."""
    p = tmp_path / "d.csv"
    p.write_text("a,b,c\n1,2,3\n4,5,6\n")
    with pytest.raises(ValueError, match=r"columns length \(2\) must match file columns \(3\)"):
        normalize_upload_input(str(p), columns=["x", "y"])


def test_array_columns_length_mismatch_raises():
    """2-D numpy array with wrong columns length must raise ValueError."""
    src = np.ones((4, 3))
    with pytest.raises(ValueError, match=r"columns length \(2\) must match data columns \(3\)"):
        normalize_upload_input(src, columns=["a", "b"])


def test_unsupported_type_raises_typeerror():
    """A plain int (unsupported type) must raise TypeError with the right message."""
    with pytest.raises(TypeError, match="numpy array, list, pandas DataFrame, dict, or file path"):
        normalize_upload_input(42, None)
