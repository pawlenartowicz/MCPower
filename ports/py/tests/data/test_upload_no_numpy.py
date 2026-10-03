"""normalize_upload_input returns column-major lists; numpy/pandas accepted via duck-typing, never imported."""
from mcpower.data.upload import normalize_upload_input


def test_dict_returns_column_major_lists():
    cols, names = normalize_upload_input({"x": [1, 2, 3], "y": [4, 5, 6]}, None)
    assert names == ["x", "y"]
    assert cols == [[1, 2, 3], [4, 5, 6]]
    assert type(cols) is list and all(type(c) is list for c in cols)


def test_list_2d_transposed_to_columns():
    cols, names = normalize_upload_input([[1, 2], [3, 4], [5, 6]], ["a", "b"])
    assert names == ["a", "b"]
    assert cols == [[1, 3, 5], [2, 4, 6]]


def test_list_1d_is_single_column():
    cols, names = normalize_upload_input([1, 2, 3], None)
    assert names == ["column_1"]
    assert cols == [[1, 2, 3]]


def test_ragged_dict_rejected():
    import pytest
    with pytest.raises(ValueError, match="same length"):
        normalize_upload_input({"x": [1, 2, 3], "y": [4, 5]}, None)


def test_string_column_preserved():
    cols, names = normalize_upload_input({"cyl": [4, 6], "origin": ["USA", "Japan"]}, None)
    assert cols[1] == ["USA", "Japan"]


def test_numpy_accepted_without_package_importing_it():
    # Fake numpy object: duck-typed by module name, converted via .tolist().
    class _FakeNDArray:
        __module__ = "numpy"

        def __init__(self, rows):
            self._rows = rows

        def tolist(self):
            return self._rows

    cols, names = normalize_upload_input(_FakeNDArray([[1.0, 2.0], [3.0, 4.0]]), ["a", "b"])
    assert names == ["a", "b"]
    assert cols == [[1.0, 3.0], [2.0, 4.0]]
