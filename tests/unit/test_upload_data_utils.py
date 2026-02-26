"""Unit tests for mcpower.utils.upload_data_utils — normalize_upload_input."""

import numpy as np
import pytest

from mcpower.utils.upload_data_utils import normalize_upload_input


class TestNormalizeUploadInput:
    """Tests for normalize_upload_input."""

    def test_dict_input(self):
        data = {"x1": [1.0, 2.0, 3.0], "x2": [4.0, 5.0, 6.0]}
        arr, cols = normalize_upload_input(data)
        assert cols == ["x1", "x2"]
        assert arr.shape == (3, 2)
        np.testing.assert_array_equal(arr[:, 0], [1.0, 2.0, 3.0])

    def test_dict_with_strings(self):
        data = {"group": ["a", "b", "a"], "x1": [1.0, 2.0, 3.0]}
        arr, cols = normalize_upload_input(data)
        assert arr.dtype == object
        assert cols == ["group", "x1"]

    def test_list_input(self):
        data = [1.0, 2.0, 3.0]
        arr, cols = normalize_upload_input(data)
        assert arr.shape == (3, 1)
        assert cols == ["column_1"]

    def test_1d_array(self):
        data = np.array([1.0, 2.0, 3.0])
        arr, cols = normalize_upload_input(data)
        assert arr.shape == (3, 1)
        assert cols == ["column_1"]

    def test_2d_array(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr, cols = normalize_upload_input(data)
        assert arr.shape == (2, 2)
        assert cols == ["column_1", "column_2"]

    def test_2d_array_with_columns(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        arr, cols = normalize_upload_input(data, columns=["a", "b"])
        assert cols == ["a", "b"]

    def test_dataframe_input(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0]})
        arr, cols = normalize_upload_input(df)
        assert cols == ["x1", "x2"]
        assert arr.shape == (2, 2)

    def test_mismatched_columns_raises(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        with pytest.raises(ValueError, match="columns length"):
            normalize_upload_input(data, columns=["a", "b", "c"])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="data must be"):
            normalize_upload_input("not valid data")
