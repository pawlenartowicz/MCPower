"""Correlation-matrix and upload validators operate on list-of-lists, no numpy."""
from mcpower.spec.validators import _validate_correlation_matrix, _validate_upload_data


def test_corr_matrix_valid_listoflists():
    m = [[1.0, 0.3], [0.3, 1.0]]
    assert _validate_correlation_matrix(m).is_valid


def test_corr_matrix_non_square():
    res = _validate_correlation_matrix([[1.0, 0.3]])
    assert not res.is_valid
    assert any("square" in e for e in res.errors)


def test_corr_matrix_bad_diagonal_and_asymmetry():
    res = _validate_correlation_matrix([[1.0, 0.3], [0.9, 2.0]])
    assert not res.is_valid
    assert any("Diagonal" in e for e in res.errors)
    assert any("symmetric" in e for e in res.errors)


def test_corr_matrix_none():
    res = _validate_correlation_matrix(None)
    assert not res.is_valid


def test_upload_nan_and_inf_detected():
    # min_rows is small in config; build enough rows. Column "a" numeric clean,
    # "b" has a NaN, "c" has an Inf, "lab" is strings (skipped).
    n = 40
    a = [float(i) for i in range(n)]
    b = [1.0] * n
    b[3] = float("nan")
    c = [2.0] * n
    c[5] = float("inf")
    lab = ["x"] * n
    res = _validate_upload_data([a, b, c, lab], ["a", "b", "c", "lab"])
    assert not res.is_valid
    assert any("NaN" in e and "b" in e for e in res.errors)
    assert any("Inf" in e and "c" in e for e in res.errors)


def test_upload_clean_passes():
    n = 40
    cols = [[float(i) for i in range(n)], ["g"] * n]
    assert _validate_upload_data(cols, ["x", "grp"]).is_valid
