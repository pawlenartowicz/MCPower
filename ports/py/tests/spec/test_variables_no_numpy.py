"""VariableRegistry stores correlation matrix as list-of-lists and effect sizes as a list."""
from mcpower.spec.variables import VariableRegistry


def test_effect_sizes_is_plain_list():
    reg = VariableRegistry("y ~ x1 + x2")
    reg.set_effect_size("x1", 0.3)
    reg.set_effect_size("x2", 0.5)
    es = reg.get_effect_sizes()
    assert es == [0.3, 0.5]
    assert type(es) is list
    assert all(type(v) is float for v in es)


def test_set_correlation_builds_identity_list():
    reg = VariableRegistry("y ~ x1 + x2 + x3")
    reg.set_correlation("x1", "x2", 0.4)
    m = reg.get_correlation_matrix()
    assert m == [[1.0, 0.4, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert type(m) is list and all(type(r) is list for r in m)


def test_set_correlation_matrix_deep_copies():
    reg = VariableRegistry("y ~ x1 + x2")
    src = [[1.0, 0.2], [0.2, 1.0]]
    reg.set_correlation_matrix(src)
    src[0][1] = 99.0  # mutating the source must not leak in
    assert reg.get_correlation_matrix() == [[1.0, 0.2], [0.2, 1.0]]
