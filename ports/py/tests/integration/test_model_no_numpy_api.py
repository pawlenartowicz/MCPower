"""Public API: correlation_matrix returns list-of-lists; numpy matrix still accepted as input."""
import mcpower


def test_correlation_matrix_getter_returns_lists():
    m = mcpower.MCPower("y ~ x1 + x2 + x3")
    m.set_correlations("corr(x1,x2)=0.3")
    m._apply()  # build registry state
    cm = m.correlation_matrix
    assert type(cm) is list and all(type(r) is list for r in cm)
    assert cm[0][1] == 0.3 and cm[1][0] == 0.3


def test_set_correlations_accepts_numpy_matrix():
    import numpy as np  # dev-installed; simulates a user passing numpy
    m = mcpower.MCPower("y ~ x1 + x2")
    m.set_correlations(np.array([[1.0, 0.5], [0.5, 1.0]]))
    m._apply()
    cm = m.correlation_matrix
    assert cm[0][1] == 0.5
    assert type(cm) is list


def test_set_correlations_accepts_listoflists():
    m = mcpower.MCPower("y ~ x1 + x2")
    m.set_correlations([[1.0, 0.4], [0.4, 1.0]])
    m._apply()
    assert m.correlation_matrix[0][1] == 0.4
