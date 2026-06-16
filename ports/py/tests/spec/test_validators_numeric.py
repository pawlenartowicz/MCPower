from mcpower.spec.validators import _validate_numeric_parameter


def test_type_and_range_messages():
    r = _validate_numeric_parameter("x", "P", min_val=0, max_val=1)
    assert not r.is_valid and "must be" in r.errors[0]
    r2 = _validate_numeric_parameter(5, "P", min_val=0, max_val=1)
    assert not r2.is_valid and ">= " not in r2.errors[0] and "<= 1" in r2.errors[0]
