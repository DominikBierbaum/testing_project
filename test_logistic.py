from numpy.testing import assert_allclose
import pytest
from logistic import f

# Add here your test for the logistic map
@pytest.mark.parametrize("x, r, expected_result", ([0.1, 2.2, 0.198], [0.2, 3.4, 0.544], [0.5, 2, 0.5]))
def test_f_generic_param(x, r, expected_result):
    result = f(x, r)
    assert_allclose(result, expected_result)

def test_f_generic():
    x = [0.1, 0.2, 0.5]
    r = [2.2, 3.4, 2]
    expected_result = [0.198, 0.544, 0.5]
    for i in range(len(x)):
        assert_allclose(f(x[i], r[i]), expected_result[i])

def test_f_corner_cases():
    # Test cases are (x, r, expected)
    cases = [
        (0, 1.1, 0),
        (1, 3.7, 0),
    ]
    for x, r, expected in cases:
        result = f(x, r)
        assert_allclose(result, expected)
