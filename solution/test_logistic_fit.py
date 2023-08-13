from numpy.testing import assert_allclose

from logistic import iterate_f
from logistic_fit import fit_r


def test_logistic_fit():
    r = 3.123
    x0 = 0.322
    it = 27
    xs = iterate_f(it, x0, r)

    assert_allclose(fit_r(xs), r, atol=1e-3)
