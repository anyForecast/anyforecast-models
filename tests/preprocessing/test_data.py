import unittest

import numpy as np
from numpy.testing import assert_array_equal

from anyforecast_models.preprocessing import IdentityTransformer
from anyforecast_models.preprocessing._data import _identity


def test_identity():
    X = np.random.rand(2, 2)
    assert_array_equal(X, _identity(X))


class TestIdentityTransform(unittest.TestCase):
    def test_fit(self):
        X = np.random.rand(2, 2)
        identity_transformer = IdentityTransformer()
        retval = identity_transformer.fit(X)
        assert isinstance(retval, IdentityTransformer)

    def test_fit_transform(self):
        X = np.random.rand(2, 2)
        Xt = IdentityTransformer().fit_transform(X)
        assert_array_equal(X, Xt)

    def test_inverse_transform(self):
        X = np.random.rand(2, 2)
        identity_transformer = IdentityTransformer()
        Xt = identity_transformer.fit_transform(X)
        Xi = identity_transformer.inverse_transform(Xt)
        assert_array_equal(X, Xi)
