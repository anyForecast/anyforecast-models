import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from deepts.preprocessing import (
    CyclicalDatetimeEncoder,
    CyclicalEncoder,
    TimeIndexEncoder,
)


class TestCyclicalEncoder(unittest.TestCase):
    def test_fit_transform_numpy(self):
        periodic_signal = [1, 2, 3] * 3
        periodic_signal = np.array(periodic_signal).reshape(-1, 1)

        encoder = CyclicalEncoder(period=3)
        encoder.fit_transform(periodic_signal)

        assert encoder.n_features_in_ == 1
        assert encoder.get_feature_names_out() is None

    def test_fit_transform_pandas(self):
        periodic_signal = [1, 2, 3] * 3
        periodic_signal = np.array(periodic_signal).reshape(-1, 1)

        frame = pd.DataFrame(periodic_signal, columns=["periodic_signal"])

        encoder = CyclicalEncoder(period=3)
        encoder.fit_transform(frame)

        assert encoder.n_features_in_ == 1
        assert np.array_equal(
            encoder.get_feature_names_out(),
            np.array(["periodic_signal_sin", "periodic_signal_cos"]),
        )

    def test_value_error(self):
        periodic_signal = [1, 2, 3] * 3
        periodic_signal = np.array(periodic_signal).reshape(1, -1)

        encoder = CyclicalEncoder(period=3)
        self.assertRaises(ValueError, encoder.fit_transform, periodic_signal)


class TestCyclicalDatetimeEncoder(unittest.TestCase):
    def test_fit_transform(self):
        dti = pd.date_range("2018-01-01", periods=20, freq="W")
        X = pd.Series(dti)

        datetime_attrs = ("day", "dayofweek", "month")
        encoder = CyclicalDatetimeEncoder(datetime_attrs)
        Xt = encoder.fit_transform(X)

        for attr in datetime_attrs:
            assert attr in encoder.encoders_
            assert isinstance(encoder.encoders_[attr], CyclicalEncoder)

        assert Xt.shape == (20, 6)

    def test_value_error(self):
        dti = pd.date_range("2018-01-01", periods=20, freq="W")
        encoder = CyclicalDatetimeEncoder()

        self.assertRaises(ValueError, encoder.fit_transform, dti)


class TestTimeIndexEncoder(unittest.TestCase):
    def test_fit_transform_numpy(self):

        # Create Numpy array
        freq = "W"
        dti = pd.date_range("2018-01-01", periods=3, freq=freq)
        X = dti.values.reshape(-1, 1)

        # Fit transform encoder
        encoder = TimeIndexEncoder(freq=freq)
        Xt = encoder.fit_transform(X)

        # Assert
        assert_array_equal(Xt, [[0], [1], [2]])

    def test_fit_transform_pandas(self):

        # Create Pandas DataFrame
        freq = "W"
        dti = pd.date_range("2018-01-01", periods=3, freq=freq)
        X = dti.to_frame(index=False)
        X = X.rename(columns={0: "date"})

        # Fit transform encoder
        encoder = TimeIndexEncoder(freq=freq)
        Xt = encoder.fit_transform(X)

        # Assert
        assert encoder.feature_names_in_.item() == "date"
        assert_array_equal(Xt, [[0], [1], [2]])

    def test_inverse_transform(self):

        # Create Numpy array
        freq = "W"
        dti = pd.date_range("2018-01-01", periods=3, freq=freq)
        X = dti.values.reshape(-1, 1)

        encoder = TimeIndexEncoder(freq=freq)
        Xt = encoder.fit_transform(X)
        Xi = encoder.inverse_transform(Xt)

        assert_array_equal(X, Xi)
