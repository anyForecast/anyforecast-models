import numpy as np
import pandas as pd
import unittest

from deepts.preprocessing import (
    CyclicalEncoder,
    CyclicalDatetimeEncoder,
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

    def test_fit_transform_pandas_(self):
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
    def test_fit_transform(self):
        freq = "W"
        dti = pd.date_range("2018-01-01", periods=20, freq=freq)
        X = pd.Series(dti)

        encoder = TimeIndexEncoder(freq=freq)
        Xt = encoder.fit_transform(X)

        assert encoder.encoding_
        assert Xt.shape == (20, 1)

    def test_inverse_transform(self):
        freq = "W"
        dti = pd.date_range("2018-01-01", periods=20, freq=freq)
        X = pd.Series(dti)

        encoder = TimeIndexEncoder(freq=freq)
        Xt = encoder.fit_transform(X)
        Xi = encoder.inverse_transform(Xt)

        assert X.equals(Xi)
