import unittest

import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from deepts.preprocessing.compose import PandasColumnTransformer


class TestPandasColumnTransformer(unittest.TestCase):
    def setUp(self) -> None:
        """Setup input data X and :class:`PandasColumnTransformer` instance."""
        data = {
            "c1": [1.0, 2.0, 3.0],
            "c2": [4.0, 5.0, 6.0],
            "c3": ["yellow", "red", "blue"],
        }
        self.X = pd.DataFrame(data)

        transformers = [
            (MinMaxScaler(), ["c1"]),
            (StandardScaler(), ["c2"]),
            (OneHotEncoder(), ["c3"]),
        ]

        ct = make_column_transformer(
            *transformers, verbose_feature_names_out=False
        )
        self.pandas_ct = PandasColumnTransformer(ct)

    def fit_transform(self) -> pd.DataFrame:
        return self.pandas_ct.fit_transform(self.X)

    def test_fit_transform(self):
        Xt = self.fit_transform().round(decimals=3)

        expected_data = {
            "c1": [0.0, 0.5, 1],
            "c2": [-1.225, 0, 1.225],
            "c3_blue": [0.0, 0.0, 1.0],
            "c3_red": [0.0, 1.0, 0.0],
            "c3_yellow": [1.0, 0.0, 0.0],
        }
        expected_frame = pd.DataFrame(expected_data)

        assert Xt.equals(expected_frame)

    def test_inverse_transform(self):
        Xt = self.fit_transform()
        Xi = self.pandas_ct.inverse_transform(Xt)

        assert self.X.equals(Xi)
