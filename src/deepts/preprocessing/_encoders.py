import numpy as np
import pandas as pd

from deepts.base import Transformer
from deepts.decorators import check, sklearn_validate
from deepts.utils import checks


class SineTransformer(Transformer):
    """Trignometric sine transformation.

    Parameters
    ----------
    period : float, default=2 * np.pi
        Sine period.
    """

    def __init__(self, period: float = 2 * np.pi):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.sin(X / self.period * 2 * np.pi)

    def _more_tags(self):
        return {"stateless": True}


class CosineTransformer(Transformer):
    """Trignometric cosine transformation.

    Parameters
    ----------
    period : float, default=2 * np.pi
        Cosine period.
    """

    def __init__(self, period: float = 2 * np.pi):
        self.period = period

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.cos(X / self.period * 2 * np.pi)

    def _more_tags(self):
        return {"stateless": True}


class CyclicalEncoder(Transformer):
    """Cyclical encoder.

    Encodes periodic features using a sine and cosine transformation with the
    matching period.

    Parameters
    ----------
    period : int, default=10
        Input data period.
    """

    def __init__(self, period: int = 10):
        self.period = period

    def fit(self, X, y=None):
        return self

    @sklearn_validate()
    @check(checks=[checks.check_1_feature])
    def transform(self, X) -> np.ndarray:
        """Transforms input data.

        Parameters
        ----------
        X : array_like, shape=(n_samples, 1)
            Input data.
        """
        sin = SineTransformer(self.period).transform(X)
        cos = CosineTransformer(self.period).transform(X)
        return np.concatenate((sin, cos), axis=1)

    def get_feature_names_out(self) -> np.ndarray:
        if hasattr(self, "feature_names_in_"):
            prefix = self.feature_names_in_[0]
            return np.array([prefix + "_sin", prefix + "_cos"])


class CyclicalDatetimeEncoder(Transformer):
    """Encodes datetime features cyclically.

    Each periodic datetime feature (day, month, dayofweek) is encoded
    cyclically using a sine and cosine transformation.

    Parameters
    ----------
    datetime_attrs : list of str
    """

    def __init__(
        self, datetime_attrs: list[str] = ("day", "dayofweek", "month")
    ):
        self.datetime_attrs = datetime_attrs

    @check(checks=[checks.check_is_series, checks.check_is_datetime])
    def fit(self, X: pd.Series, y=None):
        self.encoders_: dict[str, CyclicalEncoder] = {}
        for attr in self.datetime_attrs:
            X_dt = getattr(X.dt, attr)
            encoder = CyclicalEncoder().fit(X_dt)
            self.encoders_[attr] = encoder

        return self

    @check(checks=[checks.check_is_series, checks.check_is_datetime])
    def transform(self, X: pd.Series) -> np.ndarray:
        """Adds cyclical columns to ``X``

        Parameters
        ----------
        X : pd.Series
            Datetime pandas series with datetime accessor (i.e., X.dt).

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input
        """
        transforms: list[np.ndarray] = []
        for attr, encoder in self.encoders_.items():
            x: pd.Series = getattr(X.dt, attr)
            tansformation = encoder.transform(x.values.reshape(-1, 1))
            transforms.append(tansformation)

        return np.hstack(transforms)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        return np.concatenate(
            [v.get_feature_names_out() for _, v in self.mapping_.items()]
        )


class TimeIndexEncoder(Transformer):
    """Encodes datetime features with a time index.

    Parameters
    ---------
    start_idx : int
        Integer (including 0) where the time index will start
    """

    def __init__(
        self, start_idx: int = 0, extra_timestamps: int = 10, freq: str = "D"
    ):
        self.start_idx = start_idx
        self.extra_timestamps = extra_timestamps
        self.freq = freq

    @property
    def dtype(self) -> np.dtype:
        """Specifies dtype of transformed/encoded data.
        """
        return np.dtype("int")

    @check(checks=[checks.check_is_series, checks.check_is_datetime])
    def fit(self, X: pd.Series, y=None):
        """Fits transformer with input data.

        Parameters
        ----------
        X : pd.Series
            Datetime pandas Series.
        """
        date_range = self.make_date_range(X)
        time_index = self.make_time_index(date_range)
        self.encoding_ = dict(zip(date_range, time_index))

        self.feature_names_out_ = np.array([X.name])
        return self

    @check(
        checks=[checks.check_is_series, checks.check_is_datetime],
        check_is_fitted=True,
    )
    def transform(self, X: pd.Series) -> np.ndarray:
        """Encodes input data with a time index.

        Parameters
        ----------
        X : pd.Series
            Datetime pandas Series.
        """
        Xt = X.map(self.encoding_)
        return Xt.values.reshape(-1, 1)

    @sklearn_validate(reset=False)
    @check(checks=[checks.check_1_feature], check_is_fitted=True)
    def inverse_transform(self, X: np.ndarray) -> pd.Series:
        X: pd.Series = pd.Series(X.flatten())
        return X.map(self.inverse_encoding)

    @property
    def inverse_encoding(self) -> dict:
        return {v: k for k, v in self.encoding_.items()}

    def make_time_index(self, date_range: pd.DatetimeIndex) -> range:
        return range(self.start_idx, len(date_range) + self.start_idx)

    def make_date_range(self, X: pd.Series) -> pd.DatetimeIndex:
        date_range = pd.date_range(X.min(), X.max(), freq=self.freq)

        if self.extra_timestamps > 0:
            extra_range = self.make_extra_date_range(X)
            date_range = date_range.union(extra_range)

        return date_range

    def make_extra_date_range(self, X: pd.Series) -> pd.DatetimeIndex:
        return pd.date_range(
            X.max(),
            periods=self.extra_timestamps + 1,
            freq=self.freq,
            inclusive="right",
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        self.check_is_fitted()
        return self.feature_names_out_
