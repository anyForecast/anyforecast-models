import numpy as np
import pandas as pd

from anyforecast_models import base, decorators
from anyforecast_models.utils import checks


class SineTransformer(base.Transformer):
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


class CosineTransformer(base.Transformer):
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


class CyclicalEncoder(base.Transformer):
    """Cyclical encoder.

    Encodes periodic features using sine and cosine transformations with the
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

    @decorators.sklearn_check()
    @decorators.check_input(checks.check_1_feature)
    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Transforms input data.

        Parameters
        ----------
        X : array_like, shape=(n_samples, 1)
            Input data.

        Returns
        -------
        Xt : array_like, shape=(n_samples, 2)
            Sine and cosine transformations.
        """
        sin = SineTransformer(self.period).transform(X)
        cos = CosineTransformer(self.period).transform(X)
        return np.concatenate((sin, cos), axis=1)

    def get_feature_names_out(self) -> np.ndarray:
        if hasattr(self, "feature_names_in_"):
            prefix = self.feature_names_in_[0]
            return np.array([prefix + "_sin", prefix + "_cos"])


class CyclicalDatetimeEncoder(base.Transformer):
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

    @decorators.check_input(checks.check_is_series, checks.check_is_datetime)
    def fit(self, X: pd.Series, y=None):
        self.encoders_: dict[str, CyclicalEncoder] = {}
        for attr in self.datetime_attrs:
            X_dt = getattr(X.dt, attr)
            encoder = CyclicalEncoder().fit(X_dt)
            self.encoders_[attr] = encoder

        return self

    @decorators.check_input(checks.check_is_series, checks.check_is_datetime)
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

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        features_out = [
            v.get_feature_names_out() for _, v in self.encoders_.items()
        ]
        return np.concatenate(features_out)


class TimeIndexEncoder(base.Transformer):
    """Encodes datetime features with a time index.

    Parameters
    ---------
    start_idx : int
        Integer (including 0) where the time index will start

    Attributes
    ----------
    encoding_ : dict, pd.Timestamp -> int
        Mapping from timestamp to its associated index value.
    """

    def __init__(
        self, start_idx: int = 0, extra_timestamps: int = 10, freq: str = "D"
    ):
        self.start_idx = start_idx
        self.extra_timestamps = extra_timestamps
        self.freq = freq

    @property
    def dtype(self) -> np.dtype:
        """Specifies dtype of transformed/encoded data."""
        return np.dtype("int")

    @decorators.sklearn_check()
    @decorators.check_input(checks.check_1_feature, checks.check_is_datetime)
    def fit(self, X: pd.DataFrame | np.ndarray, y=None):
        """Fits transformer with input data.

        Parameters
        ----------
        X : array_like, shape=(n, 1)
            Datetime array.
        """
        date_range = self._make_date_range(X)
        time_index = self._make_time_index(date_range)
        self.encoding_ = dict(zip(date_range, time_index))
        return self

    @decorators.sklearn_check(reset=False)
    @decorators.check_input(checks.check_1_feature, checks.check_is_datetime)
    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Encodes input data with a time index.

        Parameters
        ----------
        X : array_like, shape=(n, 1)
            Datetime array.

        Returns
        -------
        Xt : array_like, shape=(n, 1)
            Integer array.
        """
        self.check_is_fitted()
        return (
            pd.Series(X.flatten())
            .astype(str)
            .map(self.encoding_)
            .values.reshape(-1, 1)
        )

    @decorators.sklearn_check(reset=False)
    @decorators.check_input(checks.check_1_feature)
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transfrom time index to original timestamp.

        Parameters
        ----------
        X : array_like, shape=(n, 1)
            Integer array.

        Returns
        -------
        Xi : array_like, shape=(n, 1)
            Datetime array.
        """
        self.check_is_fitted()
        Xi = pd.to_datetime(pd.Series(X.flatten()).map(self.inverse_encoding))
        return Xi.values.reshape(-1, 1)

    @property
    def inverse_encoding(self) -> dict:
        return {v: k for k, v in self.encoding_.items()}

    def _make_time_index(self, date_range: pd.DatetimeIndex) -> range:
        return range(self.start_idx, len(date_range) + self.start_idx)

    def _make_date_range(self, X: np.ndarray) -> np.ndarray:
        date_range = pd.date_range(X.min(), X.max(), freq=self.freq)

        if self.extra_timestamps > 0:
            extra_range = self._make_extra_date_range(X)
            date_range = date_range.union(extra_range)

        return date_range.astype(str)

    def _make_extra_date_range(self, X: np.ndarray) -> pd.DatetimeIndex:
        return pd.date_range(
            X.max(),
            periods=self.extra_timestamps + 1,
            freq=self.freq,
            inclusive="right",
        )

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return getattr(self, "feature_names_in_", None)
