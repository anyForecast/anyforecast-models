import numpy as np
import pandas as pd
import sklearn
from sklearn.base import TransformerMixin, BaseEstimator

from ._data import UnitCircleProjector
from ..utils import validation, pandas


class CyclicalDatesEncoder(TransformerMixin, BaseEstimator):
    """Encodes datetime features by projecting them into a unit circle.

    Parameters
    ----------
    day : bool, default=True
        Whether to transform day of month

    month : bool, default=True
        Whether to transform month number

    dayofweek : bool, default=False
        Whether to transform day of week

    References
    ----------
    https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
    """

    def __init__(self, day=True, month=True, dayofweek=False, dtype=float):
        self.day = day
        self.month = month
        self.dayofweek = dayofweek
        self.dtype = dtype

    def fit(self, X, y=None):
        validation.check_pandas(X, 'series', self)
        validation.check_is_datetime(X)

        self.mapping_ = {}
        datetime_attrs = ['day', 'month', 'dayofweek']
        for attr in datetime_attrs:
            if getattr(self, attr):
                x = getattr(X.dt, attr)
                self.mapping_[attr] = UnitCircleProjector(attr).fit(x)

        self.feature_names_in_ = [X.name]
        return self

    def transform(self, X):
        """Adds cyclical columns to ``X``

        Parameters
        ----------
        X : datetime pd.Series
            Datetime pandas series with datetime accessor (i.e., X.dt).

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_encoded_features)
            Transformed input
        """
        validation.check_pandas(X, 'series', self)
        validation.check_is_datetime(X)

        X = X.copy()
        projections = []
        for datetime_attr, unit_circle_projector in self.mapping_.items():
            x = getattr(X.dt, datetime_attr)
            x_2d = pandas.series_to_2d(x)
            proj = unit_circle_projector.transform(x_2d)
            projections.append(proj)

        return np.hstack(projections)

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation

        Returns
        -------
        feature_names_out : list of str
            Transformed feature names.
        """
        return np.concatenate(
            [v.get_feature_names_out() for _, v in self.mapping_.items()])


class TimeIndexEncoder(TransformerMixin, BaseEstimator):
    """Encodes datetime features with a time index.

    Parameters
    ---------
    start_idx : int
        Integer (including 0) where the time index will start
    """

    def __init__(self, start_idx=0, extra_timestamps=10, freq='D'):
        self.start_idx = start_idx
        self.extra_timestamps = extra_timestamps
        self.freq = freq
        self.dtype = int

    def fit(self, X, y=None):
        validation.check_pandas(X, 'series', self)
        validation.check_is_datetime(X)

        date_range = self._make_date_range(X)
        time_index = range(self.start_idx, len(date_range) + self.start_idx)
        self.mapping_ = dict(zip(date_range, time_index))
        self.feature_names_out_ = np.array([X.name])
        return self

    def _make_date_range(self, X):
        date_range = pd.date_range(X.min(), X.max(), freq=self.freq)
        if self.extra_timestamps > 0:
            extra_range = pd.date_range(
                X.max(), periods=self.extra_timestamps + 1, freq=self.freq,
                inclusive='right')
            return date_range.union(extra_range)
        return date_range

    def get_feature_names_out(self, input_features=None):
        validation.check_is_fitted(self)
        return self.feature_names_out_

    def transform(self, X):
        validation.check_pandas(X, 'series', self)
        validation.check_is_datetime(X)
        X_out = X.map(self.mapping_)
        return pandas.series_to_2d(X_out)

    def inverse_transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]

        inverse_map = {v: k for k, v in self.mapping_.items()}
        X_inv = X.map(inverse_map)
        return pandas.series_to_2d(X_inv)


class MultiColumnLabelEncoder(TransformerMixin, BaseEstimator):
    """Label encoder by columns.

    For each column, a sklearn :class:`LabelEncoder` is fitted and applied.

    Used for transforming nominal data (e.g, Holidays, IDs, etc) to a integer
    scale that goes from 0 to n-1 where n is the number of unique values inside
    the column.

    Parameters
    ----------
    columns : list
        Columns to be transformed

    Attributes
    ----------
    mapping_ : dict, str -> LabelEncoder object
        Dictionary mapping from column name to its corresponding fitted
        :class:LabelEncoder object
    """

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """Obtains a LabelEncoder object for each column for later
        transformation.

        Each LabelEncoder object contains all the necessary
        information to perform the mapping between the nominal data and its
        corresponding numerical value.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be fitted

        y : None
            This param exists to match sklearn interface, but it should never
            be used.

        Returns
        -------
        self : Fitted transformer
        """
        self.mapping_ = {}  # mapping from column to LabelEncoder object
        for col in self.columns:
            label_enc = sklearn.preprocessing.LabelEncoder()
            label_enc.fit(X[col])
            self.mapping_[col] = label_enc
        return self

    def transform(self, X):
        """Maps every value inside ``X`` to its numerical counterpart.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe having init ``columns``

        Returns
        -------
        X : pd.DataFrame
            Copy of ``X`` with ``columns`` numerically encoded
        """
        validation.check_is_fitted(self)
        X = X.copy()
        for col in self.columns:
            label_enc = self.mapping_[col]
            X[col] = label_enc.transform(X[col])
        X[self.columns] = X[self.columns].astype('category')
        return X

    def inverse_transform(self, X):
        """Undos the numerical encoding.

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        inverse transformed X
        """
        validation.check_is_fitted(self)
        X = X.copy()
        for col in self.columns:
            if col not in self.columns:
                continue
            label_enc = self.mapping_[col]
            X[col] = label_enc.inverse_transform(X[col])
        return X

    def _flip_dict(self, d):
        return {v: k for k, v in d.items()}
