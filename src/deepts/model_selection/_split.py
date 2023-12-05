from typing import Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils import validation


def make_datetime_index(
    start=None,
    end=None,
    periods=None,
    freq=None,
    inclusive: Optional[Literal["left", "right"]] = None,
    **kwargs
) -> pd.DatetimeIndex:
    """Makes pandas DatetimeIndex.

    Returns
    -------
    datetime_index : pd.DatetimeIndex
    """
    return pd.date_range(
        start, end, periods, freq=freq, inclusive=inclusive, **kwargs
    )


def timeseries_train_test_split(
    X: pd.DataFrame,
    date_col: str,
    horizon: pd.DatetimeIndex,
    train_start: Optional[Union[str, pd.Timestamp]] = None,
    encoder_length: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits time series dataframe.

    Special case of KFold where K=1 and test indices are defined
    by passed dates.

    Parameters
    ----------
    X : pd.DataFrame
        Data to split.

    date_col : str
        Date column.

    horizon : pd.DatetimeIndex
        Horizon date range.

    train_start : str or pd.Timestamp, default=None
        Training start date. Defaults to ``X[date_col].min()``.

    encoder_length : int, default=0
        Encoder sequence length.

    Returns
    -------
    split : tuple
        (X_train, X_test) tuple.
    """
    cv = TimeseriesTrainTestSplit(horizon, train_start, encoder_length)
    generator = cv.split(X[date_col])
    train_indices, test_indices = list(generator)[0]
    return X.iloc[train_indices], X.iloc[test_indices]


class TimeseriesTrainTestSplit:
    """Timeseries train test split.

    Special case of KFold where K=1 and test indices are defined
    by ``date_range``.

    Parameters
    ----------
    horizon : pd.DatetimeIndex
        Horizon date range.

    train_start : str or pd.Timestamp, default=None
        Training start date. Defaults to ``X.min()``.

    encoder_length : int, default=0
        Encoder sequence length.
    """

    def __init__(
        self,
        horizon: pd.DatetimeIndex,
        train_start: Optional[Union[str, pd.Timestamp]] = None,
        encoder_length: int = 0,
    ):
        self.horizon = horizon
        self.train_start = train_start
        self.encoder_length = encoder_length

    def split(self, X: pd.Series):
        validation.check_is_datetime(X)

        indices = np.arange(len(X))
        for train_mask, test_mask in self._iter_train_test_masks(X):
            train_index = indices[train_mask]
            test_index = indices[test_mask]
            yield train_index, test_index

    def get_n_splits(self, X=None) -> int:
        """Returns the number of splitting iterations in the cross-validator."""
        return 1

    def _iter_train_test_masks(self, X):
        """Notice a single fold (K=1) is generated."""
        train_start = X.min() if self.train_start is None else self.train_start
        train_date_range = self._get_train_date_range(train_start)
        test_date_range = self._get_test_date_range()
        yield X.isin(train_date_range), X.isin(test_date_range)

    def _get_train_date_range(self, start):
        return make_datetime_index(
            start=start,
            end=self.horizon.min(),
            freq=self.horizon.freq,
            inclusive="left",
        )

    def _get_test_date_range(self):
        encoder_date_range = self._get_encoder_date_range()
        return encoder_date_range.union(self.horizon)

    def _get_encoder_date_range(self):
        periods = self.encoder_length + 1
        return make_datetime_index(
            end=self.horizon.min(),
            periods=periods,
            inclusive="left",
            freq=self.horizon.freq,
        )
