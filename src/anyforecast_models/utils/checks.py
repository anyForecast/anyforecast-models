from typing import Any

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
from torch.nn.modules import rnn

from anyforecast_models import base, exceptions


def is_lstm(rnn_cell: Any) -> bool:
    """Returns True if ``rnn_cell`` is of type LSTM.

    Parameters
    ----------
    rnn_cell : Any

    Returns
    -------
    bool
    """
    return isinstance(rnn_cell, (rnn.LSTM, rnn.LSTMCell))


def is_invertible(trans: Any) -> bool:
    return hasattr(trans, "inverse_transform")


def is_gru(rnn_cell: Any) -> bool:
    """Returns True if ``rnn_cell`` is of type GRU.

    Parameters
    ----------
    rnn_cell : Any

    Returns
    -------
    bool
    """
    return isinstance(rnn_cell, (rnn.GRU, rnn.GRUCell))


def is_single_feature(X) -> bool:
    """Returns True if input data has a single feature.

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return X.shape[1] == 1


def is_1d(X) -> bool:
    """Returns True if input data in 1-dimensional

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return len(X.shape) == 1


def is_pandas_series(X) -> bool:
    """Returns True if input data is instance of :class:`pd.Series`.

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return isinstance(X, pd.Series)


def is_pandas_frame(X) -> bool:
    """Returns True if input data is instance of :class:`pd.DataFrame`.

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return isinstance(X, pd.DataFrame)


def is_datetime(X) -> bool:
    return is_datetime64_any_dtype(X)


def check_is_lstm_or_gru(rnn_cell: Any) -> None:
    """Checks passed object is either LSTM or GRU."""
    if not (is_lstm(rnn_cell) or is_gru(rnn_cell)):
        raise TypeError(
            "Passed RNN object must be either LSTM or GRU. Instead got"
            f" {rnn.__class__.__name__}"
        )


def check_is_1d(X) -> None:
    """Checks input data is 1-dimensional"""
    if not is_1d(X):
        raise ValueError()


def check_1_feature(X) -> None:
    """Validates input data contains a single feature."""

    if not is_single_feature(X):
        raise ValueError()


def check_is_series(X) -> None:
    """Validates input data is an instance of :class:`pd.Series`"""
    if not is_pandas_series(X):
        raise ValueError()


def check_is_frame(X) -> None:
    """Validates input data is an instance of  :class:`pd.DataFrame`"""
    if not is_pandas_frame(X):
        raise ValueError()


def check_is_datetime(X) -> None:
    """Validates input data is datetime"""
    if not is_datetime(X):
        raise ValueError()


def check_inverse_features(
    X: pd.DataFrame,
    name: str,
    trans: base.Transformer,
    features: list[str],
) -> None:
    """Checks X has the required features/columns for inverse transform.

    Raises
    ------
    InverseTransformerFeaturesError when there are missing
    features/columns in X.
    """
    missing = set(features) - set(X)

    if missing:
        raise exceptions.InverseTransformFeaturesError(
            name=name, type=type(trans).__name__, missing_features=missing
        )
