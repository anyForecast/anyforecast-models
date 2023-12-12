import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def _is_single_feature(X) -> bool:
    """Returns True if input data has a single feature.

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return X.shape[1] == 1


def _is_1d(X) -> bool:
    """Returns True if input data in 1-dimensional

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return len(X.shape) == 1


def _is_pandas_series(X) -> bool:
    """Returns True if input data is instance of :class:`pd.Series`.

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return isinstance(X, pd.Series)


def _is_pandas_frame(X) -> bool:
    """Returns True if input data is instance of :class:`pd.DataFrame`.

    Parameters
    ----------
    X : array_like

    Returns
    -------
    bool
    """
    return isinstance(X, pd.DataFrame)


def _is_datetime(X) -> bool:
    return is_datetime64_any_dtype(X)


def make_check_cols(cols: list[str]) -> callable:
    def check_cols(X: pd.DataFrame):
        missing = set(cols) - set(X)
        if missing:
            raise ValueError()

    return check_cols


def check_is_1d(X) -> None:
    """Validates input data is 1-dimensional"""
    if not _is_1d(X):
        raise ValueError()


def check_1_feature(X) -> None:
    """Validates input data contains a single feature."""

    if not _is_single_feature(X):
        raise ValueError()


def check_is_series(X) -> None:
    """Validates input data is an instance of :class:`pd.Series`"""
    if not _is_pandas_series(X):
        raise ValueError()


def check_is_frame(X) -> None:
    """Validates input data is an instance of  :class:`pd.DataFrame`"""
    if not _is_pandas_frame(X):
        raise ValueError()


def check_is_datetime(X) -> None:
    """Validates input data is datetime"""
    if not _is_datetime(X):
        raise ValueError()