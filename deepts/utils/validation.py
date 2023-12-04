import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_datetime64_any_dtype
from pytorch_forecasting.data import timeseries


def check_nd(arr: np.ndarray):
    if not arr.ndim >= 2:
        raise


def check_is_fitted(estimator: sklearn.base.BaseEstimator) -> None:
    """Perform is_fitted validation for estimator.

    Checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore) and otherwise raises
    a NotFittedError with the given message.

    Raises
    ------
    NotFittedError if not fitted.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
    """
    sklearn.utils.validation.check_is_fitted(estimator)


def bool_check_is_fitted(estimator: sklearn.base.BaseEstimator) -> bool:
    """Returns True if estimator is fitted else False.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator

    Returns
    -------
    is_fitted : bool
        True if estimator is fitted else False.
    """
    try:
        check_is_fitted(estimator)
    except sklearn.exceptions.NotFittedError:
        return False
    return True


def check_is_finite(tensor, names):
    """Checks if 2D tensor contains NAs or infinite values.

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check

    names : Union[str, List[str]]
        Name(s) of column(s) (used for error messages)

    Returns
    -------
    torch.Tensor: returns tensor if checks yield no issues
    """
    return timeseries.check_for_nonfinite(tensor, names)


def check_is_datetime(X):
    """Checks if passed pandas `series` is datetime64 compatible.
    """
    if not is_datetime64_any_dtype(X):
        raise ValueError('Series is not datetime64 compatible.')


def check_pandas(X, series_or_dataframe, transformer):
    pandas_types = {
        'series': pd.Series,
        'dataframe': pd.DataFrame
    }
    pandas_type_to_check = pandas_types[series_or_dataframe]
    if not isinstance(X, pandas_type_to_check):
        fmt = "Transformer {transformer} only accepts {correct_type}. " \
              "Instead got {wrong_type}"
        fmt_kwargs = {
            'transformer': transformer.__class__.__name__,
            'correct_type': pandas_type_to_check,
            'wrong_type': type(X)
        }
        raise TypeError(fmt.format(**fmt_kwargs))


def check_group_ids(X, group_ids):
    """Checks `group_ids` columns are present in `X`.
    """
    msg = 'group_id column {} not found in X.'
    for col in group_ids:
        if col not in X:
            raise ValueError(msg.format(col))
