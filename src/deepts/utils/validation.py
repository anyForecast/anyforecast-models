from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
import torch
from pandas.api.types import is_datetime64_any_dtype
from pytorch_forecasting.data import timeseries


def check_is_nd(arr: np.ndarray) -> None:
    """Checks array in n-dimensional."""
    if not arr.ndim >= 2:
        raise


def check_is_finite(tensor: torch.Tensor, names: str | list[str]):
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


def check_is_datetime(series: pd.Series) -> None:
    """Checks if passed pandas `series` is datetime64 compatible."""
    if not is_datetime64_any_dtype(series):
        raise ValueError("Series is not datetime64 compatible.")


def check_cols(frame: pd.DataFrame, cols: list[str]) -> None:
    """Checks cols are not missing in pandas DataFrame."""
    missing = set(cols) - set(frame)
    if missing:
        raise ValueError(f"The following columns are missing {missing}")
