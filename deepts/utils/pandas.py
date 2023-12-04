from typing import Union, List

import numpy as np
import pandas as pd


def series_to_2d(series: pd.Series) -> np.array:
    """Converts pandas Series to 2D Numpy array.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    array : np.array
        2D Numpy array
    """
    return series.values.reshape(-1, 1)


def loc_group(
        X: pd.DataFrame,
        group_cols: List[Union[int, str]],
        group_id: List[Union[int, str]]
):
    """Auxiliary for locating rows in dataframes with one or multiple group_ids.

    Parameters
    ----------
    X : pd.DataFrame
        Dataframe to filter.

    group_cols: list
        List of columns names.

    group_id : list
        Group id of the wanted group.

    Returns
    -------
    pd.DataFrame
    """
    # Broadcasted numpy comparison.
    return X[(X[group_cols].values == group_id).all(1)].copy()
