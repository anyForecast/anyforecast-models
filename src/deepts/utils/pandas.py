from typing import List, Union

import pandas as pd


def loc_group(
    X: pd.DataFrame,
    group_cols: List[Union[int, str]],
    group_name: List[Union[int, str]],
) -> pd.DataFrame:
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
    return X[(X[group_cols].values == group_name).all(1)].copy()
