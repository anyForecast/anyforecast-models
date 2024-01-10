import numpy as np
import pandas as pd


class OutputToPandasTransformer:
    """Transforms output to pandas DataFrame.

    Parameters
    ----------
    group_cols : list of str
        List of column names identifying a time series.

    output_col : str, default="output"
        Output column name for the returned dataframe.

    time_idx_col : str, default="time_idx"
        Time index column name for the returned dataframe,
    """

    def __init__(
        self,
        group_cols: list[str],
        output_col: str = "output",
        time_idx_col: str = "time_idx",
    ):
        self.group_cols = group_cols
        self.output_col = output_col
        self.time_idx_col = time_idx_col

    def transform(
        self, output: np.ndarray, decoded_index: pd.DataFrame
    ) -> pd.DataFrame:
        
        def create_df(output, time_idx, group_cols):
            """Creates dataframe for the current output."""
            df = pd.DataFrame(output, columns=[self.output_col])
            df[self.time_idx_col] = time_idx
            df[self.group_cols] = group_cols
            return df

        def gen_time_index(group: pd.DataFrame) -> range:
            """Generates time index values."""
            first_idx = group["time_idx_first_prediction"].item()
            last_idx = group["time_idx_last"].item()
            time_idx = range(first_idx, last_idx + 1)
            return time_idx

        def apply_fn(group: pd.DataFrame):
            i = group.name
            time_index = gen_time_index(group)
            group_cols = group[self.group_cols].values.flatten()
            print(i)
            print(time_index)
            print(group_cols)
            print('')
            return create_df(output[i], time_index, group_cols)

        decoded_index = decoded_index.reset_index(names="index")
        groupby = decoded_index.groupby("index", group_keys=False)
        pandas_output = groupby.apply(apply_fn)
        return pandas_output.reset_index(drop=True)
