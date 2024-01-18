import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from anyforecast_models.base import Transformer
from anyforecast_models.decorators import CheckCols, MultiCheck
from anyforecast_models.utils import checks

from ._column import PandasColumnTransformer


class GroupedColumnTransformer(Transformer):
    """Transformer that transforms by groups.

    For each group, a :class:`PandasColumnTransformer` is fitted and
    applied.

    Notes
    -----
    The order of the columns in the transformed feature matrix follows the
    order of how the columns are specified in the transformers list. Since the
    passthrough kwarg is set, columns not specified in the transformers list
    are added at the right to the output.

    Parameters
    ----------
    transformers : tuples
        Tuples of the form (transformer, columns) specifying the
        transformer objects to be applied to subsets of the data.

    group_cols : list of str
        Column names identifying each group in the data.


    Attributes
    ----------
    column_transformers_ : dict, str -> PandasColumnTransformer
        Mapping from group_id to its corresponding fitted
        :class:`PandasColumnTransformer` object.
    """

    def __init__(
        self, column_transformer: ColumnTransformer, group_cols: list[str]
    ):
        self.column_transformer = column_transformer
        self.group_cols = group_cols

    @CheckCols(cols_attr="group_cols")
    @MultiCheck(checks=[checks.check_is_frame])
    def fit(self, X: pd.DataFrame, y=None):
        """Fits a :class:`ColumnTransformer` object to each group inside X.

        In other words, each group in X gets assigned its own
        :class:`PandasColumnTransformer` instance which is then fitted to the
        data inside such group.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
            Must contain ``group_cols`` column(s).

        y : None
            This param exists for compatibility purposes with sklearn.

        Returns
        -------
        self (object): Fitted transformer.
        """
        self.column_transformers_: dict[str, PandasColumnTransformer] = {}

        groupby = X.groupby(self.group_cols, group_keys=True, observed=False)
        for group_name in groupby.groups:
            column_transformer = self.create_column_transformer()
            group = groupby.get_group(group_name)
            column_transformer.fit(group)
            self.column_transformers_[group_name] = column_transformer

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms every group in X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            Must contain ``group_ids`` column(s).

        Returns
        -------
        Xt : pd.DataFrame.
            Transformed dataframe
        """
        return self._groupwise_transform(X)

    def create_column_transformer(self) -> PandasColumnTransformer:
        """Construct a new unfitted :class:`PandasColumnTransformer`.

        Returns
        -------
        column_transformer : PandasColumnTransformer
        """
        if not hasattr(self, "_pandas_ct"):
            self._pandas_ct = PandasColumnTransformer(self.column_transformer)
        return self._pandas_ct.clone()

    @CheckCols(cols_attr="group_cols")
    @MultiCheck(checks=[checks.check_is_frame], check_is_fitted=True)
    def _groupwise_transform(
        self, X: pd.DataFrame, inverse: bool = False
    ) -> pd.DataFrame:
        """Group-wise transformation.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            Must contain ``group_ids`` column(s).

        inverse : bool, default=False
            If True, the inverse-transform function will be used.

        Returns
        -------
        Xt : pd.DataFrame.
            Transformed dataframe
        """
        groupby = X.groupby(self.group_cols, group_keys=True, observed=False)

        def apply_fn(group: pd.DataFrame) -> pd.DataFrame:
            """Applies transformation function to a single group.

            Parameters
            ----------
            group : pd.DataFrame
                Subset of X containing data for a single group.

            Returns
            -------
            pd.DataFrame
                Transformed group.
            """
            if group.name not in self.column_transformers_:
                return group

            ct = self.column_transformers_[group.name]
            transform_func = ct.inverse_transform if inverse else ct.transform
            return transform_func(group)

        return groupby.apply(apply_fn).reset_index(drop=True)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms input data.

        Notes
        -----
        Transformed columns whose corresponding transformer does not have
        implemented an :meth:`inverse_transform` method will not appear after
        calling this inverse transformation. This causes that the returned
        DataFrame might not be equal to the original X, that is, the
        expression X = f-1(f(X)) won't be satisfied.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
            Must contain ``group_ids`` column(s).

        Returns
        -------
        Xi : pd.DataFrame
            Inverse transformed dataframe.
        """
        return self._groupwise_transform(X, inverse=True)

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self._pandas_ct.feature_names_in_

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return self._pandas_ct.get_feature_names_out(input_features)
