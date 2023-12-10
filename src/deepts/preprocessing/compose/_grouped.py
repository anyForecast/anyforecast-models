import pandas as pd
from sklearn.compose import ColumnTransformer

from deepts.base import Transformer
from deepts.utils import checks, pandas
from deepts.decorators import check_cols, check

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

    @check_cols(cols="group_cols")
    @check(checks=[checks.check_is_frame])
    def fit(self, X: pd.DataFrame, y=None):
        """Fits a sklearn ColumnTransformer object to each group inside ``X``.

        In other words, each group in ``X`` gets assigned its own
        :class:`PandasColumnTransformer` instance which is then fitted to the
        data inside such group.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to fit.
            Must contain ``group_ids`` column(s).

        y : None
            This param exists for compatibility purposes with sklearn.

        Returns
        -------
        self (object): Fitted transformer.
        """
        self.column_transformers_: dict[str, PandasColumnTransformer] = {}

        groupby = X.groupby(self.group_cols)
        for group_name in groupby.groups:
            column_transformer = self.make_column_transformer()
            group = groupby.get_group(group_name)
            column_transformer.fit(group)
            self.column_transformers_[group_name] = column_transformer

        return self

    def make_column_transformer(self) -> PandasColumnTransformer:
        """Construct a new unfitted :class:`PandasColumnTransformer`.

        Returns
        -------
        column_transformer : PandasColumnTransformer
        """
        if not hasattr(self, "_pandas_ct"):
            self._pandas_ct = PandasColumnTransformer(self.column_transformer)
        return self._pandas_ct.clone()

    @check_cols(cols="group_cols")
    @check(checks=[checks.check_is_frame], check_is_fitted=True)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms every group in X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data to transform.
            Must contain ``group_ids`` column(s).

        Returns
        -------
        X_out : pd.DataFrame.
            Transformed dataframe
        """
        transformed_groups: list[pd.DataFrame] = []
        groupby = X.groupby(self.group_cols)

        for group_name in groupby.groups:
            if group_name not in self.column_transformers_:
                continue

            group = groupby.get_group(group_name)
            ct = self.column_transformers_[group_name]
            transformed_group = ct.transform(group)
            transformed_groups.append(transformed_group)

        return pd.concat(transformed_groups).reset_index(drop=True)

    @check_cols(cols="group_cols")
    @check(checks=[checks.check_is_frame], check_is_fitted=True)
    def inverse_transform(self, X):
        """Inverse transformation.

        Notes
        -----
        Transformed columns whose corresponding transformer does not have
        implemented an :meth:`inverse_transform` method will not appear after
        calling this inverse transformation. This causes that the resulting
        DataFrame ``X_out`` might not be equal to the original X, that is, the
        expression X = f-1(f(X)) wont be satisfied.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.

        Returns
        -------
        X_inv : pd.DataFrame
            Inverse transformed dataframe
        """
        inverse_transforms = []
        for group_id, column_transformer in self.column_transformers_.items():
            group = pandas.loc_group(X, self.group_cols, group_id)
            if not group.empty:
                inverse = column_transformer.inverse_transform(group)
                inverse_transforms.append(inverse)

        return pd.concat(inverse_transforms)

    def iter(self, fitted=True, replace_strings=False, column_as_strings=True):
        return self._pandas_ct.iter(fitted, replace_strings, column_as_strings)

    @property
    def feature_names_in_(self):
        return self._pandas_ct.feature_names_in_

    def get_feature_names_out(self, input_features=None):
        return self._pandas_ct.get_feature_names_out(input_features)
