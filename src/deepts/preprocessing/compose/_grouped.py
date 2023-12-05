import pandas as pd
from sklearn.base import TransformerMixin, clone
from sklearn.utils.metaestimators import _BaseComposition

from deepts.utils import validation
from deepts.utils.pandas import loc_group

from ._column import PandasColumnTransformer


class GroupedColumnTransformer(TransformerMixin, _BaseComposition):
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

    def __init__(self, transformers: list[tuple], group_cols: list[str]):
        self.transformers = transformers
        self.group_cols = group_cols

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
        validation.check_group_ids(X, self.group_ids)
        self.column_transformers_: dict[str, PandasColumnTransformer] = {}

        groups = X.groupby(self.group_ids).groups
        for g in groups:
            column_transformer = self.make_column_transformer()
            group = loc_group(X, self.group_ids, g)
            column_transformer.fit(group)
            self.column_transformers_[g] = column_transformer

        return self

    def make_column_transformer(self) -> PandasColumnTransformer:
        """Construct a new unfitted :class:`PandasColumnTransformer`.

        Returns
        -------
        column_transformer : PandasColumnTransformer
        """
        if not hasattr(self, "_column_transformer"):
            self._column_transformer = PandasColumnTransformer(
                self.transformers
            )
        return clone(self._column_transformer)

    def transform(self, X):
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
        validation.check_is_fitted(self)
        validation.check_group_ids(X, self.group_ids)

        transformed_data = []
        for group_id, column_transformer in self.column_transformers_.items():
            group = loc_group(X, self.group_ids, group_id)
            if not group.empty:
                transformed_group = column_transformer.transform(group)
                transformed_data.append(transformed_group)

        return pd.concat(transformed_data).reset_index(drop=True)

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
        validation.check_is_fitted(self)
        validation.check_group_ids(X, self.group_ids)

        inverse_transforms = []
        for group_id, column_transformer in self.column_transformers_.items():
            group = loc_group(X, self.group_ids, group_id)
            if not group.empty:
                inverse = column_transformer.inverse_transform(group)
                inverse_transforms.append(inverse)

        return pd.concat(inverse_transforms)

    def iter(self, fitted=True, replace_strings=False, column_as_strings=True):
        return self._column_transformer.iter(
            fitted, replace_strings, column_as_strings
        )

    @property
    def feature_names_in_(self):
        return self._column_transformer.feature_names_in_

    def get_feature_names_out(self, input_features=None):
        return self._column_transformer.get_feature_names_out(input_features)
