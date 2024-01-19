from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _check_feature_names_in

from anyforecast_models import base
from anyforecast_models.base import Transformer
from anyforecast_models.decorators import InputCheck
from anyforecast_models.preprocessing.compose._dtypes import (
    OutputDTypesResolver,
)
from anyforecast_models.preprocessing.compose._inverse_behaviors import (
    InverseBehavior,
    get_inverse_behavior,
)
from anyforecast_models.utils import checks


class PandasColumnTransformer(Transformer):
    """Pandas wrapper for sklearn :class:`ColumnTransformer`.

    This wrapper returns pandas DataFrames instead of numpy arrays in
    transform and inverse_transform methods.

    Parameters
    ----------
    column_transformer : sklearn.compose.ColumnTransformer
        sklearn ColumnTransformer

    Attributes
    ----------
    column_transformer_ : sklearn.compose.ColumnTransformer
        Fitted sklearn ColumnTransformer

    feature_dtypes_in_ : dict, str -> np.dtype
        Input feature dtypes.

    feature_names_in_ : np.ndarray, shape=(n,)
        Input feature names.
    """

    def __init__(self, column_transformer: ColumnTransformer):
        self.column_transformer = column_transformer

    @InputCheck(checks.check_is_frame)
    def fit(self, X: pd.DataFrame, y: None = None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data, from which specified subsets are used to fit the
            transformers.

        y : None
            Compatibility purposes.

        Returns
        -------
        self : PandasColumnTransformer
            This estimator.
        """
        column_transformer: ColumnTransformer = clone(self.column_transformer)
        column_transformer.fit(X)

        self.column_transformer_ = column_transformer
        self.feature_dtypes_in_: dict[str, np.dtype] = X.dtypes.to_dict()
        return self

    @InputCheck(checks.check_is_frame)
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input data and collects results in a pandas DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be transformed.

        Returns
        -------
        pd.DataFrame
        """
        Xt = self.column_transformer_.transform(X)
        return self.output_to_pandas(Xt)

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self.column_transformer_.feature_names_in_

    def output_to_pandas(self, output: np.ndarray) -> pd.DataFrame:
        """Converts transformed Numpy array into pandas DataFrame.

        Parameters
        ----------
        output : shape=(n, m)
            Transformed output ndarray.
        """
        columns = self.get_feature_names_out()
        dtypes = self.get_feature_dtypes_out()
        return pd.DataFrame(data=output, columns=columns).astype(dtypes)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms ``X`` separately by each transformer and
        concatenates results in a single pandas DataFrame.

        Transformed columns whose corresponding transformer does not have
        implemented an :meth:`inverse_transform` method will not appear
        after calling this inverse transformation. Hence, it is possible the
        resulting DataFrame ``X_out`` is not equal to the original X, that
        is, the expression X = f-1(f(X)) wont be satisfied.


        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed by subsets

        Returns
        -------
        X_out : pd.DataFrame
        """
        inverse_transformer = InverseColumnTransformer(self)
        return inverse_transformer.inverse_transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        return self.column_transformer_.get_feature_names_out(input_features)

    def get_feature_dtypes_out(self) -> dict[str, np.dtype]:
        """Get feature dtypes from all transformers.

        Returns
        -------
        feature_dtypes : dict, str -> dtype
        """
        self.check_is_fitted()
        resolver = OutputDTypesResolver(self.column_transformer_)
        feature_dtypes_out = resolver.resolve(self.feature_dtypes_in_)
        return feature_dtypes_out

    def iter(
        self,
        fitted: bool = False,
        column_as_labels: bool = False,
        skip_drop: bool = False,
        skip_empty_columns: bool = False,
    ):
        """Generates (name, trans, column, weight) tuples.

        Notes
        -----
        Wrapper for private method :meth:`ColumnTransformer._iter`

        Parameters
        ----------
        fitted : bool
            If True, use the fitted transformers (``self.transformers_``) to
            iterate through transformers, else use the transformers passed by
            the user (``self.transformers``).

        column_as_labels : bool
            If True, columns are returned as string labels. If False, columns
            are returned as they were given by the user. This can only be True
            if the ``ColumnTransformer`` is already fitted.

        skip_drop : bool
            If True, 'drop' transformers are filtered out.

        skip_empty_columns : bool
            If True, transformers with empty selected columns are filtered out.

        Yields
        ------
        A generator of tuples containing:
            - name : the name of the transformer
            - transformer : the transformer object
            - columns : the columns for that transformer
            - weight : the weight of the transformer
        """
        return self.column_transformer_._iter(
            fitted=fitted,
            column_as_labels=column_as_labels,
            skip_drop=skip_drop,
            skip_empty_columns=skip_empty_columns,
        )

    def check_feature_names_in(
        self, input_features: np.ndarray | None = None
    ) -> np.ndarray:
        return _check_feature_names_in(self.column_transformer_, input_features)

    def get_feature_name_out_for_transformer(
        self,
        name: str,
        trans: str,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        """Gets feature names of transformer.

        Notes
        -----
        Wrapper for private method
        :meth:`ColumnTransformer._get_feature_name_out_for_transformer`

        Returns
        -------
        feature_names_out : np.ndarray
        """

        return self.column_transformer_._get_feature_name_out_for_transformer(
            name=name, trans=trans, feature_names_in=input_features
        )


class InverseColumnTransformer:
    """Inverse transformation of a :class:`PandasColumnTransformer` instance.

    Parameters
    ----------
    column_transformer : compose.PandasColumnTransformer
        Fitted :class:`PandasColumnTransformer` instance.

    ignore_or_raise : str, {"ignore", "raise"}
        Behavior for when transformers do not have inverse_transform method.
    """

    def __init__(
        self,
        column_transformer: PandasColumnTransformer,
        ignore_or_raise: Literal["ignore", "raise"] = "ignore",
    ):
        self.column_transformer = column_transformer
        self.ignore_or_raise = ignore_or_raise

    @InputCheck(checks.check_is_frame)
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms X.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        Returns
        -------
        pd.DataFrame
        """
        self.column_transformer.check_is_fitted()
        input_features = self.column_transformer.check_feature_names_in(
            input_features=None
        )

        inverse_transforms: list[pd.DataFrame] = []
        for name, trans, columns, _ in self.column_transformer.iter(
            fitted=True,
            column_as_labels=True,
            skip_drop=True,
            skip_empty_columns=True,
        ):
            feature_names_out = self.get_feature_name_out_for_transformer(
                name=name, trans=trans, input_features=input_features
            )

            if feature_names_out is None:
                continue

            inverse = self.inverse_transform_features(
                X=X, name=name, trans=trans, features=feature_names_out
            )

            inverse = self.inverse_to_pandas(inverse, columns)
            inverse_transforms.append(inverse)

        return self.merge_inverse_transforms(inverse_transforms)

    def inverse_to_pandas(
        self,
        inverse: pd.DataFrame | np.ndarray,
        columns: np.ndarray | list[str],
    ) -> pd.DataFrame:
        """Converts inverse transformed data to a pandas DataFrame.

        Parameters
        ----------
        inverse : np.ndarray or pd.DataFrame
            Numpy 2-D array or pandas DataFrame.

        columns : array_like of str objects.
            Columns to assing when inverse data is a Numpy ndarray.
        """

        if checks.is_pandas_frame(inverse):
            inverse = inverse.reset_index(drop=True)
        else:
            inverse = pd.DataFrame(inverse, columns=columns)

        return inverse

    def inverse_transform_features(
        self,
        X: pd.DataFrame,
        name: str,
        trans: base.Transformer,
        features: np.ndarray,
    ) -> np.ndarray | pd.DataFrame:
        """Inverse transforms columns of a single transformer.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        name : str
            Transformer name.

        trans : Transformer
            Transformer instance.

        features : array_like of str objects.
            Features inside input data X to be inverse transformed.

        Returns
        -------
        inverse : pd.DataFrame
            Inverse transformed dataframe.
        """
        inverse_transformer = FeaturesInverseTransformer(
            name=name,
            trans=trans,
            features=features,
            ignore_or_raise=self.ignore_or_raise,
        )
        return inverse_transformer.inverse_transform(X)

    def merge_inverse_transforms(
        self, inverse_transforms: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Concatenates inverse transformed DataFrames into a single one.

        Parameters
        ----------
        inverse_transforms : list of pd.DataFrame
            List of inverse transformed dataframes.

        Returns
        -------
        merged : pd.DataFrame
            Merged inverse transformed pandas DataFrame.
        """
        merged = pd.concat(inverse_transforms, axis=1)
        dtypes = self.get_dtypes_in()

        # Select only dtypes inside merged data.
        dtypes = {col: dtype for col, dtype in dtypes.items() if col in merged}

        return merged.astype(dtypes)

    def get_feature_name_out_for_transformer(
        self,
        name: str,
        trans: base.Transformer,
        input_features: np.ndarray,
    ) -> np.ndarray:
        return self.column_transformer.get_feature_name_out_for_transformer(
            name=name, trans=trans, input_features=input_features
        )

    def get_dtypes_in(self) -> dict[str, np.dtype]:
        return self.column_transformer.feature_dtypes_in_


class FeaturesInverseTransformer:
    """Inverse transforms given features inside the input data.

    The inverse transform behavior is inferred from the given transformer
    (see :meth:`get_inverse_behavior`).

    Parameters
    ----------
    name : str
        Transformer name.

    trans : transformer estimator or {"passthrough", "drop"}
        Transformer instance or strings "passthrough" and "drop".

    features : array of str objects.
        Features inside input data X to be inverse transformed.

    ignore_or_raise : str, {"ignore", "raise"}
        Behavior for when transformers do not have inverse_transform method.
    """

    def __init__(
        self,
        name: str,
        trans: base.Transformer | Literal["passthrough"],
        features: np.ndarray,
        ignore_or_raise: Literal["ignore", "raise"],
    ) -> None:
        self.name = name
        self.trans = trans
        self.features = features
        self.ignore_or_raise = ignore_or_raise

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        """Inverse transforms input data.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        Returns
        -------
        Xi : 2-D numpy array
            Array with inverse transformed data.
        """
        inverse_behavior = self.get_inverse_behavior()
        return inverse_behavior.inverse_transform(X)

    def get_inverse_behavior(self) -> InverseBehavior:
        """Returns inverse behavior based on the given transformer.

        Returns
        -------
        inverse_behavior : InverseBehavior
        """
        cls = get_inverse_behavior(self.trans, self.ignore_or_raise)
        return cls(name=self.name, trans=self.trans, features=self.features)
