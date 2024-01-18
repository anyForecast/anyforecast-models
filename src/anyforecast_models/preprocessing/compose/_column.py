from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _check_feature_names_in

from anyforecast_models import base
from anyforecast_models.base import Transformer
from anyforecast_models.decorators import MultiCheck
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

    @MultiCheck(checks=[checks.check_is_frame])
    def fit(self, X: pd.DataFrame, y: None = None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data, from which specified subsets are used to fit the
            transformers.

        y : None

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

    @MultiCheck(checks=[checks.check_is_frame])
    def transform(
        self, X: pd.DataFrame, to_pandas: bool = True
    ) -> pd.DataFrame:
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
        return self.output_to_pandas(Xt) if to_pandas else Xt

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

    def get_feature_names_out(self, input_features=None) -> np.array:
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
        self, fitted=False, replace_strings=False, column_as_strings=False
    ):
        """Generates (name, trans, column, weight) tuples.

        Notes
        -----
        Wrapper for private method :meth:`ColumnTransformer._iter`
        """
        return self.column_transformer_._iter(
            fitted=fitted,
            replace_strings=replace_strings,
            column_as_strings=column_as_strings,
        )

    def get_feature_name_out_for_transformer(
        self,
        name: str,
        trans: str,
        column: list[str],
        input_features: list[str] = None,
    ):
        """Gets feature names of transformer.

        Used in conjunction with self.iter(fitted=True)

        Notes
        -----
        Wrapper for private method
        :meth:`ColumnTransformer._get_feature_name_out_for_transformer`

        Returns
        -------
        feature_names_out : list
        """
        input_features = _check_feature_names_in(
            self.column_transformer_, input_features
        )
        return self.column_transformer_._get_feature_name_out_for_transformer(
            name, trans, column, input_features
        )


class InverseTransformer:
    """Inverse transforms input data using __init__ transformer.

    The inverse transfrom behavior is inferred from the given transformer
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
        trans: base.Transformer | Literal["drop", "passthrough"],
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
        column_transformer.check_is_fitted()
        self.column_transformer = column_transformer
        self.ignore_or_raise = ignore_or_raise

    @MultiCheck(checks=[checks.check_is_frame])
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms X.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        Returns
        -------
        Xi : pd.DataFrame or dict[str, pd.DataFrame]
        """
        inverse_transforms: list[pd.DataFrame] = []

        for name, trans, features_in, _ in self.column_transformer.iter(
            replace_strings=False, fitted=True, column_as_strings=True
        ):
            Xi = self._inverse_transform(X, name, trans, features_in)

            if checks.is_pandas_frame(Xi):
                Xi = Xi.reset_index(drop=True)
            else:
                Xi = pd.DataFrame(Xi, columns=features_in)

            inverse_transforms.append(Xi)

        return self.merge_inverse_transforms(inverse_transforms)

    def _inverse_transform(
        self,
        X: pd.DataFrame,
        name: str,
        trans: base.Transformer,
        features_in: np.ndarray,
    ) -> np.ndarray:
        """Inverse transforms ``trans`` features.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        name : str
            Transformer name.

        trans : Transformer
            Transformer instance.

        features_in : np.ndarray
            Input features. These are the features originally
            transformed by the given ``trans``.

        Returns
        -------
        Xi : pd.DataFrame
            Inverse transformed dataframe.
        """
        features_out = self.get_features_out(name, trans, features_in)
        inverse_trans = InverseTransformer(
            name, trans, features_out, self.ignore_or_raise
        )
        return inverse_trans.inverse_transform(X)

    def get_features_out(
        self,
        name: str,
        trans: base.Transformer,
        column: list[str],
    ) -> np.ndarray:
        """Returns output features for transformer."""
        return self.column_transformer.get_feature_name_out_for_transformer(
            name=name, trans=trans, column=column
        )

    def get_dtypes_in(self) -> dict[str, np.dtype]:
        return self.column_transformer.feature_dtypes_in_

    def merge_inverse_transforms(
        self, inverse_transforms: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """Concatenates inverse transformed DataFrames into a single one."""
        merged = pd.concat(inverse_transforms, axis=1)
        dtypes = self.get_dtypes_in()

        # Select only dtypes inside merged data.
        dtypes = {col: dtype for col, dtype in dtypes.items() if col in merged}

        return merged.astype(dtypes)
