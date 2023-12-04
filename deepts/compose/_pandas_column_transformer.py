from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.compose._column_transformer import (
    _check_feature_names_in,
    _is_empty_column_selection,
)
from sklearn.utils.metaestimators import _BaseComposition

from ..exceptions import InverseTransformFeaturesError
from ..preprocessing import Transformer
from ..utils import data, validation


class PandasColumnTransformer(TransformerMixin, _BaseComposition):
    """Pandas wrapper for sklearn :class:`ColumnTransformer`.

    This wrapper returns pandas DataFrames instead of numpy arrays in
    transform and inverse_transform methods.

    Parameters
    ----------
     transformers : list of 2-tuples
        Tuples of the form (transformer, columns) specifying the
        transformer objects to be applied to subsets of the data.

    Attributes
    ----------
    column_transformer_ : sklearn.compose.ColumnTransformer
    """

    def __init__(self, transformers: list[tuple], int_to_float: bool = False):
        self.transformers = transformers
        self.int_to_float = int_to_float

        self._inverse_transformer = ColumnTransformerInverseTransformer(self)
        self._dtypes_inferencer = DTypesInferencer(self, int_to_float)

    def fit(self, X: pd.DataFrame, y: None = None) -> PandasColumnTransformer:
        """Fit all transformers using X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data, of which specified subsets are used to fit the
            transformers.

        y : None

        Returns
        -------
        self : PandasColumnTransformer
            This estimator.
        """
        self.column_transformer_ = self.make_column_transformer()
        self.column_transformer_.fit(X)
        self.feature_dtypes_in_ = X.dtypes.to_dict()
        return self

    @property
    def feature_names_in_(self):
        return self.column_transformer_.feature_names_in_

    def make_pandas(self, X: np.ndarray) -> pd.DataFrame:
        """Returns pandas DataFrame using data from X.

        Columns and dtypes are obtained from methods
        :meth:`get_feature_names_out` and :meth:`get_feature_dtypes_out`,
        respectively.
        """
        columns = self.get_feature_names_out()
        dtypes = self.get_feature_dtypes_out()
        return pd.DataFrame(X, columns=columns).astype(dtypes)

    def make_column_transformer(
        self,
        sparse_threshold=0,
        remainder="passthrough",
        verbose_feature_names_out=False,
    ) -> ColumnTransformer:
        """Factory for :class:`sklearn.compose.ColumnTransformer` instances.

        Returns
        -------
        column_transformer
        """
        return make_column_transformer(
            *self.transformers,
            sparse_threshold=sparse_threshold,
            remainder=remainder,
            verbose_feature_names_out=verbose_feature_names_out,
        )

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms ``X`` separately by each transformer and concatenates
        results in a single pandas DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be transformed by subsets.

        Returns
        -------
        pd.DataFrame
        """
        Xt = self.column_transformer_.transform(X)
        return self.make_pandas(Xt)

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
        return self._inverse_transformer.transform(X)

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
        return self._dtypes_inferencer.get_feature_dtypes_out()

    def iter(self, fitted=True, replace_strings=False, column_as_strings=True):
        """Generates (name, trans, column, weight) tuples.

        Notes
        -----
        This is a wrapper for the sklearn :meth:`ColumnTransformer._iter`
        private method.

        Yields
        ------
        (name, trans, column, weight) tuples
        """
        return self.column_transformer_._iter(
            fitted=fitted,
            replace_strings=replace_strings,
            column_as_strings=column_as_strings,
        )

    def get_feature_name_out_for_transformer(
        self,
        name: str,
        trans: str | Transformer,
        column: list[str],
        input_features: list[str] = None,
    ):
        """Gets feature names of transformer.

        Used in conjunction with self.iter(fitted=True)

        Notes
        -----
        Thi is wrapper for the sklearn
        :class:`ColumnTransformer._get_feature_name_out_for_transformer`
        private method.

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


class DTypesInferencer:
    """Infers output dtypes for a :class:`ColumnTransformer` instance.

    Output dtypes are either inferred from the column transformer input dtypes
    i.e., ``column_transformer.feature_dtypes_in_``, or from the actual fitted
    transformers in case they implement a ``dtype`` attribute.

    Parameters
    ----------
    column_transformer : ColumnTransformer or PandasColumnTransformer
        ColumnTransformer or PandasColumnTransformer object.

    int_to_float : bool
        If True, convert int dtypes to float.
    """

    def __init__(
        self,
        column_transformer: PandasColumnTransformer | ColumnTransformer,
        int_to_float: bool = True,
    ):
        self.column_transformer = column_transformer
        self.int_to_float = int_to_float

    @property
    def feature_dtypes_in_(self):
        """Input features dtypes."""
        return self.column_transformer.feature_dtypes_in_

    def get_feature_dtypes_out(self) -> dict[str, np.dtype]:
        """Returns feature dtypes from all transformers.

        Returns
        -------
        dtypes : dict, str -> dtype
        """
        dtypes_out = {}
        for name, trans, column, _ in self.column_transformer.iter():
            trans_dtype_out = self.get_feature_dtypes_out_for_transformer(
                name, trans, column
            )
            dtypes_out.update(trans_dtype_out)

        return dtypes_out

    def get_feature_dtypes_out_for_transformer(
        self, name: str, trans: str | Transformer, column: list[str]
    ) -> dict[str, np.dtype]:
        """Gets feature dtypes of transformer.

        Used in conjunction with self._iter(fitted=True) in
        get_feature_dtypes_out.

        Same logic as in sklearn private method
        :meth:`ColumnTransformer._get_feature_name_out_for_transformer`

        Parameters
        ----------
        name : str
            Transformer name.

        trans : str or Transformer
            String ("drop" or "passthrough") or an actual transformer object.

        column : str
            Column names transformed.
        """
        if trans == "drop" or _is_empty_column_selection(column):
            dtypes_out = {}
        elif trans == "passthrough":
            dtypes_out = {c: self.feature_dtypes_in_[c] for c in column}
        else:  # An actual estimator/transformer object.
            dtypes_out = self.get_dtypes_out_for_estimator(name, trans)

        return dtypes_out

    def get_dtypes_out_for_estimator(
        self, name: str, trans: Transformer
    ) -> dict[str, np.dtype]:
        """Returns dtypes for an actual transformer object.

        Parameters
        ----------
        name : str
            Transformer name.

        trans : Transformer
            An actual transformer/estimator.

        Raises
        ------
        AttributeError when dtypes cannot be inferred.
        """
        dtypes_out = {}
        features_out = trans.get_feature_names_out()
        for feature in features_out:
            if not hasattr(trans, "dtype"):
                if feature not in self.feature_dtypes_in_:
                    raise AttributeError(
                        f"Cannot obtain dtype for Transformer "
                        f"{name} (type {type(trans).__name__}) since it "
                        f"does not provide attribute `dtype` and "
                        "features_names_in_ != feature_names_out_."
                    )
                dtypes_out[feature] = self.feature_dtypes_in_[feature]
            else:
                dtypes_out[feature] = trans.dtype

        if self.int_to_float:
            dtypes_out = self.map_int_to_float(dtypes_out)

        return dtypes_out

    def map_int_to_float(
        self, dtypes: dict[str, np.dtype]
    ) -> dict[str, np.dtype]:
        """Maps int dtypes to floats."""
        return {
            k: np.dtype("float") if data.is_int(dtype) else dtype
            for k, dtype in dtypes.items()
        }


class ColumnTransformerInverseTransformer:
    """Inverse transformation of a :class:`PandasColumnTransformer` instance.

    Parameters
    ----------
    column_transformer : PandasColumnTransformer
        Fitted :class:`PandasColumnTransformer` instance.
    """

    def __init__(
        self,
        column_transformer: PandasColumnTransformer,
        non_inverse_transformers: str = "ignore",
    ):
        self.column_transformer = column_transformer
        self.non_inverse_transformers = non_inverse_transformers

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transforms X.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        Returns
        -------
        inverse : pd.DataFrame
        """
        validation.check_is_fitted(self.column_transformer)

        inverse_transforms = []
        columns = []
        for name, trans, features_in, _ in self.column_transformer.iter():
            if isinstance(features_in, str):
                features_in = [features_in]

            features_out = self.get_feature_name_out_for_transformer(
                name, trans, features_in
            )

            if features_out is not None:
                x = self.check_transformer_features(
                    X, name, trans, features_out
                )

                # Since we allow "passthrough" to continue with missing
                # features, is possible that len(x.columns) < len(features_in).
                # Continuing without changes will cause the final column
                # labels for pandas to be more than needed, so set
                # ``features_in`` to whatever is in x.
                if trans == "passthrough":
                    features_in = x.columns

                # Only consider non-empty inverse transformations. Empty
                # inverse transformations are possible for transformers without
                # an :meth:`inverse_transform` and
                # :attr:`non_inverse_transformers` is set to "ignore".
                inverse_transform = self.inverse_transform(x, name, trans)
                if inverse_transform.size != 0:
                    inverse_transforms.append(inverse_transform)
                    columns.extend(features_in)

        return self.make_pandas(inverse_transforms, columns)

    def get_feature_name_out_for_transformer(
        self, name: str, trans: str | Transformer, features_in: list[str]
    ) -> np.ndarray:
        """Gets feature names of transformer."""
        return self.column_transformer.get_feature_name_out_for_transformer(
            name, trans, column=features_in, input_features=None
        )

    def check_transformer_features(
        self,
        X: pd.DataFrame,
        name: str,
        trans: str | Transformer,
        features: np.ndarray | list[str],
    ) -> pd.DataFrame:
        """Checks ``X`` contains the required features/columns for inverse
        transformation.

        Raises
        ------
        InverseTransformerFeaturesError when there are missing
        features/columns in X.

        Notes
        -----
        Only "passthrough" transformation is allowed to have missing features.

        Returns
        -------
        X : pd.DataFrame
            Subset of X containing the passed features.
        """
        missing_features = self.get_missing_features(X, features)
        if missing_features:
            if trans == "passthrough":
                # Only "passthrough" transformation is allowed to have
                # missing features (only the ones present in X are returned).
                intersection = self.get_intersection(X, features)
                return X[intersection]

            raise InverseTransformFeaturesError(
                name=name,
                type=type(trans).__name__,
                missing_features=missing_features,
            )

        return X[features]

    def make_pandas(
        self, inverse_transforms: list[np.ndarray], columns: list[np.ndarray]
    ) -> pd.DataFrame:
        dtypes = self.get_feature_dtypes_in(columns=columns)
        array = data.hstack(inverse_transforms)
        return pd.DataFrame(array, columns=columns).astype(dtypes)

    def get_feature_dtypes_in(self, columns=None):
        """Returns feature dtypes in from ``self.pd_column_transformer``."""
        dtypes_in = self.column_transformer.feature_dtypes_in_

        if columns is not None:
            dtypes_in = {
                col: dtype for col, dtype in dtypes_in.items() if col in columns
            }
        return dtypes_in

    def get_missing_features(self, X, features):
        """Returns features/columns present in ``features`` but not in  ``X``."""
        return list(set(features) - set(X))

    def get_intersection(self, X, features):
        """Returns features present in both ``features`` and ``X``."""
        return list(set(features).intersection(set(X)))

    def inverse_transform(
        self, X: pd.DataFrame, name: str, trans: str | Transformer
    ) -> np.ndarray:
        """Performs an inverse transformation.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        name : str
            Transformer name.

        trans : transformer estimator or 'passthrough'
            Transformer.

        Returns
        -------
        inverse_transform. : 2-D numpy array
            Array with inverse transformed data.
        """
        if trans == "passthrough":
            return X.values

        if hasattr(trans, "inverse_transform"):
            return trans.inverse_transform(X)
        else:
            if self.non_inverse_transformers == "ignore":
                # Return empty array.
                return np.ndarray(shape=(len(X), 0))

            elif self.non_inverse_transformers == "raise":
                raise AttributeError(
                    f"Transformer {name} (type {type(trans).__name__}) does "
                    "not provide `inverse_transform` method."
                )

            else:
                raise ValueError(
                    '`non_inverse_transformers` param can either be "ignore" '
                    f'or "raise". Instead got {self.non_inverse_transformers}'
                )
