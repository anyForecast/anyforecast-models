from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from deepts.base import Transformer
from deepts.decorators import MultiCheck
from deepts.utils import checks

from ._dtypes import OutputDTypesResolver
from ._inverse import ColumnTransformerInverseTransform


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
    """

    def __init__(self, column_transformer: ColumnTransformer):
        self.column_transformer = column_transformer

    @MultiCheck(checks=[checks.check_is_frame])
    def fit(self, X: pd.DataFrame, y: None = None):
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
        column_transformer: ColumnTransformer = clone(self.column_transformer)
        column_transformer.fit(X)

        self.column_transformer_ = column_transformer
        self.feature_dtypes_in_: dict[str, np.dtype] = X.dtypes.to_dict()
        return self

    @MultiCheck(checks=[checks.check_is_frame])
    def transform(self, X: pd.DataFrame, to_frame: bool = True) -> pd.DataFrame:
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
        return self.to_frame(Xt) if to_frame else Xt

    def to_frame(self, Xt: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(data=Xt, columns=self.get_feature_names_out())
        return frame.astype(self.get_feature_dtypes_out())

    @property
    def feature_names_in_(self) -> np.ndarray:
        return self.column_transformer_.feature_names_in_

    def inverse_transform(
        self,
        X: pd.DataFrame,
        non_inverse_behavior: Literal["raise", "ignore"] = "ignore",
    ) -> pd.DataFrame:
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
        inverse_transformer = ColumnTransformerInverseTransform(
            column_transformer=self.column_transformer_,
            non_inverse_behavior=non_inverse_behavior,
        )
        return inverse_transformer.transform(X)

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
