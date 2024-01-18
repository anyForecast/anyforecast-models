from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone


class Transformer(BaseEstimator, TransformerMixin):
    """Base Transformer."""

    def get_feature_names_out(self) -> np.ndarray:
        """Returns transformed feature names."""
        pass

    def clone(self) -> Transformer:
        """Construct a new unfitted transformer with the same parameters.

        Calls :meth:`sklearn.base.clone` on this transformer instance.
        """
        return clone(self)

    def validate_data(
        self,
        X: np.ndarray | pd.DataFrame,
        reset: bool = True,
        cast_to_ndarray: bool = True,
        **check_params,
    ) -> np.ndarray:
        """Validate input data and set or check the `n_features_in_` attribute.

        Calls private method :meth:`sklearn.base.BaseEstimator._validate_data`.

        Parameters
        ----------
        X : array_like
            Data to validate.

        reset : bool, default=True
            Whether to reset the `n_features_in_` attribute.

        cast_to_ndarray : bool, default=True
            Cast `X` to ndarray with checks in `check_params`. If
            `False`, `X` is unchanged and only `feature_names_in_` and
            `n_features_in_` are checked.

        **check_params : kwargs
            Parameters passed to :func:`sklearn.utils.check_array`
        """
        return self._validate_data(
            X=X, reset=reset, cast_to_ndarray=cast_to_ndarray, **check_params
        )

    def check_is_fitted(self):
        """Perform is_fitted validation for this transformer.

        Checks if this transformer is fitted by verifying the presence of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError

        Raises
        ------
        NotFittedError if not fitted.
        """
        sklearn.utils.validation.check_is_fitted(self)


class InvertibleTransformer(Transformer):
    """Transformer with inverse transform method."""

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform body.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.
        """
        pass
