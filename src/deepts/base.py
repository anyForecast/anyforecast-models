from __future__ import annotations

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone


class Transformer(BaseEstimator, TransformerMixin):
    """Base Transformer."""

    def get_feature_names_out(self):
        pass

    def clone(self) -> Transformer:
        """Construct a new unfitted transformer with the same parameters.

        Calls :meth:`sklearn.base.clone` on this transformer instance.
        """
        return clone(self)

    def validate_data(
        self,
        X,
        reset: bool = True,
        force_all_finite: bool = True,
        cast_to_ndarray: bool = True,
    ) -> np.ndarray:
        """Validate input data and set or check the `n_features_in_` attribute.

        Calls private method :meth:`sklearn.base.BaseEstimator._validate_data`.

        Parameters
        ----------

        """
        return self._validate_data(
            X=X,
            reset=reset,
            force_all_finite=force_all_finite,
            cast_to_ndarray=cast_to_ndarray,
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
