from __future__ import annotations

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone


class Transformer(BaseEstimator, TransformerMixin):
    """Base Transformer."""

    def get_feature_names_out(self):
        pass

    def clone(self) -> Transformer:
        return clone(self)

    def validate_data(
        self, X, reset: bool = True, force_all_finite: bool = True
    ) -> np.ndarray:
        """Wrapper for private sklearn data validation method."""
        return self._validate_data(
            X, reset=reset, force_all_finite=force_all_finite
        )

    def check_is_fitted(self) -> None:
        """Perform is_fitted validation for this transformer.

        Checks if this transformer is fitted by verifying the presence of
        fitted attributes (ending with a trailing underscore) and otherwise
        raises a NotFittedError

        Raises
        ------
        NotFittedError if not fitted.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator
        """
        sklearn.utils.validation.check_is_fitted(self)

    def is_fitted(self) -> bool:
        """Returns True if estimator is fitted else False.

        Parameters
        ----------
        estimator : sklearn.base.BaseEstimator

        Returns
        -------
        is_fitted : bool
            True if estimator is fitted else False.
        """
        try:
            self.check_is_fitted()
        except sklearn.exceptions.NotFittedError:
            return False
        return True
