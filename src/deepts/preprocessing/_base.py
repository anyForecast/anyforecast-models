from __future__ import annotations

from typing import Protocol


from sklearn.base import BaseEstimator, TransformerMixin
import sklearn


class Check(Protocol):
    def __call__(self, X) -> None: ...


class Transformer(BaseEstimator, TransformerMixin):
    """Transformer interface.

    Used primarily for type hint.
    """

    def __init__(self, checks: list[Check] = ()):
        self.checks = checks

    def check_X(self, X):
        for check in self.checks:
            check(X)

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
