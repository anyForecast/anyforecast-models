from __future__ import annotations

import abc
from typing import Literal, Type

import numpy as np
import pandas as pd

from anyforecast_models import base, exceptions
from anyforecast_models.utils import checks


class InverseBehavior(abc.ABC):
    """Base abstract class for inverse transform behavior.

    Parameters
    ----------
    name : str
        Transformer name.

    trans : Transformer
        Transformer instance.

    features : array_like of str objects
        Features to be inverse transformed.
    """

    def __init__(
        self,
        name: str,
        trans: base.Transformer | base.InvertibleTransformer,
        features: np.ndarray,
    ):
        self.name = name
        self.trans = trans
        self.features = features

    @abc.abstractmethod
    def inverse_transform(self, X):
        """Inverse transforms input data X.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
        """
        pass

    def check_features(self, X: pd.DataFrame) -> None:
        """Checks X has the required features/columns for inverse transform.

        Raises
        ------
        InverseTransformerFeaturesError when there are missing
        features/columns in X.
        """
        missing = set(self.features) - set(X)

        if missing:
            raise exceptions.InverseTransformFeaturesError(
                name=self.name,
                type=type(self.trans).__name__,
                missing_features=missing,
            )


def get_inverse_behavior(
    trans: base.Transformer,
    ignore_or_raise: Literal["ignore", "raise"],
) -> Type[InverseBehavior]:
    """Returns inverse behavior based on the given transformer.

    Parameters
    ----------
    trans : trans : transformer estimator or {"passthrough", "drop"}
        Transformer instance or strings "passthrough" and "drop".

    ignore_or_raise : str, {"ignore", "raise"}
        Behavior for when transformers do not have inverse_transform method.

    Returns
    -------
    InverseBehavior : class
    """
    if checks.is_invertible(trans):
        return InvertibleBehavior

    return IgnoreBehavior if ignore_or_raise == "ignore" else RaiseBehavior


class InvertibleBehavior(InverseBehavior):
    """Invertible behavior.

    Directly applies :meth:`trans.inverse_transformation` to input data
    using the configured features.

    Parameters
    ----------
    check_features : bool, default=True
        If True, checks for missing features in X.
    """

    def inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Directly applies :meth:`trans.inverse_transformation` to X.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
        """
        self.check_features(X)
        return self.trans.inverse_transform(X[self.features])


class RaiseBehavior(InverseBehavior):
    """Raises AttributeError."""

    def inverse_transform(self, X: pd.DataFrame):
        """Raises AttributeError when called.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
        """
        raise AttributeError(
            f"Transformer {self.name} (type {type(self.trans).__name__}) does"
            " not provide `inverse_transform` method."
        )


class IdentityBehavior(InverseBehavior):
    """Identity behavior."""

    def inverse_transform(self, X: pd.DataFrame):
        """Identity function.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
        """
        return X


class RemainderBehavior(InverseBehavior):
    """Returns features present in both X and self.features."""

    def inverse_transform(self, X: pd.DataFrame):
        """Returns only features present in both X and self.features.

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
        """
        features = list(set(self.features).intersection(set(X)))
        return X[features]


class IgnoreBehavior(InverseBehavior):
    """Returns empty Numpy array."""

    def inverse_transform(self, X: pd.DataFrame):
        """Returns empty Numpy array of shape=len(X)

        Parameters
        ----------
        X : pd.DataFrame
            Dataframe to be inverse transformed.
        """
        return np.ndarray(shape=(len(X), 0))
