from __future__ import annotations

import abc
from typing import Literal

import numpy as np
import pandas as pd
from numpy import ndarray

from deepts import base
from deepts.exceptions import InverseTransformFeaturesError
from deepts.utils import checks


def get_inverse_behavior(
    name: str,
    trans: base.Transformer | base.InvertibleTransformer,
    features: np.ndarray,
    ignore_or_raise: Literal["ignore", "raise"],
) -> InverseBehavior:
    """Returns inverse behavior based on the given transformer.

    Parameters
    ----------
    name : str
        Transformer name.

    trans : Transformer
        Transformer instance.

    features : array_like
        Features to be inverse transformed.

    ignore_or_raise : str, {"ignore", "raise"}
        Behavior for when transformers do not have inverse_transform method.
    """
    if checks.is_invertible(trans):
        cls = InvertibleBehavior

    else:  # Transformer does not has inverse transform method.
        cls = IgnoreBehavior if ignore_or_raise == "ignore" else RaiseBehavior

    return cls(name=name, trans=trans, features=features)


class InverseBehavior(abc.ABC):
    """Inverse transform behavior.

    Parameters
    ----------
    name : str
        Transformer name.

    trans : Transformer
        Transformer instance.

    features : array_like
        Features to be inverse transformed.
    """

    def __init__(
        self,
        name: str,
        trans: base.Transformer | base.InvertibleTransformer,
        features: np.ndarray,
    ) -> None:
        self.name = name
        self.trans = trans
        self.features = features

    @abc.abstractmethod
    def inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Inverse transformation.

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
            raise InverseTransformFeaturesError(
                name=self.name,
                type=type(self.trans).__name__,
                missing_features=missing,
            )


class InvertibleBehavior(InverseBehavior):
    """Normal inverse transformation.

    Directly applies :meth:`trans.inverse_transformation` to input data
    using the configured features.
    """

    def inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.check_features(X)
        X = X[self.features].values
        return self.trans.inverse_transform(X)


class IdentityBehavior(InverseBehavior):
    """Identity behavior."""

    def inverse_transform(self, X: pd.DataFrame) -> ndarray:
        return X


class IgnoreBehavior(InverseBehavior):
    """Returns empty Numpy array."""

    def inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.ndarray(shape=(len(X), 0))


class RaiseBehavior(InverseBehavior):
    """Raises AttributeError."""

    def inverse_transform(self, X: pd.DataFrame):
        msg = (
            f"Transformer {self.name} (type {type(self.trans).__name__}) does"
            " not provide `inverse_transform` method."
        )
        raise AttributeError(msg)
