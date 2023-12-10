from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from deepts import base, exceptions
from deepts.decorators import check
from deepts.utils import checks

from ._wrapper import ColumnTransformerWrapper


class NonInverseBehavior(Protocol):
    def __call__(self, X, name: str, trans: base.Transformer) -> Any: ...


def _ignore_behavior(X, name, trans) -> np.ndarray:
    return np.ndarray(shape=(len(X), 0))


def _raise_behavior(X, name, trans):
    raise AttributeError(
        f"Transformer {name} (type {type(trans).__name__}) does "
        "not provide `inverse_transform` method."
    )


def get_non_inverse_behavior(
    name: Literal["raise", "ignore"]
) -> NonInverseBehavior:
    behaviors = {"ignore": _ignore_behavior, "raise": _raise_behavior}
    return behaviors[name]


def _validate_X_for_inverse_transform(
    X: pd.DataFrame,
    name: str,
    trans: str | base.Transformer,
    features: np.ndarray | list[str],
):
    """Validtes X contains the required features/columns for inverse transform.

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
    missing = set(features) - set(X)

    if missing:
        if trans == "passthrough":

            # Only "passthrough" transformation is allowed to have
            # missing features (only the ones present in X are returned).
            intersection = set(features).intersection(set(X))
            return X[list(intersection)]

        raise exceptions.InverseTransformFeaturesError(
            name=name,
            type=type(trans).__name__,
            missing_features=missing,
        )

    return X[features]


def _inverse_transform(
    X: pd.DataFrame,
    name: str,
    trans: str | base.Transformer,
    non_inverse_behavior: Literal["raise", "ignore"] = "ignore",
) -> np.ndarray:
    """Inverse transforms input data.

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
    inverse_transform : 2-D numpy array
        Array with inverse transformed data.
    """
    if trans == "passthrough":
        return X.values

    if hasattr(trans, "inverse_transform"):
        return trans.inverse_transform(X)

    return get_non_inverse_behavior(non_inverse_behavior)(
        X=X, name=name, trans=trans
    )


class ColumnTransformerInverseTransform:
    """Inverse transformation of a sklearn :class:`ColumnTransformer` instance.

    Parameters
    ----------
    column_transformer : sklearn.compose.ColumnTransformer
        Fitted sklearn :class:`ColumnTransformer` instance.
    """

    def __init__(
        self,
        ct: ColumnTransformer,
        non_inverse_behavior: Literal["raise", "ignore"] = "ignore",
    ):
        self.ct = ColumnTransformerWrapper(ct)
        self.non_inverse_behavior = non_inverse_behavior

    @check(checks=[checks.check_is_frame])
    def transform(
        self, X: pd.DataFrame, to_frame: bool = True
    ) -> pd.DataFrame | dict[str, np.ndarray]:
        """Inverse transforms X.

        Parameters
        ----------
        X : pd.DataFrame
            DataFrame to be inverse transformed.

        Returns
        -------
        inverse : pd.DataFrame
        """
        inverse_transforms: dict[str, np.ndarray] = {}

        for name, trans, features_in, _ in self.ct.iter():

            if isinstance(features_in, str):
                features_in = [features_in]

            features_out = self.ct.get_feature_name_out_for_transformer(
                name=name, trans=trans, column=features_in
            )
            if features_out is None:
                continue

            X_validated = _validate_X_for_inverse_transform(
                X=X, name=name, trans=trans, features=features_out
            )

            if trans == "passthrough":
                features_in = X_validated.columns

            inverse_transform: np.ndarray = _inverse_transform(
                X=X_validated,
                name=name,
                trans=trans,
                non_inverse_behavior=self.non_inverse_behavior,
            )

            # Only consider non-empty inverse transformations. Empty
            # inverse transformations are possible when transformers do not have
            # the method :meth:`inverse_transform` and attribute
            # ``non_inverse_behavior`` is set to "ignore".
            if inverse_transform.size != 0:
                for i, col_name in enumerate(features_in):
                    inverse_transforms[col_name] = inverse_transform[:, i]

        return (
            pd.DataFrame.from_dict(inverse_transforms)
            if to_frame
            else inverse_transforms
        )
