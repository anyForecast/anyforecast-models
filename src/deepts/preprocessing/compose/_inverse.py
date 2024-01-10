from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from deepts import base
from deepts.decorators import MultiCheck
from deepts.preprocessing.compose import ColumnTransformerWrapper
from deepts.preprocessing.compose._inverse_behaviors import (
    InverseBehavior,
    get_inverse_behavior,
)
from deepts.utils import checks


class _InverseTransformer:
    """Inverse transforms input data using __init__ transformer.

    Parameters
    ----------
    name : str
        Transformer name.

    trans : transformer estimator or "passthrough"
        Transformer instance.

    features : list of str
        Features inside input data X to be inverse transformed.

    ignore_or_raise : str, {"ignore", "raise"}
        Behavior for when transformers do not have inverse_transform method.
    """

    def __init__(
        self,
        name: str,
        trans: base.Transformer,
        features: np.ndarray | list[str],
        ignore_or_raise: Literal["ignore", "raise"],
    ) -> None:
        self.name = name
        self.trans = trans
        self.features = features
        self.ignore_or_raise = ignore_or_raise

    def inverse_transform(self, X: pd.DataFrame) -> np.ndarray:
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
        return self.get_inverse_behavior().inverse_transform(X)

    def get_inverse_behavior(self) -> InverseBehavior:
        return get_inverse_behavior(
            name=self.name,
            trans=self.trans,
            features=self.features,
            ignore_or_raise=self.ignore_or_raise,
        )


class InverseColumnTransformer:
    """Inverse transformation of a sklearn :class:`ColumnTransformer` instance.

    Parameters
    ----------
    column_transformer : sklearn.compose.ColumnTransformer
        Fitted sklearn :class:`ColumnTransformer` instance.

    ignore_or_raise : str, {"ignore", "raise"}
        Behavior for when transformers do not have inverse_transform method.
    """

    def __init__(
        self,
        column_transformer: ColumnTransformer,
        ignore_or_raise: Literal["ignore", "raise"] = "ignore",
    ):
        self.column_transformer = ColumnTransformerWrapper(column_transformer)
        self.ignore_or_raise = ignore_or_raise

    def get_features_out(
        self,
        name: str,
        trans: base.Transformer,
        column: list[str],
    ) -> np.ndarray:
        """Returns features_out for transformer."""
        return self.column_transformer.get_feature_name_out_for_transformer(
            name=name, trans=trans, column=column
        )

    @MultiCheck(checks=[checks.check_is_frame])
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
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

        for name, trans, features_in, _ in self.column_transformer.iter():
            features_out = self.get_features_out(name, trans, features_in)

            # It is possible X does not contain all features/columns of
            # "passthrough" or "drop", so only the ones present are used.
            if checks.is_passthrough_or_drop(trans):
                features_out = set(features_out).intersection(set(X))
                features_in = features_out

            inverse_trans = _InverseTransformer(
                name=name,
                trans=trans,
                features=features_out,
                ignore_or_raise=self.ignore_or_raise,
            )

            Xi = inverse_trans.inverse_transform(X)

            # Only consider non-empty inverse transformations.
            if Xi.size != 0:
                for i, feature in enumerate(features_in):
                    inverse_transforms[feature] = Xi[:, i]

        return pd.DataFrame.from_dict(inverse_transforms)
