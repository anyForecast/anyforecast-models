import numpy as np
from sklearn.compose import ColumnTransformer

from anyforecast_models import base


class OutputDTypesResolver:
    """Output dtypes resolver.

    Parameters
    ----------
    feature_dtypes_in : dict, str -> dtype, default=None
        Input feature dtypes.
    """

    def __init__(self, column_transformer: ColumnTransformer):
        self.column_transformer = column_transformer

    def resolve(
        self, feature_dtypes_in: dict[str, np.dtype] | None = None
    ) -> dict[str, np.dtype]:
        """Resolves feature dtypes from all transformers.

        Returns
        -------
        dtypes : dict, str -> dtype
        """
        dtypes_out: dict[str, np.dtype] = {}

        for name, trans, column, _ in self.column_transformer._iter(
            fitted=True, column_as_labels=True, skip_drop=True, skip_empty_columns=True
        ):
            trans_dtype_out = self.get_feature_dtypes_out_for_transformer(
                name=name,
                trans=trans,
                column=column,
                feature_dtypes_in=feature_dtypes_in,
            )

            dtypes_out.update(trans_dtype_out)

        return dtypes_out

    def get_feature_dtypes_out_for_transformer(
        self,
        name: str,
        trans: str | base.Transformer,
        column: list[str],
        feature_dtypes_in: dict[str, np.dtype] | None = None,
    ) -> dict[str, np.dtype]:
        """Returns transformer feature dtypes.

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
        if feature_dtypes_in is None:
            feature_dtypes_in = {}

        if trans == "passthrough":
            dtypes_out = {c: feature_dtypes_in[c] for c in column}

        else:  # An actual estimator/transformer object.
            dtypes_out = self.get_dtypes_out_for_estimator(
                name=name, trans=trans, feature_dtypes_in=feature_dtypes_in
            )

        return dtypes_out

    def get_dtypes_out_for_estimator(
        self,
        name: str,
        trans: base.Transformer,
        feature_dtypes_in: dict[str, np.dtype] | None = None,
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
        dtypes_out: dict[str, np.dtype] = {}
        features_out = trans.get_feature_names_out()

        for feature in features_out:
            if hasattr(trans, "dtype"):
                dtypes_out[feature] = trans.dtype

            else:
                if feature not in feature_dtypes_in:
                    raise AttributeError(
                        "Cannot obtain dtype for Transformer "
                        f"{name} (type {type(trans).__name__}) since it "
                        "does not provide attribute `dtype` and "
                        "features_names_in != feature_names_out."
                    )

                dtypes_out[feature] = feature_dtypes_in[feature]

        return dtypes_out
