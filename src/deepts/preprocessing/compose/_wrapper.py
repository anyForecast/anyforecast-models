from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _check_feature_names_in


class ColumnTransformerWrapper:
    def __init__(self, column_transformer: ColumnTransformer):
        self.column_transformer = column_transformer

    def iter(self, fitted=True, replace_strings=False, column_as_strings=True):
        return self.column_transformer._iter(
            fitted=fitted,
            replace_strings=replace_strings,
            column_as_strings=column_as_strings,
        )

    def get_feature_name_out_for_transformer(
        self,
        name: str,
        trans: str,
        column: list[str],
        input_features: list[str] = None,
    ):
        """Gets feature names of transformer.

        Used in conjunction with self.iter(fitted=True)

        Notes
        -----
        Thi is wrapper for the sklearn
        :class:`ColumnTransformer._get_feature_name_out_for_transformer`
        private method.

        Returns
        -------
        feature_names_out : list
        """
        input_features = _check_feature_names_in(
            self.column_transformer, input_features
        )
        return self.column_transformer._get_feature_name_out_for_transformer(
            name, trans, column, input_features
        )
