from sklearn import pipeline
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from deepts.base import Transformer
from deepts.preprocessing import TimeIndexEncoder
from deepts.preprocessing.compose import (
    GroupedColumnTransformer,
    PandasColumnTransformer,
)

__all__ = ("make_preprocessor",)


def make_sk_column_transformer(transformers) -> ColumnTransformer:
    return make_column_transformer(
        *transformers, verbose_feature_names_out=False, remainder="passthrough"
    )


def make_numeric_selector(pattern: str | None = None) -> callable:
    return make_column_selector(pattern, dtype_include=(int, float))


def make_categorical_selector(pattern: str | None = None) -> callable:
    return make_column_selector(pattern, dtype_include=object)


def make_datetime_encoder(
    datetime_col: str, freq: str
) -> PandasColumnTransformer:
    transformers = [(TimeIndexEncoder(freq=freq), [datetime_col])]
    ct = make_sk_column_transformer(transformers)
    return PandasColumnTransformer(ct)


def make_features_transformer(
    group_cols: str | list[str],
    target_col: str,
    scaler: Transformer,
    encoder: Transformer,
) -> GroupedColumnTransformer:
    pattern = f"^(?!{target_col}).*$"  # Exclude ``target_col`` from selection.
    num_selector = make_numeric_selector(pattern)
    cat_selector = make_categorical_selector()
    transformers = [(scaler, num_selector), (encoder, cat_selector)]
    ct = make_sk_column_transformer(transformers)
    return GroupedColumnTransformer(ct, group_cols=group_cols)


def make_target_transformer(
    group_cols: str | list[str], target_col: str, scaler: Transformer
) -> GroupedColumnTransformer:
    transformers = [(scaler, [target_col])]
    ct = make_sk_column_transformer(transformers)
    return GroupedColumnTransformer(ct, group_cols=group_cols)


def make_preprocessor(
    group_cols: str | list[str],
    datetime_col: str,
    target_col: str,
    freq: str = "D",
    scaler: Transformer = MinMaxScaler(),
    encoder: Transformer = OneHotEncoder(),
) -> pipeline.Pipeline:
    datetime_encoder = make_datetime_encoder(
        datetime_col=datetime_col, freq=freq
    )

    features_trans = make_features_transformer(
        group_cols=group_cols,
        target_col=target_col,
        scaler=scaler,
        encoder=encoder,
    )
    target_trans = make_target_transformer(
        group_cols=group_cols, target_col=target_col, scaler=scaler
    )

    steps = [
        ("features", features_trans),
        ("target", target_trans),
        ("datetime", datetime_encoder),
    ]

    return pipeline.Pipeline(steps)
