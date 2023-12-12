from sklearn import pipeline
from sklearn.compose import (
    ColumnTransformer,
    make_column_selector,
    make_column_transformer,
)
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from deepts.base import Transformer

from ._encoders import TimeIndexEncoder
from .compose import GroupedColumnTransformer, PandasColumnTransformer

__all__ = ("make_preprocessor",)


def make_sk_column_transformer(transformers) -> ColumnTransformer:
    return make_column_transformer(
        *transformers, verbose_feature_names_out=False, remainder="passthrough"
    )


def make_numeric_selector(pattern: str | None = None) -> callable:
    return make_column_selector(pattern, dtype_include=(int, float))


def make_categorical_selector(pattern: str | None = None) -> callable:
    return make_column_selector(pattern, dtype_include=object)


def make_timestamp_transformer(
    timestamp: str, freq: str
) -> PandasColumnTransformer:
    transformers = [(TimeIndexEncoder(freq=freq), timestamp)]
    ct = make_sk_column_transformer(transformers)
    return PandasColumnTransformer(ct)


def make_features_transformer(
    group_ids: str | list[str],
    target: str,
    scaler: Transformer,
    encoder: Transformer,
) -> GroupedColumnTransformer:
    pattern = f"^(?!{target}).*$"  # Exclude ``target`` from selection.
    num_selector = make_numeric_selector(pattern)
    cat_selector = make_categorical_selector()
    transformers = [(scaler, num_selector), (encoder, cat_selector)]
    ct = make_sk_column_transformer(transformers)
    return GroupedColumnTransformer(ct, group_cols=group_ids)


def make_target_transformer(
    group_ids: str | list[str], target: str, scaler: Transformer
) -> GroupedColumnTransformer:
    transformers = [(scaler, [target])]
    ct = make_sk_column_transformer(transformers)
    return GroupedColumnTransformer(ct, group_cols=group_ids)


def make_preprocessor(
    group_ids: str | list[str],
    timestamp: str,
    target: str,
    freq: str = "D",
    scaler: Transformer = MinMaxScaler(),
    encoder: Transformer = OneHotEncoder(),
) -> pipeline.Pipeline:
    timestamp_trans = make_timestamp_transformer(timestamp=timestamp, freq=freq)

    features_trans = make_features_transformer(
        group_ids=group_ids,
        target=target,
        scaler=scaler,
        encoder=encoder,
    )
    target_trans = make_target_transformer(
        group_ids=group_ids, target=target, scaler=scaler
    )

    steps = [
        ("features", features_trans),
        ("target", target_trans),
        ("timestamp", timestamp_trans),
    ]

    return pipeline.Pipeline(steps)
