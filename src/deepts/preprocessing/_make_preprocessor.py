import sklearn
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from . import _base, _encoders, compose

__all__ = ("make_preprocessor",)


def make_numeric_selector(pattern: str | None = None) -> callable:
    return make_column_selector(pattern, dtype_include=(int, float))


def make_categorical_selector(pattern: str | None = None) -> callable:
    return make_column_selector(pattern, dtype_include=object)


def make_timestamp_transformer(
    timestamp: str, freq: str
) -> compose.PandasColumnTransformer:
    transformers = [(_encoders.TimeIndexEncoder(freq=freq), timestamp)]
    return compose.PandasColumnTransformer(transformers, int_to_float=False)


def make_features_transformer(
    group_ids: str | list[str],
    target: str,
    scaler: _base.Transformer,
    encoder: _base.Transformer,
) -> compose.GroupedColumnTransformer:
    pattern = f"^(?!{target}).*$"  # Exclude ``target`` from selection.
    num_selector = make_numeric_selector(pattern)
    cat_selector = make_categorical_selector()
    transformers = [(scaler, num_selector), (encoder, cat_selector)]

    return compose.GroupedColumnTransformer(transformers, group_ids)


def make_target_transformer(
    group_ids: str | list[str], target: str, scaler: _base.Transformer
) -> compose.GroupedColumnTransformer:
    transformers = [(scaler, [target])]
    return compose.GroupedColumnTransformer(transformers, group_ids)


def make_preprocessor(
    group_ids: str | list[str],
    timestamp: str,
    target: str,
    freq: str = "D",
    scaler: _base.Transformer = MinMaxScaler(),
    encoder: _base.Transformer = OneHotEncoder(),
) -> Pipeline:
    timestamp_trans = make_timestamp_transformer(timestamp, freq)
    features_trans = make_features_transformer(
        group_ids, target, scaler, encoder
    )
    target_trans = make_target_transformer(group_ids, target, scaler)
    steps = [
        ("features", features_trans),
        ("target", target_trans),
        ("timestamp", timestamp_trans),
    ]

    return Pipeline(steps)
