import os
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from deepts.definitions import ROOT_DIR

DATA_DIR = os.path.join(ROOT_DIR, "datasets/data")


@dataclass
class TimeseriesDataset:
    X: pd.DataFrame
    target: list[str]
    group_cols: list[str]
    datetime: str
    feature_names: list[str]
    freq: str
    filepath: str


def get_filepath(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def load_csv(
    filepath,
    names: list[str] | None = None,
    header: int | Literal["infer"] | None = "infer",
) -> pd.DataFrame:
    return pd.read_csv(filepath, names=names, header=header)


def load_stallion() -> TimeseriesDataset:
    """Load and return the iris dataset (time series)."""

    filepath = get_filepath("stallion.csv")

    feature_names = [
        "agency",
        "sku",
        "date",
        "industry_volume",
        "price_regular",
        "price_actual",
        "discount",
    ]

    target = "volume"
    group_cols = ["agency", "sku"]
    datetime = "date"
    freq = "MS"

    X = load_csv(filepath)

    return TimeseriesDataset(
        X=X,
        target=target,
        group_cols=group_cols,
        datetime=datetime,
        freq=freq,
        feature_names=feature_names,
        filepath=filepath,
    )
