from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

import pandas as pd
import torch
from pytorch_forecasting.data import encoders, timeseries
from torch.utils.data import Dataset as TorchDataset

from deepts import base
from deepts.preprocessing import IdentityTransformer


class TimeseriesDataset(TorchDataset):
    """Dataset for time series models.

    Wrapper for :class:`pytorch_forecasting.data.timeseries.TimeSeriesDataset`.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with time series data. Each row can be identified with
        ``date`` and the ``group_ids``.

    group_ids : list of str
        List of column names identifying a time series. This means that the
        ``group_ids`` identify a sample together with ``date``. If you
        have only one times series, set this to the name of column that is
        constant.

    time_idx : str
        Time index column.

    target : str or list of str
        Target columns.

    max_prediction_length : int
        Maximum prediction/decoder length. Usually this is defined by the
        difference between forecasting dates.

    max_encoder_length : int, default=None
        Maximum length to encode (also known as `input sequence length`). This
        is the maximum history length used by the time series dataset.

    time_varying_known_reals : list of str
        List of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product).

    time_varying_unknown_reals : list of str
        List of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here.

    static_reals : list of str
        List of continuous variables that do not change over time

    static_categoricals : list of str
        List of categorical variables that do not change over time (also known
        as `time independent variables`). You might want to include your
        ``group_ids`` here for the learning algorithm to distinguish between
        different time series.

    add_encoder_length : bool, default=True
        If True, adds encoder length to list of static real variables.
        Recommended if ``min_encoder_length != max_encoder_length``.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        time_idx: str,
        target: str | list[str],
        group_ids: list[str],
        max_encoder_length: int = 30,
        min_encoder_length: int = None,
        min_prediction_length: int = None,
        max_prediction_length: int = 1,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_known_categoricals: list[str] | None = None,
        time_varying_known_reals: list[str] | None = None,
        time_varying_unknown_categoricals: list[str] | None = None,
        time_varying_unknown_reals: list[str] | None = None,
        randomize_length: None | tuple[float, float] | bool = False,
        add_encoder_length: bool = True,
        categorical_encoders: dict[str, base.Transformer] | None = None,
        predict_mode: bool = False,
    ):
        self.time_idx = time_idx
        self.target = target
        self.group_ids = group_ids
        self.max_encoder_length = max_encoder_length
        self.min_encoder_length = min_encoder_length
        self.min_prediction_length = min_prediction_length
        self.max_prediction_length = max_prediction_length
        self.static_categoricals = static_categoricals or []
        self.static_reals = static_reals or []
        self.time_varying_known_categoricals = (
            time_varying_known_categoricals or []
        )
        self.time_varying_known_reals = time_varying_known_reals or []
        self.time_varying_unknown_categoricals = (
            time_varying_unknown_categoricals or []
        )
        self.time_varying_unknown_reals = time_varying_unknown_reals or []
        self.randomize_length = randomize_length
        self.add_encoder_length = add_encoder_length
        self.predict_mode = predict_mode

        self.categorical_encoders = (
            categorical_encoders or self.get_default_encoders()
        )

        self.pfds = self.create_pytorchforecasting(data)

    @property
    def target_names(self) -> list[str]:
        """Returns list of target names.

        Returns
        -------
        list of str
        """
        return self.target if isinstance(self.target, list) else [self.target]

    @property
    def reals(self) -> list[str]:
        """Continuous variables as used for modelling.

        Returns
        -------
        List[str]: list of variables
        """
        return (
            self.static_reals
            + self.time_varying_known_reals
            + self.time_varying_unknown_reals
        )

    @property
    def categoricals(self) -> list[str]:
        """Categorical variables as used for modelling.

        Returns
        -------
        List[str]: list of variables
        """
        return (
            self.static_categoricals
            + self.time_varying_known_categoricals
            + self.time_varying_unknown_categoricals
        )

    def create_pytorchforecasting(
        self, data: pd.DataFrame
    ) -> timeseries.TimeSeriesDataSet:
        """Creates pytorch-forecasting :class:`timeseries.TimeSeriesDataset`.

        Returns
        -------
        pytorch_forecasting.data.timeseries.TimeSeriesDataset
        """
        if (
            self.add_encoder_length
            and "encoder_length" not in self.features.reals
        ):
            self.static_reals.append("encoder_length")

        return timeseries.TimeSeriesDataSet(
            data=data,
            time_idx=self.time_idx,
            group_ids=self.group_ids,
            max_prediction_length=self.max_prediction_length,
            max_encoder_length=self.max_encoder_length,
            min_encoder_length=self.min_encoder_length,
            min_prediction_length=self.min_prediction_length,
            randomize_length=self.randomize_length,
            scalers=self.get_default_scalers(),
            categorical_encoders=self.categorical_encoders,
            predict_mode=self.predict_mode,
            **self.features.as_dict(),
        )

    @property
    def decoded_index(self) -> pd.DataFrame:
        """Returns pytorch-forecasting decoded index.
        """
        self.pfds.decoded_index

    def get_parameters(self) -> dict[str, Any]:
        """Get parameters that can be used with :py:meth:`~from_parameters` to
        create a new dataset with the same scalers.

        Returns
        -------
        Dict[str, Any]: dictionary of parameters
        """
        init_signature = inspect.signature(self.__class__.__init__)
        to_exclude = ("self", "data")

        return {
            name: getattr(self, name)
            for name in init_signature.parameters.keys()
            if name not in to_exclude
        }

    @classmethod
    def from_parameters(
        cls, parameters: dict[str, Any], data: pd.DataFrame, **kwargs
    ) -> TimeseriesDataset:
        """Generate dataset with different underlying data but same variable
        encoders and scalers, etc.

        Returns
        -------
        TimeseriesDataset
        """
        parameters = deepcopy(parameters)
        parameters.update(kwargs)
        new = cls(data, **parameters)
        return new

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Returns data sample."""
        return self.pfds[idx]

    def __len__(self):
        """Returns dataset length."""
        return len(self.pfds)

    def get_default_scalers(self) -> dict[str, IdentityTransformer]:
        """Returns dictionary from real variable (excluding target) to
        IdentityTransformer.

        Returns
        -------
        Dict[str, None]
        """
        return {
            r: IdentityTransformer()
            for r in self.reals
            if r not in self.target_names
        }

    def get_default_encoders(self) -> dict[str, encoders.NaNLabelEncoder]:
        """Returns dictionary from categorical variable to NanLabelEncoder.

        Returns
        -------
        Dict[str, NaNLabelEncoder]
        """
        return {
            cat: encoders.NaNLabelEncoder(warn=True)
            for cat in self.categoricals
        }
