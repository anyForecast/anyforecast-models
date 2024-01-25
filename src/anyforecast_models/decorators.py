import abc
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from anyforecast_models import base
from anyforecast_models.utils import checks


class FitTransformCallable(Protocol):
    """Fit/transform signature."""

    def __call__(
        trans: base.Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ): ...


class check_decorator(abc.ABC):
    @abc.abstractmethod
    def check(
        self,
        trans: base.Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        pass

    def __call__(self, fit_transform: FitTransformCallable) -> Any:
        def fit_transform_wrapper(
            trans: base.Transformer,
            X: np.ndarray | pd.DataFrame,
            *args,
            **kwargs,
        ):
            X = self.check(trans, X=X, *args, **kwargs)
            return fit_transform(trans, X=X, *args, **kwargs)

        return fit_transform_wrapper


class sklearn_check(check_decorator):
    def __init__(
        self,
        reset: bool = True,
        force_all_finite: bool = True,
        cast_to_ndarray: bool = True,
    ):
        self.reset = reset
        self.force_all_finite = force_all_finite
        self.cast_to_ndarray = cast_to_ndarray

    def check(
        self,
        trans: base.Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        return trans.validate_data(
            X=X,
            reset=self.reset,
            force_all_finite=self.force_all_finite,
            cast_to_ndarray=self.cast_to_ndarray,
        )


class check_columns_exist(check_decorator):
    def __init__(self, cols_attr: str):
        self.cols_attr = cols_attr

    def check(
        self,
        trans: base.Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        cols = getattr(trans, self.cols_attr)
        for col in cols:
            if col not in X:
                raise ValueError()

        return X


class check_is_pandas(check_decorator):
    def __init__(
        self, frame_or_series: Literal["frame", "series"] = "frame"
    ) -> None:
        self.frame_or_series = frame_or_series

    def check(
        self,
        trans: base.Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        check_fn = (
            checks.is_pandas_frame
            if self.frame_or_series == "frame"
            else checks.check_is_series
        )

        check_fn(X)
        return X


class check_input(check_decorator):
    def __init__(self, *checks, check_is_fitted: bool = False):
        self.checks = checks
        self.check_is_fitted = check_is_fitted

    def check(
        self,
        transformer: base.Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        if self.check_is_fitted:
            transformer.check_is_fitted()

        for check in self.checks:
            check(X)

        return X
