import abc
from typing import Any, Protocol

import numpy as np
import pandas as pd

from anyforecast_models.base import Transformer


class Check(Protocol):
    def __call__(self, X) -> None: ...


class FitTransformCallable(Protocol):
    """Fit/transform signature."""

    def __call__(
        transformer: Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ): ...


class CheckDecorator(abc.ABC):
    @abc.abstractmethod
    def check(
        self,
        transformer: Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        pass

    def __call__(self, fit_transform: FitTransformCallable) -> Any:
        def fit_transform_wrapper(
            transformer: Transformer,
            X: np.ndarray | pd.DataFrame,
            *args,
            **kwargs,
        ):
            X = self.check(transformer, X=X, *args, **kwargs)
            return fit_transform(transformer, X=X, *args, **kwargs)

        return fit_transform_wrapper


class SklearnCheck(CheckDecorator):
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
        transformer: Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        return transformer.validate_data(
            X=X,
            reset=self.reset,
            force_all_finite=self.force_all_finite,
            cast_to_ndarray=self.cast_to_ndarray,
        )


class CheckCols(CheckDecorator):
    def __init__(self, cols_attr: str):
        self.cols_attr = cols_attr

    def check(
        self,
        transformer: Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        cols = getattr(transformer, self.cols_attr)
        for col in cols:
            if col not in X:
                raise ValueError()

        return X


class InputCheck(CheckDecorator):
    def __init__(self, fn: Check, check_is_fitted: bool = False) -> None:
        self.fn = fn
        self.check_is_fitted = check_is_fitted

    def check(
        self,
        transformer: Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        if self.check_is_fitted:
            transformer.check_is_fitted()

        self.fn(X)

        return X


class MultiCheck(CheckDecorator):
    def __init__(self, checks: list[Check], check_is_fitted: bool = False):
        self.checks = checks
        self.check_is_fitted = check_is_fitted

    def check(
        self,
        transformer: Transformer,
        X: np.ndarray | pd.DataFrame,
        *args,
        **kwargs,
    ):
        if self.check_is_fitted:
            transformer.check_is_fitted()

        for check in self.checks:
            check(X)

        return X
