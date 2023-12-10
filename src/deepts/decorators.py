from typing import Any, Protocol

from deepts.base import Transformer


class Check(Protocol):
    def __call__(self, X) -> None: ...


class FitTransformCallable(Protocol):
    """Fit/transform signature."""

    def __call__(self: Transformer, X, *args, **kwargs) -> Any: ...


class FitTransformDecorator(Protocol):
    """Extends behavior of callables with :class:`FitTransformCallable` signature."""

    def __call__(self, f: FitTransformCallable) -> FitTransformCallable: ...


def check_cols(cols: str) -> FitTransformDecorator:
    """Adds column check.

    Parameters
    ----------
    """

    def decorator(f: FitTransformCallable) -> FitTransformCallable:
        def inner_f(self: Transformer, X, *args, **kwargs) -> Any:
            cols_ = getattr(self, cols)
            for col in cols_:
                if col not in X:
                    raise ValueError()

            return f(self, X, *args, **kwargs)

        return inner_f

    return decorator


def sklearn_validate(
    reset: bool = True, force_all_finite: bool = True
) -> FitTransformDecorator:
    """Adds sklearn validation.

    Parameters
    ----------
    """

    def decorator(f: FitTransformCallable) -> FitTransformCallable:
        def inner_f(self: Transformer, X, *args, **kwargs) -> Any:
            X = self.validate_data(
                X, reset=reset, force_all_finite=force_all_finite
            )
            return f(self, X, *args, **kwargs)

        return inner_f

    return decorator


def check(
    checks: list[Check], check_is_fitted: bool = False
) -> FitTransformDecorator:
    """Adds arbitrary checks.

    Parameters
    ----------
    """

    def decorator(f: FitTransformCallable) -> FitTransformCallable:
        def inner_fun(self: Transformer, X, *args, **kwargs) -> Any:
            if check_is_fitted:
                self.check_is_fitted()

            for check in checks:
                check(X)

            return f(self, X, *args, **kwargs)

        return inner_fun

    return decorator
