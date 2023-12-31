from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from deepts.data import TimeseriesDataset


class ModuleFactory(ABC):
    """Base class for Module factories."""

    @abstractmethod
    def create(self, ds: TimeseriesDataset, **kwargs) -> BaseModule:
        """Creates Module object from passed dataset.

        Parameters
        ----------
        ds : TimeseriesDataset
            Dataset object from which init parameters will be obtained.

        kwargs : keyword arguments
            Additional arguments such as hyperparameters for model.
        """
        pass

    def __call__(self, ds: TimeseriesDataset, **kwargs) -> BaseModule:
        return self.create(ds, **kwargs)


class BaseModule(torch.nn.Module):
    """Base class for deepsts modules."""

    factory: ModuleFactory | None = None

    @classmethod
    def from_dataset(cls, ds: TimeseriesDataset, **kwargs) -> BaseModule:
        if cls.factory is None:
            raise NotImplementedError(
                f"`factory` for module {cls.__name__} is not implemented."
            )

        return cls.factory(ds, **kwargs)
