from typing import Type

import skorch

__all__ = ["SkorchAdapter"]


class SkorchAdapter:
    """Mixin adapter class for skorch models."""

    # Defines the name of the attribute containing the skorch estimator.
    _estimator_attr = "_skorch"

    def _get_skorch_class(self) -> Type:
        """Returns skorch class.

        The default returns skorch.NeuralNet
        """
        return skorch.NeuralNet

    def _get_skorch_object(self) -> object:
        """Abstract method to initialize skorch object.

        The default initializes result of _get_skorch_class
        with self.get_params.
        """
        cls = self._get_skorch_class()
        return cls(**self.get_params())

    def _init_skorch_object(self) -> object:
        """Method to initialize skorch object and set to _estimator_attr."""
        obj = self._get_skorch_object()
        setattr(self, self._estimator_attr, obj)
        return getattr(self, self._estimator_attr)
