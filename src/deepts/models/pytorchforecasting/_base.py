from typing import Literal, Type

import pytorch_forecasting
import torch

from deepts.data import TimeseriesDataset

from .. import _base


def _output_class(net=None, **res):
    """Auxiliary for pytorch-forecasting modules output.

    pytorch-forecasting requires an output_class method that gets called at the
    end of every forward pass and deals with all the output information from
    the model. Since skorch only requires the actual prediction to compute the
    loss, that is the only thing we extract.

    Notes
    -----
    Notice this is not actually a class. The name follows from the private
    attribute ``_output_class`` (which is where this function is assigned to)
    in all pytorch_forecasting models.

    Parameters
    ----------
    net : pytorch-forecasting model
        Compatability purposes (equivalent to self)

    **res : dict
        Dictionary containing info about the results

    Returns
    -------
    predictions : torch.tensor
    """
    return res["prediction"].squeeze(-1)


def _output_transformer(module, out=None):
    """Auxiliary for pytorch-forecasting modules output.

    pytorch-forecasting modules require a pickable callable that takes network
    output and transforms it to prediction space. Since we leave the
    preprocessing to the user (for our purpose predictions, already are in the
    prediction space), we leave predictions untouched.

    Parameters
    ----------
    module : pytorch-forecasting model
        Compatability purposes (equivalent to self)

    out : dict
        Dictionary containing info about the results

    Returns
    -------
    predictions : torch.tensor
    """
    if isinstance(module, dict):
        return module["prediction"]
    return out["prediction"]


class PyTorchForecastingCollateFn:
    """Collate fn for temporal fusion transformer (TFT).

    Wrapper to :meth:`pytorch_forecasting.TimeSeriesDataSet._collate_fn`
    from pytorch_forecasting library to match skorch conventions.
    """

    def __call__(self, batch):
        X, y = pytorch_forecasting.TimeSeriesDataSet._collate_fn(batch)

        # Modifications in order to satisfy skorch convention:
        # - weights are ignored.
        # - X is a nested dict.
        # - y is reshaped to 2D.

        batch_size = len(batch)
        y = y[0].reshape(batch_size, -1)
        return {"x": X}, y


class PytorchForecastingNeuralNet(_base.TimeseriesNeuralNet):
    """Base class for pytorch_forecasting models that collects common
    methods between them.

    .. note::
        This class should not be used directly. Use derived classes instead.
    """

    def __init__(
        self,
        module: Type[torch.nn.Module],
        group_ids: list[str],
        time_idx: str,
        target: str,
        max_encoder_length: int,
        max_prediction_length: int,
        criterion: pytorch_forecasting.metrics.MultiHorizonMetric,
        optimizer: torch.optim.Optimizer,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_known_categoricals: list[str] | None = None,
        time_varying_known_reals: list[str] | None = None,
        time_varying_unknown_categoricals: list[str] | None = None,
        time_varying_unknown_reals: list[str] | None = None,
        min_encoder_length: int | None = None,
        min_prediction_length: int | None = None,
        max_epochs: int = 10,
        batch_size: int = 128,
        warm_start: bool = False,
        verbose: int = 1,
        device: Literal["cpu", "cuda"] = "cpu",
        train_split: callable | None = None,
        callbacks: list | None = None,
        **kwargs
    ):
        super().__init__(
            module=module,
            group_ids=group_ids,
            time_idx=time_idx,
            target=target,
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            criterion=criterion,
            optimizer=optimizer,
            max_epochs=max_epochs,
            batch_size=batch_size,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            train_split=train_split,
            min_prediction_length=min_prediction_length,
            min_encoder_length=min_encoder_length,
            callbacks=callbacks,
            iterator_train__collate_fn=PyTorchForecastingCollateFn(),
            iterator_valid__collate_fn=PyTorchForecastingCollateFn(),
            **kwargs
        )
        self.loss = self.criterion()

    def _initialize_module(self, ds: TimeseriesDataset) -> torch.nn.Module:
        """Instantiates pytorch module using object (self) attributes and
        training dataset.

        Overrides :meth:`super()._initialize_module` to assign functions
        `output_transformer` and `output_class`.

        Returns
        -------
        module : torch neural net object
            Instantiated neural net
        """
        ignore = ("static_categoricals", "static_reals")
        module_kwargs = self.get_kwargs_for("module", ignore)
        module_kwargs["output_transformer"] = _output_transformer
        pfds = ds.get_pytorch_forecasting_ds()
        module = self.module.from_dataset(pfds, **module_kwargs)
        module._output_class = _output_class
        return module
