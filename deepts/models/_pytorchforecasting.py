from typing import (
    Optional,
    Type,
    List,
    Callable,
    Literal
)

import pytorch_forecasting as pf
import torch

from . import base
from ..data import TimeseriesDataset


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
    return res['prediction'].squeeze(-1)


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
        return module['prediction']
    return out['prediction']


class PytorchForecastingCollateFn:
    """Collate fn for temporal fusion transformer (TFT).

    Wrapper to :meth:`pytorch_forecasting.TimeSeriesDataSet._collate_fn`
    from pytorch_forecasting library to match skorch conventions.
    """

    def __call__(self, batch):
        X, y = pf.TimeSeriesDataSet._collate_fn(batch)

        # Modifications in order to satisfy skorch convention:
        # - weights are ignored.
        # - X is a nested dict.
        # - y is reshaped to 2D.

        batch_size = len(batch)
        y = y[0].reshape(batch_size, -1)
        return {'x': X}, y


class PFNeuralNet(base.TimeseriesNeuralNet):
    """Base class for pytorch_forecasting models that collects common
    methods between them.

    .. note::
        This class should not be used directly. Use derived classes instead.
    """

    def __init__(
            self,
            module: Type[torch.nn.Module],
            group_ids: List[str],
            time_idx: str,
            target: str,
            max_encoder_length: int,
            max_prediction_length: int,
            criterion: pf.metrics.MultiHorizonMetric,
            optimizer: torch.optim.Optimizer,
            static_categoricals: Optional[List[str]] = None,
            static_reals: Optional[List[str]] = None,
            time_varying_known_categoricals: Optional[List[str]] = None,
            time_varying_known_reals: Optional[List[str]] = None,
            time_varying_unknown_categoricals: Optional[List[str]] = None,
            time_varying_unknown_reals: Optional[List[str]] = None,
            min_encoder_length: Optional[int] = None,
            min_prediction_length: Optional[int] = None,
            max_epochs: int = 10, batch_size: int = 128,
            warm_start: bool = False, verbose: int = 1,
            device: Literal["cpu", "cuda"] = 'cpu',
            train_split: Optional[Callable] = None,
            callbacks: Optional[List] = None,
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
            iterator_train__collate_fn=PytorchForecastingCollateFn(),
            iterator_valid__collate_fn=PytorchForecastingCollateFn(),
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
        ignore = ('static_categoricals', 'static_reals')
        module_kwargs = self.get_kwargs_for('module', ignore)
        module_kwargs['output_transformer'] = _output_transformer
        pfds = ds.get_pytorch_forecasting_ds()
        module = self.module.from_dataset(pfds, **module_kwargs)
        module._output_class = _output_class
        return module


class TemporalFusionTransformer(PFNeuralNet):
    """Temporal Fusion Transformer.

    Implementation obtained from pytorch_forecasting.

    Parameters
    ----------
    group_ids : list of str
        List of column names identifying a time series. This means that the
        ``group_ids`` identify a sample together with the ``date``. If you
        have only one times eries, set this to the name of column that is
        constant.

    time_idx : str
        Time index column. This column is used to determine the sequence of
        samples.

    target : str
        Target column. Column containing the values to be predicted.

    max_prediction_length : int
        Maximum prediction/decoder length. Usually this is defined by the
        difference between forecasting dates.

    max_encoder_length : int, default=None
        Maximum length to encode (also known as `input sequence length`). This
        is the maximum history length used by the time series dataset. If None,
        3 times the ``max_prediction_length`` is used.

    time_varying_known_reals : list of str, default=None
        List of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product). If None,
        every numeric column excluding ``target`` is used.

    time_varying_unknown_reals : list of str
        List of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here. If None,
        only ``target`` is used.

    static_categoricals : list of str
        List of categorical variables that do not change over time (also known
        as `time independent variables`). You might want to include your
        ``group_ids`` here for the learning algorithm to distinguish between
        different time series. If None, only ``group_ids`` is used.

    criterion : class, default=None
        The uninitialized criterion (loss) used to optimize the module. If
        None, the :class:`.RMSE` is used.

    optimizer : class, default=None
        The uninitialized optimizer (update rule) used to optimize the
        module. if None, :class:`.Adam` optimizer is used.

    lr : float, default=1e-5
        Learning rate passed to the optimizer.

    max_epochs : int, default=10
        The number of epochs to train for each :meth:`fit` call. Note that you
        may keyboard-interrupt training at any time.

    batch_size : int, default=64
        Mini-batch size. If ``batch_size`` is -1, a single batch with
        all the data will be used during training and validation.

    callbacks: None, “disable”, or list of Callback instances, default=None
        Which callbacks to enable.


    emb_dim : int, default=10
        Dimension of every embedding table

    hidden_size : int, default=16
        Size of the context vector

    hidden_continuous_size : int, default=8
        Hidden size for processing continuous variables

    lstm_layers : int, default=2
        Number of LSTM layers (2 is mostly optimal)

    dropout : float, default=0.1
        Dropout rate

    verbose : int, default=1
        This parameter controls how much print output is generated by
        the net and its callbacks. By setting this value to 0, e.g. the
        summary scores at the end of each epoch are no longer printed.
        This can be useful when running a hyperparameter search. The
        summary scores are always logged in the history attribute,
        regardless of the verbose setting.

    device : str, torch.device, default="cpu"
        The compute device to be used. If set to "cuda", data in torch
        tensors will be pushed to cuda tensors before being sent to the
        module. If set to None, then all compute devices will be left
        unmodified.
    """

    def __init__(
            self,
            group_ids: List[str],
            time_idx: str,
            target: str,
            max_prediction_length: int,
            static_categoricals: Optional[List[str]] = None,
            static_reals: Optional[List[str]] = None,
            time_varying_known_categoricals: Optional[List[str]] = None,
            time_varying_known_reals: Optional[List[str]] = None,
            time_varying_unknown_categoricals: Optional[List[str]] = None,
            time_varying_unknown_reals: Optional[List[str]] = None,
            criterion: pf.metrics.MultiHorizonMetric = pf.metrics.RMSE,
            optimizer: torch.optim.Optimizer = torch.optim.Adam,
            max_encoder_length: Optional[int] = None,
            min_encoder_length: Optional[int] = None,
            min_prediction_length: Optional[int] = None,
            train_split: Callable = None,
            callbacks: List = None,
            max_epochs: int = 10,
            batch_size: int = 64,
            lstm_layers: int = 1,
            dropout: float = 0.1,
            output_size: int = 1,
            hidden_continuous_size: int = 8,
            emb_dim: int = 10,
            hidden_size: int = 16,
            warm_start: bool = False,
            verbose: int = 1,
            device: Literal["cpu", "cuda"] = "cpu",
            **kwargs
    ):
        super().__init__(
            module=pf.TemporalFusionTransformer,
            group_ids=group_ids,
            time_idx=time_idx,
            target=target,
            max_prediction_length=max_prediction_length,
            static_reals=static_reals,
            static_categoricals=static_categoricals,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_categoricals=time_varying_unknown_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            max_encoder_length=max_encoder_length,
            min_encoder_length=min_encoder_length,
            min_prediction_length=min_prediction_length,
            max_epochs=max_epochs, batch_size=batch_size,
            optimizer=optimizer, train_split=train_split, criterion=criterion,
            callbacks=callbacks,
            warm_start=warm_start, verbose=verbose, device=device, **kwargs
        )
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.hidden_continuous_size = hidden_continuous_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.output_size = output_size
