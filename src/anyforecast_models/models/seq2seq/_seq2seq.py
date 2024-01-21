from typing import Callable, Literal

import torch
from pytorch_forecasting import metrics
from skorch.callbacks import Callback

from anyforecast_models.models.base import TimeseriesNeuralNet
from anyforecast_models.models.seq2seq._collate import Seq2SeqCollateFn
from anyforecast_models.models.seq2seq._module import Seq2SeqModule


class Seq2Seq(TimeseriesNeuralNet):
    """Seq2Seq architecture.

    This model applies a sequence to sequence learning architecture to solve
    the multivariate multistep time series forecasting problem. An additional
    embedding layer allows to condition the encoder module on
    time independent/static categorical data, making it possible to learn and
    predict multiple time series using a single model (i.e. a single non linear
    mapping).

    Notes
    -----
    This model only accepts the following types of covariates:
    - Time varying known reals
    - Time varying unknown reals
    - Static reals
    - Static categoricals

    Time varying known categoricals and time varying unknown categoricals can
    still be modeled by first preprocessing them into a numerical space
    (e.g., one hot encoding) and passing them as time varying known reals and
    time varying unknown reals, respectively.


    Parameters
    ----------
    group_ids : list of str
        list of column names identifying a time series. This means that the
        ``group_ids`` identify a sample together with the ``date``. If you
        have only one times series, set this to the name of column that is
        constant.

    time_idx : str
        Time index column. This column is used to determine the sequence of
        samples.

    target : str
        Target column. Column containing the values to be predicted.

    max_prediction_length : int
        Maximum prediction/decoder length. Usually this is defined by
        the difference between forecasting dates.

    max_encoder_length : int
        Maximum length to encode. This is the maximum history length used by
        the time series dataset.

    time_varying_known_reals : list of str
        list of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product). If None,
        every numeric column excluding ``target`` is used.

    time_varying_unknown_reals : list of str
        list of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here. If None,
        only ``target`` is used.

    static_reals : list of str
        list of continuous variables that do not change over time

    static_categoricals : list of str
        list of categorical variables that do not change over time (also known
        as `time independent variables`). You might want to include your
        ``group_ids`` here for the learning algorithm to distinguish between
        different time series.

    criterion : class, default=None
        The uninitialized criterion (loss) used to optimize the module. If
        None, the :class:`.RMSE` (root mean squared error) is used.

    optimizer : class, default=None
        The uninitialized optimizer (update rule) used to optimize the
        module. if None, :class:`.Adam` optimizer is used.

    max_epochs : int, default=10
        The number of epochs to train for each :meth:`fit` call. Note that you
        may keyboard-interrupt training at any time.

    batch_size : int, default=64
        Mini-batch size. If ``batch_size`` is -1, a single batch with all the
        data will be used during training and validation.

    warm_start: bool, default=False
        Whether each fit call should lead to a re-initialization of the module
        (cold start) or whether the module should be trained further
        (warm start).

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

    embedding_dim : int, default=10
        Size of embedding tables.

    hidden_size : int, default=16
        Size of the context vector.

    tf_ratio : float, default=0.2
        For each forward pass, if the sampling from a standard uniform
        distribution is less than ``tf_ratio``, teacher forcing is used.

    num_layers : int, default=1
        Number of hidden layers.

    cell : str, {'lstm', 'gru}, default='lstm'
        Recurrent unit to be used for both encoder and decoder

    train_split, None or callable, default=None
        If ``None``, there is no train/validation split. Else, ``train_split``
        should be a function or callable that is called with X and y
        data and should return the tuple ``dataset_train, dataset_valid``.
        The validation data may be ``None``.

    min_encoder_length : int, default=None
        Minimum allowed length to encode. If None, defaults to
        ``max_encoder_length``.

    callbacks: None, “disable”, or list of Callback instances, default=None
        Which callbacks to enable.
    """

    def __init__(
        self,
        group_ids: list[str],
        time_idx: str,
        target: str,
        max_prediction_length: int,
        static_categoricals: list[str] | None = None,
        static_reals: list[str] | None = None,
        time_varying_known_reals: list[str] | None = None,
        time_varying_unknown_reals: list[str] | None = None,
        criterion: metrics.MultiHorizonMetric = metrics.RMSE,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        max_encoder_length: int | None = None,
        min_encoder_length: int | None = None,
        min_prediction_length: int | None = None,
        train_split: Callable | None = None,
        callbacks: list[Callback] | None = None,
        max_epochs: int = 10,
        batch_size: int = 64,
        embedding_dim: int = 10,
        hidden_size: int = 16,
        tf_ratio: float = 0.2,
        num_layers: int = 1,
        cell: str = "LSTM",
        warm_start: bool = False,
        verbose: int = 1,
        device: Literal["cpu", "cuda"] = "cpu",
        **kwargs,
    ):
        super().__init__(
            module=Seq2SeqModule,
            group_ids=group_ids,
            time_idx=time_idx,
            target=target,
            max_prediction_length=max_prediction_length,
            max_encoder_length=max_encoder_length,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            criterion=criterion,
            optimizer=optimizer,
            min_encoder_length=min_encoder_length,
            min_prediction_length=min_prediction_length,
            max_epochs=max_epochs,
            batch_size=batch_size,
            warm_start=warm_start,
            verbose=verbose,
            device=device,
            train_split=train_split,
            callbacks=callbacks,
            iterator_train__collate_fn=Seq2SeqCollateFn(),
            iterator_valid__collate_fn=Seq2SeqCollateFn(),
            **kwargs,
        )

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.tf_ratio = tf_ratio
        self.num_layers = num_layers
        self.cell = cell