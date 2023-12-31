from __future__ import annotations

import inspect
from typing import Any, Literal, Type

import numpy as np
import pandas as pd
import skorch
import torch

from deepts.adapters import SkorchAdapter
from deepts.base import Transformer
from deepts.data import TimeseriesDataset
from deepts.models.base import BaseModule
from deepts.preprocessing import OutputToPandasTransformer


class InitParams:
    """Infers init parameters from the given context.

    Parameters
    ----------
    context : Any
        Context object from which init parameters values will be inferred.

    kwargs : key-word arguments
        Additional arguments from which to also infer init parameters.
    """

    def __init__(self, context: Any, **kwargs):
        self.context = context
        self.kwargs = kwargs

    def get(self, cls: Type[Any]) -> dict[str, Any]:
        """Infers cls init parameters from the given context.

        Parameters
        ----------
        cls : Type[Any]
            Class whose init parameters will be obtained.

        Returns
        -------
        kwargs : dict
        """
        init_signature = self._get_init_signature(cls)
        kwargs = self._get_init_kwargs(init_signature)
        return kwargs

    def _get_init_signature(self, cls: Type[Any]) -> inspect.Signature:
        """Retrieves __init__ signature from ``cls``.

        Parameters
        ----------
        cls : class

        Returns
        -------
        Signature
        """
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        return inspect.signature(init)

    def _get_init_kwargs(
        self, init_signature: inspect.Signature
    ) -> dict[str, Any]:
        """Returns init kwargs inferred from :attr:`context`."""
        context_dict = {**self.context.__dict__, **self.kwargs}
        return {
            k: v
            for k, v in context_dict.items()
            if k in init_signature.parameters
        }


class TimeseriesNeuralNet(Transformer, SkorchAdapter):
    """Base class for time series neural nets.

    In addition to the parameters listed below, there are parameters
    with specific prefixes that are handled separately. To illustrate
    this, here is an example:

    >>> net = TimeseriesNeuralNet(
    ...    [...],
    ...    optimizer=torch.optim.SGD,
    ...    optimizer__momentum=0.95,
    ...)

    This way, when ``optimizer`` is initialized, :class:`.NeuralNet`
    will take care of setting the ``momentum`` parameter to 0.95.
    (Note that the double underscore notation in
    ``optimizer__momentum`` means that the parameter ``momentum``
    should be set on the object ``optimizer``. This is the same
    semantic as used by sklearn.). Supported prefixes include:
    ['module',
    'iterator_train',
    'iterator_valid',
    'optimizer',
    'criterion',
    'callbacks',
    'dataset']

    .. note::
        This class should not be used directly. Use derived classes instead.

    Parameters
    ----------
    module : class
        Neural network class that inherits from :class:`BaseModule`

    group_ids : list of str
        List of column names identifying a time series. This means that the
        group_ids identify a sample together with the time_idx. If you have
        only one time series, set this to the name of column that is constant.

    time_idx : str
        Time index column. This column is used to determine the sequence of
        samples.

    target : str
        Target column. Column containing the values to be predicted.

    max_prediction_length : int
        Maximum prediction/decoder length (choose this not too short as it can
        help convergence)

    max_encoder_length : int
        Maximum length to encode. This is the maximum history length used by
        the time series dataset.

    time_varying_known_reals : list of str
        List of continuous variables that change over time and are known in the
        future (e.g. price of a product, but not demand of a product). If None,
        every numeric column excluding ``target`` is used.

    time_varying_unknown_reals : list of str
        List of continuous variables that change over time and are not known in
        the future. You might want to include your ``target`` here. If None,
        only ``target`` is used.

    static_reals : list of str
        List of continuous variables that do not change over time

    static_categoricals : list of str
        List of categorical variables that do not change over time (also known
        as `time independent variables`). You might want to include your
        ``group_ids`` here for the learning algorithm to distinguish between
        different time series. If None, only ``group_ids`` is used.

    min_encoder_length : int, default=None
        Minimum allowed length to encode. If None, defaults to
        ``max_encoder_length``.

    criterion : class, default=None
        The uninitialized criterion (loss) used to optimize the
        module. If None, the root mean squared error (:class:`RMSE`) is used.

    optimizer : class, default=None
        The uninitialized optimizer (update rule) used to optimize the
        module. If None, :class:`Adam` optimizer is used.

    min_encoder_length : int, default=None
        Minimum allowed length to encode. Defaults to ``max_encoder_length``.

    min_prediction_length : int, default=None
        Minimum prediction/decoder length. Defaults to
        ``max_prediction_length``.

    lr : float, default=1e-5
        Learning rate passed to the optimizer.

    max_epochs : int, default=10
        The number of epochs to train for each :meth:`fit` call. Note that you
        may keyboard-interrupt training at any time.

    batch_size : int, default=64
        Mini-batch size. If ``batch_size`` is -1, a single batch with all the
        data will be used during training and validation.

    callbacks: None, “disable”, or list of Callback instances, default=None
        Which callbacks to enable.

    warm_start: bool, default=False
        Whether each fit call should lead to a re-initialization of the module
        (cold start) or whether the module should be trained further
        (warm start).

    verbose : int, default=1
        This parameter controls how much print output is generated by
        the net and its callbacks. By setting this value to 0, e.g. the
        summary scores at the end of each epoch are no longer printed.
        This can be useful when running a hyperparameters search. The
        summary scores are always logged in the history attribute,
        regardless of the verbose setting.

    device : str, torch.device, default='cpu'
        The compute device to be used. If set to 'cuda', data in torch
        tensors will be pushed to cuda tensors before being sent to the
        module. If set to None, then all compute devices will be left
        unmodified.

    kwargs : key-word args.
       Extra prefixed parameters (see list of supported prefixes above).

    Attributes
    ----------
    dataset_params_ : dict
        Training dataset parameters.

    skorch_ : skorch.NeuralNet
        Fitted skorch neural network.
    """

    prefixes = [
        "module",
        "iterator_train",
        "iterator_valid",
        "optimizer",
        "criterion",
        "callbacks",
        "dataset",
    ]

    def __init__(
        self,
        module: Type[BaseModule],
        group_ids: list[str],
        time_idx: str,
        target: str,
        max_encoder_length: int,
        max_prediction_length: int,
        criterion,
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
        dataset: Type[TimeseriesDataset] = TimeseriesDataset,
        output_transformer: OutputToPandasTransformer | None = None,
        **prefix_kwargs,
    ):
        self.module = module
        self.dataset = dataset
        self.group_ids = group_ids
        self.time_idx = time_idx
        self.target = target
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.static_categoricals = static_categoricals
        self.static_reals = static_reals
        self.time_varying_known_categoricals = time_varying_known_categoricals
        self.time_varying_known_reals = time_varying_known_reals
        self.time_varying_unknown_categoricals = (
            time_varying_unknown_categoricals
        )
        self.time_varying_unknown_reals = time_varying_unknown_reals
        self.criterion = criterion
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.warm_start = warm_start
        self.verbose = verbose
        self.device = device
        self.train_split = train_split
        self.min_encoder_length = min_encoder_length
        self.min_prediction_length = min_prediction_length
        self.callbacks = callbacks
        self.output_transformer = output_transformer
        self.prefix_kwargs = prefix_kwargs
        self._initialized = False

    def _get_param_names(self):
        """This method has been overriden since the one from super ignores
        prefixed param names.
        """
        return (k for k in self.__dict__ if not k.endswith("_"))

    def get_params(self, deep: bool = True) -> dict:
        params = super().get_params(deep=deep)

        # Don't include the following attributes.
        to_exclude = {"module", "dataset"}
        return {k: v for k, v in params.items() if k not in to_exclude}

    def get_prefix_kwargs(self, prefix: str) -> dict:
        """Extracts kwargs containing the given prefix.

        This is useful for obtaining parameters that belong to a submodule.

        Parameters
        ----------
        prefix : str
            Kwargs with this prefix are returned.

        Returns
        -------
        kwargs : dict
            Kwargs containing the given prefix.
        """
        if not prefix.endswith("__"):
            prefix += "__"

        return {
            key[len(prefix) :]: val
            for key, val in self.prefix_kwargs.items()
            if key.startswith(prefix)
        }

    def get_init_kwargs(self, cls: Type) -> dict:
        """Infers __init__ parameters for ``cls``.

        Parameters
        ----------
        cls : Type[Any]
            Class whose __init__ parameters will be obtained.

        Returns
        -------
        init_kwargs : dict
        """
        return InitParams(self).get(cls)

    def get_kwargs_for(self, name: str, ignore: tuple = ()) -> dict:
        """Collects __init__ kwargs for an attribute.

        The returned kwargs are obtained by inspecting the __init__ method
        from the passed attribute (e.g., module.__init__()) and from prefixed
        kwargs (double underscore notation, e.g., 'module__something').

        Parameters
        ----------
        name : str
            The name of the attribute whose arguments should be
            returned. E.g. for the module, it should be ``'module'``.

        ignore : tuple
            Collection of keys ot ignore.

        Returns
        -------
        kwargs : dict
        """

        cls = getattr(self, name)
        init_kwargs = self.get_init_kwargs(cls)
        prefix_kwargs = self.get_prefix_kwargs(name)
        kwargs = {**init_kwargs, **prefix_kwargs}
        return {k: v for k, v in kwargs.items() if k not in ignore}

    def fit(self, X: pd.DataFrame | TimeseriesDataset, y: None = None):
        """Initializes and fits the module.

        If the module was already initialized, by calling fit, the module
        will be re-initialized (unless warm_start is True).

        Parameters
        ----------
        X : pd.DataFrame
            The input data

        y : None
            This parameter only exists for sklearn compatibility and must
            be left in None.

        Returns
        -------
        self (object)
        """
        X = self.get_dataset(X)

        if not self._initialized:
            self._initialize(X)

        self._skorch.fit(X, y)

        return self

    def _initialize(self, ds: TimeseriesDataset):
        """Initializes all components."""

        self.dataset_params_ = ds.get_parameters()
        self.module_ = self._initialize_module(ds)
        self._init_skorch_object()

        self._initialized = True

    def skorch_predict(
        self,
        X: pd.DataFrame | TimeseriesDataset,
        return_dataset: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, TimeseriesDataset]:
        """Predicts input data X.

        Parameters
        ----------
        X : pd.DataFrame
            Input values.

        return_dataset : bool, default=False
            If True, input dataset is also returned.

        Returns
        -------
        output : np.ndarray.
            Raw predicted values.
        """
        self.check_is_fitted()
        dataset = self.get_predict_dataset(X)
        output = self._skorch.predict(dataset)

        if return_dataset:
            return output, dataset
        return output

    def predict(self, X: pd.DataFrame | TimeseriesDataset) -> pd.DataFrame:
        """Predicts and transforms input data.

        Predictions are transformed using the configured ``output_transformer``.

        Parameters
        ----------
        X : pd.DataFrame
            Input values.

        Returns
        -------
        trans_output : object
            Output after transformation.
        """
        output_transformer = self.get_output_transformer()
        output, ds = self.skorch_predict(X, return_dataset=True)
        return output_transformer.transform(output, ds.decoded_index)

    def get_output_transformer(self) -> OutputToPandasTransformer:
        if self.output_transformer is None:
            return OutputToPandasTransformer(
                group_cols=self.group_ids,
                output_col=self.target,
                time_idx_col=self.time_idx,
            )

        return self.output_transformer

    def get_predict_dataset(self, X: pd.DataFrame) -> TimeseriesDataset:
        """Returns dataset in prediction mode.

        Parameters
        ----------
        X : pd.DataFrame
            Input values.

        Returns
        -------
        dataset : torch.utils.data.Dataset
        """
        self.check_is_fitted()
        return self.get_dataset(X, self.dataset_params_, predict_mode=True)

    def get_dataset(
        self, X: pd.DataFrame, params: dict | None = None, **kwargs
    ) -> TimeseriesDataset:
        """Constructs torch dataset using the input data ``X``

        Parameters
        ----------
        X : pd.DataFrame
            Input data

        params : dict, default=None
            If given, generates torch dataset using this parameters. Otherwise,
            the parameters are obtained from the object (self) attributes.

        **kwargs : key-word arguments
            Additional parameters passed to dataset class. If given,
            kwargs will override values given to ``params``.

        Returns
        -------
        dataset: torch dataset
            The initialized dataset.
        """
        # Return ``X`` if already is a dataset
        if isinstance(X, TimeseriesDataset):
            return X

        if params is not None:
            return self.dataset.from_parameters(params, X, **kwargs)

        dataset_params = self.get_kwargs_for("dataset")
        dataset_params.update(kwargs)
        return self.dataset(X, **dataset_params)

    def get_history(
        self, name: str, step_every: Literal["batch", "epoch"] = "epoch"
    ) -> list:
        """Obtains history.

        Parameters
        ----------
        name : str
            Name of history.

        step_every : str {'batch', 'epoch'}, default='epoch'

        Returns
        -------
        history : list
        """
        self.check_is_fitted()
        if step_every == "epoch":
            history = self._skorch.history_[:, name]

        elif step_every == "batch":
            history = []
            for epoch in self._skorch.history_:
                for batch in epoch["batches"]:
                    if batch:
                        val = batch[name]
                        history.append(val)

        else:
            raise ValueError('`step_every` can be either "epoch" or "batch".')

        return history

    def _get_skorch_object(self) -> skorch.NeuralNet:
        """Instantiates skorch :class:`NeuralNet`.

        Returns
        -------
        models : skorch.NeuralNet
            Initialized skorch neural net.
        """
        cls = self._get_skorch_class()
        return cls(
            module=self.module_,
            criterion=self.criterion,
            optimizer=self.optimizer,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            callbacks=self.callbacks,
            verbose=self.verbose,
            device=self.device,
            warm_start=self.warm_start,
            train_split=self.train_split,
            **self.prefix_kwargs,
        )

    def _initialize_module(self, ds: TimeseriesDataset) -> torch.nn.Module:
        """Instantiates pytorch module using object (self) attributes and
        training dataset.

        Parameters
        ----------
        ds : TimeseriesDataset
            Training dataset. Used as input in :meth:`from_dataset` factory.

        Returns
        -------
        module : torch.nn.Module
            Instantiated torch neural net.
        """
        kwargs = self.get_kwargs_for("module")
        module = self.module.from_dataset(ds, **kwargs)
        return module.to(self.device)
