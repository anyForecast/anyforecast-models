from __future__ import annotations

from typing import Any

import torch
from torch.nn.modules.rnn import RNNBase

from deepts.data import TimeseriesDataset
from deepts.models.base import BaseModule, ModuleFactory
from deepts.models.nn import embeddings, rnn


class Seq2SeqModuleFactory(ModuleFactory):
    """Creates an instance of :class:`Seq2SeqModule` from the passed dataset.

    Module arguments obtained from the dataset include:
    * embedding sizes
    * encoder feature names
    * decoder feature names
    * target names
    """

    def get_embedding_sizes(self, ds: TimeseriesDataset) -> list[tuple]:
        embedding_sizes = torch.max(ds.data["categoricals"], dim=0)
        embedding_sizes = tuple(embedding_sizes.values.numpy() + 1)
        return embedding_sizes

    def get_time_varying_reals_encoder(
        self, ds: TimeseriesDataset
    ) -> list[str]:
        # All continuous variables, i.e., time_varying_known_reals,
        # time_varying_known_reals and static_reals, are encoded.
        return ds.reals

    def get_time_varying_reals_decoder(
        self, ds: TimeseriesDataset
    ) -> list[str]:
        # Only known variables are used for decoding.
        return ds.time_varying_known_reals

    def get_target(self, ds: TimeseriesDataset):
        return ds.target_names

    def get_kwargs(self, ds: TimeseriesDataset, **kwargs) -> dict[str, Any]:
        return dict(
            embedding_sizes=self.get_embedding_sizes(ds),
            time_varying_reals_encoder=self.get_time_varying_reals_encoder(ds),
            time_varying_reals_decoder=self.get_time_varying_reals_decoder(ds),
            target=self.get_target(ds),
            **kwargs,
        )

    def create(self, ds: TimeseriesDataset, **kwargs) -> Seq2SeqModule:
        kwargs = self.get_kwargs(ds, **kwargs)
        return Seq2SeqModule(**kwargs)


class Seq2SeqModule(BaseModule):
    """Encoder-decoder architecture applied to time series.

    An encoder network condenses an input sequence into a vector,
    and a decoder network unfolds that vector into a new sequence.
    Additionally, an initial multi-embedding layer is used to condition the
    encoder network with time independent/statical features.

    Parameters
    ----------
    embedding_sizes : tuple
        Size of each embedding table.

    hidden_size : int
        Size of the context vector.

    target : list of str
        List of target names.

    time_varying_reals_encoder : list of str
        List of variables to be encoded.

    time_varying_reals_decoder : list of str
        List of variables used when decoding.

    cell : str, {"LSTM", "GRU"}, default="LSTM"
        RNN cell type.

    num_layers : int, default=1
        Number of hidden layers.
    """

    factory = Seq2SeqModuleFactory

    def __init__(
        self,
        embedding_sizes: tuple[int],
        hidden_size: int,
        target: list[str],
        time_varying_reals_encoder: list[str],
        time_varying_reals_decoder: list[str],
        teacher_forcing_ratio: float = 0.2,
        cell: str = "LSTM",
        num_layers: int = 1,
        dropout: float = 0,
    ):
        super().__init__()

        self.embedding_sizes = embedding_sizes
        self.hidden_size = hidden_size
        self.target = target
        self.time_varying_reals_encoder = time_varying_reals_encoder
        self.time_varying_reals_decoder = time_varying_reals_decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.cell = cell
        self.num_layers = num_layers
        self.dropout = dropout

        self.multi_embedding = embeddings.MultiEmbedding(
            embedding_sizes, concat=True
        )
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()

    def create_rnn(self, input_size: int) -> RNNBase:
        """Creates RNN using init args."""
        return rnn.make_rnn(
            cell_type=self.cell,
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
        )

    def create_encoder(self) -> rnn.ConditionalRNN:
        """Creates Encoder."""
        return rnn.ConditionalRNN(
            rnn_unit=self.create_rnn(input_size=self.encoder_size),
            in_context_features=self.multi_embedding.output_size,
        )

    def create_decoder(self) -> rnn.AutoRegressiveRNN:
        """Creates Decoder."""

        return rnn.AutoRegressiveRNN(
            rnn_unit=self.create_rnn(input_size=self.decoder_size),
            teacher_forcing_ratio=self.teacher_forcing_ratio,
            output_size=1,
        )

    @property
    def encoder_size(self) -> int:
        return len(self.time_varying_reals_encoder)

    @property
    def decoder_size(self) -> int:
        return len(self.time_varying_reals_decoder + self.target)

    @property
    def target_positions(self) -> torch.LongTensor:
        """Position of target variables in covariates.

        Returns
        -------
        target_positions : torch.LongTensor
        """
        return torch.LongTensor(
            [self.covariates.index(name) for name in self.target]
        )

    @property
    def covariates(self) -> list[str]:
        """list of all covariates used by the model.

        Returns
        -------
        covariates : list of str
        """
        return self.time_varying_reals_encoder

    @property
    def decoder_positions(self) -> torch.LongTensor:
        """Position of decoder variables in covariates.

        Returns
        -------
        decoder_positions : torch.LongTensor
        """
        return torch.LongTensor(
            [self.covariates.index(x) for x in self.time_varying_reals_decoder]
        )

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forwards pass.

        Notes
        -----
        1. The output from the MultiEmbedding module is used to initialize the
        hidden state of the Encoder.

        2. We directly use the hidden state at the final time step of the
        encoder to initialize the hidden state of the decoder.

        3. In addition to sharing the encoder last hidden state and to further
        incorporate the encoded input sequence information, the last Encoder
        output is used as first Decoder input.

        Parameters
        ----------
        x : dict, str -> torch.Tensor
            Input dictionary.

        Returns
        ------
        output : torch.Tensor
        """
        encoder_output, hidden_state = self.encode(x)
        output = self.decode(x, hidden_state, encoder_output[:, -1, -1:])

        return output

    def encode(
        self, x: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, rnn.HiddenState]:
        """Transforms an input sequence of variable length into a fixed-shape
        context variable.

        Returns
        -------
        (encoder_output, hidden_state) : tuple
            tuple of Tensors.
        """
        static_embeddings = self.multi_embedding(x["categoricals"])
        encoder_output, hidden_state = self.encoder(
            input=x["encoder_cont"],
            context=static_embeddings,
            lengths=x["encoder_lengths"],
        )

        return encoder_output, hidden_state

    def decode(
        self,
        x: dict[str, torch.Tensor],
        hidden_state: rnn.HiddenState,
        first_input: torch.Tensor,
    ) -> torch.Tensor:
        """Maps the fixed-shape encoded state into a variable-length sequence.

        Returns
        -------
        output : torch.Tensor
            Predicted output.
        """
        positions = torch.cat((self.target_positions, self.decoder_positions))
        input = x["decoder_cont"][..., positions]
        lengths = x["decoder_lengths"]

        return self.decoder(
            input=input,
            first_input=first_input,
            lengths=lengths,
            hidden_state=hidden_state,
        )
