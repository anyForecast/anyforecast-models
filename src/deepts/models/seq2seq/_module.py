from __future__ import annotations

import random

import torch
from torch import nn

from deepts.data import TimeseriesDataset
from deepts.utils import rnn

HiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Seq2SeqModule(torch.nn.Module):
    """Encoder-decoder architecture applied to timeseries.

    An encoder network condenses an input sequence into a vector,
    and a decoder network unfolds that vector into a new sequence.
    Also, an additional embedding layer is used to condition the encoder network
    with time independent/statical features.

    Parameters
    ----------
    embedding_dim : int
        Dimension for every embedding table.

    embedding_sizes : tuple
        Size of each embedding table.

    hidden_size : int
        Size of the context vector.

    target : list of str
        list of target names.

    time_varying_reals_encoder : list of str
        list of variables to be encoded.

    time_varying_reals_decoder : list of str
        list of variables used when decoding.

    cell : str, {"LSTM", "GRU"}, default="LSTM"
        Cell type.

    num_layers : int, default=1
        Number of hidden layers.
    """

    def __init__(
        self,
        embedding_dim: int,
        embedding_sizes: tuple[int],
        hidden_size: int,
        target: list[str],
        time_varying_reals_encoder: list[str],
        time_varying_reals_decoder: list[str],
        cell: str = "LSTM",
        num_layers: int = 1,
        tf_ratio: float = 0.2,
    ):
        super().__init__()

        self.target = target
        self.time_varying_reals_encoder = time_varying_reals_encoder
        self.time_varying_reals_decoder = time_varying_reals_decoder
        self.cell = cell
        self.tf_ratio = tf_ratio

        self.embedding = MultiEmbedding(
            embedding_dim=embedding_dim,
            embedding_sizes=embedding_sizes,
            encoder_hidden_size=hidden_size,
            encoder_num_layers=num_layers,
        )

        self.encoder = rnn.make_rnn(
            cell_type=cell,
            input_size=len(time_varying_reals_encoder),
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.decoder = rnn.make_rnn(
            cell_type=cell,
            input_size=len(time_varying_reals_decoder + target),
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        self.decoder_projection = nn.Linear(
            in_features=hidden_size, out_features=len(target)
        )

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

    def use_teacher(self) -> bool:
        """Decides whether to use teacher forcing or not.

        Returns
        -------
        use_teacher : bool
        """
        return random.random() < self.tf_ratio and self.training

    def forward(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forwards pass.

        Notes
        -----
        * We directly use the embedding output to initialize the hidden state
        of the encoder.

        * We directly use the hidden state at the final time step of the
        encoder to initialize the hidden state of the decoder.

        * In addition to sharing the encoder last hidden state and to further
        incorporate the encoded input sequence information, we
        directly use the last encoder output as first decoder input.

        Parameters
        ----------
        x : dict, str -> torch.Tensor
            Input dictionary.

        Returns
        ------
        output : torch.Tensor
        """
        initial_state = self.get_initial_state(x)
        encoder_output, hidden_state = self.encode(x, initial_state)
        output = self.decode(x, hidden_state, encoder_output[:, -1, -1:])

        return output

    def get_initial_state(self, x: dict[str, torch.Tensor]) -> HiddenState:
        """Returns encoder initial state.

        Returns
        -------
        initial_state : torch.Tensor or tuple of torch.Tensor
        """
        h0 = self.embedding(x["categoricals"])

        if self.cell == "LSTM":
            c0 = torch.zeros_like(h0, device=h0.device)
            initial_state = (h0, c0)
        else:  # cell == "GRU"
            initial_state = h0

        return initial_state

    def encode(
        self, x: dict[str, torch.Tensor], hidden_state: HiddenState
    ) -> tuple[torch.Tensor, HiddenState]:
        """Transforms an input sequence of variable length into a fixed-shape
        context variable.

        Returns
        -------
        (encoder_output, hidden_state) : tuple
            tuple of Tensors.
        """
        input_vector = x["encoder_cont"]
        lengths = x["encoder_lengths"]
        encoder_output, hidden_state = self.encoder(
            input_vector, hidden_state, lengths, enforce_sorted=False
        )

        return encoder_output, hidden_state

    def decode(
        self,
        x: dict[str, torch.Tensor],
        hidden_state: HiddenState,
        first_input: torch.Tensor,
    ) -> torch.Tensor:
        """Maps the fixed-shape encoded state into a variable-length sequence.

        Returns
        -------
        output : torch.Tensor
            Predicted output.
        """
        positions = torch.cat((self.target_positions, self.decoder_positions))
        input_vector = x["decoder_cont"][..., positions]

        if self.use_teacher():
            # When using teacher forcing, the official “ground-truth” is
            # used as input (along with the rest of decoder covariates)
            # at every step.
            lengths = x["decoder_lengths"]
            output, _ = self.decoder(
                input_vector,
                hidden_state,
                lengths=lengths,
                enforce_sorted=False,
            )

            return self.decoder_projection(output).squeeze(-1)

        def decode_single_step(
            index: int, previous_output: torch.Tensor, hidden_state: HiddenState
        ) -> tuple[torch.Tensor, HiddenState]:
            """Decodes single step.

            Parameters
            ----------
            index : int
                Index of decoding step.

            previous_output : torch.Tensor
                Previous output.

            hidden_state : torch.Tensor or tuple of torch.Tensor
                Hidden state used for decoding.

            Returns
            -------
            (decoder_output, hidden_state) : tuple
                tuple of Tensors.
            """
            x = input_vector[:, [index]]

            # Overwrite target values with previous output.
            # Target values are located at zero index.
            x[..., 0] = previous_output

            decoder_output, hidden_state = self.decoder(x, hidden_state)
            output = self.decoder_projection(decoder_output).squeeze(-1)
            return output, hidden_state

        return self.decode_autoregressive(
            decode_one=decode_single_step,
            first_input=first_input,
            first_hidden_state=hidden_state,
            n_decoder_steps=input_vector.size(1),
        )

    def decode_autoregressive(
        self,
        decode_one: callable,
        first_input: torch.Tensor,
        first_hidden_state: HiddenState,
        n_decoder_steps: int,
    ) -> torch.Tensor:
        """Make predictions in autoregressive manner.

        To predict the output sequence at each step, the predicted output from
        the previous time step is fed into the decoder as an input.

        Parameters
        ----------
         decode_one : callable
            Single step decoder.

        first_input : torch.Tensor
            First decoder input. Popular options for ``first_input`` are:
            - Last encoder target.
                Full autoregressive behaviour.

            - Last encoder output.
                Helps to further incorporate the encoded input sequence
                information.

        first_hidden_state : torch.Tensor or tuple of torch.Tensor
            First hidden state used for decoding

        n_decoder_steps : int
            Number of decoding/prediction steps
        """

        output = []
        current_output = first_input
        current_hidden_state = first_hidden_state

        for index in range(n_decoder_steps):
            current_output, current_hidden_state = decode_one(
                index=index,
                previous_output=current_output,
                hidden_state=current_hidden_state,
            )

            output.append(current_output)

        output = torch.stack(output, dim=1)

        return output

    @classmethod
    def from_dataset(cls, ds: TimeseriesDataset, **kwargs) -> Seq2SeqModule:
        """Creates module from dataset.

        Parameters
        ----------
        ds : TimeseriesDataset
            Dataset object from which init parameters will be obtained.

        kwargs : keyword arguments
            Additional arguments such as hyperparameters for model
            (see ``__init__()``).

        Returns
        -------
        Seq2SeqModule object
        """
        # Get embedding sizes from categorical data.
        pfds = ds.get_pytorchforecasting()
        embedding_sizes = torch.max(pfds.data["categoricals"], dim=0)
        embedding_sizes = tuple(embedding_sizes.values.numpy() + 1)

        # All continuous variables, i.e., time_varying_known_reals,
        # time_varying_known_reals and static_reals, are encoded.
        time_varying_reals_encoder = ds.features.reals

        # Only known variables are used for decoding.
        time_varying_reals_decoder = ds.features.time_varying_known_reals

        new_kwargs = dict(
            embedding_sizes=embedding_sizes,
            target=ds.features.target_names,
            time_varying_reals_encoder=time_varying_reals_encoder,
            time_varying_reals_decoder=time_varying_reals_decoder,
        )

        kwargs.update(new_kwargs)
        return cls(**kwargs)


class MultiEmbedding(nn.Module):
    """MultiEmbedding layer.

    Conditions the Encoder with time independents/static categorical
    variables.

    Parameters
    ----------
    embedding_dim : int
        Embedding dimension for all tables.

    embedding_sizes : tuple
        Embedding size for each table.

    encoder_hidden_size : int
        Encoder hidden size.

    encoder_num_layers : int
        Encoder number of layers.
    """

    def __init__(
        self,
        embedding_dim: int,
        embedding_sizes: tuple[int],
        encoder_hidden_size: int,
        encoder_num_layers: int,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embedding_sizes = embedding_sizes
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_hidden_size

        _module_dict = {}
        for i, size in enumerate(embedding_sizes):
            # Assign a :class:`nn.Sequential` model to each embedding.
            _module_dict[str(i)] = nn.Sequential(
                nn.Embedding(size, embedding_dim),
                nn.Linear(embedding_dim, encoder_hidden_size),
            )

        self.linear = nn.Linear(len(embedding_sizes), encoder_num_layers)
        self.module_dict = nn.ModuleDict(_module_dict)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size = len(tokens)

        # Pre-allocate output memory.
        output_shape = (
            batch_size,
            self.encoder_hidden_size,
            len(self.embedding_sizes),
        )
        output = torch.zeros(output_shape, device=tokens.device)

        # Forward tokens for each embedding.
        for i, sequential in self.module_dict.items():
            i = int(i)
            ith_tokens = tokens[:, i]
            ith_output = sequential(ith_tokens)
            output[:, :, i] = ith_output

        # Apply linear transformation.
        output = self.linear(output)

        # Permute output since it must match encoder h0 input shape format.
        # - Current shape: (batch_size, hidden_size, num_layers)
        # - Target shape: (num_layers, batch_size, hidden_size).
        output = output.permute(2, 0, 1)
        return output
