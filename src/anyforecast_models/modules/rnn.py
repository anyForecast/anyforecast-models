import random
from typing import Literal

import torch
from pytorch_forecasting.models.nn import get_rnn
from torch.nn.modules import rnn

from anyforecast_models.utils import checks

HiddenState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def make_rnn(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    cell_type: Literal["LSTM", "GRU"] = "LSTM",
    dropout: float = 0,
    batch_first: bool = True,
) -> rnn.RNNBase:
    """Returns RNN cell unit.

    Parameters
    ----------
    input_size: int
        Input size.

    hidden_size: int
        Hidden size.

    num_layers: int
        Number of layers.

    cell_type : str, {"LSTM", "GRU"}
        Rnn cell unit.

    batch_first : bool, default=True
        If ``True``, then the input and output tensors are provided
        as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
    """
    cls = get_rnn(cell_type)
    return cls(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        dropout=dropout,
    )


class ConditionalRNN(torch.nn.Module):
    """Conditional RNN.

    Initializes RNN hidden states with a learned representation of
    conditional/context data. Useful for time series with external inputs that
    do not depend on time.

    Parameters
    ----------
    rnn_unit : RNNBase
        PyTorch RNN.

    in_context_features : int
        Number of context/conditional features.
    """

    def __init__(self, rnn_cell: rnn.RNNBase, in_context_features: int):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.in_context_features = in_context_features
        checks.check_is_lstm_or_gru(rnn_cell)

        self.h0_linear = torch.nn.Linear(
            in_features=in_context_features,
            out_features=rnn_cell.hidden_size,
        )
        self.c0_linear = torch.nn.Linear(
            in_features=in_context_features,
            out_features=rnn_cell.hidden_size,
        )

    def preprocess_context(self, context: torch.Tensor) -> torch.Tensor:
        """Preprocess context tensor.

        Unsequeezes, permutes and expands to match torch hidden tensors
        definition.

        Returns
        -------
        context : torch.Tensor
            Preprocessed context tensor of shape
            (num_layers, batch_size, in_context_features)
        """
        # Unsqueeze.
        # Each context feature vector must be a row on its own.
        # Shape: (batch_size, 1, in_context_features)
        context.unsqueeze_(1)

        # Permute.
        # Permute dimensions to match PyTorch hidden tensor definition.
        # Shape: (1, batch_size, in_context_features)
        context = context.permute(1, 0, 2)

        # Expand.
        # Expand first dimension to the number of hidden layers.
        # Shape: (num_layers, batch_size, in_context_features)
        context = context.expand(self.rnn_cell.num_layers, -1, -1)

        return context

    def get_initial_state(self, context: torch.Tensor) -> HiddenState:
        """Returns RNN initial state.

        Parameters
        ----------
        context : torch.Tensor, shape=(batch_size, in_context_features)
            Context input tensor.

        Returns
        -------
        initial_state : torch.Tensor or tuple of torch.Tensor
        """
        context = self.preprocess_context(context)
        h0 = self.h0_linear(context)

        if checks.is_lstm(self.rnn_cell):
            c0 = self.c0_linear(context)
            initial_state = (h0, c0)

        else:  # self.rnn_cell is GRU
            initial_state = h0

        return initial_state

    def forward(
        self,
        input: rnn.PackedSequence | torch.Tensor,
        context: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, HiddenState]:
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor, shape=(batch_size, sequence_len, input_size)
            Input tensor.

        context : torch.Tensor, shape=(batch_size, in_context_features)
            Context/conditional input tensor.

        lengths : torch.Tensor, shape=(n,)
            Lengths of sequences. Used for pack padded sequences.

        Returns
        -------
        (encoder_output, hidden_state) : tuple
        """
        hidden_state = self.get_initial_state(context)

        encoder_output, hidden_state = self.rnn_cell(
            x=input,
            hx=hidden_state,
            lengths=lengths,
            enforce_sorted=False,
        )

        return encoder_output, hidden_state


class AutoRegressiveRNN(torch.nn.Module):
    """AutoRegressiveRNN.

    Typically used as a Decoder in Encoder-Decoder arquitectures. That is,
    to map the fixed-shape encoded state into a variable-length sequence.

    By autoregressive it is meant that in order to predict the output sequence
    at each step, the predicted output from the previous time step is fed into
    the rnn as an input.

    Notes
    -----
    When using teacher forcing, the official “ground-truth” is used as input
    (along with the rest of decoder covariates) at every step.

    Parameters
    ----------
    rnn : RNNBase
        Recurrent neural network.

    output_size : int
        Output size.
    """

    def __init__(
        self,
        rnn_cell: rnn.RNNBase,
        teacher_forcing_ratio: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()
        self.rnn_cell = rnn_cell
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.output_size = output_size

        # The output layer transforms the latent representation
        # back to a single prediction.
        self.linear = torch.nn.Linear(
            in_features=rnn_cell.hidden_size, out_features=output_size
        )

    def use_teacher(self) -> bool:
        """Decides whether to use teacher forcing or not.

        Returns
        -------
        use_teacher : bool
        """
        return random.random() < self.teacher_forcing_ratio and self.training

    def forward(
        self,
        input: torch.Tensor,
        first_input: torch.Tensor,
        lengths: torch.Tensor,
        hidden_state: HiddenState,
        enforce_sorted: bool = False,
        target_index: int = 0,
    ):
        """Forward pass.

        Parameters
        ----------
        input : torch.Tensor, shape=(batch_size, sequence_len, input_size)
            Input tensor.

        lengths : torch.Tensor, shape=(n,)
            Lengths of sequences. Used for pack padded sequences.
        """

        if self.use_teacher():
            output, _ = self.rnn_cell(
                x=input,
                hx=hidden_state,
                lengths=lengths,
                enforce_sorted=enforce_sorted,
            )

            return self.linear(output).squeeze(-1)

        def forward_single_step(
            index: int,
            previous_output: torch.Tensor,
            hidden_state: HiddenState,
        ) -> tuple[torch.Tensor, HiddenState]:
            """Forwards single step.

            Parameters
            ----------
            index : int
                Index of forward step.

            previous_output : torch.Tensor
                Previous output.

            hidden_state : torch.Tensor or tuple of torch.Tensor
                Hidden state used for rnn forward pass.

            Returns
            -------
            (step_output, hidden_state) : tuple
            """
            x = input[:, [index]]

            # Overwrite target values with previous output.
            # Target values are located at ``target_index`` .
            x[..., target_index] = previous_output

            step_output, hidden_state = self.rnn_cell(x, hidden_state)
            step_output = self.linear(step_output).squeeze(-1)
            return step_output, hidden_state

        return self.autoregressive(
            forward_one=forward_single_step,
            first_input=first_input,
            first_hidden_state=hidden_state,
            n_steps=input.size(1),
        )

    def autoregressive(
        self,
        forward_one: callable,
        first_input: torch.Tensor,
        first_hidden_state: HiddenState,
        n_steps: int,
    ):
        """Makes autogresive predictions.

        Parameters
        ----------
        forward_one : callable
            Single step forward pass function.

        first_input : torch.Tensor
            First rnn input.

        first_hidden_state : torch.Tensor or tuple of torch.Tensor
            First hidden state used for decoding

        n_steps : int
            Number of prediction steps
        """
        output: list[torch.Tensor] = []
        current_output = first_input
        current_hidden_state = first_hidden_state

        # At each step, the predicted output from the previous time step
        # is fed into the rnn.
        for index in range(n_steps):
            current_output, current_hidden_state = forward_one(
                index=index,
                previous_output=current_output,
                hidden_state=current_hidden_state,
            )
            output.append(current_output)

        output = torch.stack(output, dim=1)

        return output
