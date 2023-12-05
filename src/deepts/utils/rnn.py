from typing import (
    List,
    Union,
    Optional
)

import torch
from pytorch_forecasting.models.nn import rnn
from torch import nn


def make_rnn(
        cell_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: Optional[bool] = True,
) -> rnn.RNN:
    """Factory for rnn cells.

    Parameters
    ----------
    cell_type : str, {"LSTM", "GRU"}
        Rnn cell unit.

    input_size: int
        Input size.

    hidden_size: int
        Hidden size.

    num_layers: int
        Number of layers.

    batch_first: If ``True``, then the input and output tensors are provided
            as `(batch, seq, feature)` instead of `(seq, batch, feature)`.
            Note that this does not apply to hidden or cell states. See the
            Inputs/Outputs sections below for details.  Default: ``False``
    """
    cls = rnn.get_rnn(cell_type)
    return cls(input_size=input_size, hidden_size=hidden_size,
               num_layers=num_layers, batch_first=batch_first)


def pad_sequence(
        sequences: List[torch.Tensor],
        batch_first: bool = False,
        padding_value: float = 0.0
) -> torch.Tensor:
    """Pad a list of variable length Tensors with ``padding_value``.

    Wrapper for :pyfunc:`torch.models.utils.rnn.pad_sequence`
    """
    return nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)


def pack_padded_sequence(
        input: torch.Tensor,
        lengths: Union[torch.Tensor, List[int]],
        batch_first: bool = False,
        enforce_sorted: bool = True
):
    """Packs a Tensor containing padded sequences of variable length.

    Wrapper for :pyfunc:`torch.models.utils.rnn.pack_padded_sequence`
    """
    return nn.utils.rnn.pack_padded_sequence(
        input, lengths, batch_first, enforce_sorted)


def pad_packed_sequence(
        sequence: nn.utils.rnn.PackedSequence,
        batch_first: bool = False,
        padding_value: float = 0.0,
        total_length: Optional[int] = None
):
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to pack_padded_sequence(). Wrapper for
    :pyfunc:`torch.models.utils.rnn.pad_packed_sequence`
    """
    return nn.utils.rnn.pad_packed_sequence(
        sequence, batch_first, padding_value, total_length)
