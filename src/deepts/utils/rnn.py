import torch
from pytorch_forecasting.utils import nn


def pad_sequence(
    sequences: list[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """Pad a list of variable length Tensors with ``padding_value``.

    Wrapper for :pyfunc:`torch.models.utils.rnn.pad_sequence`
    """
    return nn.utils.rnn.pad_sequence(sequences, batch_first, padding_value)


def pack_padded_sequence(
    input: torch.Tensor,
    lengths: torch.Tensor | list[int],
    batch_first: bool = False,
    enforce_sorted: bool = True,
):
    """Packs a Tensor containing padded sequences of variable length.

    Wrapper for :pyfunc:`torch.models.utils.rnn.pack_padded_sequence`
    """
    return nn.utils.rnn.pack_padded_sequence(
        input, lengths, batch_first, enforce_sorted
    )


def pad_packed_sequence(
    sequence: nn.utils.rnn.PackedSequence,
    batch_first: bool = False,
    padding_value: float = 0.0,
    total_length: int | None = None,
):
    """Pads a packed batch of variable length sequences.

    It is an inverse operation to pack_padded_sequence(). Wrapper for
    :pyfunc:`torch.models.utils.rnn.pad_packed_sequence`
    """
    return nn.utils.rnn.pad_packed_sequence(
        sequence, batch_first, padding_value, total_length
    )
