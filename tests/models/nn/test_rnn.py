import unittest
from typing import Literal

import torch

from deepts.models.nn.rnn import ConditionalRNN, make_rnn

INPUT_SIZE = 2
HIDDEN_SIZE = 4
NUM_LAYERS = 10
BATCH_SIZE = 10


RNN_KWARGS = {
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "num_layers": NUM_LAYERS,
    "dropout": 0,
    "batch_first": True,
}


def create_rnn_cell(
    cell_type: Literal["LSTM", "GRU"] = "LSTM"
) -> torch.nn.modules.RNNBase:
    return make_rnn(cell_type=cell_type, **RNN_KWARGS)


def create_conditional_rnn(
    in_context_features: int, cell_type: Literal["LSTM", "GRU"] = "LSTM"
) -> ConditionalRNN:
    rnn_cell = create_rnn_cell(cell_type)
    return ConditionalRNN(rnn_cell, in_context_features)


class TestConditionalRNN(unittest.TestCase):
    def test_get_initial_state_lstm(self):
        in_context_features = 2

        conditional_rnn = create_conditional_rnn(
            in_context_features, cell_type="LSTM"
        )
        context = torch.rand((BATCH_SIZE, in_context_features))
        initial_state = conditional_rnn.get_initial_state(context)

        assert isinstance(initial_state, tuple)
        assert len(initial_state) == 2
        h0, c0 = initial_state

        expected_shape = (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        assert h0.shape == expected_shape
        assert c0.shape == expected_shape

    def test_get_initial_state_gru(self):
        in_context_features = 2

        conditional_rnn = create_conditional_rnn(
            in_context_features, cell_type="GRU"
        )
        context = torch.rand((BATCH_SIZE, in_context_features))
        h0 = conditional_rnn.get_initial_state(context)

        # Expected shape: (num_layers, batch_size, hidden_size)
        expected_shape = (NUM_LAYERS, BATCH_SIZE, HIDDEN_SIZE)
        assert h0.shape == expected_shape

    def test_preprocess_context(self):
        in_context_features = 2

        conditional_rnn = create_conditional_rnn(in_context_features)
        context = torch.rand((BATCH_SIZE, in_context_features))
        context = conditional_rnn.preprocess_context(context)
        assert context.shape == (NUM_LAYERS, BATCH_SIZE, in_context_features)
