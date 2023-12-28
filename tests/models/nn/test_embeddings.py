import unittest

import torch

from deepts.models.nn import MultiEmbedding


class TestMultiEmbedding(unittest.TestCase):
    def test_concat_output(self):
        batch_size = 10
        num_categorical_features = 2
        embedding_sizes = [(10, 3), (20, 2)]
        num_dimensions = 5  # 3 + 2

        multi_embedding = MultiEmbedding(embedding_sizes, concat=True)
        tokens = torch.randint(
            low=0, high=10, size=(batch_size, num_categorical_features)
        )
        output: torch.Tensor = multi_embedding(tokens)

        assert output.shape == (batch_size, num_dimensions)

    def test_dictionary_output(self):
        batch_size = 10
        num_categorical_features = 2
        embedding_sizes = [(10, 3), (20, 2)]

        multi_embedding = MultiEmbedding(embedding_sizes)
        tokens = torch.randint(
            low=0, high=10, size=(batch_size, num_categorical_features)
        )
        output: dict[str, torch.Tensor] = multi_embedding(tokens)
        output_values = list(output.values())

        assert isinstance(output, dict)
        assert len(output) == num_categorical_features
        assert output_values[0].shape == (batch_size, 3)
        assert output_values[1].shape == (batch_size, 2)
