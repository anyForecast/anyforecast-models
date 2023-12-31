import unittest

import torch

from deepts.models.nn import MultiEmbedding

BATCH_SIZE = 10


def create_random_tokens(size: tuple[int, int]) -> torch.Tensor:
    return torch.randint(low=0, high=10, size=size)


def create_multi_embedding(
    embedding_sizes: list[tuple], concat: bool = False
) -> MultiEmbedding:
    return MultiEmbedding(embedding_sizes, concat=concat)


class TestMultiEmbedding(unittest.TestCase):
    def test_forward_with_concat_true(self):
        num_categorical_features = 2
        embedding_sizes = [(10, 3), (20, 2)]

        multi_embedding = create_multi_embedding(embedding_sizes, concat=True)
        tokens = create_random_tokens((BATCH_SIZE, num_categorical_features))
        output: torch.Tensor = multi_embedding(tokens)

        output_size = 5  # 3 + 2
        assert output.shape == (BATCH_SIZE, output_size)
        assert multi_embedding.output_size == output_size

    def test_forward_with_concat_false(self):
        num_categorical_features = 2
        embedding_sizes = [(10, 3), (20, 2)]

        multi_embedding = create_multi_embedding(embedding_sizes, concat=False)
        tokens = create_random_tokens((BATCH_SIZE, num_categorical_features))
        output: dict[str, torch.Tensor] = multi_embedding(tokens)
        assert isinstance(output, dict)
        assert len(output) == num_categorical_features

        output_values = list(output.values())
        assert output_values[0].shape == (BATCH_SIZE, 3)
        assert output_values[1].shape == (BATCH_SIZE, 2)
