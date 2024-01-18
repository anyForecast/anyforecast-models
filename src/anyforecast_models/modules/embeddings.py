import torch
from torch import nn


def get_embedding_size(n: int, max_size: int = 100) -> int:
    """Determine empirically good embedding sizes (formula taken from fastai).

    Parameters
    ----------
    n : int
        Number of classes.

    max_size : int
        Maximum embedding size. Defaults to 100.

    Returns
    -------
    embedding_size : int
    """
    if n > 2:
        return min(round(1.6 * n**0.56), max_size)
    else:
        return 1


class MultiEmbedding(nn.Module):
    """MultiEmbedding layer.

    Embeddings are named based on order unless ``embedding_names`` is
    specified.

    Parameters
    ----------
    embedding_sizes : list of tuple
        List of categorical and embedding sizes.
        For example, ``[(10, 3), (20, 2)]`` indicates that the first categorical
        variable has 10 unique values which are mapped to 3 embedding
        dimensions. Similarly for the second.

    embedding_names : list of str
        Emebedding names for output dictionary.
    """

    def __init__(
        self,
        embedding_sizes: list[tuple[int, int]],
        embedding_names: list[str] | None = None,
        concat: bool = False,
    ):
        super().__init__()

        self.embedding_sizes = embedding_sizes
        self.embedding_names = embedding_names
        self.concat = concat
        self.init_embeddings()

    @property
    def output_size(self) -> int | dict[str, int]:
        if self.concat:
            return sum([s[1] for s in self.embedding_sizes])

        return {
            name: emb.embedding_dim for name, emb in self.embeddings.items()
        }

    def init_embeddings(self):
        """Initializes :class:`torch.nn.Embedding` modules."""
        self.embeddings: dict[str, nn.Embedding] = nn.ModuleDict()

        for i, embedding_sizes in enumerate(self.embedding_sizes):
            num_embeddings, embedding_dim = embedding_sizes
            name = self.get_embedding_name(i)
            self.embeddings[name] = nn.Embedding(num_embeddings, embedding_dim)

    def get_embedding_name(self, i: int) -> str:
        """Returns ith Embedding name.

        Embeddings are named based on order unless ``embedding_names`` is
        specified.
        """
        return (
            self.embedding_names[i]
            if self.embedding_names is not None
            else str(i)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output: dict[str, torch.Tensor] = {}

        for i, (name, emb) in enumerate(self.embeddings.items()):
            output[name] = emb(tokens[:, i])

        if self.concat:
            output = torch.cat(list(output.values()), dim=-1)

        return output
