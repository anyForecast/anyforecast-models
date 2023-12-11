from deepts.base import Transformer


def _identity(X):
    """The identity function."""
    return X


class IdentityTransformer(Transformer):
    """Identity Transformer"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _identity(X)

    def inverse_transform(self, X):
        return _identity(X)
