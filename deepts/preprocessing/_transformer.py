from typing import Protocol


class Transformer(Protocol):
    """Transformer interface.

    Used primarily for type checking.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass

    def get_feature_names_out(self):
        pass
