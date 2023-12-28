import pandas as pd
from sklearn.pipeline import make_pipeline

from deepts.base import Transformer


class PreprocessorEstimatorPipeline(Transformer):
    def __init__(
        self,
        preprocessor: Transformer,
        estimator: Transformer,
        inverse_steps: list | None,
    ):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.inverse_steps = inverse_steps

    def fit(self, X: pd.DataFrame, y=None):
        pipeline = make_pipeline(self.preprocessor, self.estimator)
        pipeline.fit(X)
        self.pipeline_ = pipeline
        return self

    def transform(self, X: pd.DataFrame):
        self.check_is_fitted()
        return self.pipeline_.transform(X)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        self.check_is_fitted()
        output = self.pipeline_.predict(X)
        return self.inverse_transform(output)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.inverse_steps is None:
            return X

        Xi = X.copy()
        for step in self.inverse_steps:
            transformer = self.preprocessor[step]
            Xi = transformer.inverse_transform(Xi)

        return Xi
