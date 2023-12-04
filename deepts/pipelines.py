from typing import Union, Optional

import numpy as np
import pandas as pd
import sklearn

from .preprocessing import Transformer
from .utils import validation


class PreprocessorEstimatorPipeline(sklearn.base.BaseEstimator):
    """Wraps preprocessor and estimator into a single sklearn Pipeline.

    Parameters
    ----------
    estimator : Estimator
        Fitted estimator (implementing `fit`/`predict`).

    preprocessor : Transformer
        Fitted transformer (implementing `fit`/`transform`).
    """

    def __init__(
            self,
            preprocessor: Transformer,
            estimator: sklearn.base.BaseEstimator,
            predict_function: str = 'predict',
            inverse_transformer: Optional = None
    ):
        self.preprocessor = preprocessor
        self.estimator = estimator
        self.predict_function = predict_function
        self.inverse_transformer = inverse_transformer

    def fit(self, X: pd.DataFrame, y=None):
        """Fits pipeline composed by both the preprocessor and estimator on X.

        Parameters
        ----------
        X : pd.DataFrame
            Input data

        y : None
            Compatibility purposes.

        Returns
        -------
        self (object)
        """
        self.estimator.fit(self.preprocessor.fit_transform(X))
        return self

    def predict(
            self, X: pd.DataFrame, **kwargs
    ) -> Union[pd.DataFrame, np.array]:
        predictor = getattr(self.estimator, self.predict_function)
        output = predictor(self.preprocessor.transform(X), **kwargs)

        if self.inverse_transformer is not None:
            return self.inverse_transformer.inverse_transform(output)

        return output

    def __sklearn_is_fitted__(self):
        return (
                validation.bool_check_is_fitted(self.estimator) and
                validation.bool_check_is_fitted(self.preprocessor)
        )
