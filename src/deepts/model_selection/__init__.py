"""
The :mod:`skorch_forecasting.model_selection` module includes classes and
functions to split the data based on a preset strategy.
"""

from ._split import TimeseriesTrainTestSplit
from ._split import timeseries_train_test_split
from ._walkforward import WalkForwardCV

__all__ = [
    'TimeseriesTrainTestSplit',
    'WalkForwardCV',
    'timeseries_train_test_split'
]
