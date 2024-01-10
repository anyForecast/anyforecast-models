"""
This example shows the use of `deepts.preprocessing.make_preprocessor` 
when creating time series preprocessors. The `deepts.models.Seq2Seq` model is 
then appended to it to have a full prediction pipeline.
"""

import pandas as pd

from deepts.datasets import load_stallion
from deepts.models import Seq2Seq
from deepts.pipelines import PreprocessorEstimatorPipeline
from deepts.preprocessing import make_preprocessor

ts_dataset = load_stallion()

# Constants
GROUP_COLS = ts_dataset.group_cols
DATETIME_COL = ts_dataset.datetime
TARGET_COL = ts_dataset.target
FREQ = ts_dataset.freq
X = ts_dataset.X


# Create time series preprocessor.
preprocessor = make_preprocessor(
    group_cols=GROUP_COLS,
    datetime_col=DATETIME_COL,
    target_col=TARGET_COL,
    freq=FREQ,
)

# Define model.
max_prediction_length = 6
max_encoder_length = 24
model_kwargs = {
    "group_ids": GROUP_COLS,
    "time_idx": "date",
    "target": TARGET_COL,
    "min_encoder_length": max_encoder_length // 2,
    "max_encoder_length": max_encoder_length,
    "min_prediction_length": 1,
    "max_prediction_length": max_prediction_length,
    "static_categoricals": GROUP_COLS,
    "time_varying_known_reals": ["discount"],
    "time_varying_unknown_reals": [
        "volume",
        "industry_volume",
    ],
}
model = Seq2Seq(**model_kwargs)

# Combine model and preprocessor into a single prediction pipeline.
inverse_steps = ["datetime", "target"]
pipeline = PreprocessorEstimatorPipeline(preprocessor, model, inverse_steps)

X[GROUP_COLS] = X[GROUP_COLS].astype("category")
X[DATETIME_COL] = pd.to_datetime(X[DATETIME_COL])
X["industry_volume"] = X["industry_volume"].astype(float)

pipeline.fit(X)
output = pipeline.predict(X)
