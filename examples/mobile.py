"""
This example shows the use of :pyfunc:`make_preprocessor` when creating
time series preprocessors. The `Seq2Seq` model is then appended to
the preprocessor to have a full prediction pipeline.
"""

from sklearn.pipeline import Pipeline

from deepts.datasets import get_mobile_data
from deepts.models import Seq2Seq
from deepts.preprocessing import make_preprocessor

TARGET = "target"
GROUP_COLS = ["item_id"]
DATETIME = "date"

# Load data.
X = get_mobile_data()


# Create time series preprocessor.
preprocessor = make_preprocessor(
    group_ids=GROUP_COLS, datetime=DATETIME, target=TARGET
)

# Define model.
max_prediction_length = 6
max_encoder_length = 24
model_kwargs = {
    "group_ids": GROUP_COLS,
    "time_idx": "time_idx",
    "target": TARGET,
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
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

#pipeline.fit(X)
#output = pipeline.predict(X)
