"""
This example shows the use of `GroupWiseColumnTransformer` when creating
time series preprocessors. The `Seq2Seq` model is then appended to
the preprocessor to have a full prediction pipeline.
"""

import numpy as np
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from skorch_forecasting.datasets import get_mobile_data

from skorch_forecasting.compose import GroupWiseColumnTransformer
from skorch_forecasting.models import Seq2Seq

X = get_mobile_data()

target = 'target'
group_ids = ['item_id']

# Create a group wise column transformer for both numeric and categorical data.
# Notice ``numeric_selector`` excludes target from selection.
exclude = f"^(?!{target}).*$"
numeric_selector = make_column_selector(exclude, dtype_include=np.numeric)
object_selector = make_column_selector(dtype_include=object)
transformers = [
    (MinMaxScaler(), numeric_selector),
    (OneHotEncoder(), object_selector)
]
features_transformer = GroupWiseColumnTransformer(
    transformers, group_ids)

# Create target transformer.
target_transformer = GroupWiseColumnTransformer(
    group_ids=group_ids,
    transformers=[(MinMaxScaler(), [target])]
)

# Place both features and target transformers into a single pipeline.
preprocessor = Pipeline(
    steps=[('features', features_transformer), ('target', target_transformer)]
)

# Define model.
max_prediction_length = 6
max_encoder_length = 24
model_kwargs = {
    'group_ids': group_ids,
    'time_idx': "time_idx",
    'target': target,
    'min_encoder_length': max_encoder_length // 2,
    'max_encoder_length': max_encoder_length,
    'min_prediction_length': 1,
    'max_prediction_length': max_prediction_length,
    'static_categoricals': group_ids,
    'time_varying_known_reals': ["discount"],
    'time_varying_unknown_reals': [
        "volume",
        "industry_volume",
    ]
}
model = Seq2Seq(**model_kwargs)

# Append model to preprocessor to have a full prediction pipeline.
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("model", model)]
)

pipeline.fit(X)
output = pipeline.predict(X)

