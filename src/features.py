"""Reusable preprocessing + model pipelines (works for both datasets)."""
import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(cat_cols, num_cols) -> ColumnTransformer:
    """One-hot encode categoricals and standardise numerics.

    handle_unknown="ignore" makes unseen categories (e.g. a new country in the
    test years) encode as all-zeros instead of crashing. Scaling matters for
    linear models and the neural net; it is harmless for trees.
    """
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(cat_cols)),
        ("num", StandardScaler(), list(num_cols)),
    ])


def build_pipeline(model, cat_cols, num_cols, log_target: bool = False):
    """Preprocessor + model, optionally wrapped so the target is log1p-transformed.

    TransformedTargetRegressor fits on log1p(y) but predict() returns values
    already inverted with expm1, so every metric is in ORIGINAL units.
    """
    pipe = Pipeline([("prep", build_preprocessor(cat_cols, num_cols)), ("model", model)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def inner_pipeline(fitted):
    """Return the fitted Pipeline whether or not it is wrapped for log-target."""
    return fitted.regressor_ if isinstance(fitted, TransformedTargetRegressor) else fitted
