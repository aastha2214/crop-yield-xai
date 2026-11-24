"""SHAP additivity: base value + sum of SHAP values must equal the model's prediction (local accuracy)."""
import numpy as np

from src import config as C, india
from src.explain_global import load_global_explainer


def test_global_explainer_is_additive():
    ex, test, _ = load_global_explainer()
    X = test[C.G_FEATURES].sample(200, random_state=7)
    total = ex.base_value + ex.shap_grouped(X).sum(axis=1).to_numpy()
    assert np.abs(total - ex.fitted.predict(X)).max() < 1.0          # hg/ha (float32 noise only)


def test_india_explainer_is_additive():
    ex, test, _ = india.load_india_explainer()
    X = test[C.I_FEATURES].sample(200, random_state=7)
    total = ex.base_value + ex.shap_grouped(X).sum(axis=1).to_numpy()
    assert np.abs(total - ex.fitted.predict(X)).max() < 1e-3          # t/ha
