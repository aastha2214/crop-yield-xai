"""Generate all global SHAP figures + text (used by notebook 03 and callable alone)."""
import joblib
import pandas as pd

from . import config as C, data
from .explain import Explainer, pick_examples


def load_global_explainer(n_background=2000):
    """Build the global Explainer and return (explainer, df_test, meta)."""
    meta = joblib.load(C.GLOBAL_META_PATH)
    model = joblib.load(C.GLOBAL_MODEL_PATH)
    df = data.clean(verbose=False)
    _, test = data.time_split(df)
    q = {c: tuple(v) for c, v in meta["quantiles"].items()}
    ex = Explainer(model, C.G_CAT, C.G_NUM, C.G_LABELS, C.G_UNITS, q, test.head(n_background))
    return ex, test, meta


def build_all():
    """Create every global XAI plot; return dict with example sentences."""
    ex, test, _ = load_global_explainer()
    X = test[C.G_FEATURES]
    sample = X.sample(1500, random_state=C.RANDOM_STATE)
    ex.beeswarm(sample, C.FIG_DIR / "shap_beeswarm.png")
    ex.bar_plot(sample, C.FIG_DIR / "shap_bar.png")
    cmp = ex.compare_importance(sample, C.FIG_DIR / "shap_vs_builtin.png")
    for col, nm in [(C.G_RAIN, "rainfall"), (C.G_TEMP, "temperature"), (C.G_PEST, "pesticides")]:
        ex.dependence(sample, col, C.FIG_DIR / f"shap_dependence_{nm}.png")
    pred = ex.fitted.predict(X)
    out = {}
    for name, i in pick_examples(pred).items():
        row = X.iloc[[i]]
        ex.waterfall(row, C.FIG_DIR / f"shap_waterfall_{name}.png",
                     title=f"{name.capitalize()}-yield example")
        out[name] = {"row": test.iloc[i].to_dict(), "sentence": ex.sentence(row)}
    return out, cmp
