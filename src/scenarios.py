"""Decision-support logic used by the app: climate what-if scenarios and a crop recommender.

IMPORTANT: both tools query a model trained on historical associations. They show what the
model associates with a change in inputs - they are NOT guaranteed forecasts or agronomic
prescriptions.
"""
import numpy as np
import pandas as pd

from . import config as C

TEMP_STEPS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]     # °C warming
RAIN_STEPS = [-30, -15, 0, 15, 30]                   # % change in annual rainfall


# ----------------------------------------------------------------- baselines
def global_baseline(df: pd.DataFrame, country: str, crop: str, last_year: int) -> dict:
    """'Today' for a country-crop: climate/pesticide averaged over its 3 most recent years.

    Averaging 3 years (not just the last) avoids anchoring on one unusual year.
    """
    sub = df[(df[C.G_COUNTRY] == country) & (df[C.G_CROP] == crop)]
    recent = sub[sub[C.G_YEAR] >= sub[C.G_YEAR].max() - 2]
    return {C.G_COUNTRY: country, C.G_CROP: crop, C.G_YEAR: last_year,
            C.G_RAIN: float(recent[C.G_RAIN].mean()), C.G_TEMP: float(recent[C.G_TEMP].mean()),
            C.G_PEST: float(recent[C.G_PEST].mean())}


def last_full_year(df: pd.DataFrame, min_rows: int = 100) -> int:
    """Latest India year with enough rows (2020 has only 37, so it is not a reliable 'today')."""
    counts = df[C.I_YEAR].value_counts()
    return int(counts[counts >= min_rows].index.max())


def india_baseline(df: pd.DataFrame, state: str, crop: str, season: str, year: int) -> dict:
    """'Today' for a state-crop-season using medians of its last 5 years (fallback: all years)."""
    sub = df[(df[C.I_STATE] == state) & (df[C.I_CROP] == crop) & (df[C.I_SEASON] == season)]
    if sub.empty:
        sub = df[(df[C.I_STATE] == state) & (df[C.I_CROP] == crop)]
    recent = sub[sub[C.I_YEAR] >= sub[C.I_YEAR].max() - 4]
    m = recent.median(numeric_only=True)
    return {C.I_CROP: crop, C.I_SEASON: season, C.I_STATE: state, C.I_YEAR: year,
            C.I_AREA: float(m[C.I_AREA]), C.I_RAIN: float(m[C.I_RAIN]),
            C.I_FERT_HA: float(m[C.I_FERT_HA]), C.I_PEST_HA: float(m[C.I_PEST_HA])}


# ----------------------------------------------------------------- scenarios
def apply_change(base: dict, rain_col: str, temp_col, d_temp: float, rain_pct: float) -> dict:
    """Copy of `base` with temperature raised by d_temp °C and rainfall scaled by rain_pct %."""
    row = dict(base)
    row[rain_col] = max(0.0, base[rain_col] * (1 + rain_pct / 100))
    if temp_col is not None:
        row[temp_col] = base[temp_col] + d_temp
    return row


def climate_grid(model, base: dict, features: list, rain_col: str, temp_col,
                 temps=TEMP_STEPS, rains=RAIN_STEPS):
    """Predicted yield % change vs the baseline for every (rain change, temperature change).

    Returns (baseline_prediction, DataFrame[index=rain %, columns=temp °C] of % change).
    With temp_col=None (India model has no temperature) only temps=[0] is used.
    """
    temps = temps if temp_col is not None else [0.0]
    rows = [apply_change(base, rain_col, temp_col, t, r) for r in rains for t in temps]
    preds = model.predict(pd.DataFrame(rows)[features])
    base_pred = float(model.predict(pd.DataFrame([base])[features])[0])
    grid = pd.DataFrame(np.array(preds).reshape(len(rains), len(temps)), index=rains, columns=temps)
    return base_pred, (grid / base_pred - 1) * 100


# ----------------------------------------------------------------- recommender
def recommend(model, df: pd.DataFrame, state: str, season: str, rainfall: float, area: float,
              year: int, min_rows: int = 5) -> pd.DataFrame:
    """Predict yield for every crop grown in `state` during `season`.

    Fertilizer/pesticide per hectare are set to each crop's own typical (median) level in that
    state and season, because the user does not specify them. Returns one row per crop with the
    prediction, the crop's historical mean yield there, and the % difference between them.
    """
    sub = df[(df[C.I_STATE] == state) & (df[C.I_SEASON] == season)]
    counts = sub[C.I_CROP].value_counts()
    crops = counts[counts >= min_rows].index
    rows, meta = [], []
    for crop in crops:
        c = sub[sub[C.I_CROP] == crop]
        rows.append({C.I_CROP: crop, C.I_SEASON: season, C.I_STATE: state, C.I_YEAR: year,
                     C.I_AREA: area, C.I_RAIN: rainfall,
                     C.I_FERT_HA: float(c[C.I_FERT_HA].median()), C.I_PEST_HA: float(c[C.I_PEST_HA].median())})
        meta.append((crop, float(c[C.I_TARGET].mean()), len(c)))
    if not rows:
        return pd.DataFrame(columns=["crop", "predicted", "hist_avg", "vs_usual_pct", "n_rows", "row"])
    X = pd.DataFrame(rows)[C.I_FEATURES]
    pred = model.predict(X)
    out = pd.DataFrame({"crop": [m[0] for m in meta], "predicted": pred,
                        "hist_avg": [m[1] for m in meta], "n_rows": [m[2] for m in meta]})
    out["vs_usual_pct"] = (out.predicted / out.hist_avg - 1) * 100
    out["row"] = list(X.to_dict("records"))
    return out.sort_values("predicted", ascending=False).reset_index(drop=True)
