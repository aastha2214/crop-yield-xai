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
def recommend(model, df: pd.DataFrame, state: str, season: str, rainfall: float, area=None,
              year: int = 2019, min_rows: int = 5) -> pd.DataFrame:
    """Predict yield for every crop grown in `state` during `season`.

    Each crop is described by ITS OWN typical conditions there: median area, and median fertilizer and
    pesticide per hectare. Pass `area` to override the area for ALL crops (the app's optional input).
    Areas differ hugely between crops (state-wide totals; e.g. rice is cultivated on far more hectares
    than sesame), so one shared area would unfairly favour some crops.
    Returns one row per crop: prediction, area used, the crop's historical mean yield there and the
    % difference between them.
    """
    sub = df[(df[C.I_STATE] == state) & (df[C.I_SEASON] == season)]
    counts = sub[C.I_CROP].value_counts()
    crops = counts[counts >= min_rows].index
    rows, meta = [], []
    for crop in crops:
        c = sub[sub[C.I_CROP] == crop]
        rows.append({C.I_CROP: crop, C.I_SEASON: season, C.I_STATE: state, C.I_YEAR: year,
                     C.I_AREA: float(c[C.I_AREA].median()) if area is None else float(area), C.I_RAIN: rainfall,
                     C.I_FERT_HA: float(c[C.I_FERT_HA].median()), C.I_PEST_HA: float(c[C.I_PEST_HA].median())})
        meta.append((crop, float(c[C.I_TARGET].mean()), len(c)))
    if not rows:
        return pd.DataFrame(columns=["crop", "predicted", "hist_avg", "vs_usual_pct", "n_rows", "area_used", "row"])
    X = pd.DataFrame(rows)[C.I_FEATURES]
    pred = model.predict(X)
    out = pd.DataFrame({"crop": [m[0] for m in meta], "predicted": pred, "area_used": X[C.I_AREA].to_numpy(),
                        "hist_avg": [m[1] for m in meta], "n_rows": [m[2] for m in meta]})
    out["vs_usual_pct"] = (out.predicted / out.hist_avg - 1) * 100
    out["row"] = list(X.to_dict("records"))
    return out.sort_values("predicted", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- default-scenario scan
def behaviour_metrics(grid: pd.DataFrame) -> dict:
    """How 'intuitive and smooth' a scenario grid is (grid = % change vs today, rain x temp).

    dry_drop  : % the yield FALLS at -30 % rain (temperature unchanged)   - want >= 5
    hot_drop  : % the yield FALLS at +3 °C (rainfall unchanged)           - want >= 5 (needs a temp input)
    viol_rain : total % points by which yield wrongly FALLS as rain rises from -30 % to 0 %
    viol_heat : total % points by which yield wrongly RISES as temperature rises
    rough     : mean |second difference| along both axes (0 = perfectly straight curves)
    score     : viol_rain + viol_heat + rough  (lower = more intuitive and smoother)
    """
    dry = grid.loc[[-30, -15, 0]]                         # dry side only: wetter should never hurt here
    d_rain = dry.diff().iloc[1:]                          # change when rain increases, per temperature column
    viol_rain = float((-d_rain).clip(lower=0).to_numpy().sum())
    rough_parts = [dry.diff().diff().iloc[2:].abs().to_numpy().mean()]
    out = {"dry_drop": float(-grid.loc[-30, 0.0]), "viol_rain": viol_rain}
    if grid.shape[1] > 1:                                 # a temperature axis exists
        d_heat = grid.diff(axis=1).iloc[:, 1:]            # change when temperature rises, per rain row
        out["hot_drop"] = float(-grid.loc[0, grid.columns[-1]])
        out["viol_heat"] = float(d_heat.clip(lower=0).to_numpy().sum())
        rough_parts.append(grid.diff(axis=1).diff(axis=1).iloc[:, 2:].abs().to_numpy().mean())
    else:
        out["hot_drop"], out["viol_heat"] = float("nan"), 0.0
    out["rough"] = float(np.mean(rough_parts))
    out["score"] = out["viol_rain"] + out["viol_heat"] + out["rough"]
    return out


def scan_global(model, df: pd.DataFrame, last_year: int, min_rows: int = 8) -> pd.DataFrame:
    """Behaviour metrics for EVERY country-crop pair in the global dataset."""
    rows = []
    for (country, crop), g in df.groupby([C.G_COUNTRY, C.G_CROP]):
        if len(g) < min_rows:
            continue
        base = global_baseline(df, country, crop, last_year)
        base_pred, grid = climate_grid(model, base, C.G_FEATURES, C.G_RAIN, C.G_TEMP)
        rows.append({"model": "global", "country_or_state": country, "crop": crop, "season": "",
                     "baseline": base_pred, **behaviour_metrics(grid)})
    return pd.DataFrame(rows)


def scan_india(model, df: pd.DataFrame, year: int, min_rows: int = 8) -> pd.DataFrame:
    """Behaviour metrics for EVERY state-crop-season in the India data (rainfall axis only)."""
    rows = []
    for (state, crop, season), g in df.groupby([C.I_STATE, C.I_CROP, C.I_SEASON]):
        if len(g) < min_rows:
            continue
        base = india_baseline(df, state, crop, season, year)
        base_pred, grid = climate_grid(model, base, C.I_FEATURES, C.I_RAIN, None)
        rows.append({"model": "india", "country_or_state": state, "crop": crop, "season": season,
                     "baseline": base_pred, **behaviour_metrics(grid)})
    return pd.DataFrame(rows)


def qualifying(scan: pd.DataFrame, min_drop: float = 5.0) -> pd.DataFrame:
    """Rows whose yield falls >= min_drop % with less rain AND (if a temp axis exists) with +3 °C, best first."""
    ok = scan[(scan.dry_drop >= min_drop) & ((scan.hot_drop >= min_drop) | scan.hot_drop.isna())]
    return ok.sort_values("score").reset_index(drop=True)


def main():
    """Scan every combination and save outputs/results/scenario_scan_*.csv (python -m src.scenarios)."""
    import joblib
    from . import data, india
    gmodel, gdf, meta = joblib.load(C.GLOBAL_MODEL_PATH), data.clean(verbose=False), joblib.load(C.GLOBAL_META_PATH)
    imodel, idf = joblib.load(C.INDIA_MODEL_PATH), india.clean(verbose=False)
    g = scan_global(gmodel, gdf, meta["year_range"][1]).round(3)
    i = scan_india(imodel, idf, last_full_year(idf)).round(3)
    g.to_csv(C.RESULTS_DIR / "scenario_scan_global.csv", index=False)
    i.to_csv(C.RESULTS_DIR / "scenario_scan_india.csv", index=False)
    print(f"global combos scanned: {len(g)}, qualifying (both >=5%): {len(qualifying(g))}")
    print(f"india combos scanned: {len(i)}, qualifying on rainfall (>=5%): {len(qualifying(i))}")


if __name__ == "__main__":
    main()
