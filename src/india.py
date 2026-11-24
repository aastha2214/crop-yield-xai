"""India case study: cleaning, modelling, SHAP and global comparison."""
import json

import joblib
import numpy as np
import pandas as pd

from . import config as C, data, evaluate, train
from .explain import Explainer, pick_examples


# ------------------------------------------------------------------ cleaning
def load_raw() -> pd.DataFrame:
    """Read the India CSV as shipped."""
    return pd.read_csv(C.INDIA_CSV)


def remove_yield_outliers(df: pd.DataFrame, k: float = 3.0, min_rows: int = 20):
    """Drop extreme yields PER CROP using Tukey fences with a wide multiplier (k=3).

    Why per crop: yields differ by orders of magnitude between crops (coconut is
    counted in nuts, sugarcane in tonnes), so one global fence would be meaningless.
    Why k=3 (not 1.5): only 'extreme' points (typically data-entry errors) are removed;
    legitimately high-yielding states are kept. Crops with < min_rows rows are left alone.
    """
    keep = pd.Series(True, index=df.index)
    for _, idx in df.groupby(C.I_CROP).groups.items():
        y = df.loc[idx, C.I_TARGET]
        if len(y) < min_rows:
            continue
        q1, q3 = y.quantile([0.25, 0.75]); iqr = q3 - q1
        keep[idx] = (y >= q1 - k * iqr) & (y <= q3 + k * iqr)
    return df[keep], int((~keep).sum())


def clean(verbose: bool = True) -> pd.DataFrame:
    """Cleaned India dataframe. Target = Production / Area (recomputed).

    Steps: strip text, drop missing / zero Area or Production, exclude crops measured in
    a different unit (coconut = nuts), recompute yield and
    compare it with the shipped Yield column, then remove extreme per-crop outliers.
    """
    df = load_raw()
    n0 = len(df)
    for c in C.I_CAT:
        df[c] = df[c].str.strip()           # Season/State carry trailing spaces
    df = df.dropna(subset=[C.I_AREA, C.I_PROD])
    df = df[(df[C.I_AREA] > 0) & (df[C.I_PROD] > 0)]
    n1 = len(df)
    n_excl = int(df[C.I_CROP].isin(C.I_EXCLUDE_CROPS).sum())
    df = df[~df[C.I_CROP].isin(C.I_EXCLUDE_CROPS)]      # different unit (nuts), see config
    recomputed = df[C.I_PROD] / df[C.I_AREA]
    diff = (recomputed - df[C.I_TARGET]).abs()
    rel = diff / recomputed
    info = {"rows_raw": n0, "after_zero_missing_drop": n1, "excluded_other_unit_rows": n_excl,
            "median_abs_diff": float(diff.median()), "max_abs_diff": float(diff.max()),
            "pct_rows_rel_diff_gt_1pct": float((rel > 0.01).mean() * 100)}
    df = df.assign(**{C.I_TARGET: recomputed}).drop_duplicates()
    df, n_out = remove_yield_outliers(df)
    info["outliers_removed"] = n_out
    info["rows_final"] = len(df)
    df = df.reset_index(drop=True)
    # Per-hectare intensities: totals mostly reflect farm size, intensity reflects practice.
    df[C.I_FERT_HA] = df[C.I_FERT] / df[C.I_AREA]
    df[C.I_PEST_HA] = df[C.I_PEST] / df[C.I_AREA]
    if verbose:
        print(json.dumps(info, indent=1))
    clean.info = info
    return df


# ------------------------------------------------------------------ modelling
def main():
    """Train the top-2 Phase-5 model types on India under both splits; save best."""
    top = json.loads((C.RESULTS_DIR / "top_models.json").read_text())
    types = top["top2_types"]
    df = clean()
    splits = {"random": data.random_split(df), "time": data.time_split(df, C.I_YEAR, C.I_TEST_YEARS)}
    tr_t, te_t = splits["time"]
    models = {k: v for k, v in train.get_models().items() if k in types}
    log_choice, _ = train.log_target_experiment(tr_t[C.I_FEATURES], tr_t[C.I_TARGET],
                                                te_t[C.I_FEATURES], te_t[C.I_TARGET],
                                                C.I_CAT, C.I_NUM, models)
    print("India log-target choice:", log_choice, " models:", types)
    res_all, fits = [], {}
    for split, (tr, te) in splits.items():
        res, fitted, params = train.run_comparison(tr, te, C.I_CAT, C.I_NUM, C.I_TARGET,
                                                   C.I_YEAR, split, log_choice, only=types, tune_models=types)
        res_all.append(res); fits[split] = fitted
    comp = pd.concat(res_all, ignore_index=True)
    comp = comp.round(3)
    comp.to_csv(C.INDIA_COMPARISON_CSV, index=False)
    t = comp[comp.split == "time"]
    best_name = t.loc[t.cv_R2_mean.idxmax(), "model"]
    best = fits["time"][best_name]
    pred = best.predict(te_t[C.I_FEATURES])
    evaluate.plot_pred_vs_actual(te_t[C.I_TARGET], pred, f"India: {best_name} (time-based test)",
                                 C.FIG_DIR / "india_pred_vs_actual.png")
    evaluate.plot_comparison(comp, C.FIG_DIR / "india_model_comparison.png", "India model comparison (test R²)")
    joblib.dump(best, C.INDIA_MODEL_PATH)
    meta = {"best_name": best_name, "log_target": log_choice,
            "states": sorted(df[C.I_STATE].unique()), "crops": sorted(df[C.I_CROP].unique()),
            "seasons": sorted(df[C.I_SEASON].unique()),
            "medians": df[C.I_NUM].median().to_dict(),
            "quantiles": {c: df[c].quantile([0.33, 0.67]).tolist() for c in [C.I_RAIN, C.I_FERT_HA, C.I_PEST_HA]},
            "year_range": [int(df[C.I_YEAR].min()), int(df[C.I_YEAR].max())],
            "cleaning": clean.info}
    joblib.dump(meta, C.INDIA_META_PATH)
    print("INDIA BEST:", best_name)
    print(comp.round(3).to_string())


# ------------------------------------------------------------------ SHAP
def load_india_explainer():
    """Explainer for the saved India model; returns (explainer, test_df, meta)."""
    meta = joblib.load(C.INDIA_META_PATH)
    model = joblib.load(C.INDIA_MODEL_PATH)
    df = clean(verbose=False)
    _, test = data.time_split(df, C.I_YEAR, C.I_TEST_YEARS)
    q = {c: tuple(v) for c, v in meta["quantiles"].items()}
    return Explainer(model, C.I_CAT, C.I_NUM, C.I_LABELS, C.I_UNITS, q, test.head(1000)), test, meta


def example_row(test_or_df, crop="Rice", state="Punjab", season="Kharif"):
    """First matching row (latest year) as a one-row DataFrame of features."""
    sub = test_or_df[(test_or_df[C.I_CROP] == crop) & (test_or_df[C.I_STATE] == state)
                     & (test_or_df[C.I_SEASON] == season)].sort_values(C.I_YEAR)
    return sub.tail(1)[C.I_FEATURES]


def global_comparison() -> pd.DataFrame:
    """India vs world, per shared crop, using yield_df.csv rows where Area == 'India'.

    Units: both converted to t/ha (hg/ha * 1e-4). 'Rice, paddy' (FAO) ~ 'Rice' (India).
    """
    g = data.clean(verbose=False)
    gi = g[g[C.G_COUNTRY] == "India"]
    mapping = {"Rice, paddy": "Rice", "Wheat": "Wheat", "Maize": "Maize", "Potatoes": "Potato",
               "Sweet potatoes": "Sweet potato", "Sorghum": "Jowar", "Soybeans": "Soyabean"}
    rows = []
    for fao, ind in mapping.items():
        w = g[g[C.G_CROP] == fao].groupby(C.G_YEAR)[C.G_TARGET].mean() * C.HG_PER_HA_TO_T_PER_HA
        i = gi[gi[C.G_CROP] == fao].groupby(C.G_YEAR)[C.G_TARGET].mean() * C.HG_PER_HA_TO_T_PER_HA
        for y in i.index:
            rows.append({"crop": ind, "year": y, "india_t_ha": i[y], "world_avg_t_ha": w.get(y, np.nan)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
