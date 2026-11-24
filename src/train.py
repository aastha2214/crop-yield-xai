"""Train + compare models on the GLOBAL dataset (and reusable for India).

Run end-to-end with:  python -m src.train
"""
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from . import config as C
from . import data, evaluate, features

RS = C.RANDOM_STATE
# MLP hitting max_iter on some CV folds is expected and harmless here (early stopping
# is on); silence the repeated warning so the training log stays readable.
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def get_models() -> dict:
    """The five model families required by the brief (fresh, unfitted objects)."""
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree": DecisionTreeRegressor(random_state=RS),
        "RandomForest": RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=RS),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.1, max_depth=6,
                                n_jobs=-1, random_state=RS),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), early_stopping=True,
                            max_iter=300, random_state=RS),
    }


def param_grids() -> dict:
    """Search spaces for RandomizedSearchCV (keys use the Pipeline 'model__' prefix)."""
    return {
        "RandomForest": {"model__n_estimators": randint(100, 400),
                         "model__max_depth": [None, 10, 20, 30],
                         "model__min_samples_leaf": randint(1, 6),
                         "model__max_features": ["sqrt", 0.5, 1.0]},
        "XGBoost": {"model__n_estimators": randint(200, 700),
                    "model__max_depth": randint(3, 11),
                    "model__learning_rate": loguniform(0.02, 0.3),
                    "model__subsample": uniform(0.6, 0.4),
                    "model__colsample_bytree": uniform(0.5, 0.5)},
        "DecisionTree": {"model__max_depth": randint(4, 30),
                         "model__min_samples_leaf": randint(1, 20)},
        "MLP": {"model__hidden_layer_sizes": [(64,), (128, 64), (256, 128), (128, 64, 32)],
                "model__alpha": loguniform(1e-5, 1e-1),
                "model__learning_rate_init": loguniform(1e-4, 1e-2)},
    }


def folds_for(split: str, X, years, n=5):
    """5-fold CV: shuffled KFold for the random split, forward-chaining for time."""
    if split == "random":
        return list(KFold(n, shuffle=True, random_state=RS).split(X))
    return evaluate.year_forward_splits(years, n)


def fit_eval(name, pipe, Xtr, ytr, Xte, yte, folds) -> tuple:
    """CV on train, fit on full train, score on test. Returns (row, fitted)."""
    cv_mean, cv_std = evaluate.cv_r2(pipe, Xtr, ytr, folds)
    t0 = time.time()
    fitted = clone(pipe).fit(Xtr, ytr)
    train_time = time.time() - t0
    m = evaluate.regression_metrics(yte, fitted.predict(Xte))
    row = {"model": name, "cv_R2_mean": cv_mean, "cv_R2_std": cv_std,
           "test_R2": m["R2"], "test_MAE": m["MAE"], "test_RMSE": m["RMSE"],
           "train_time_s": train_time}
    return row, fitted


def log_target_experiment(Xtr, ytr, Xte, yte, cat, num, models) -> dict:
    """Decide per model whether a log1p target helps (judged on 3-fold
    forward-chaining CV R² of the TRAIN set - the test set is never used)."""
    folds = evaluate.year_forward_splits(Xtr[C_YEAR_OF[tuple(cat)]], 3)
    rows, choice = [], {}
    for name, mdl in models.items():
        sc = {}
        for flag in (False, True):
            pipe = features.build_pipeline(clone(mdl), cat, num, flag)
            sc[flag] = evaluate.cv_r2(pipe, Xtr, ytr, folds)[0]
            rows.append({"model": name, "log_target": flag, "cv_R2": sc[flag]})
        choice[name] = bool(sc[True] > sc[False] + 0.005)   # require a real gain
    return choice, pd.DataFrame(rows)


# which column holds the year, keyed by the categorical feature list
C_YEAR_OF = {tuple(C.G_CAT): C.G_YEAR, tuple(C.I_CAT): C.I_YEAR}


def tune(name, base_pipe, Xtr, ytr, folds, n_iter=20):
    """RandomizedSearchCV with the split-appropriate CV folds."""
    rs = RandomizedSearchCV(base_pipe, {k: v for k, v in param_grids()[name].items()},
                            n_iter=n_iter, cv=folds, scoring="r2", n_jobs=1,
                            random_state=RS, refit=True)
    rs.fit(Xtr, ytr)
    return rs.best_estimator_, rs.best_params_


def run_comparison(train, test, cat, num, target, year_col, split, log_choice, n_iter=20, only=None):
    """Evaluate all models (+ tuned top-2) for one split.

    Returns (results DataFrame, dict of fitted pipelines, tuned-params dict).
    """
    feats = list(cat) + list(num)
    Xtr, ytr, Xte, yte = train[feats], train[target], test[feats], test[target]
    folds = folds_for(split, Xtr, train[year_col])
    rows, fitted, models = [], {}, get_models()
    if only:                      # e.g. India re-uses just the top-2 model types
        models = {k: v for k, v in models.items() if k in only}
    for name, mdl in models.items():
        pipe = features.build_pipeline(mdl, cat, num, log_choice[name])
        row, fit = fit_eval(name, pipe, Xtr, ytr, Xte, yte, folds)
        rows.append({**row, "split": split, "log_target": log_choice[name], "tuned": False})
        fitted[name] = fit
        print(f"  [{split}] {name:16s} cvR2={row['cv_R2_mean']:.3f} testR2={row['test_R2']:.3f}")
    res = pd.DataFrame(rows)
    # top-2 by CV R² (no peeking at the test set), excluding the baseline
    top2 = res[res.model != "LinearRegression"].nlargest(2, "cv_R2_mean").model.tolist()
    tuned_params = {}
    for name in top2:
        base = features.build_pipeline(models[name], cat, num, log_choice[name])
        # when the target is log-transformed we search the inner pipeline on log1p(y)
        grid_pipe = base.regressor if log_choice[name] else base
        t0 = time.time()
        if log_choice[name]:
            best_inner, params = tune(name, grid_pipe, Xtr, np.log1p(ytr), folds, n_iter)
            best = features.build_pipeline(clone(models[name]), cat, num, True)
            best.regressor.set_params(**params)
        else:
            best_inner, params = tune(name, grid_pipe, Xtr, ytr, folds, n_iter)
            best = best_inner
        cv_mean, cv_std = evaluate.cv_r2(best, Xtr, ytr, folds)
        t1 = time.time()
        fit = clone(best).fit(Xtr, ytr)
        tt = time.time() - t1
        m = evaluate.regression_metrics(yte, fit.predict(Xte))
        res.loc[len(res)] = {"model": f"{name} (tuned)", "cv_R2_mean": cv_mean, "cv_R2_std": cv_std,
                             "test_R2": m["R2"], "test_MAE": m["MAE"], "test_RMSE": m["RMSE"],
                             "train_time_s": tt, "split": split,
                             "log_target": log_choice[name], "tuned": True}
        fitted[f"{name} (tuned)"] = fit
        tuned_params[name] = {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}
        print(f"  [{split}] {name} tuned in {time.time()-t0:.0f}s  cvR2={cv_mean:.3f} testR2={m['R2']:.3f}")
    return res, fitted, tuned_params


def main():
    """Full global pipeline: clean -> both splits -> compare -> tune -> save best."""
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = data.clean()
    splits = {"random": data.random_split(df), "time": data.time_split(df)}

    # Log-target decision uses only the time-split TRAIN years.
    tr_t, te_t = splits["time"]
    log_choice, log_tbl = log_target_experiment(tr_t[C.G_FEATURES], tr_t[C.G_TARGET],
                                                te_t[C.G_FEATURES], te_t[C.G_TARGET],
                                                C.G_CAT, C.G_NUM, get_models())
    log_tbl.to_csv(C.RESULTS_DIR / "log_target_experiment.csv", index=False)
    print("log-target choice:", log_choice)

    all_res, all_fit, all_params = [], {}, {}
    for split, (tr, te) in splits.items():
        res, fitted, params = run_comparison(tr, te, C.G_CAT, C.G_NUM, C.G_TARGET,
                                             C.G_YEAR, split, log_choice)
        all_res.append(res); all_fit[split] = fitted; all_params[split] = params
    comp = pd.concat(all_res, ignore_index=True)
    comp.to_csv(C.GLOBAL_COMPARISON_CSV, index=False)
    evaluate.plot_comparison(comp, C.FIG_DIR / "model_comparison.png")

    # ---- best model: highest time-split CV R² (honest, forward-looking criterion)
    t = comp[comp.split == "time"]
    best_name = t.loc[t.cv_R2_mean.idxmax(), "model"]
    best = all_fit["time"][best_name]
    tr, te = splits["time"]
    pred = best.predict(te[C.G_FEATURES])
    evaluate.plot_pred_vs_actual(te[C.G_TARGET], pred, f"{best_name} - time-based test set",
                                 C.FIG_DIR / "best_pred_vs_actual.png")
    joblib.dump(best, C.GLOBAL_MODEL_PATH)

    # ---- metadata used by the explainer / advisor / app
    meta = {"best_name": best_name, "log_target": log_choice, "tuned_params": all_params,
            "countries": sorted(df[C.G_COUNTRY].unique()), "crops": sorted(df[C.G_CROP].unique()),
            "medians": df[C.G_NUM].median().to_dict(),
            "quantiles": {c: df[c].quantile([0.33, 0.67]).tolist() for c in [C.G_RAIN, C.G_TEMP, C.G_PEST]},
            "year_range": [int(df.Year.min()), int(df.Year.max())]}
    joblib.dump(meta, C.GLOBAL_META_PATH)
    # top-2 model *types* (used again for India)
    base_names = [n.replace(" (tuned)", "") for n in t[t.tuned == False].nlargest(3, "cv_R2_mean").model]
    top2 = [n for n in base_names if n != "LinearRegression"][:2]
    (C.RESULTS_DIR / "top_models.json").write_text(json.dumps(
        {"best_global": best_name, "top2_types": top2, "log_choice": log_choice,
         "tuned_params": all_params}, indent=2, default=str))
    print("\nBEST (time split, by CV R²):", best_name)
    print(comp.round(3).to_string())


if __name__ == "__main__":
    main()
