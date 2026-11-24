"""Metrics, cross-validation schemes and diagnostic plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C


def regression_metrics(y_true, y_pred) -> dict:
    """R², MAE and RMSE in the original target units."""
    return {"R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred)))}


def year_forward_splits(years, n_splits: int = 5):
    """Forward-chaining CV folds by year: train on earlier years, validate on the
    next block. This is the time-aware analogue of 5-fold CV and never lets a
    model see the future."""
    years = np.asarray(years)
    uniq = np.sort(np.unique(years))
    chunks = np.array_split(uniq, n_splits + 1)
    folds = []
    for i in range(n_splits):
        tr = np.where(np.isin(years, np.concatenate(chunks[: i + 1])))[0]
        va = np.where(np.isin(years, chunks[i + 1]))[0]
        folds.append((tr, va))
    return folds


def cv_r2(estimator, X, y, folds) -> tuple:
    """Mean and std of validation R² over explicit folds (clone per fold)."""
    scores = []
    for tr, va in folds:
        m = clone(estimator).fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores)), float(np.std(scores))


def plot_pred_vs_actual(y_true, y_pred, title, path):
    """Predicted-vs-actual scatter and residual plots for one model."""
    resid = y_true - y_pred
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].scatter(y_true, y_pred, s=6, alpha=0.4)
    lim = [0, max(y_true.max(), y_pred.max())]
    ax[0].plot(lim, lim, "r--", label="perfect")
    ax[0].set(xlabel="Actual (hg/ha)", ylabel="Predicted (hg/ha)", title="Predicted vs actual")
    ax[0].legend()
    ax[1].scatter(y_pred, resid, s=6, alpha=0.4)
    ax[1].axhline(0, color="r", ls="--")
    ax[1].set(xlabel="Predicted (hg/ha)", ylabel="Residual", title="Residuals vs predicted")
    ax[2].hist(resid, bins=60)
    ax[2].set(xlabel="Residual (hg/ha)", title="Residual distribution")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_comparison(df: pd.DataFrame, path, title="Model comparison (test R²)"):
    """Grouped bar chart of test R² per model for both splits."""
    piv = df.pivot(index="model", columns="split", values="test_R2")
    ax = piv.plot.bar(figsize=(11, 5), rot=30)
    ax.set(ylabel="Test R²", title=title)
    ax.set_ylim(min(0, piv.min().min()) - 0.02, 1)
    ax.legend(title="split", loc="upper left", bbox_to_anchor=(1.01, 1))   # keep legend off the bars
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
