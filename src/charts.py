"""Language-aware charts for the app (English / Hindi).

The notebooks save fixed English PNGs; the app draws the same charts live through these
functions so titles and axis labels can follow the language toggle. Crop, state and season
names stay as in the data. Hindi uses a Devanagari font via i18n.rc(); if none is installed,
callers pass lang='en' (see i18n.plot_lang) and the labels stay in English.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C, i18n

MAJOR_CROPS = ["Rice", "Wheat", "Maize", "Sugarcane", "Potato", "Groundnut"]
HEAT_CROPS = ["Rice", "Wheat", "Maize", "Sugarcane", "Potato", "Groundnut", "Bajra", "Cotton(lint)"]


def _t(key, lang, **kw):
    """Shorthand for the translation lookup."""
    return i18n.tr(key, lang, **kw)


def model_label(name: str, lang: str) -> str:
    """'XGBoost (tuned)' -> 'XGBoost (ट्यून किया हुआ)' in Hindi; model names themselves stay English."""
    return name.replace("(tuned)", f"({_t('model.tuned', lang)})")


def top_states(df: pd.DataFrame, lang: str, figsize=(7, 4.2)):
    """Top 10 states by total production."""
    tp = df.groupby(C.I_STATE)[C.I_PROD].sum().sort_values(ascending=False).head(10)
    with plt.rc_context(i18n.rc(lang)):
        fig, ax = plt.subplots(figsize=figsize)
        (tp[::-1] / 1e6).plot.barh(color="#e9c46a", ax=ax)
        ax.set(xlabel=_t("ch_top_states_x", lang), ylabel=_t("state", lang), title=_t("ch_top_states", lang))
        fig.tight_layout()
    plt.close(fig)
    return fig


def yield_trends(df: pd.DataFrame, lang: str, figsize=(10, 5.2)):
    """Median yield per year for the major crops (2 x 3 small multiples)."""
    tr = df[df[C.I_CROP].isin(MAJOR_CROPS)].groupby([C.I_YEAR, C.I_CROP])[C.I_TARGET].median().unstack()
    with plt.rc_context(i18n.rc(lang)):
        fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True)
        for ax, crop in zip(axes.ravel(), MAJOR_CROPS):
            ax.plot(tr.index, tr[crop], marker="o", ms=3)
            ax.set_title(crop, fontsize=10)
        for ax in axes[:, 0]:
            ax.set_ylabel(_t("ch_trends_y", lang), fontsize=8)
        for ax in axes[1]:
            ax.set_xlabel(_t("year", lang))
        fig.suptitle(_t("ch_trends", lang))
        fig.tight_layout()
    plt.close(fig)
    return fig


def state_crop_heatmap(df: pd.DataFrame, lang: str, figsize=(8, 5.8)):
    """Median yield for the 15 best-covered states x 8 crops."""
    states = df[C.I_STATE].value_counts().head(15).index
    hm = df[df[C.I_CROP].isin(HEAT_CROPS) & df[C.I_STATE].isin(states)].pivot_table(
        index=C.I_STATE, columns=C.I_CROP, values=C.I_TARGET, aggfunc="median")
    with plt.rc_context(i18n.rc(lang)):
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(hm, annot=True, fmt=".1f", cmap="YlGn", ax=ax, cbar_kws={"label": _t("ch_heat_cbar", lang)})
        ax.set(title=_t("ch_heat", lang), xlabel=_t("crop", lang), ylabel=_t("state", lang))
        fig.tight_layout()
    plt.close(fig)
    return fig


def india_vs_world(cmp: pd.DataFrame, lang: str, figsize=(12, 5.2)):
    """India's yield vs the world average for the crops both datasets share."""
    crops = list(cmp.crop.unique())
    with plt.rc_context(i18n.rc(lang)):
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        for ax, crop in zip(axes.ravel(), crops):
            g = cmp[cmp.crop == crop]
            ax.plot(g.year, g.india_t_ha, marker="o", ms=3, label=_t("ch_india", lang))
            ax.plot(g.year, g.world_avg_t_ha, marker="s", ms=3, label=_t("ch_world", lang))
            ax.set_title(crop, fontsize=10)
        axes.ravel()[0].legend(fontsize=7)
        for ax in axes.ravel()[len(crops):]:
            ax.axis("off")
        fig.suptitle(_t("ch_vs_world", lang))
        fig.tight_layout()
    plt.close(fig)
    return fig


def comparison_bars(comp: pd.DataFrame, title_key: str, lang: str, figsize=(10, 4.6)):
    """Grouped bars of test R² for the random vs time-based split."""
    piv = comp.pivot(index="model", columns="split", values="test_R2")
    piv = piv.reindex(comp.model.drop_duplicates())          # keep the table's model order
    with plt.rc_context(i18n.rc(lang)):
        fig, ax = plt.subplots(figsize=figsize)
        x = np.arange(len(piv))
        for i, split in enumerate(["random", "time"]):
            ax.bar(x + (i - 0.5) * 0.38, piv[split], 0.38, label=_t(f"split.{split}", lang))
        ax.set_xticks(x, [model_label(m, lang) for m in piv.index], rotation=30, ha="right")
        ax.set_ylim(min(0, piv.min().min()) - 0.02, 1)
        ax.set(ylabel=_t("chart_cmp_y", lang), xlabel=_t("chart_cmp_x", lang), title=_t(title_key, lang))
        ax.legend(title=_t("chart_split", lang), loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
    plt.close(fig)
    return fig
