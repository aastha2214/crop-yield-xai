"""Loading, cleaning and splitting the GLOBAL crop-yield dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def load_raw() -> pd.DataFrame:
    """Read the global CSV exactly as shipped (including the junk index column)."""
    return pd.read_csv(C.GLOBAL_CSV)


def iqr_outlier_report(s: pd.Series, k: float = 1.5) -> dict:
    """Count values outside [Q1 - k*IQR, Q3 + k*IQR] and return the fences."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    n_out = int(((s < lo) | (s > hi)).sum())
    return {"q1": q1, "q3": q3, "lower": lo, "upper": hi,
            "n_outliers": n_out, "pct": 100 * n_out / len(s)}


def clean(verbose: bool = True) -> pd.DataFrame:
    """Return the cleaned global dataframe and (optionally) print a cleaning log.

    Decisions
    ---------
    * Drop the unnamed index column (it carries no information).
    * Strip whitespace on text columns (defensive; keeps category labels exact).
    * Missing values: none expected; if any appear we drop the rows (tiny share).
    * Exact duplicate rows are dropped: they inflate the sample and, worse, can
      appear in both train and test, which would flatter the scores.
    * Target outliers are REPORTED but NOT removed. IQR flags high-yield crops
      (potatoes, cassava, sweet potatoes: 20-40 t/ha) as "outliers" relative to
      cereals, but these are real, legitimate values. Deleting them would make
      the model unable to predict root crops at all.
    """
    df = load_raw()
    n0 = len(df)
    df = df.drop(columns=[c for c in [C.G_INDEX] if c in df.columns])
    for c in C.G_CAT:
        df[c] = df[c].str.strip()
    n_missing = int(df.isna().sum().sum())
    df = df.dropna()
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if verbose:
        out = iqr_outlier_report(df[C.G_TARGET])
        print(f"rows raw={n0}  missing cells={n_missing}  duplicates dropped={n_dup}  final={len(df)}")
        print(f"target IQR outliers: {out['n_outliers']} ({out['pct']:.1f}%) "
              f"above {out['upper']:.0f} hg/ha -> KEPT (legitimate root-crop yields)")
    return df


def random_split(df: pd.DataFrame, test_size: float = 0.2):
    """(a) Random 80/20 split. Optimistic: neighbouring years of the same
    country-crop pair end up on both sides (leakage)."""
    train, test = train_test_split(df, test_size=test_size, random_state=C.RANDOM_STATE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def time_split(df: pd.DataFrame, year_col: str = C.G_YEAR, n_test_years: int = C.G_TEST_YEARS):
    """(b) Time-based split: train on early years, test on the most recent
    `n_test_years` years. Mimics real forecasting: the future is unseen."""
    cutoff = df[year_col].max() - n_test_years
    train = df[df[year_col] <= cutoff].reset_index(drop=True)
    test = df[df[year_col] > cutoff].reset_index(drop=True)
    return train, test


def xy(df: pd.DataFrame, features=C.G_FEATURES, target=C.G_TARGET):
    """Split a dataframe into feature matrix X and target vector y."""
    return df[features], df[target]
