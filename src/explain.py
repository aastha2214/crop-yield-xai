"""SHAP-based explainability for the tree models (global + India).

Key idea: after one-hot encoding, a country becomes ~100 binary columns. SHAP values
are ADDITIVE, so summing the SHAP values of all one-hot columns that belong to the
same original feature gives that feature's total contribution. We use this to show
readable explanations such as "Crop: Potatoes" instead of 100 dummy columns.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import features, i18n


class Explainer:
    """Wraps a fitted tree pipeline and exposes SHAP values per ORIGINAL feature."""

    def __init__(self, fitted, cat_cols, num_cols, labels, units, quantiles, background):
        """
        fitted    : fitted Pipeline (prep -> tree model)
        quantiles : {numeric_col: (q33, q67)} used to call a value low/typical/high
        background: DataFrame sample (raw columns) kept for reference/medians
        """
        self.pipe = features.inner_pipeline(fitted)
        self.fitted = fitted
        self.cat, self.num = list(cat_cols), list(num_cols)
        self.labels, self.units, self.q = labels, units, quantiles
        self.prep, self.model = self.pipe.named_steps["prep"], self.pipe.named_steps["model"]
        self.explainer = shap.TreeExplainer(self.model)
        raw_names = list(self.prep.get_feature_names_out())      # e.g. cat__Area_India
        self.full_names = [self._readable(n) for n in raw_names]
        self.group_of = [self._group(n) for n in raw_names]       # original column
        self.groups = self.cat + self.num
        self.base_value = float(np.ravel(self.explainer.expected_value)[0])

    # ---------------------------------------------------------- name helpers
    def _group(self, name: str) -> str:
        """Map a transformed column name back to its original feature."""
        kind, rest = name.split("__", 1)
        if kind == "num":
            return rest
        return next(c for c in self.cat if rest.startswith(c + "_"))

    def _readable(self, name: str) -> str:
        """'cat__Area_India' -> 'Country = India';  'num__avg_temp' -> 'Temperature'."""
        kind, rest = name.split("__", 1)
        if kind == "num":
            return self.labels[rest]
        col = next(c for c in self.cat if rest.startswith(c + "_"))
        return f"{self.labels[col]} = {rest[len(col) + 1:]}"

    # ------------------------------------------------------------ SHAP maths
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Apply the fitted preprocessor."""
        return self.prep.transform(X)

    def shap_full(self, X: pd.DataFrame):
        """(transformed matrix, SHAP values on every transformed column)."""
        Xt = self.transform(X)
        return Xt, self.explainer.shap_values(Xt)

    def shap_grouped(self, X: pd.DataFrame) -> pd.DataFrame:
        """SHAP per ORIGINAL feature (one-hot columns summed)."""
        _, sv = self.shap_full(X)
        out = pd.DataFrame(0.0, index=range(len(X)), columns=self.groups)
        for j, g in enumerate(self.group_of):
            out[g] += sv[:, j]
        return out

    # ------------------------------------------------------------ plotting
    def beeswarm(self, X, path, max_display=15):
        """Global beeswarm over the transformed (one-hot) features, readable names."""
        Xt, sv = self.shap_full(X)
        plt.figure()
        shap.summary_plot(sv, Xt, feature_names=self.full_names, max_display=max_display, show=False)
        plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()

    def mean_abs_grouped(self, X) -> pd.Series:
        """Mean |SHAP| per original feature, sorted."""
        return self.shap_grouped(X).abs().mean().sort_values(ascending=False)

    def bar_plot(self, X, path):
        """Mean |SHAP| bar plot per original feature."""
        s = self.mean_abs_grouped(X)
        s.index = [self.labels[i] for i in s.index]
        ax = s[::-1].plot.barh(figsize=(7, 4), color="#2a9d8f")
        ax.set(xlabel="mean |SHAP value| (impact on predicted yield)", title="Global feature importance (SHAP)")
        plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()

    def builtin_importance(self) -> pd.Series:
        """Model's own feature_importances_, summed to original features."""
        imp = pd.Series(self.model.feature_importances_).groupby(self.group_of).sum()
        return imp.reindex(self.groups).sort_values(ascending=False)

    def compare_importance(self, X, path) -> pd.DataFrame:
        """Side-by-side normalised SHAP vs built-in importance (saved as a figure)."""
        shap_i = self.mean_abs_grouped(X)
        shap_i = shap_i / shap_i.sum()
        bi = self.builtin_importance(); bi = bi / bi.sum()
        df = pd.DataFrame({"SHAP (mean |value|)": shap_i, "Built-in importance": bi}).sort_values(
            "SHAP (mean |value|)")
        df.index = [self.labels[i] for i in df.index]
        df.plot.barh(figsize=(8, 4.5)); plt.xlabel("share of total importance")
        plt.title("SHAP vs built-in feature importance"); plt.tight_layout()
        plt.savefig(path, dpi=130); plt.close()
        return df.sort_values("SHAP (mean |value|)", ascending=False)

    def dependence(self, X, col, path):
        """SHAP dependence plot for one numeric feature (SHAP vs its value)."""
        grouped = self.shap_grouped(X)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(X[col].values, grouped[col].values, s=6, alpha=0.4)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(xlabel=f"{self.labels[col]} ({self.units.get(col, '')})",
               ylabel="SHAP value (change in predicted yield)", title=f"Dependence: {self.labels[col]}")
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    def _display(self, row: pd.Series, col: str) -> str:
        """Value as shown to humans, e.g. 'India' or '1485 mm/yr'."""
        v = row[col]
        if col in self.cat:
            return str(v)
        if "year" in col.lower():            # years must not get thousands separators
            return str(int(v))
        u = self.units.get(col, "")
        txt = f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}".rstrip("0").rstrip(".")
        return f"{txt} {u}".strip()

    def waterfall(self, x_row: pd.DataFrame, path=None, title=None):
        """Waterfall plot for ONE prediction, with grouped (readable) features.

        Saves to `path` if given; with path=None the open Figure is returned
        (used by Streamlit's st.pyplot)."""
        g = self.shap_grouped(x_row).iloc[0]
        row = x_row.iloc[0]
        disp = np.array([self._display(row, c) for c in self.groups], dtype=object)
        exp = shap.Explanation(values=g.values, base_values=self.base_value, data=disp,
                               feature_names=[self.labels[c] for c in self.groups])
        plt.figure()
        shap.plots.waterfall(exp, show=False, max_display=len(self.groups))
        if title:
            plt.title(title, fontsize=10)
        fig = plt.gcf()
        if path is None:
            return fig
        fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)

    # --------------------------------------------------------- plain English
    def level(self, col, value) -> str:
        """'low' / 'typical' / 'high' relative to the dataset's 33rd/67th percentiles."""
        if col not in self.q:
            return ""
        lo, hi = self.q[col]
        return "low" if value < lo else "high" if value > hi else "typical"

    @staticmethod
    def _num(v: float) -> str:
        """Format a yield amount: whole numbers when large, 2 decimals when small."""
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"

    def sentence(self, x_row: pd.DataFrame, top_k=3, unit="hg/ha", lang="en") -> str:
        """Turn the top-k SHAP contributions into a plain-English (or Hindi, lang='hi') sentence."""
        g = self.shap_grouped(x_row).iloc[0]
        row = x_row.iloc[0]
        pred = float(self.fitted.predict(x_row)[0])
        hi = lang == "hi"
        parts = []
        for col in g.abs().sort_values(ascending=False).index[:top_k]:
            amt = self._num(abs(g[col]))
            up = g[col] > 0
            lvl = self.level(col, row[col]) if col in self.num else ""
            name = self.labels[col]
            if hi:
                name_hi = i18n.LABELS_HI.get(name, name)
                if col in self.cat:
                    subj = f"{name_hi} '{row[col]}'"
                elif lvl:
                    subj = f"{i18n.LEVELS_HI[lvl]} {name_hi} ({self._display(row, col)})"
                else:
                    subj = f"{name_hi} ({self._display(row, col)})"
                parts.append(f"{subj} ने अनुमानित उपज को {amt} {unit} {'बढ़ाया' if up else 'घटाया'}")
                continue
            verb = "increased" if up else "reduced"
            if col in self.cat:
                subj = f"{name} '{row[col]}'"
            elif lvl:
                subj = f"{lvl.capitalize()} {name.lower()} ({self._display(row, col)})"
            else:
                subj = f"{name} ({self._display(row, col)})"
            parts.append(f"{subj} {verb} the predicted yield by {amt} {unit}")
        if hi:
            return (f"मॉडल का अनुमान {self._num(pred)} {unit} है (औसत अनुमान: {self._num(self.base_value)})। "
                    + "; ".join(parts) + "।")
        return (f"The model predicts {self._num(pred)} {unit} (average prediction: {self._num(self.base_value)}). "
                + "; ".join(parts) + ".")

    def contributions(self, x_row: pd.DataFrame) -> pd.Series:
        """Grouped SHAP values for a single row, sorted by magnitude."""
        g = self.shap_grouped(x_row).iloc[0]
        return g.reindex(g.abs().sort_values(ascending=False).index)


def pick_examples(pred: np.ndarray) -> dict:
    """Indices of a high / medium / low predicted-yield example (90th/50th/10th pct)."""
    out = {}
    for name, q in {"high": 0.9, "medium": 0.5, "low": 0.1}.items():
        out[name] = int(np.argmin(np.abs(pred - np.quantile(pred, q))))
    return out
