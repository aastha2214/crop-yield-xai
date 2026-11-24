"""End-to-end tests of the Streamlit app (headless) in English and Hindi."""
import re

import pytest
from streamlit.testing.v1 import AppTest

from src import config as C

APP = str(C.ROOT / "app.py")

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# Names, acronyms and units that are deliberately shown as in the data / by convention.
ALLOWED = set("SHAP KVK PMFBY Soil Health Card CV MAE RMSE tonnes year Kharif Rabi XGBoost India Rice paddy "
              "Punjab Language".split())


@pytest.fixture(scope="module")
def app_en():
    return AppTest.from_file(APP, default_timeout=300).run()


@pytest.fixture(scope="module")
def app_hi():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.radio(key="lang").set_value("हिन्दी").run()
    for key in ("g_drought", "i_drought"):          # make the expert-system rules fire in both tabs
        next(b for b in at.button if b.key == key).click()
        at.run()
    at.radio(key="sc_mode__hi").set_value("india").run()
    return at


def visible_strings(at):
    """Every visible label/heading/caption/option string in the app."""
    out = []
    for kind in ["title", "header", "subheader", "markdown", "caption", "info", "warning", "success"]:
        out += [e.value for e in getattr(at, kind)]
    out += [m.label for m in at.metric]
    for kind in ["selectbox", "number_input", "slider", "select_slider", "radio", "button"]:
        out += [e.label for e in getattr(at, kind)]
    out += [t.label for t in at.tabs] + [x.label for x in at.expander]
    for r in at.radio:
        out += list(r.options)
    return out


def test_english_app_runs(app_en):
    assert not app_en.exception
    assert [t.label for t in app_en.tabs] == ["Predict & Explain", "India Insights", "Climate Scenarios", "Model Evaluation"]
    assert "Predict · Explain · Advise" in app_en.caption[0].value


def test_hindi_app_runs_and_translates_tabs(app_hi):
    assert not app_hi.exception
    assert not any(t.label.isascii() for t in app_hi.tabs)
    assert len([b for b in app_hi.button if DEVANAGARI.search(b.label)]) == 2


def test_no_english_leaks_in_hindi_mode(app_hi):
    leaks = []
    for s in visible_strings(app_hi):
        if s == "English":                                   # the language switch itself
            continue
        words = [w for w in re.findall(r"[A-Za-z]{3,}", s) if w not in ALLOWED]
        if not DEVANAGARI.search(s) and re.search(r"[A-Za-z]{3,}", s) or (DEVANAGARI.search(s) and words):
            leaks.append(s[:80])
    assert not leaks, leaks


def test_language_switch_keeps_choices(app_hi):
    assert app_hi.radio(key="sc_mode__hi").value == "india"          # radio selection survives translation
    assert app_hi.select_slider(key="i_outlook__hi").value == "high"  # drought click survives too


def test_recommender_area_default_follows_state_and_season(app_en):
    """Area box defaults to the state+season median (refreshing on change); typical-area checkbox is on."""
    from src import india
    df = india.clean(verbose=False)

    def median(state, season):
        return float(df[(df.State == state) & (df.Season == season)].Area.median())

    at = app_en
    assert at.checkbox(key="rec_use_typical__en").value is True
    box = at.number_input(key="rec_area_Punjab_Kharif")
    assert box.value == pytest.approx(median("Punjab", "Kharif")) and box.disabled     # optional, off by default
    at.selectbox(key="rec_state").set_value("Maharashtra").run()
    at.selectbox(key="rec_season").set_value("Rabi").run()
    assert at.number_input(key="rec_area_Maharashtra_Rabi").value == pytest.approx(median("Maharashtra", "Rabi"))
    assert any("Typical area for Maharashtra, Rabi" in c.value for c in at.caption)


def test_unchecking_typical_area_enables_the_override(app_en):
    at = app_en
    at.checkbox(key="rec_use_typical__en").uncheck().run()
    assert not at.exception
    assert not at.number_input(key="rec_area_Maharashtra_Rabi").disabled
    at.checkbox(key="rec_use_typical__en").check().run()
    assert at.number_input(key="rec_area_Maharashtra_Rabi").disabled


def test_predict_tab_metrics_share_one_unit(app_en):
    """Predicted yield and Historical average are both shown in t/ha, with the hg/ha value as a caption."""
    import re
    pred, hist = app_en.metric[0], app_en.metric[1]            # the first two metrics are the Predict & Explain tab's
    assert pred.label == "Predicted yield" and hist.label.startswith("Historical average")
    assert re.fullmatch(r"\d+\.\d{2} t/ha", pred.value) and re.fullmatch(r"\d+\.\d{2} t/ha", hist.value)
    caps = [c.value for c in app_en.caption]
    pred_hg = round(float(pred.value.split()[0]) * 1e4, -2)           # t/ha -> hg/ha (to the nearest 100)
    assert any(re.fullmatch(r"= [\d,]+ hg/ha", c) for c in caps)
    assert any(abs(int(c[2:].split()[0].replace(",", "")) - pred_hg) <= 100 for c in caps if c.startswith("= "))
    assert not any(x.label.startswith("= ") for x in app_en.metric)   # the old separate tonnes metric is gone
