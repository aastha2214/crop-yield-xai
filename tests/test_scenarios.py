"""Tests for the scenario / recommender logic and the Hindi translations (use the saved models)."""
import joblib
import pytest

from src import advisor as A, config as C, data, i18n, india, scenarios as S


@pytest.fixture(scope="module")
def gstuff():
    return joblib.load(C.GLOBAL_MODEL_PATH), data.clean(verbose=False), joblib.load(C.GLOBAL_META_PATH)


@pytest.fixture(scope="module")
def istuff():
    return joblib.load(C.INDIA_MODEL_PATH), india.clean(verbose=False)


def test_zero_change_is_zero_percent(gstuff):
    model, df, meta = gstuff
    base = S.global_baseline(df, "India", "Wheat", meta["year_range"][1])
    _, grid = S.climate_grid(model, base, C.G_FEATURES, C.G_RAIN, C.G_TEMP)
    assert grid.loc[0, 0.0] == pytest.approx(0.0)
    assert grid.shape == (len(S.RAIN_STEPS), len(S.TEMP_STEPS))


def test_rainfall_never_negative():
    row = S.apply_change({"r": 10.0, "t": 20.0}, "r", "t", 2.0, -30)
    assert row["r"] == pytest.approx(7.0) and row["t"] == 22.0


def test_india_grid_has_no_temperature(istuff):
    model, df = istuff
    base = S.india_baseline(df, "Punjab", "Rice", "Kharif", S.last_full_year(df))
    b, grid = S.climate_grid(model, base, C.I_FEATURES, C.I_RAIN, None)
    assert grid.shape == (len(S.RAIN_STEPS), 1) and b > 0


def test_recommender_sorted_and_valid(istuff):
    model, df = istuff
    rec = S.recommend(model, df, "Punjab", "Kharif", 700.0, 50000.0, 2019)
    assert len(rec) >= 5 and rec.predicted.is_monotonic_decreasing
    sub = df[(df.State == "Punjab") & (df.Season == "Kharif")]
    assert set(rec.crop) <= set(sub.Crop)          # only crops actually grown there


def test_every_rule_has_hindi():
    for r in A.RULES:
        assert any("ऀ" <= ch <= "ॿ" for ch in r.text_hi), r.name   # Devanagari present


def test_advice_from_trace_languages():
    _, _, trace = A.forward_chain({"rainfall": "low", "temp": "high"})
    en, hi = A.advice_from_trace(trace), A.advice_from_trace(trace, "hi")
    assert len(en) == len(hi) >= 1 and en != hi


def test_every_rule_fact_is_translated():
    """All fact names/values used by rules (except crop/season names) have a Hindi entry."""
    for r in A.RULES:
        for k, v in list(r.conditions.items()) + list(r.conclusions.items()):
            assert f"factk.{k}" in i18n.T, k
            if k not in ("season", "crop"):
                assert f"factv.{v}" in i18n.T, (k, v)


def test_every_ui_string_has_both_languages():
    """The single dictionary is complete: every non-rule key has English and Hindi text."""
    for key, entry in i18n.T.items():
        if key.startswith("rule."):
            assert entry.get("hi")
        else:
            assert entry.get("en") and entry.get("hi"), key


def test_translation_formatting():
    assert "{" not in i18n.tr("rec_chart_title", "hi", state="Punjab", season="Kharif", n=11)
    assert i18n.tr("sc_rain_line", "hi", r=-30) == "वर्षा -30%"


def test_default_scenario_crop_behaves_intuitively(gstuff):
    """The app's default scenario (India, Cassava): less rain and +3 °C both lower the predicted yield."""
    model, df, meta = gstuff
    base = S.global_baseline(df, "India", "Cassava", meta["year_range"][1])
    _, grid = S.climate_grid(model, base, C.G_FEATURES, C.G_RAIN, C.G_TEMP)
    assert (grid.loc[-30] < 0).all()          # 30 % less rain: below today at every warming level
    assert grid.loc[0, 3.0] < 0               # +3 °C at unchanged rain: below today
    assert grid.loc[-30, 3.0] < grid.loc[0, 0.0]   # dry AND hot is worse than today
