"""Tests for the scenario / recommender logic and the Hindi translations (use the saved models)."""
import joblib
import pytest

from src import advisor as A, config as C, data, india, scenarios as S


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
