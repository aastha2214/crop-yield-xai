"""Tests for the scenario / recommender logic and the Hindi translations (use the saved models)."""
import joblib
import pandas as pd
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


def test_default_scenarios_match_the_scan():
    """App defaults (Tajikistan Maize; Punjab Groundnut Kharif) are the best rows of the saved full scans."""
    g = pd.read_csv(C.RESULTS_DIR / "scenario_scan_global.csv")
    i = pd.read_csv(C.RESULTS_DIR / "scenario_scan_india.csv")
    assert len(g) > 500 and len(i) > 900                       # every country-crop and state-crop-season was scanned
    best_g, best_i = S.qualifying(g).iloc[0], S.qualifying(i).iloc[0]
    assert (best_g.country_or_state, best_g.crop) == ("Tajikistan", "Maize")
    assert (best_i.country_or_state, best_i.crop, best_i.season) == ("Punjab", "Groundnut", "Kharif")


def test_default_scenarios_fall_at_least_5_percent(gstuff, istuff):
    model, df, meta = gstuff
    base = S.global_baseline(df, "Tajikistan", "Maize", meta["year_range"][1])
    _, grid = S.climate_grid(model, base, C.G_FEATURES, C.G_RAIN, C.G_TEMP)
    m = S.behaviour_metrics(grid)
    assert m["dry_drop"] >= 5 and m["hot_drop"] >= 5           # less rain AND +3 °C both cut the yield by >= 5 %
    imodel, idf = istuff
    ib = S.india_baseline(idf, "Punjab", "Groundnut", "Kharif", S.last_full_year(idf))
    _, igrid = S.climate_grid(imodel, ib, C.I_FEATURES, C.I_RAIN, None)
    assert S.behaviour_metrics(igrid)["dry_drop"] >= 5          # India model has no temperature input


def test_behaviour_metrics_on_synthetic_grids():
    rains, temps = S.RAIN_STEPS, S.TEMP_STEPS
    ideal = pd.DataFrame([[ (r / 30) * 10 - t * 3 for t in temps] for r in rains], index=rains, columns=temps)
    m = S.behaviour_metrics(ideal)                              # straight, intuitive surface
    assert m["viol_rain"] == 0 and m["viol_heat"] == 0 and m["rough"] == pytest.approx(0, abs=1e-9)
    assert m["dry_drop"] == pytest.approx(10) and m["hot_drop"] == pytest.approx(9)
    wrong = ideal.copy(); wrong.loc[:, :] = -ideal.values       # more rain/heat gives MORE yield: counter-intuitive
    assert S.behaviour_metrics(wrong)["viol_rain"] > 0 and S.behaviour_metrics(wrong)["viol_heat"] > 0


def test_recommender_uses_each_crops_own_area(istuff):
    model, df = istuff
    sub = df[(df.State == "Punjab") & (df.Season == "Kharif")]
    rec = S.recommend(model, df, "Punjab", "Kharif", float(sub.Annual_Rainfall.median()), None, 2019)
    assert rec.area_used.nunique() > 5                          # areas differ by crop
    for crop, area in zip(rec.crop, rec.area_used):
        assert area == pytest.approx(float(sub[sub.Crop == crop].Area.median()))
    rice = rec[rec.crop == "Rice"].iloc[0]
    assert abs(rice.vs_usual_pct) < 15                          # Punjab Kharif rice close to its usual yield
    shared = S.recommend(model, df, "Punjab", "Kharif", float(sub.Annual_Rainfall.median()), 8000.0, 2019)
    assert (shared.area_used == 8000.0).all()                   # the optional override still works
