"""Streamlit demo: Smart-agriculture yield prediction with explanations and advice.

Run:  streamlit run app.py
"""
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import advisor, config as C, data, india
from src.explain import Explainer

st.set_page_config(page_title="Crop Yield XAI", page_icon="🌾", layout="wide")


# ----------------------------------------------------------------- cached loading
@st.cache_resource(show_spinner="Loading global model...")
def load_global():
    """Global model, SHAP explainer, cleaned data and discretisation quantiles."""
    meta = joblib.load(C.GLOBAL_META_PATH)
    model = joblib.load(C.GLOBAL_MODEL_PATH)
    df = data.clean(verbose=False)
    q = {c: tuple(v) for c, v in meta["quantiles"].items()}
    ex = Explainer(model, C.G_CAT, C.G_NUM, C.G_LABELS, C.G_UNITS, q, df.sample(500, random_state=C.RANDOM_STATE))
    return ex, df, meta


@st.cache_resource(show_spinner="Loading India model...")
def load_india():
    """India model, explainer, cleaned data and per-hectare quantiles for the rules."""
    meta = joblib.load(C.INDIA_META_PATH)
    model = joblib.load(C.INDIA_MODEL_PATH)
    df = india.clean(verbose=False)
    q = {c: tuple(v) for c, v in meta["quantiles"].items()}
    ex = Explainer(model, C.I_CAT, C.I_NUM, C.I_LABELS, C.I_UNITS, q, df.sample(500, random_state=C.RANDOM_STATE))
    per_ha = {  # fertilizer/pesticide are totals; rules judge intensity per hectare
        "fertilizer": tuple((df[C.I_FERT] / df[C.I_AREA]).quantile([0.33, 0.67])),
        "pesticides": tuple((df[C.I_PEST] / df[C.I_AREA]).quantile([0.33, 0.67])),
        "rainfall": tuple(df[C.I_RAIN].quantile([0.33, 0.67]))}
    return ex, df, meta, per_ha


@st.cache_data
def load_results():
    """Saved comparison tables."""
    return (pd.read_csv(C.GLOBAL_COMPARISON_CSV), pd.read_csv(C.INDIA_COMPARISON_CSV))


def show_advice(facts: dict):
    """Run forward chaining and display advice + the reasoning trace."""
    all_facts, advice, trace = advisor.forward_chain(facts)
    st.subheader("Expert-system advice")
    st.caption("Facts: " + ", ".join(f"{k} = {v}" for k, v in facts.items()))
    if advice:
        for a in advice:
            st.success(a)
    else:
        st.info("No rule fired: conditions look unremarkable, so no special advice.")
    with st.expander("Reasoning trace (rules fired, in order)"):
        if not trace:
            st.write("No rules fired.")
        for i, t in enumerate(trace, 1):
            cond = " AND ".join(f"{k} = {v}" for k, v in t["because"].items())
            der = ", ".join(f"{k} = {v}" for k, v in t["derived"].items())
            st.markdown(f"**{i}. {t['rule']}**: IF {cond} → THEN {der}  \n_{t['text']}_")


def show_explanation(ex: Explainer, row: pd.DataFrame, unit: str, title: str):
    """Waterfall plot and the plain-English sentence."""
    st.subheader("Why this prediction? (SHAP)")
    fig = ex.waterfall(row, title=title)
    st.pyplot(fig, clear_figure=True)
    plt.close("all")
    st.info(ex.sentence(row, unit=unit))
    st.caption("SHAP shows what the model associates with the prediction, not proven causes.")


st.title("🌾 Crop Yield Prediction with Explainable AI")
st.caption("Aastha Kumari · Thapar Institute of Engineering and Technology")
tab1, tab2, tab3 = st.tabs(["Predict & Explain", "India Insights", "Model Evaluation"])

# =================================================================== TAB 1
with tab1:
    ex, gdf, gmeta = load_global()
    st.markdown(f"Model: **{gmeta['best_name']}** (best on the time-based split).")
    c1, c2, c3 = st.columns(3)
    country = c1.selectbox("Country", gmeta["countries"], index=gmeta["countries"].index("India"))
    crop = c2.selectbox("Crop", gmeta["crops"], index=gmeta["crops"].index("Rice, paddy"))
    year = c3.slider("Year", gmeta["year_range"][0], 2030, int(gmeta["medians"][C.G_YEAR]))
    if year > gmeta["year_range"][1]:
        st.warning("Years beyond the training data extrapolate: tree models hold the last-seen trend flat.")
    m = gmeta["medians"]
    c4, c5, c6 = st.columns(3)
    rain = c4.number_input("Rainfall (mm/year)", 0.0, 4000.0, float(m[C.G_RAIN]), step=50.0)
    temp = c5.number_input("Avg temperature (°C)", -5.0, 40.0, float(m[C.G_TEMP]), step=0.5)
    pest = c6.number_input("Pesticides (tonnes)", 0.0, 400000.0, float(m[C.G_PEST]), step=100.0)

    row = pd.DataFrame([{C.G_COUNTRY: country, C.G_CROP: crop, C.G_YEAR: year,
                         C.G_RAIN: rain, C.G_TEMP: temp, C.G_PEST: pest}])[C.G_FEATURES]
    pred = float(ex.fitted.predict(row)[0])
    hist = gdf[(gdf[C.G_COUNTRY] == country) & (gdf[C.G_CROP] == crop)][C.G_TARGET]
    hist_avg = hist.mean() if len(hist) else gdf[gdf[C.G_CROP] == crop][C.G_TARGET].mean()
    basis = f"{country} · {crop}" if len(hist) else f"{crop} (all countries; no {country} history)"
    k1, k2, k3 = st.columns(3)
    k1.metric("Predicted yield", f"{pred:,.0f} hg/ha")
    k2.metric("= tonnes per hectare", f"{pred * C.HG_PER_HA_TO_T_PER_HA:.2f} t/ha")
    k3.metric(f"Historical average ({basis})", f"{hist_avg:,.0f} hg/ha", f"{(pred / hist_avg - 1) * 100:+.1f}%")

    show_explanation(ex, row, "hg/ha", f"{crop} in {country}, {year}")

    q = gmeta["quantiles"]
    facts = advisor.build_facts({"rainfall": q[C.G_RAIN], "temp": q[C.G_TEMP], "pesticides": q[C.G_PEST]},
                                rainfall=rain, temp=temp, pesticides=pest,
                                predicted=pred, crop_avg=hist_avg, crop=crop)
    show_advice(facts)

# =================================================================== TAB 2
with tab2:
    iex, idf, imeta, per_ha = load_india()
    st.header("India insights")
    figs = [("india_top_states.png", "Top states by production"), ("india_yield_trends.png", "Yield trends, major crops"),
            ("india_state_crop_heatmap.png", "State × crop yield"), ("india_vs_world.png", "India vs world average")]
    cols = st.columns(2)
    for i, (f, cap) in enumerate(figs):
        if (C.FIG_DIR / f).exists():
            cols[i % 2].image(str(C.FIG_DIR / f), caption=cap)
    st.divider()
    st.subheader("Predict yield for an Indian state")
    d1, d2, d3, d4 = st.columns(4)
    state = d1.selectbox("State", imeta["states"], index=imeta["states"].index("Punjab"))
    icrop = d2.selectbox("Crop", imeta["crops"], index=imeta["crops"].index("Rice"))
    season = d3.selectbox("Season", imeta["seasons"], index=imeta["seasons"].index("Kharif"))
    iyear = d4.slider("Year", imeta["year_range"][0], 2030, 2019)
    sub = idf[(idf[C.I_CROP] == icrop) & (idf[C.I_STATE] == state)]
    ref = sub if len(sub) else idf[idf[C.I_CROP] == icrop]
    e1, e2, e3, e4 = st.columns(4)
    area = e1.number_input("Area (ha)", 1.0, 5e7, float(ref[C.I_AREA].median()), step=1000.0)
    irain = e2.number_input("Annual rainfall (mm)", 0.0, 7000.0, float(ref[C.I_RAIN].median()), step=50.0)
    fert = e3.number_input("Fertilizer (kg, total)", 0.0, 5e9, float(ref[C.I_FERT].median()), step=1e5, format="%.0f")
    ipest = e4.number_input("Pesticide (kg, total)", 0.0, 2e7, float(ref[C.I_PEST].median()), step=1e3, format="%.0f")
    outlook = st.select_slider("Temperature outlook (advisory only - the India model has no temperature input)",
                               ["low", "normal", "high"], value="normal")

    irow = pd.DataFrame([{C.I_CROP: icrop, C.I_SEASON: season, C.I_STATE: state, C.I_YEAR: iyear,
                          C.I_AREA: area, C.I_RAIN: irain, C.I_FERT: fert, C.I_PEST: ipest}])[C.I_FEATURES]
    ipred = float(iex.fitted.predict(irow)[0])
    iavg = ref[C.I_TARGET].mean()
    m1, m2 = st.columns(2)
    m1.metric("Predicted yield", f"{ipred:.2f} t/ha")
    m2.metric(f"Historical average ({'state · crop' if len(sub) else 'crop, all states'})", f"{iavg:.2f} t/ha",
              f"{(ipred / iavg - 1) * 100:+.1f}%")
    show_explanation(iex, irow, "t/ha", f"{icrop}, {state}, {season}, {iyear}")
    ifacts = advisor.build_facts(per_ha, rainfall=irain, pesticides=ipest / area, fertilizer=fert / area,
                                 predicted=ipred, crop_avg=iavg, crop=icrop, season=season)
    ifacts["temp"] = outlook
    show_advice(ifacts)

# =================================================================== TAB 3
with tab3:
    gcomp, icomp = load_results()
    cols = ["model", "split", "cv_R2_mean", "test_R2", "test_MAE", "test_RMSE", "train_time_s"]
    st.header("Model evaluation")
    for title, comp, unit in [("Global dataset (yield in hg/ha)", gcomp, "hg/ha"), ("India dataset (yield in t/ha)", icomp, "t/ha")]:
        st.subheader(title)
        a, b = st.columns(2)
        a.markdown("**Random 80/20 split**"); a.dataframe(comp[comp.split == "random"][cols].round(3), hide_index=True)
        b.markdown("**Time-based split (latest 4 years held out)**"); b.dataframe(comp[comp.split == "time"][cols].round(3), hide_index=True)
    st.subheader("Why does the time-based split score lower?")
    st.write("In a random split, the same country/state-crop series appears on both sides: for most test rows the model "
             "has already seen the previous and next year of that series, so it can nearly look the answer up (leakage). "
             "In the time-based split the test years are strictly in the future, so the model must forecast, "
             "which is what a real user needs. The time-based score is the honest one; the random-split score is an optimistic upper bound.")
    for f in ["model_comparison.png", "india_model_comparison.png"]:
        if (C.FIG_DIR / f).exists():
            st.image(str(C.FIG_DIR / f))
