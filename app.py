"""Streamlit demo: Smart-agriculture yield prediction with explanations and advice.

Run:  streamlit run app.py
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import advisor, config as C, data, i18n, india, scenarios
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
    # rule thresholds (fertilizer / pesticide are per-hectare intensities)
    rule_q = {"rainfall": q[C.I_RAIN], "fertilizer": q[C.I_FERT_HA], "pesticides": q[C.I_PEST_HA]}
    return ex, df, meta, rule_q


@st.cache_data
def load_results():
    """Saved comparison tables."""
    return (pd.read_csv(C.GLOBAL_COMPARISON_CSV), pd.read_csv(C.INDIA_COMPARISON_CSV))


def pick(label, options, key, default=None, container=st):
    """Selectbox whose options can change: resets an invalid stored choice instead of crashing."""
    options = list(options)
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    return container.selectbox(label, options, key=key)


def bar_chart(labels, values, xlabel, title, fmt="{:.2f}", highlight_first=True):
    """Horizontal ranked bar chart (best at the top)."""
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(labels) + 1.2))
    colors = ["#2a9d8f" if (i == 0 and highlight_first) else "#8ecae6" for i in range(len(labels))]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    for y, v in enumerate(values[::-1]):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9)
    ax.set(xlabel=xlabel, title=title)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def show_advice(facts: dict, lang: str = "en"):
    """Run forward chaining and display advice (English or Hindi) + the reasoning trace."""
    _, _, trace = advisor.forward_chain(facts)
    advice = advisor.advice_from_trace(trace, lang)
    st.subheader(i18n.ui("advice_title", lang))
    st.caption(i18n.ui("facts", lang) + ": " + ", ".join(i18n.fact_text(k, v, lang) for k, v in facts.items()))
    if advice:
        for a in advice:
            st.success(a)
    else:
        st.info(i18n.ui("no_rule", lang))
    with st.expander(i18n.ui("trace", lang)):
        if not trace:
            st.write(i18n.ui("no_rule", lang))
        IF, THEN, AND = (i18n.ui(k, lang) for k in ("if", "then", "and"))
        for i, t in enumerate(trace, 1):
            cond = f" {AND} ".join(i18n.fact_text(k, v, lang) for k, v in t["because"].items())
            der = ", ".join(i18n.fact_text(k, v, lang) for k, v in t["derived"].items())
            txt = t["text_hi"] if lang == "hi" else t["text"]
            st.markdown(f"**{i}. {t['rule']}**: {IF} {cond} → {THEN} {der}  \n_{txt}_")


def show_explanation(ex: Explainer, row: pd.DataFrame, unit: str, title: str, lang: str = "en"):
    """Waterfall plot, the E[f(X)] note and the plain-language sentence (English or Hindi)."""
    st.subheader(i18n.ui("why", lang))
    fig = ex.waterfall(row, title=title)
    st.pyplot(fig, clear_figure=True)
    plt.close("all")
    st.caption(i18n.ui("waterfall_note", lang))
    st.info(ex.sentence(row, unit=unit, lang=lang))
    st.caption(i18n.ui("shap_caption", lang))


st.title("🌾 Crop Yield Prediction with Explainable AI")
st.caption("Predict · Explain · Advise: an explainable AI system for smarter farming")
lang = i18n.LANGS[st.radio("Language / भाषा", list(i18n.LANGS), horizontal=True, key="lang")]
tab1, tab2, tab_sc, tab3 = st.tabs(["Predict & Explain", "India Insights", "Climate Scenarios", "Model Evaluation"])

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
    gq = gmeta["quantiles"]
    for k, v in {"g_rain": float(m[C.G_RAIN]), "g_temp": float(m[C.G_TEMP]), "g_pest": float(m[C.G_PEST])}.items():
        st.session_state.setdefault(k, v)          # defaults = dataset medians

    def set_global_drought():
        """Button callback: fill inputs with a low-rainfall / high-temperature scenario."""
        st.session_state["g_rain"] = round(gq[C.G_RAIN][0] * 0.5, -1)   # half of the 'low' cut-off
        st.session_state["g_temp"] = round(gq[C.G_TEMP][1] + 3, 1)      # 3 °C above the 'high' cut-off

    c4, c5, c6 = st.columns(3)
    rain = c4.number_input("Rainfall (mm/year)", 0.0, 4000.0, step=50.0, key="g_rain")
    temp = c5.number_input("Avg temperature (°C)", -5.0, 40.0, step=0.5, key="g_temp")
    pest = c6.number_input("Pesticides (tonnes)", 0.0, 400000.0, step=100.0, key="g_pest")
    st.button("Try example: drought scenario", key="g_drought", on_click=set_global_drought,
              help="Sets low rainfall and high temperature so the expert-system rules fire.")

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

    show_explanation(ex, row, "hg/ha", f"{crop} in {country}, {year}", lang)

    q = gmeta["quantiles"]
    facts = advisor.build_facts({"rainfall": q[C.G_RAIN], "temp": q[C.G_TEMP], "pesticides": q[C.G_PEST]},
                                rainfall=rain, temp=temp, pesticides=pest,
                                predicted=pred, crop_avg=hist_avg, crop=crop)
    show_advice(facts, lang)

# =================================================================== TAB 2
with tab2:
    iex, idf, imeta, rule_q = load_india()
    st.header("India insights")
    figs = [("india_top_states.png", "Top states by production"), ("india_yield_trends.png", "Yield trends, major crops"),
            ("india_state_crop_heatmap.png", "State × crop yield"), ("india_vs_world.png", "India vs world average")]
    cols = st.columns(2)
    for i, (f, cap) in enumerate(figs):
        if (C.FIG_DIR / f).exists():
            cols[i % 2].image(str(C.FIG_DIR / f), caption=cap)
    st.divider()
    st.subheader("Predict yield for an Indian state")
    # defaults live in session_state (not `index=`) because the drought button also sets them
    for k, v in {"i_state": "Punjab", "i_crop": "Rice", "i_season": "Kharif"}.items():
        st.session_state.setdefault(k, v)
    d1, d2, d3, d4 = st.columns(4)
    state = d1.selectbox("State", imeta["states"], key="i_state")
    icrop = d2.selectbox("Crop", imeta["crops"], key="i_crop")
    season = d3.selectbox("Season", imeta["seasons"], key="i_season")
    iyear = d4.slider("Year", imeta["year_range"][0], 2030, 2019)
    sub = idf[(idf[C.I_CROP] == icrop) & (idf[C.I_STATE] == state)]
    ref = sub if len(sub) else idf[idf[C.I_CROP] == icrop]
    rain_key = f"i_rain_{state}_{icrop}"            # new default whenever state/crop changes
    st.session_state.setdefault(rain_key, float(ref[C.I_RAIN].median()))
    st.session_state.setdefault("i_outlook", "normal")

    def set_india_drought():
        """Button callback: Kharif season, low rainfall, high temperature outlook."""
        s_, c_ = st.session_state["i_state"], st.session_state["i_crop"]
        st.session_state["i_season"] = "Kharif"
        st.session_state[f"i_rain_{s_}_{c_}"] = round(rule_q["rainfall"][0] * 0.6, -1)
        st.session_state["i_outlook"] = "high"

    e1, e2, e3, e4 = st.columns(4)
    area = e1.number_input("Area under crop (ha)", 1.0, 5e7, float(ref[C.I_AREA].median()), step=1000.0, help="In the dataset, Area is the total area under that crop in the whole state (not a single farm), so very small values are outside what the model has seen.")
    irain = e2.number_input("Annual rainfall (mm)", 0.0, 7000.0, step=50.0, key=rain_key)
    fert = e3.number_input("Fertilizer (kg/ha)", 0.0, 5000.0, float(ref[C.I_FERT_HA].median()), step=5.0, format="%.2f")
    ipest = e4.number_input("Pesticide (kg/ha)", 0.0, 100.0, float(ref[C.I_PEST_HA].median()), step=0.05, format="%.3f")
    outlook = st.select_slider("Temperature outlook (advisory only - the India model has no temperature input)",
                               ["low", "normal", "high"], key="i_outlook")
    st.button("Try example: drought scenario", key="i_drought", on_click=set_india_drought,
              help="Sets Kharif season, low rainfall and a high temperature outlook so the expert-system rules fire.")

    irow = pd.DataFrame([{C.I_CROP: icrop, C.I_SEASON: season, C.I_STATE: state, C.I_YEAR: iyear,
                          C.I_AREA: area, C.I_RAIN: irain, C.I_FERT_HA: fert, C.I_PEST_HA: ipest}])[C.I_FEATURES]
    ipred = float(iex.fitted.predict(irow)[0])
    iavg = ref[C.I_TARGET].mean()
    m1, m2 = st.columns(2)
    m1.metric("Predicted yield", f"{ipred:.2f} t/ha")
    m2.metric(f"Historical average ({'state · crop' if len(sub) else 'crop, all states'})", f"{iavg:.2f} t/ha",
              f"{(ipred / iavg - 1) * 100:+.1f}%")
    show_explanation(iex, irow, "t/ha", f"{icrop}, {state}, {season}, {iyear}", lang)
    ifacts = advisor.build_facts(rule_q, rainfall=irain, pesticides=ipest, fertilizer=fert,
                                 predicted=ipred, crop_avg=iavg, crop=icrop, season=season)
    ifacts["temp"] = outlook
    show_advice(ifacts, lang)

    # ------------------------------------------------------------ Crop Recommender
    st.divider()
    st.subheader("🌱 Crop Recommender")
    st.caption("Choose a state, season, rainfall and area. The model predicts the yield of every crop grown in that "
               "state and season and ranks the top 5. Fertilizer and pesticide per hectare are set to each crop's "
               "typical level there.")
    last_year = scenarios.last_full_year(idf)
    r1, r2, r3, r4 = st.columns(4)
    rstate = pick("State", imeta["states"], "rec_state", "Punjab", r1)
    rseason = pick("Season", sorted(idf[idf[C.I_STATE] == rstate][C.I_SEASON].unique()), "rec_season", "Kharif", r2)
    rsub = idf[(idf[C.I_STATE] == rstate) & (idf[C.I_SEASON] == rseason)]
    rkey = f"rec_rain_{rstate}_{rseason}"
    st.session_state.setdefault(rkey, float(rsub[C.I_RAIN].median()))
    rrain = r3.number_input("Annual rainfall (mm)", 0.0, 7000.0, step=50.0, key=rkey)
    rarea = r4.number_input("Area under crop (ha)", 1.0, 5e7, float(rsub[C.I_AREA].median()), step=1000.0, key=f"rec_area_{rstate}_{rseason}", help="In the dataset, Area is the total area under that crop in the whole state (not a single farm), so very small values are outside what the model has seen.")
    rank_by = st.radio("Rank crops by", ["Predicted yield (t/ha)", "Predicted yield vs. the crop's usual yield here (%)"],
                       horizontal=True, key="rec_rank")
    rec = scenarios.recommend(iex.fitted, idf, rstate, rseason, rrain, rarea, last_year)
    if rec.empty:
        st.info("Not enough historical records for this state and season.")
    else:
        by_pct = rank_by.startswith("Predicted yield vs")
        ranked = rec.sort_values("vs_usual_pct" if by_pct else "predicted", ascending=False).reset_index(drop=True)
        top5 = ranked.head(5)
        vals = top5.vs_usual_pct if by_pct else top5.predicted
        fig = bar_chart(list(top5.crop), list(vals), "% above/below the usual yield" if by_pct else "predicted yield (t/ha)",
                        f"Top 5 crops: {rstate}, {rseason} ({len(rec)} crops considered)",
                        fmt="{:+.1f}%" if by_pct else "{:.2f}")
        st.pyplot(fig, clear_figure=True); plt.close("all")
        st.dataframe(top5[["crop", "predicted", "hist_avg", "vs_usual_pct", "n_rows"]].rename(columns={
            "crop": "Crop", "predicted": "Predicted (t/ha)", "hist_avg": "Usual yield here (t/ha)",
            "vs_usual_pct": "Vs usual (%)", "n_rows": "Past records"}).style.format(
            {"Predicted (t/ha)": "{:.3f}", "Usual yield here (t/ha)": "{:.3f}", "Vs usual (%)": "{:+.1f}"}), hide_index=True)
        st.warning("Yields in t/ha are not comparable across crops in value or effort (e.g. sugarcane vs pulses), and this ranking "
                   "ignores prices, soil, water availability, labour and crop rotation. It shows what the model predicts, "
                   "not what you should plant.")
        best = top5.iloc[0]
        st.markdown(f"**SHAP explanation for the top crop: {best.crop}**")
        show_explanation(iex, pd.DataFrame([best.row])[C.I_FEATURES], "t/ha", f"{best.crop}, {rstate}, {rseason}", lang)


# =================================================================== CLIMATE SCENARIOS
with tab_sc:
    st.header("Climate Scenarios")
    st.warning("This shows what the model associates with climate in historical data, not a guaranteed forecast. "
               "Real outcomes also depend on irrigation, crop varieties, adaptation, pests and extreme events that the "
               "model cannot see (its climate inputs are annual averages).")
    gex, gdf, gmeta = load_global()
    iex, idf, imeta, _ = load_india()
    mode = st.radio("Model", ["Global model (country)", "India model (state)"], horizontal=True, key="sc_mode")
    use_global = mode.startswith("Global")
    k1, k2, k3 = st.columns(3)
    if use_global:
        country = pick("Country", gmeta["countries"], "sc_country", "India", k1)
        crop_s = pick("Crop", sorted(gdf[gdf[C.G_COUNTRY] == country][C.G_CROP].unique()), "sc_crop", "Wheat", k2)
        base = scenarios.global_baseline(gdf, country, crop_s, gmeta["year_range"][1])
        model, feats, rain_col, temp_col, unit, scale = gex.fitted, C.G_FEATURES, C.G_RAIN, C.G_TEMP, "hg/ha", 1.0
        where = f"{crop_s} in {country}"
    else:
        st_s = pick("State", imeta["states"], "sc_state", "Punjab", k1)
        crop_s = pick("Crop", sorted(idf[idf[C.I_STATE] == st_s][C.I_CROP].unique()), "sc_icrop", "Rice", k2)
        season_s = pick("Season", sorted(idf[(idf[C.I_STATE] == st_s) & (idf[C.I_CROP] == crop_s)][C.I_SEASON].unique()),
                        "sc_iseason", "Kharif", k3)
        base = scenarios.india_baseline(idf, st_s, crop_s, season_s, scenarios.last_full_year(idf))
        model, feats, rain_col, temp_col, unit, scale = iex.fitted, C.I_FEATURES, C.I_RAIN, None, "t/ha", 1.0
        where = f"{crop_s} in {st_s} ({season_s})"
    s1, s2 = st.columns(2)
    d_temp = s1.slider("Temperature change (°C)", 0.0, 3.0, 1.0, 0.5, key="sc_dt", disabled=temp_col is None)
    d_rain = s2.slider("Rainfall change (%)", -30, 30, -10, 5, key="sc_dr")
    if temp_col is None:
        st.info("The India dataset has no temperature variable, so the India model can only respond to rainfall. "
                "For temperature scenarios choose the global model with country = India.")
    base_pred, grid = scenarios.climate_grid(model, base, feats, rain_col, temp_col)
    scen_row = scenarios.apply_change(base, rain_col, temp_col, d_temp if temp_col else 0.0, d_rain)
    scen_pred = float(model.predict(pd.DataFrame([scen_row])[feats])[0])
    chg = (scen_pred / base_pred - 1) * 100
    fmt = "{:,.0f}" if use_global else "{:.2f}"
    m1, m2, m3 = st.columns(3)
    m1.metric("Today (model, recent climate)", f"{fmt.format(base_pred)} {unit}")
    m2.metric("Scenario", f"{fmt.format(scen_pred)} {unit}")
    m3.metric("Change vs today", f"{chg:+.1f}%")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if use_global:
        cols = plt.cm.RdYlBu(np.linspace(0.05, 0.95, len(grid.index)))
        for (r, row_), c in zip(grid.iterrows(), cols):
            ax.plot(row_.index, row_.values, marker="o", ms=4, color=c, label=f"rainfall {r:+d}%")
        ax.scatter([d_temp], [chg], marker="*", s=260, color="black", zorder=5, label="your scenario")
        ax.set_xlabel("Temperature change (°C)"); ax.legend(fontsize=8, ncol=2)
    else:
        vals = grid[0.0]
        ax.bar(vals.index, vals.values, width=10, color=["#c1121f" if v < 0 else "#2a9d8f" for v in vals.values])
        ax.scatter([d_rain], [chg], marker="*", s=260, color="black", zorder=5, label="your scenario"); ax.legend()
        ax.set_xlabel("Rainfall change (%)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(ylabel="Predicted yield change vs today (%)", title=f"{where}: predicted yield change across scenarios")
    fig.tight_layout(); st.pyplot(fig, clear_figure=True); plt.close("all")
    with st.expander("Full scenario table (% change vs today)"):
        tbl = grid.copy(); tbl.index = [f"rainfall {r:+d}%" for r in tbl.index]; tbl.columns = [f"+{t:g}°C" for t in tbl.columns]
        st.dataframe(tbl.style.format("{:+.1f}%"))
    st.caption("Tree-based models change in steps, so the curves can be flat or non-monotonic; treat small differences "
               "as noise. Results can look counter-intuitive (for example less rain giving a higher yield) because the model learned "
               "patterns across many countries/states and years, not the physiology of one crop. 'Today' = the model's prediction "
               "using this location's last observed years of climate.")

# =================================================================== TAB 3
with tab3:
    gcomp, icomp = load_results()
    cols = ["model", "cv_R2_mean", "test_R2", "test_MAE", "test_RMSE", "train_time_s"]
    fmt = {c: "{:.3f}" for c in cols[1:]}          # every number in these tables: 3 decimals
    st.header("Model evaluation")
    for title, comp, unit in [("Global dataset (yield in hg/ha)", gcomp, "hg/ha"), ("India dataset (yield in t/ha)", icomp, "t/ha")]:
        st.subheader(title)
        a, b = st.columns(2)
        a.markdown("**Random 80/20 split**"); a.dataframe(comp[comp.split == "random"][cols].style.format(fmt), hide_index=True)
        b.markdown("**Time-based split (latest 4 years held out)**"); b.dataframe(comp[comp.split == "time"][cols].style.format(fmt), hide_index=True)
    st.subheader("Why does the time-based split score lower?")
    st.write("In a random split, the same country/state-crop series appears on both sides: for most test rows the model "
             "has already seen the previous and next year of that series, so it can nearly look the answer up (leakage). "
             "In the time-based split the test years are strictly in the future, so the model must forecast, "
             "which is what a real user needs. The time-based score is the honest one; the random-split score is an optimistic upper bound.")
    for f in ["model_comparison.png", "india_model_comparison.png"]:
        if (C.FIG_DIR / f).exists():
            st.image(str(C.FIG_DIR / f))
