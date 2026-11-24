"""Streamlit demo: Smart-agriculture yield prediction with explanations and advice.

Run:  streamlit run app.py

Every visible string goes through tr() -> src/i18n.py, the single English/Hindi dictionary.
Crop, country and state names, numbers and units are shown as they are in the data.
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src import advisor, charts, config as C, data, i18n, india, scenarios
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


@st.cache_resource(show_spinner=False)
def cached_chart(name: str, plang: str):
    """Charts of fixed data are drawn once per language ('en'/'hi') and then reused."""
    _, idf, _, _ = load_india()
    gcomp, icomp = load_results()
    if name == "top_states":
        return charts.top_states(idf, plang)
    if name == "trends":
        return charts.yield_trends(idf, plang)
    if name == "heatmap":
        return charts.state_crop_heatmap(idf, plang)
    if name == "vs_world":
        return charts.india_vs_world(india.global_comparison(), plang)
    if name == "cmp_global":
        return charts.comparison_bars(gcomp, "chart_cmp_global", plang)
    return charts.comparison_bars(icomp, "chart_cmp_india", plang)


# ----------------------------------------------------------------- language + helpers
# The radio stores its label in session_state, so the chosen language is known before it is drawn.
lang = i18n.LANGS.get(st.session_state.get("lang", "English"), "en")
plang = i18n.plot_lang(lang)        # language used INSIDE plots ('en' if no Devanagari font installed)


def tr(key: str, **kw) -> str:
    """Translate a dictionary key into the current language."""
    return i18n.tr(key, lang, **kw)


def pick(label, options, key, default=None, container=st):
    """Selectbox whose options can change: resets an invalid stored choice instead of crashing."""
    options = list(options)
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    return container.selectbox(label, options, key=key)


def stable_choice(widget, label, options, state_key, format_func, **kw):
    """Radio / select_slider whose selection survives a language switch.

    Streamlit derives a widget's identity from its labels, so a translated widget would be re-created
    (and lose its selection) on every language switch. We keep the choice in session_state[state_key]
    and give the widget a per-language key that is re-seeded from it on every run.
    """
    wkey = f"{state_key}__{lang}"
    st.session_state.setdefault(state_key, options[0])
    st.session_state[wkey] = st.session_state[state_key]
    return widget(label, options, format_func=format_func, key=wkey,
                  on_change=lambda: st.session_state.__setitem__(state_key, st.session_state[wkey]), **kw)


def stable_radio(label, options, state_key, format_func):
    """Horizontal radio with a language-stable selection."""
    return stable_choice(st.radio, label, options, state_key, format_func, horizontal=True)


def level_fmt(v):
    """Display low/normal/high in the current language."""
    return tr(f"level.{v}")


def bar_chart(labels, values, xlabel, title, fmt="{:.2f}"):
    """Horizontal ranked bar chart (best at the top)."""
    with plt.rc_context(i18n.rc(plang)):
        fig, ax = plt.subplots(figsize=(7, 0.55 * len(labels) + 1.2))
        colors = ["#2a9d8f" if i == 0 else "#8ecae6" for i in range(len(labels))]
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        for y, v in enumerate(values[::-1]):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=9)
        ax.set(xlabel=xlabel, title=title)
        ax.margins(x=0.15)
        fig.tight_layout()
    return fig


def show_fig(fig):
    """Render a matplotlib figure and release it."""
    st.pyplot(fig, clear_figure=True)
    plt.close("all")


def show_advice(facts: dict):
    """Run forward chaining and display advice (English or Hindi) + the reasoning trace."""
    _, _, trace = advisor.forward_chain(facts)
    advice = advisor.advice_from_trace(trace, lang)
    st.subheader(tr("advice_title"))
    st.caption(tr("facts") + ": " + ", ".join(i18n.fact_text(k, v, lang) for k, v in facts.items()))
    if advice:
        for a in advice:
            st.success(a)
    else:
        st.info(tr("no_rule"))
    with st.expander(tr("trace")):
        if not trace:
            st.write(tr("no_rule"))
        for i, t in enumerate(trace, 1):
            cond = f" {tr('and')} ".join(i18n.fact_text(k, v, lang) for k, v in t["because"].items())
            der = ", ".join(i18n.fact_text(k, v, lang) for k, v in t["derived"].items())
            txt = t["text_hi"] if lang == "hi" else t["text"]
            st.markdown(f"**{i}. {t['rule']}**: {tr('if')} {cond} → {tr('then')} {der}  \n_{txt}_")


def show_explanation(ex: Explainer, row: pd.DataFrame, unit: str, title: str):
    """SHAP waterfall, the E[f(X)] note and the plain-language sentence."""
    st.subheader(tr("why"))
    show_fig(ex.waterfall(row, title=title, lang=plang))
    st.caption(tr("waterfall_note"))
    st.info(ex.sentence(row, unit=unit, lang=lang))
    st.caption(tr("shap_caption"))


# ----------------------------------------------------------------- header
st.title(tr("title"))
st.caption(tr("subtitle"))
st.radio(tr("lang_label"), list(i18n.LANGS), horizontal=True, key="lang")
if lang == "hi" and not i18n.hindi_font():
    st.caption("No Devanagari font found for charts: chart labels stay in English. / चार्ट के लेबल अंग्रेज़ी में रहेंगे।")
tab1, tab2, tab_sc, tab3 = st.tabs([tr("tab_predict"), tr("tab_india"), tr("tab_climate"), tr("tab_eval")])

# =================================================================== TAB 1
with tab1:
    ex, gdf, gmeta = load_global()
    st.markdown(tr("model_line", name=charts.model_label(gmeta["best_name"], lang)))
    for k, v in {"g_country": "India", "g_crop": "Rice, paddy", "g_year": int(gmeta["medians"][C.G_YEAR])}.items():
        st.session_state.setdefault(k, v)
    c1, c2, c3 = st.columns(3)
    country = c1.selectbox(tr("country"), gmeta["countries"], key="g_country")
    crop = c2.selectbox(tr("crop"), gmeta["crops"], key="g_crop")
    year = c3.slider(tr("year"), gmeta["year_range"][0], 2030, key="g_year")
    if year > gmeta["year_range"][1]:
        st.warning(tr("year_warn"))
    m = gmeta["medians"]
    gq = gmeta["quantiles"]
    for k, v in {"g_rain": float(m[C.G_RAIN]), "g_temp": float(m[C.G_TEMP]), "g_pest": float(m[C.G_PEST])}.items():
        st.session_state.setdefault(k, v)          # defaults = dataset medians

    def set_global_drought():
        """Button callback: fill inputs with a low-rainfall / high-temperature scenario."""
        st.session_state["g_rain"] = round(gq[C.G_RAIN][0] * 0.5, -1)   # half of the 'low' cut-off
        st.session_state["g_temp"] = round(gq[C.G_TEMP][1] + 3, 1)      # 3 °C above the 'high' cut-off

    c4, c5, c6 = st.columns(3)
    rain = c4.number_input(tr("rain_mm_year"), 0.0, 4000.0, step=50.0, key="g_rain")
    temp = c5.number_input(tr("temp_avg"), -5.0, 40.0, step=0.5, key="g_temp")
    pest = c6.number_input(tr("pest_tonnes"), 0.0, 400000.0, step=100.0, key="g_pest")
    st.button(tr("drought_btn"), key="g_drought", on_click=set_global_drought, help=tr("drought_help_global"))

    row = pd.DataFrame([{C.G_COUNTRY: country, C.G_CROP: crop, C.G_YEAR: year,
                         C.G_RAIN: rain, C.G_TEMP: temp, C.G_PEST: pest}])[C.G_FEATURES]
    pred = float(ex.fitted.predict(row)[0])
    hist = gdf[(gdf[C.G_COUNTRY] == country) & (gdf[C.G_CROP] == crop)][C.G_TARGET]
    hist_avg = hist.mean() if len(hist) else gdf[gdf[C.G_CROP] == crop][C.G_TARGET].mean()
    basis = f"{country} · {crop}" if len(hist) else tr("basis_no_history", crop=crop, country=country)
    k1, k2, k3 = st.columns(3)
    k1.metric(tr("pred_yield"), f"{pred:,.0f} hg/ha")
    k2.metric(tr("tonnes_per_ha"), f"{pred * C.HG_PER_HA_TO_T_PER_HA:.2f} t/ha")
    k3.metric(tr("hist_avg", basis=basis), f"{hist_avg:,.0f} hg/ha", f"{(pred / hist_avg - 1) * 100:+.1f}%")

    show_explanation(ex, row, "hg/ha", f"{crop} · {country}, {year}")

    q = gmeta["quantiles"]
    facts = advisor.build_facts({"rainfall": q[C.G_RAIN], "temp": q[C.G_TEMP], "pesticides": q[C.G_PEST]},
                                rainfall=rain, temp=temp, pesticides=pest,
                                predicted=pred, crop_avg=hist_avg, crop=crop)
    show_advice(facts)

# =================================================================== TAB 2
with tab2:
    iex, idf, imeta, rule_q = load_india()
    st.header(tr("india_header"))
    cols = st.columns(2)
    for i, name in enumerate(["top_states", "trends", "heatmap", "vs_world"]):
        cols[i % 2].pyplot(cached_chart(name, plang), clear_figure=False)
    st.divider()
    st.subheader(tr("india_predict_sub"))
    # defaults live in session_state (not `index=`) because the drought button also sets them
    for k, v in {"i_state": "Punjab", "i_crop": "Rice", "i_season": "Kharif", "i_year": 2019}.items():
        st.session_state.setdefault(k, v)
    d1, d2, d3, d4 = st.columns(4)
    state = d1.selectbox(tr("state"), imeta["states"], key="i_state")
    icrop = d2.selectbox(tr("crop"), imeta["crops"], key="i_crop")
    season = d3.selectbox(tr("season"), imeta["seasons"], key="i_season")
    iyear = d4.slider(tr("year"), imeta["year_range"][0], 2030, key="i_year")
    sub = idf[(idf[C.I_CROP] == icrop) & (idf[C.I_STATE] == state)]
    ref = sub if len(sub) else idf[idf[C.I_CROP] == icrop]
    sc = f"{state}_{icrop}"                           # new defaults whenever state/crop changes
    rain_key = f"i_rain_{sc}"
    st.session_state.setdefault(rain_key, float(ref[C.I_RAIN].median()))
    st.session_state.setdefault("i_outlook", "normal")     # also set by the drought button

    def set_india_drought():
        """Button callback: Kharif season, low rainfall, high temperature outlook."""
        s_, c_ = st.session_state["i_state"], st.session_state["i_crop"]
        st.session_state["i_season"] = "Kharif"
        st.session_state[f"i_rain_{s_}_{c_}"] = round(rule_q["rainfall"][0] * 0.6, -1)
        st.session_state["i_outlook"] = "high"

    e1, e2, e3, e4 = st.columns(4)
    area = e1.number_input(tr("area_crop"), 1.0, 5e7, float(ref[C.I_AREA].median()), step=1000.0,
                           help=tr("area_help"), key=f"i_area_{sc}")
    irain = e2.number_input(tr("rain_annual_mm"), 0.0, 7000.0, step=50.0, key=rain_key)
    fert = e3.number_input(tr("fert_kg_ha"), 0.0, 5000.0, float(ref[C.I_FERT_HA].median()), step=5.0,
                           format="%.2f", key=f"i_fert_{sc}")
    ipest = e4.number_input(tr("pest_kg_ha"), 0.0, 100.0, float(ref[C.I_PEST_HA].median()), step=0.05,
                            format="%.3f", key=f"i_pest_{sc}")
    outlook = stable_choice(st.select_slider, tr("temp_outlook"), ["low", "normal", "high"], "i_outlook", level_fmt)
    st.button(tr("drought_btn"), key="i_drought", on_click=set_india_drought, help=tr("drought_help_india"))

    irow = pd.DataFrame([{C.I_CROP: icrop, C.I_SEASON: season, C.I_STATE: state, C.I_YEAR: iyear,
                          C.I_AREA: area, C.I_RAIN: irain, C.I_FERT_HA: fert, C.I_PEST_HA: ipest}])[C.I_FEATURES]
    ipred = float(iex.fitted.predict(irow)[0])
    iavg = ref[C.I_TARGET].mean()
    m1, m2 = st.columns(2)
    m1.metric(tr("pred_yield"), f"{ipred:.2f} t/ha")
    m2.metric(tr("hist_avg", basis=tr("basis_state_crop") if len(sub) else tr("basis_crop_all_states")),
              f"{iavg:.2f} t/ha", f"{(ipred / iavg - 1) * 100:+.1f}%")
    show_explanation(iex, irow, "t/ha", f"{icrop} · {state}, {season}, {iyear}")
    ifacts = advisor.build_facts(rule_q, rainfall=irain, pesticides=ipest, fertilizer=fert,
                                 predicted=ipred, crop_avg=iavg, crop=icrop, season=season)
    ifacts["temp"] = outlook
    show_advice(ifacts)

    # ------------------------------------------------------------ Crop Recommender
    st.divider()
    st.subheader(tr("rec_title"))
    st.caption(tr("rec_caption"))
    last_year = scenarios.last_full_year(idf)
    r1, r2, r3, r4 = st.columns(4)
    rstate = pick(tr("state"), imeta["states"], "rec_state", "Punjab", r1)
    rseason = pick(tr("season"), sorted(idf[idf[C.I_STATE] == rstate][C.I_SEASON].unique()), "rec_season", "Kharif", r2)
    rsub = idf[(idf[C.I_STATE] == rstate) & (idf[C.I_SEASON] == rseason)]
    rkey = f"rec_rain_{rstate}_{rseason}"
    st.session_state.setdefault(rkey, float(rsub[C.I_RAIN].median()))
    rrain = r3.number_input(tr("rain_annual_mm"), 0.0, 7000.0, step=50.0, key=rkey)
    rarea = r4.number_input(tr("area_crop"), 1.0, 5e7, float(rsub[C.I_AREA].median()), step=1000.0,
                            key=f"rec_area_{rstate}_{rseason}", help=tr("area_help"))
    rank_by = stable_radio(tr("rec_rank_by"), ["pred", "pct"], "rec_rank",
                           lambda v: tr("rec_rank_pred" if v == "pred" else "rec_rank_pct"))
    rec = scenarios.recommend(iex.fitted, idf, rstate, rseason, rrain, rarea, last_year)
    if rec.empty:
        st.info(tr("rec_none"))
    else:
        by_pct = rank_by == "pct"
        ranked = rec.sort_values("vs_usual_pct" if by_pct else "predicted", ascending=False).reset_index(drop=True)
        top5 = ranked.head(5)
        vals = top5.vs_usual_pct if by_pct else top5.predicted
        show_fig(bar_chart(list(top5.crop), list(vals), tr("rec_x_pct" if by_pct else "rec_x_pred"),
                           tr("rec_chart_title", state=rstate, season=rseason, n=len(rec)),
                           fmt="{:+.1f}%" if by_pct else "{:.2f}"))
        names = {"crop": tr("col.crop"), "predicted": tr("col.predicted_tha"), "hist_avg": tr("col.usual_tha"),
                 "vs_usual_pct": tr("col.vs_usual"), "n_rows": tr("col.past_records")}
        st.dataframe(top5[list(names)].rename(columns=names).style.format(
            {names["predicted"]: "{:.3f}", names["hist_avg"]: "{:.3f}", names["vs_usual_pct"]: "{:+.1f}"}), hide_index=True)
        st.warning(tr("rec_warn"))
        best = top5.iloc[0]
        st.markdown(f"**{tr('rec_shap_top', crop=best.crop)}**")
        show_explanation(iex, pd.DataFrame([best.row])[C.I_FEATURES], "t/ha", f"{best.crop} · {rstate}, {rseason}")

# =================================================================== CLIMATE SCENARIOS
with tab_sc:
    st.header(tr("sc_header"))
    st.warning(tr("sc_warn"))
    gex, gdf, gmeta = load_global()
    iex, idf, imeta, _ = load_india()
    mode = stable_radio(tr("sc_model"), ["global", "india"], "sc_mode",
                        lambda v: tr("sc_model_global" if v == "global" else "sc_model_india"))
    use_global = mode == "global"
    k1, k2, k3 = st.columns(3)
    if use_global:
        country = pick(tr("country"), gmeta["countries"], "sc_country", "India", k1)
        crop_s = pick(tr("crop"), sorted(gdf[gdf[C.G_COUNTRY] == country][C.G_CROP].unique()), "sc_crop", "Wheat", k2)
        base = scenarios.global_baseline(gdf, country, crop_s, gmeta["year_range"][1])
        model, feats, rain_col, temp_col, unit = gex.fitted, C.G_FEATURES, C.G_RAIN, C.G_TEMP, "hg/ha"
        where = f"{crop_s} · {country}"
    else:
        st_s = pick(tr("state"), imeta["states"], "sc_state", "Punjab", k1)
        crop_s = pick(tr("crop"), sorted(idf[idf[C.I_STATE] == st_s][C.I_CROP].unique()), "sc_icrop", "Rice", k2)
        season_s = pick(tr("season"), sorted(idf[(idf[C.I_STATE] == st_s) & (idf[C.I_CROP] == crop_s)][C.I_SEASON].unique()),
                        "sc_iseason", "Kharif", k3)
        base = scenarios.india_baseline(idf, st_s, crop_s, season_s, scenarios.last_full_year(idf))
        model, feats, rain_col, temp_col, unit = iex.fitted, C.I_FEATURES, C.I_RAIN, None, "t/ha"
        where = f"{crop_s} · {st_s} ({season_s})"
    s1, s2 = st.columns(2)
    d_temp = s1.slider(tr("sc_dtemp"), 0.0, 3.0, 1.0, 0.5, key="sc_dt", disabled=temp_col is None)
    d_rain = s2.slider(tr("sc_drain"), -30, 30, -10, 5, key="sc_dr")
    if temp_col is None:
        st.info(tr("sc_no_temp"))
    base_pred, grid = scenarios.climate_grid(model, base, feats, rain_col, temp_col)
    scen_row = scenarios.apply_change(base, rain_col, temp_col, d_temp if temp_col else 0.0, d_rain)
    scen_pred = float(model.predict(pd.DataFrame([scen_row])[feats])[0])
    chg = (scen_pred / base_pred - 1) * 100
    fmt = "{:,.0f}" if use_global else "{:.2f}"
    m1, m2, m3 = st.columns(3)
    m1.metric(tr("sc_today"), f"{fmt.format(base_pred)} {unit}")
    m2.metric(tr("sc_scenario"), f"{fmt.format(scen_pred)} {unit}")
    m3.metric(tr("sc_change"), f"{chg:+.1f}%")
    with plt.rc_context(i18n.rc(plang)):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        if use_global:
            cols_ = plt.cm.RdYlBu(np.linspace(0.05, 0.95, len(grid.index)))
            for (r, row_), c in zip(grid.iterrows(), cols_):
                ax.plot(row_.index, row_.values, marker="o", ms=4, color=c,
                        label=i18n.tr("sc_rain_line", plang, r=int(r)))
            ax.scatter([d_temp], [chg], marker="*", s=260, color="black", zorder=5, label=i18n.tr("sc_yours", plang))
            ax.set_xlabel(i18n.tr("sc_x_temp", plang)); ax.legend(fontsize=8, ncol=2)
        else:
            vals = grid[0.0]
            ax.bar(vals.index, vals.values, width=10, color=["#c1121f" if v < 0 else "#2a9d8f" for v in vals.values])
            ax.scatter([d_rain], [chg], marker="*", s=260, color="black", zorder=5, label=i18n.tr("sc_yours", plang))
            ax.legend()
            ax.set_xlabel(i18n.tr("sc_x_rain", plang))
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(ylabel=i18n.tr("sc_y", plang), title=i18n.tr("sc_chart_title", plang, where=where))
        fig.tight_layout()
    show_fig(fig)
    with st.expander(tr("sc_table")):
        tbl = grid.copy()
        tbl.index = [tr("sc_rain_line", r=int(r)) for r in tbl.index]
        tbl.columns = [f"+{t:g}°C" for t in tbl.columns]
        st.dataframe(tbl.style.format("{:+.1f}%"))
    st.caption(tr("sc_caption"))

# =================================================================== MODEL EVALUATION
with tab3:
    gcomp, icomp = load_results()
    cols = ["model", "cv_R2_mean", "test_R2", "test_MAE", "test_RMSE", "train_time_s"]
    names = {c: tr(f"col.{c}") for c in cols}


    def table(comp: pd.DataFrame, split: str):
        """One evaluation table, headers and '(tuned)' translated, every number at 3 decimals."""
        v = comp[comp.split == split][cols].copy()
        v["model"] = v["model"].map(lambda mname: charts.model_label(mname, lang))
        return v.rename(columns=names).style.format({names[c]: "{:.3f}" for c in cols[1:]})


    st.header(tr("eval_header"))
    for title_key, comp in [("eval_global", gcomp), ("eval_india", icomp)]:
        st.subheader(tr(title_key))
        a, b = st.columns(2)
        a.markdown(f"**{tr('eval_random')}**"); a.dataframe(table(comp, "random"), hide_index=True)
        b.markdown(f"**{tr('eval_time')}**"); b.dataframe(table(comp, "time"), hide_index=True)
    st.subheader(tr("eval_why"))
    st.write(tr("eval_why_text"))
    for name in ["cmp_global", "cmp_india"]:
        st.pyplot(cached_chart(name, plang), clear_figure=False)
