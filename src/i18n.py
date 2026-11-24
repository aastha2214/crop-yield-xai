"""ONE translation dictionary (English / Hindi) for every user-facing string in the app.

Layout: T[key] = {"en": ..., "hi": ...}. Key prefixes group related strings:
  label.*  feature names used in SHAP sentences     level.*  low / typical / high
  factk.*  expert-system fact names                 factv.*  expert-system fact values
  rule.*   Hindi text of each expert-system rule    (English lives on the Rule objects)
  col.* / model.*  table headers and model-name parts
  everything else: headings, widget labels, buttons, notes, captions, chart text.

Crop / country / state / season names, numbers and units (mm/year, °C, t/ha, hg/ha, kg/ha, ha,
tonnes) are NEVER translated - they appear as in the datasets.
"""
from functools import lru_cache

LANGS = {"English": "en", "हिन्दी": "hi"}
T: dict = {}


def _add(key: str, en: str, hi: str) -> None:
    """Register one string in both languages."""
    T[key] = {"en": en, "hi": hi}


def _hi_only(key: str, hi: str) -> None:
    """Register a Hindi-only string (the English text lives elsewhere, e.g. on a Rule)."""
    T[key] = {"hi": hi}


# ------------------------------------------------------------------ page chrome
_add("title", "🌾 Crop Yield Prediction with Explainable AI", "🌾 समझने योग्य AI से फसल उपज का अनुमान")
_add("subtitle", "Predict · Explain · Advise: an explainable AI system for smarter farming",
     "अनुमान · व्याख्या · सलाह: बेहतर खेती के लिए एक समझने योग्य AI प्रणाली")
_add("lang_label", "Language / भाषा", "Language / भाषा")
_add("tab_predict", "Predict & Explain", "अनुमान और व्याख्या")
_add("tab_india", "India Insights", "भारत की जानकारी")
_add("tab_climate", "Climate Scenarios", "जलवायु परिदृश्य")
_add("tab_eval", "Model Evaluation", "मॉडल मूल्यांकन")

# ------------------------------------------------------------------ shared widgets / metrics
_add("country", "Country", "देश")
_add("crop", "Crop", "फसल")
_add("year", "Year", "वर्ष")
_add("state", "State", "राज्य")
_add("season", "Season", "सीज़न")
_add("rain_mm_year", "Rainfall (mm/year)", "वर्षा (mm/year)")
_add("temp_avg", "Avg temperature (°C)", "औसत तापमान (°C)")
_add("pest_tonnes", "Pesticides (tonnes)", "कीटनाशक (tonnes)")
_add("area_crop", "Area under crop (ha)", "फसल का क्षेत्रफल (ha)")
_add("area_help", "In the dataset, Area is the total area under that crop in the whole state (not a single farm), so very small values are outside what the model has seen.",
     "डेटासेट में क्षेत्रफल का मतलब पूरे राज्य में उस फसल का कुल क्षेत्रफल है (किसी एक खेत का नहीं), इसलिए बहुत छोटे मान मॉडल के देखे हुए दायरे से बाहर हैं।")
_add("rain_annual_mm", "Annual rainfall (mm)", "वार्षिक वर्षा (mm)")
_add("fert_kg_ha", "Fertilizer (kg/ha)", "उर्वरक (kg/ha)")
_add("pest_kg_ha", "Pesticide (kg/ha)", "कीटनाशक (kg/ha)")
_add("drought_btn", "Try example: drought scenario", "उदाहरण आज़माएँ: सूखे की स्थिति")
_add("drought_help_global", "Sets low rainfall and high temperature so the expert-system rules fire.",
     "कम वर्षा और अधिक तापमान भर देता है, ताकि विशेषज्ञ प्रणाली के नियम लागू हों।")
_add("drought_help_india", "Sets Kharif season, low rainfall and a high temperature outlook so the expert-system rules fire.",
     "खरीफ सीज़न, कम वर्षा और अधिक तापमान का अनुमान भर देता है, ताकि विशेषज्ञ प्रणाली के नियम लागू हों।")
_add("pred_yield", "Predicted yield", "अनुमानित उपज")
_add("hist_avg", "Historical average ({basis})", "ऐतिहासिक औसत ({basis})")
_add("basis_no_history", "{crop} (all countries; no {country} history)", "{crop} (सभी देश; {country} का कोई इतिहास नहीं)")
_add("basis_state_crop", "state · crop", "राज्य · फसल")
_add("basis_crop_all_states", "crop, all states", "फसल, सभी राज्य")

# ------------------------------------------------------------------ tab 1
_add("model_line", "Model: **{name}** (best on the time-based split).", "मॉडल: **{name}** (समय-आधारित विभाजन पर सबसे अच्छा)।")
_add("year_warn", "Years beyond the training data extrapolate: tree models hold the last-seen trend flat.",
     "प्रशिक्षण डेटा से आगे के वर्षों के अनुमान भरोसेमंद नहीं हैं: ट्री मॉडल आख़िरी देखे गए रुझान को ही स्थिर मान लेते हैं।")

# ------------------------------------------------------------------ explanation + advice
_add("advice_title", "Expert-system advice", "विशेषज्ञ प्रणाली की सलाह")
_add("facts", "Facts", "तथ्य")
_add("no_rule", "No rule fired: conditions look unremarkable, so no special advice.",
     "कोई नियम लागू नहीं हुआ: स्थितियाँ सामान्य लगती हैं, इसलिए कोई विशेष सलाह नहीं।")
_add("trace", "Reasoning trace (rules fired, in order)", "तर्क-क्रम (लागू हुए नियम, क्रम से)")
_add("if", "IF", "यदि")
_add("then", "THEN", "तो")
_add("and", "AND", "और")
_add("why", "Why this prediction? (SHAP)", "यह अनुमान क्यों? (SHAP)")
_add("waterfall_note",
     "The starting value E[f(X)] is the average prediction across all crops, so a crop with naturally lower yield per hectare shows a negative contribution.",
     "शुरुआती मान E[f(X)] सभी फसलों का औसत अनुमान है, इसलिए जिस फसल की प्रति हेक्टेयर उपज स्वभाव से कम होती है, उसका योगदान ऋणात्मक दिखता है।")
_add("shap_caption", "SHAP shows what the model associates with the prediction, not proven causes.",
     "SHAP यह दिखाता है कि मॉडल किन बातों को अनुमान से जोड़ता है; यह सिद्ध कारण नहीं है।")

# ------------------------------------------------------------------ tab 2: India insights
_add("india_header", "India insights", "भारत की जानकारी")
_add("india_predict_sub", "Predict yield for an Indian state", "भारतीय राज्य के लिए उपज का अनुमान")
_add("temp_outlook", "Temperature outlook (advisory only - the India model has no temperature input)",
     "तापमान का अनुमान (केवल सलाह के लिए - भारत मॉडल में तापमान इनपुट नहीं है)")
_add("rec_title", "🌱 Crop Recommender", "🌱 फसल सुझावक")
_add("rec_caption",
     "Choose a state, season, rainfall and area. The model predicts the yield of every crop grown in that state and season and ranks the top 5. Fertilizer and pesticide per hectare, and by default the area too, are set to each crop's own typical level there.",
     "राज्य, सीज़न, वर्षा और क्षेत्रफल चुनें। मॉडल उस राज्य और सीज़न में उगाई जाने वाली हर फसल की उपज का अनुमान लगाकर शीर्ष 5 फसलों को क्रम से दिखाता है। प्रति हेक्टेयर उर्वरक और कीटनाशक, और (डिफ़ॉल्ट रूप से) क्षेत्रफल भी, वहाँ उस फसल के अपने सामान्य स्तर पर रखे गए हैं।")
_add("rec_area_caption",
     "Typical area for {state}, {season}: median {med} ha; the middle half of records lies between {lo} and {hi} ha (state-wide totals per crop, from the dataset).",
     "{state}, {season} के लिए सामान्य क्षेत्रफल: माध्यिका {med} ha; आधे रिकॉर्ड {lo} से {hi} ha के बीच हैं (डेटासेट में हर फसल के राज्य-स्तरीय कुल)।")
_add("rec_use_typical", "Use each crop's typical area", "हर फसल का सामान्य क्षेत्रफल इस्तेमाल करें")
_add("rec_use_typical_help",
     "Checked: every crop is predicted with ITS OWN median area in this state and season. Unchecked: the Area box below is applied to all crops.",
     "चुना हुआ: हर फसल का अनुमान इस राज्य और सीज़न में उसके अपने माध्यिका क्षेत्रफल से लगाया जाता है। न चुना हुआ: नीचे दिया गया क्षेत्रफल सभी फसलों पर लागू होता है।")
_add("rec_area_disabled", "(not used while 'Use each crop's typical area' is checked)", "('हर फसल का सामान्य क्षेत्रफल' चुने रहने पर इसका उपयोग नहीं होता)")
_add("rec_rank_by", "Rank crops by", "फसलों को इस आधार पर क्रम दें")
_add("rec_rank_pred", "Predicted yield (t/ha)", "अनुमानित उपज (t/ha)")
_add("rec_rank_pct", "Predicted yield vs. the crop's usual yield here (%)", "अनुमानित उपज, यहाँ उस फसल की सामान्य उपज की तुलना में (%)")
_add("rec_chart_title", "Top 5 crops: {state}, {season} ({n} crops considered)", "शीर्ष 5 फसलें: {state}, {season} ({n} फसलें देखी गईं)")
_add("rec_x_pred", "predicted yield (t/ha)", "अनुमानित उपज (t/ha)")
_add("rec_x_pct", "% above/below the usual yield", "सामान्य उपज से % ऊपर/नीचे")
_add("rec_none", "Not enough historical records for this state and season.", "इस राज्य और सीज़न के लिए पर्याप्त ऐतिहासिक रिकॉर्ड नहीं हैं।")
_add("rec_warn",
     "Yields in t/ha are not comparable across crops in value or effort (e.g. sugarcane vs pulses), and this ranking ignores prices, soil, water availability, labour and crop rotation. It shows what the model predicts, not what you should plant.",
     "t/ha में उपज अलग-अलग फसलों के बीच मूल्य या मेहनत के हिसाब से तुलनीय नहीं है (जैसे गन्ना और दालें), और यह क्रम कीमत, मिट्टी, पानी की उपलब्धता, मज़दूरी और फसल-चक्र को नहीं देखता। यह दिखाता है कि मॉडल क्या अनुमान लगाता है, यह नहीं कि आपको क्या बोना चाहिए।")
_add("rec_shap_top", "SHAP explanation for the top crop: {crop}", "शीर्ष फसल के लिए SHAP व्याख्या: {crop}")
_add("col.crop", "Crop", "फसल")
_add("col.predicted_tha", "Predicted (t/ha)", "अनुमानित (t/ha)")
_add("col.usual_tha", "Usual yield here (t/ha)", "यहाँ सामान्य उपज (t/ha)")
_add("col.vs_usual", "Vs usual (%)", "सामान्य की तुलना में (%)")
_add("col.area_used", "Area used (ha)", "उपयोग किया गया क्षेत्रफल (ha)")
_add("col.past_records", "Past records", "पिछले रिकॉर्ड")

# ------------------------------------------------------------------ tab 3: climate scenarios
_add("sc_header", "Climate Scenarios", "जलवायु परिदृश्य")
_add("sc_warn",
     "This shows what the model associates with climate in historical data, not a guaranteed forecast. Real outcomes also depend on irrigation, crop varieties, adaptation, pests and extreme events that the model cannot see (its climate inputs are annual averages).",
     "यह दिखाता है कि पुराने डेटा में मॉडल जलवायु को किस चीज़ से जोड़ता है; यह कोई पक्का पूर्वानुमान नहीं है। असली नतीजे सिंचाई, फसल की किस्मों, अनुकूलन, कीटों और चरम घटनाओं पर भी निर्भर करते हैं, जिन्हें मॉडल नहीं देख सकता (इसके जलवायु इनपुट वार्षिक औसत हैं)।")
_add("sc_model", "Model", "मॉडल")
_add("sc_model_global", "Global model (country)", "ग्लोबल मॉडल (देश)")
_add("sc_model_india", "India model (state)", "भारत मॉडल (राज्य)")
_add("sc_dtemp", "Temperature change (°C)", "तापमान में बदलाव (°C)")
_add("sc_drain", "Rainfall change (%)", "वर्षा में बदलाव (%)")
_add("sc_no_temp", "The India dataset has no temperature variable, so the India model can only respond to rainfall. For temperature scenarios choose the global model with country = India.",
     "भारत के डेटासेट में तापमान का कोई चर नहीं है, इसलिए भारत मॉडल केवल वर्षा पर प्रतिक्रिया दे सकता है। तापमान के परिदृश्यों के लिए ग्लोबल मॉडल चुनें और देश = India रखें।")
_add("sc_today", "Today (model, recent climate)", "आज (मॉडल, हाल की जलवायु)")
_add("sc_scenario", "Scenario", "परिदृश्य")
_add("sc_change", "Change vs today", "आज की तुलना में बदलाव")
_add("sc_chart_title", "{where}: predicted yield change across scenarios", "{where}: परिदृश्यों में उपज में अनुमानित बदलाव")
_add("sc_rain_line", "rainfall {r:+d}%", "वर्षा {r:+d}%")
_add("sc_yours", "your scenario", "आपका परिदृश्य")
_add("sc_x_temp", "Temperature change (°C)", "तापमान में बदलाव (°C)")
_add("sc_x_rain", "Rainfall change (%)", "वर्षा में बदलाव (%)")
_add("sc_y", "Predicted yield change vs today (%)", "आज की तुलना में उपज में अनुमानित बदलाव (%)")
_add("sc_table", "Full scenario table (% change vs today)", "पूरी परिदृश्य तालिका (आज की तुलना में % बदलाव)")
_add("sc_caption",
     "Tree-based models change in steps, so the curves can be flat or non-monotonic; treat small differences as noise. Results can look counter-intuitive (for example less rain giving a higher yield) because the model learned patterns across many countries/states and years, not the physiology of one crop. 'Today' = the model's prediction using this location's last observed years of climate.",
     "ट्री-आधारित मॉडल सीढ़ियों की तरह बदलते हैं, इसलिए वक्र सपाट या ऊपर-नीचे हो सकते हैं; छोटे अंतर को शोर समझें। नतीजे उलटे भी लग सकते हैं (जैसे कम वर्षा पर अधिक उपज), क्योंकि मॉडल ने कई देशों/राज्यों और वर्षों के पैटर्न सीखे हैं, किसी एक फसल की जीव-विज्ञान नहीं। 'आज' = इस स्थान की हाल के वर्षों की जलवायु पर मॉडल का अनुमान।")

# ------------------------------------------------------------------ tab 4: evaluation
_add("eval_header", "Model evaluation", "मॉडल मूल्यांकन")
_add("eval_global", "Global dataset (yield in hg/ha)", "ग्लोबल डेटासेट (उपज hg/ha में)")
_add("eval_india", "India dataset (yield in t/ha)", "भारत डेटासेट (उपज t/ha में)")
_add("eval_random", "Random 80/20 split", "यादृच्छिक 80/20 विभाजन")
_add("eval_time", "Time-based split (latest 4 years held out)", "समय-आधारित विभाजन (अंतिम 4 वर्ष परीक्षण के लिए रखे गए)")
_add("eval_why", "Why does the time-based split score lower?", "समय-आधारित विभाजन में स्कोर कम क्यों आता है?")
_add("eval_why_text",
     "In a random split, the same country/state-crop series appears on both sides: for most test rows the model has already seen the previous and next year of that series, so it can nearly look the answer up (leakage). In the time-based split the test years are strictly in the future, so the model must forecast, which is what a real user needs. The time-based score is the honest one; the random-split score is an optimistic upper bound.",
     "यादृच्छिक विभाजन में एक ही देश/राज्य-फसल की श्रृंखला प्रशिक्षण और परीक्षण दोनों में आ जाती है: ज़्यादातर परीक्षण पंक्तियों के लिए मॉडल उस श्रृंखला का पिछला और अगला वर्ष पहले ही देख चुका होता है, इसलिए वह लगभग उत्तर \"खोज\" लेता है (डेटा रिसाव)। समय-आधारित विभाजन में परीक्षण के वर्ष पूरी तरह भविष्य में होते हैं, इसलिए मॉडल को सच में पूर्वानुमान लगाना पड़ता है, जो असली उपयोगकर्ता को चाहिए। समय-आधारित स्कोर ही ईमानदार है; यादृच्छिक विभाजन का स्कोर आशावादी ऊपरी सीमा है।")
_add("col.model", "model", "मॉडल")
_add("col.cv_R2_mean", "cv_R2_mean", "CV R² (औसत)")
_add("col.test_R2", "test_R2", "परीक्षण R²")
_add("col.test_MAE", "test_MAE", "परीक्षण MAE")
_add("col.test_RMSE", "test_RMSE", "परीक्षण RMSE")
_add("col.train_time_s", "train_time_s", "प्रशिक्षण (s)")
_add("model.tuned", "tuned", "ट्यून किया हुआ")
_add("chart_cmp_global", "Model comparison (test R²)", "मॉडल तुलना (परीक्षण R²)")
_add("chart_cmp_india", "India model comparison (test R²)", "भारत मॉडल तुलना (परीक्षण R²)")
_add("chart_cmp_y", "Test R²", "परीक्षण R²")
_add("chart_cmp_x", "model", "मॉडल")
_add("chart_split", "split", "विभाजन")
_add("split.random", "random", "यादृच्छिक")
_add("split.time", "time", "समय-आधारित")

# ------------------------------------------------------------------ India charts (live-drawn)
_add("ch_top_states", "Top 10 states by production (coconut already excluded)", "उत्पादन के आधार पर शीर्ष 10 राज्य (नारियल शामिल नहीं)")
_add("ch_top_states_x", "total production, millions (units as in dataset)", "कुल उत्पादन, मिलियन (डेटासेट की इकाइयों में)")
_add("ch_trends", "Yield trends, major crops", "प्रमुख फसलों की उपज का रुझान")
_add("ch_trends_y", "median yield (t/ha)", "माध्यिका उपज (t/ha)")
_add("ch_heat", "Median yield by state and crop", "राज्य और फसल के अनुसार माध्यिका उपज")
_add("ch_heat_cbar", "median yield", "माध्यिका उपज")
_add("ch_vs_world", "India vs world average yield (t/ha)", "भारत बनाम विश्व औसत उपज (t/ha)")
_add("ch_india", "India", "भारत")
_add("ch_world", "World avg", "विश्व औसत")

# ------------------------------------------------------------------ SHAP sentence pieces
for _k, _en, _hi in [("Country", "Country", "देश"), ("Crop", "Crop", "फसल"), ("Year", "Year", "वर्ष"),
                     ("Rainfall", "Rainfall", "वर्षा"), ("Temperature", "Temperature", "तापमान"),
                     ("Pesticide use", "Pesticide use", "कीटनाशक का उपयोग"), ("Season", "Season", "सीज़न"),
                     ("State", "State", "राज्य"), ("Cultivated area", "Cultivated area", "खेती का क्षेत्रफल"),
                     ("Fertilizer use", "Fertilizer use", "उर्वरक का उपयोग")]:
    _add(f"label.{_k}", _en, _hi)
for _k, _en, _hi in [("low", "low", "कम"), ("typical", "typical", "सामान्य"), ("normal", "normal", "सामान्य"), ("high", "high", "अधिक")]:
    _add(f"level.{_k}", _en, _hi)
_add("sent.head", "The model predicts {pred} {unit} (average prediction: {base}). ", "मॉडल का अनुमान {pred} {unit} है (औसत अनुमान: {base})। ")

# ------------------------------------------------------------------ expert-system facts
for _k, _hi in {"rainfall": "वर्षा", "temp": "तापमान", "pesticides": "कीटनाशक", "fertilizer": "उर्वरक",
                "predicted_yield": "अनुमानित उपज", "crop": "फसल", "season": "सीज़न", "drought_risk": "सूखे का जोखिम",
                "disease_risk": "रोग का जोखिम", "waterlogging_risk": "जलभराव का जोखिम", "heat_stress": "गर्मी का तनाव",
                "monsoon_deficit": "मानसून की कमी", "terminal_heat_risk": "अंतिम चरण में गर्मी का जोखिम"}.items():
    _add(f"factk.{_k}", _k, _hi)
for _k, _hi in {"advise_irrigation": "सलाह: सिंचाई", "advise_pest_review": "सलाह: कीट प्रबंधन की समीक्षा",
                "advise_disease_monitoring": "सलाह: रोग की निगरानी", "advise_drainage": "सलाह: जल-निकासी सुधार",
                "advise_sowing_window": "सलाह: बुवाई का समय", "advise_maintain": "सलाह: मौजूदा पद्धति जारी रखें",
                "advise_water_conservation": "सलाह: जल संरक्षण", "advise_soil_test": "सलाह: मृदा जाँच",
                "advise_early_sowing": "सलाह: जल्दी बुवाई", "advise_insurance": "सलाह: फसल बीमा"}.items():
    _add(f"factk.{_k}", _k, _hi)
for _k, _hi in {"low": "कम", "normal": "सामान्य", "high": "अधिक", "below_avg": "औसत से कम",
                "above_avg": "औसत से अधिक", "average": "औसत", "True": "हाँ"}.items():
    _add(f"factv.{_k}", _k, _hi)

# ------------------------------------------------------------------ expert-system rule texts (Hindi)
for _k, _hi in {
    "R1": "कम बारिश और अधिक तापमान: सूखे का खतरा अधिक है।",
    "R2": "गर्म और नम मौसम में फफूंद और कीट-रोग फैलने की संभावना अधिक होती है।",
    "R3": "सिंचाई का समय-सारणी बनाएँ (जैसे ड्रिप सिंचाई) और सूखा सहने वाली किस्में चुनें।",
    "R4": "कीटनाशक अधिक डालने पर भी उपज कम है: कीट प्रबंधन की समीक्षा करें (खेत की निगरानी, सही मात्रा, एकीकृत कीट प्रबंधन)।",
    "R5": "खेतों में फफूंद रोग की निगरानी करें, पानी की निकासी ठीक रखें और पौधों के बीच उचित दूरी रखें।",
    "R6": "अधिक बारिश और कम अनुमानित उपज: खेत में जलभराव की आशंका है।",
    "R7": "खेत की जल-निकासी सुधारें और ऊँची क्यारियाँ बनाने पर विचार करें।",
    "R8": "अधिक तापमान और कम अनुमानित उपज: फसल पर गर्मी का तनाव हो सकता है।",
    "R9": "बुवाई का समय बदलें या गर्मी सहने वाली किस्में लगाएँ; मिट्टी को ठंडा रखने के लिए मल्चिंग करें।",
    "R10": "स्थितियाँ अनुकूल हैं: अपनी मौजूदा खेती की पद्धति जारी रखें और जो तरीका काम करे उसे लिखकर रखें।",
    "R11": "खरीफ में कम बारिश का मतलब है मानसून की कमी।",
    "R12": "पानी बचाने के उपाय अपनाएँ (मल्चिंग, खेत-तालाब) और अपने नज़दीकी कृषि विज्ञान केंद्र (KVK) से सलाह लें।",
    "R13": "और उर्वरक डालने से पहले मृदा स्वास्थ्य कार्ड (Soil Health Card) के लिए मिट्टी की जाँच कराएँ।",
    "R14": "रबी में अधिक तापमान (जैसे गेहूँ में दाना बनते समय) का मतलब है अंतिम चरण में गर्मी का खतरा।",
    "R15": "गेहूँ/रबी की फसल जल्दी बोएँ, दाना भरते समय सिंचाई करें और गर्मी सहने वाली किस्में लगाएँ।",
    "R16": "प्रधानमंत्री फसल बीमा योजना (PMFBY) में नामांकन कराने पर विचार करें।",
}.items():
    _hi_only(f"rule.{_k}", _hi)


# ------------------------------------------------------------------ lookup helpers
def tr(key: str, lang: str = "en", **kw) -> str:
    """Translate `key` into `lang` ('en' | 'hi'); falls back to English; applies str.format(**kw)."""
    entry = T[key]
    text = entry.get(lang) or entry.get("en", key)
    return text.format(**kw) if kw else text


ui = tr   # backwards-compatible alias


def fact_text(key, value, lang: str = "en") -> str:
    """'rainfall = low' in English, or 'वर्षा = कम' in Hindi."""
    if lang != "hi":
        return f"{key} = {value}"
    k = T.get(f"factk.{key}", {}).get("hi", key)
    v = T.get(f"factv.{value}", {}).get("hi", value)
    return f"{k} = {v}"


# ------------------------------------------------------------------ Devanagari font for plots
_FONT_CANDIDATES = ["Kohinoor Devanagari", "Noto Sans Devanagari", "Devanagari Sangam MN", "Lohit Devanagari", "Mukta"]


@lru_cache(maxsize=1)
def hindi_font():
    """Name of an installed Devanagari-capable font, or None (then plots stay in English)."""
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in _FONT_CANDIDATES if f in installed), None)


def plot_lang(lang: str) -> str:
    """Language to use INSIDE matplotlib figures: Hindi only if a Devanagari font exists."""
    return "hi" if lang == "hi" and hindi_font() else "en"


def rc(lang: str) -> dict:
    """matplotlib rcParams for the language (Devanagari font first, DejaVu as per-glyph fallback)."""
    font = hindi_font() if lang == "hi" else None
    return {"font.family": [font, "DejaVu Sans"]} if font else {}
