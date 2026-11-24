"""Small English / Hindi (हिन्दी) translation tables for the app, advisor and SHAP sentence.

Only user-facing text is translated. Crop/state/country names, units and model names stay
in English because that is how they appear in the datasets.
"""

LANGS = {"English": "en", "हिन्दी": "hi"}

# feature labels used in SHAP sentences (keys = English labels from config)
LABELS_HI = {
    "Country": "देश", "Crop": "फसल", "Year": "वर्ष", "Rainfall": "वर्षा",
    "Temperature": "तापमान", "Pesticide use": "कीटनाशक का उपयोग", "Season": "सीज़न",
    "State": "राज्य", "Cultivated area": "खेती का क्षेत्रफल", "Fertilizer use": "उर्वरक का उपयोग",
}
LEVELS_HI = {"low": "कम", "typical": "सामान्य", "high": "अधिक"}

# expert-system fact names and values
FACT_KEYS_HI = {
    "rainfall": "वर्षा", "temp": "तापमान", "pesticides": "कीटनाशक", "fertilizer": "उर्वरक",
    "predicted_yield": "अनुमानित उपज", "crop": "फसल", "season": "सीज़न",
    "drought_risk": "सूखे का जोखिम", "disease_risk": "रोग का जोखिम", "waterlogging_risk": "जलभराव का जोखिम",
    "heat_stress": "गर्मी का तनाव", "monsoon_deficit": "मानसून की कमी", "terminal_heat_risk": "अंतिम चरण में गर्मी का जोखिम",
}
FACT_VALUES_HI = {
    "low": "कम", "normal": "सामान्य", "high": "अधिक", "below_avg": "औसत से कम",
    "above_avg": "औसत से अधिक", "average": "औसत", True: "हाँ", "True": "हाँ",
}

UI = {  # key -> (English, Hindi)
    "advice_title": ("Expert-system advice", "विशेषज्ञ प्रणाली की सलाह"),
    "facts": ("Facts", "तथ्य"),
    "no_rule": ("No rule fired: conditions look unremarkable, so no special advice.",
                "कोई नियम लागू नहीं हुआ: स्थितियाँ सामान्य लगती हैं, इसलिए कोई विशेष सलाह नहीं।"),
    "trace": ("Reasoning trace (rules fired, in order)", "तर्क-क्रम (लागू हुए नियम, क्रम से)"),
    "if": ("IF", "यदि"), "then": ("THEN", "तो"), "and": ("AND", "और"),
    "why": ("Why this prediction? (SHAP)", "यह अनुमान क्यों? (SHAP)"),
    "waterfall_note": (
        "The starting value E[f(X)] is the average prediction across all crops, so a crop with naturally lower yield per hectare shows a negative contribution.",
        "शुरुआती मान E[f(X)] सभी फसलों का औसत अनुमान है, इसलिए जिस फसल की प्रति हेक्टेयर उपज स्वभाव से कम होती है, उसका योगदान ऋणात्मक दिखता है।"),
    "shap_caption": ("SHAP shows what the model associates with the prediction, not proven causes.",
                     "SHAP यह दिखाता है कि मॉडल किन बातों को अनुमान से जोड़ता है; यह सिद्ध कारण नहीं है।"),
}


def ui(key: str, lang: str = "en") -> str:
    """Look up a UI string in the chosen language ('en' or 'hi')."""
    en, hi = UI[key]
    return hi if lang == "hi" else en


def fact_text(key, value, lang: str = "en") -> str:
    """'rainfall = low' in English, or 'वर्षा = कम' in Hindi."""
    if lang != "hi":
        return f"{key} = {value}"
    return f"{FACT_KEYS_HI.get(key, key)} = {FACT_VALUES_HI.get(value, value)}"
