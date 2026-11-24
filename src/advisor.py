"""Rule-based expert system (knowledge representation) for farmer advice.

Knowledge is stored as DATA: each rule is a (name, conditions, conclusions) record.
An inference engine (forward / backward chaining) operates on a fact dictionary, so
adding a rule never requires touching the engine code.

Facts are simple key -> value pairs, e.g. {"rainfall": "low", "temp": "high"}.
"""
from dataclasses import dataclass, field, replace

import numpy as np

# Every conclusion starting with "advise_" is a piece of advice shown to the farmer;
# other conclusions (drought_risk, monsoon_deficit...) are intermediate facts that
# can trigger further rules - this is what makes chaining non-trivial.


@dataclass(frozen=True)
class Rule:
    """IF all `conditions` hold THEN add all `conclusions` to the facts."""
    name: str
    conditions: dict
    conclusions: dict
    text: str = ""            # human-readable advice / explanation (English)
    text_hi: str = ""         # the same, in simple Hindi


RULES = [
    # ---- general agronomy rules
    Rule("R1", {"rainfall": "low", "temp": "high"}, {"drought_risk": "high"},
         "Low rain + high temperature -> high drought risk"),
    Rule("R2", {"rainfall": "high", "temp": "high"}, {"disease_risk": "high"},
         "Warm and wet conditions favour fungal/pest outbreaks"),
    Rule("R3", {"drought_risk": "high"}, {"advise_irrigation": True},
         "Plan irrigation scheduling (e.g. drip) and choose drought-tolerant varieties."),
    Rule("R4", {"pesticides": "high", "predicted_yield": "below_avg"}, {"advise_pest_review": True},
         "Heavy pesticide use but low yield: review pest management (IPM, scouting, correct dose)."),
    Rule("R5", {"disease_risk": "high"}, {"advise_disease_monitoring": True},
         "Monitor fields for fungal disease and ensure good drainage and spacing."),
    Rule("R6", {"rainfall": "high", "predicted_yield": "below_avg"}, {"waterlogging_risk": "high"},
         "Excess rain with low expected yield suggests waterlogging."),
    Rule("R7", {"waterlogging_risk": "high"}, {"advise_drainage": True},
         "Improve field drainage and consider raised beds."),
    Rule("R8", {"temp": "high", "predicted_yield": "below_avg"}, {"heat_stress": "high"},
         "High temperature with low expected yield suggests heat stress."),
    Rule("R9", {"heat_stress": "high"}, {"advise_sowing_window": True},
         "Shift the sowing window or use heat-tolerant varieties; mulch to cool soil."),
    Rule("R10", {"predicted_yield": "above_avg", "rainfall": "normal", "temp": "normal"},
         {"advise_maintain": True},
         "Conditions look favourable: maintain current practice and record what works."),
    # ---- India-specific rules
    Rule("R11", {"season": "Kharif", "rainfall": "low"}, {"monsoon_deficit": True},
         "Low rainfall in the Kharif (monsoon) season = monsoon deficit."),
    Rule("R12", {"monsoon_deficit": True}, {"advise_water_conservation": True},
         "Adopt water conservation (mulching, farm ponds) and consult your local Krishi Vigyan Kendra (KVK)."),
    Rule("R13", {"fertilizer": "high", "predicted_yield": "below_avg"}, {"advise_soil_test": True},
         "Get a Soil Health Card test before adding more fertilizer."),
    Rule("R14", {"season": "Rabi", "temp": "high"}, {"terminal_heat_risk": True},
         "High temperature during Rabi (e.g. wheat grain filling) = terminal heat risk."),
    Rule("R15", {"terminal_heat_risk": True}, {"advise_early_sowing": True},
         "Sow wheat/Rabi crops early and irrigate at grain filling; use heat-tolerant varieties."),
    Rule("R16", {"drought_risk": "high", "predicted_yield": "below_avg"}, {"advise_insurance": True},
         "Consider enrolling in PMFBY (Pradhan Mantri Fasal Bima Yojana) crop insurance."),
]


# Simple-Hindi version of every rule text (keyed by rule name). Kept separate from RULES so the
# English knowledge base stays easy to read; merged into the Rule objects just below.
RULE_TEXT_HI = {
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
}
RULES = [replace(r, text_hi=RULE_TEXT_HI[r.name]) for r in RULES]


# ------------------------------------------------------------ discretisation
def discretize(value: float, low_cut: float, high_cut: float) -> str:
    """Map a number to 'low' / 'normal' / 'high' using dataset quantile cut-offs."""
    return "low" if value < low_cut else "high" if value > high_cut else "normal"


def build_facts(quantiles: dict, rainfall, temp=None, pesticides=None, fertilizer=None,
                predicted=None, crop_avg=None, crop=None, season=None, tol=0.10) -> dict:
    """Create the fact dictionary from raw numbers.

    quantiles : {"rainfall": (q33,q67), "temp": (..), "pesticides": (..), "fertilizer": (..)}
    predicted_yield becomes above/below/average relative to `crop_avg` (+/- tol).
    """
    f = {"rainfall": discretize(rainfall, *quantiles["rainfall"])}
    if temp is not None:
        f["temp"] = discretize(temp, *quantiles["temp"])
    if pesticides is not None:
        f["pesticides"] = discretize(pesticides, *quantiles["pesticides"])
    if fertilizer is not None:
        f["fertilizer"] = discretize(fertilizer, *quantiles["fertilizer"])
    if predicted is not None and crop_avg:
        r = predicted / crop_avg
        f["predicted_yield"] = "below_avg" if r < 1 - tol else "above_avg" if r > 1 + tol else "average"
    if crop:
        f["crop"] = crop
    if season:
        f["season"] = season
    return f


# ------------------------------------------------------------ inference engine
def _holds(rule: Rule, facts: dict) -> bool:
    """True if every condition of the rule matches the current facts."""
    return all(facts.get(k) == v for k, v in rule.conditions.items())


def forward_chain(facts: dict, rules=RULES):
    """Data-driven inference: keep firing rules until nothing new can be derived.

    Returns (all_facts, advice_list, trace) where trace lists each fired rule in
    order with the new facts it produced.
    """
    facts = dict(facts)
    fired, trace = set(), []
    changed = True
    while changed:                       # loop until a fixed point is reached
        changed = False
        for r in rules:
            if r.name in fired or not _holds(r, facts):
                continue
            new = {k: v for k, v in r.conclusions.items() if facts.get(k) != v}
            fired.add(r.name)
            if new:                      # fire only if it adds knowledge
                facts.update(new)
                trace.append({"rule": r.name, "because": dict(r.conditions),
                              "derived": new, "text": r.text, "text_hi": r.text_hi})
                changed = True
    advice = [t["text"] for t in trace if any(k.startswith("advise_") for k in t["derived"])]
    return facts, advice, trace


def backward_chain(goal: dict, facts: dict, rules=RULES, _seen=None) -> bool:
    """Goal-driven inference: can `goal` ({key: value}) be PROVEN from the facts?

    A goal is true if it is already a fact, or some rule concludes it and ALL of
    that rule's conditions can themselves be proven (recursively). `_seen` blocks
    infinite loops on cyclic rule sets.
    """
    _seen = set() if _seen is None else _seen
    (key, val), = goal.items()
    if facts.get(key) == val:
        return True
    if (key, val) in _seen:
        return False
    _seen = _seen | {(key, val)}
    for r in rules:
        if r.conclusions.get(key) == val:
            if all(backward_chain({k: v}, facts, rules, _seen) for k, v in r.conditions.items()):
                return True
    return False


def advise(facts: dict):
    """Convenience wrapper: forward-chain and return (advice, trace, all_facts)."""
    allf, advice, trace = forward_chain(facts)
    return advice, trace, allf


def advice_from_trace(trace: list, lang: str = "en") -> list:
    """Advice sentences (conclusions named advise_*) from a trace, in English or Hindi ('hi')."""
    key = "text_hi" if lang == "hi" else "text"
    return [t[key] for t in trace if any(k.startswith("advise_") for k in t["derived"])]
