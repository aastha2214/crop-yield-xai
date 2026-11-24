"""Unit tests for the expert system."""
from src import advisor as A


def test_discretize():
    assert A.discretize(5, 10, 20) == "low"
    assert A.discretize(15, 10, 20) == "normal"
    assert A.discretize(25, 10, 20) == "high"


def test_chaining_drought():
    facts, advice, trace = A.forward_chain({"rainfall": "low", "temp": "high"})
    assert facts["drought_risk"] == "high"
    assert any("irrigation" in a.lower() for a in advice)
    assert [t["rule"] for t in trace][:2] == ["R1", "R3"]          # R1 must precede R3


def test_monsoon_deficit_chain():
    facts, advice, _ = A.forward_chain({"season": "Kharif", "rainfall": "low"})
    assert facts["monsoon_deficit"] is True
    assert any("Krishi Vigyan Kendra" in a for a in advice)


def test_fertilizer_and_pesticide_rules():
    _, advice, _ = A.forward_chain({"fertilizer": "high", "predicted_yield": "below_avg"})
    assert any("Soil Health Card" in a for a in advice)
    _, advice, _ = A.forward_chain({"pesticides": "high", "predicted_yield": "below_avg"})
    assert any("pest management" in a for a in advice)


def test_no_rule_fires():
    facts, advice, trace = A.forward_chain({"rainfall": "normal", "temp": "normal"})
    assert advice == [] and trace == []


def test_forward_chain_does_not_mutate_input():
    f = {"rainfall": "low", "temp": "high"}
    A.forward_chain(f)
    assert f == {"rainfall": "low", "temp": "high"}


def test_backward_chain():
    f = {"rainfall": "low", "temp": "high"}
    assert A.backward_chain({"advise_irrigation": True}, f)         # needs 2-step proof
    assert not A.backward_chain({"advise_soil_test": True}, f)
    assert A.backward_chain({"rainfall": "low"}, f)                 # already a fact


def test_backward_chain_terminates_on_cycle():
    cyc = [A.Rule("a", {"x": 1}, {"y": 1}), A.Rule("b", {"y": 1}, {"x": 1})]
    assert not A.backward_chain({"y": 1}, {}, cyc)


def test_build_facts():
    q = {"rainfall": (500, 1500), "temp": (10, 20)}
    f = A.build_facts(q, rainfall=100, temp=25, predicted=70, crop_avg=100, season="Kharif")
    assert f["rainfall"] == "low" and f["temp"] == "high"
    assert f["predicted_yield"] == "below_avg" and f["season"] == "Kharif"


def test_rules_are_data():
    assert 12 <= len(A.RULES) <= 18
    assert all(isinstance(r.conditions, dict) and r.conclusions for r in A.RULES)
