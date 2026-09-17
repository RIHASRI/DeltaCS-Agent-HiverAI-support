import pytest
from src.escalation_engine import DeltaEscalationEngine

def test_distress_escalation():
    engine = DeltaEscalationEngine()
    query = "Stuck on tarmac in MSP for 3 hours on DL1104, passengers are starving and need medical attention!"
    res = engine.evaluate_query(query, "flight_delay_cancellation", confidence=0.90)
    assert res["decision"] == "HUMAN_ESCALATION"
    assert len(res["flagged_keywords"]) > 0

def test_routine_auto_reply():
    engine = DeltaEscalationEngine()
    query = "Is DL601 operating on time from SEA to LAX today?"
    res = engine.evaluate_query(query, "flight_delay_cancellation", confidence=0.95, complexity="low", urgency="low", sentiment="neutral")
    assert res["decision"] == "AUTO_REPLY"
