import pytest
from src.data_pipeline import clean_tweet_text, build_golden_dataset, INTENT_TAXONOMY

def test_clean_tweet_text():
    raw = "  My   flight  DL102   is delayed!   "
    cleaned = clean_tweet_text(raw)
    assert cleaned == "My flight DL102 is delayed!"

def test_golden_dataset_structure():
    dataset = build_golden_dataset()
    assert len(dataset) >= 200
    
    first = dataset[0]
    assert "id" in first
    assert "query" in first
    assert "intent" in first
    assert first["intent"] in INTENT_TAXONOMY
    assert "expected_escalation" in first
    assert first["expected_escalation"] in ["AUTO_REPLY", "HUMAN_ESCALATION"]
