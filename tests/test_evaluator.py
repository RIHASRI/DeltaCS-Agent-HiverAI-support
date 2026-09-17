import pytest
from src.evaluator import DeltaEvaluator

def test_jaccard_and_rouge():
    evaluator = DeltaEvaluator()
    ref = "Hello! We apologize for the flight delay. Please check the Fly Delta app."
    cand = "Hello! We apologize for your delayed flight. Check Fly Delta app."
    
    jaccard = evaluator.compute_jaccard_similarity(cand, ref)
    rouge = evaluator.compute_rouge_l(cand, ref)
    
    assert jaccard > 0.40
    assert rouge > 0.50

def test_llm_judge_rubric():
    evaluator = DeltaEvaluator()
    ref = "Hello! Check delta.com/rebook for options."
    cand = "Hello! We apologize. You can check http://www.delta.com/rebook for rebooking."
    query = "DL102 delayed!"
    
    scores = evaluator.evaluate_llm_judge(cand, query, "flight_delay_cancellation", ref)
    assert scores["overall_score"] >= 4.0
    assert scores["relevance"] == 5.0
    assert scores["tone_empathy"] == 5.0
