import pytest
from src.intent_classifier import DeltaIntentClassifier
from src.data_pipeline import build_golden_dataset

@pytest.fixture
def trained_classifier():
    dataset = build_golden_dataset()
    clf = DeltaIntentClassifier()
    clf.train(dataset)
    return clf

def test_predict_delay(trained_classifier):
    query = "My flight DL1421 is delayed by 4 hours in Atlanta!"
    intent, conf = trained_classifier.predict(query)
    assert intent == "flight_delay_cancellation"
    assert conf > 0.30

def test_predict_baggage(trained_classifier):
    query = "My checked suitcase was lost on flight DL502 tag #DL98122!"
    intent, conf = trained_classifier.predict(query)
    assert intent == "baggage_issue"
    assert conf > 0.30

def test_metrics_computation(trained_classifier):
    y_true = ["baggage_issue", "flight_delay_cancellation"]
    y_pred = ["baggage_issue", "flight_delay_cancellation"]
    metrics = trained_classifier.compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
