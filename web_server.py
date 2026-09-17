import os
import json
from flask import Flask, render_template, request, jsonify

from src.data_pipeline import build_golden_dataset, save_datasets
from src.intent_classifier import DeltaIntentClassifier
from src.response_generator import DeltaResponseGenerator
from src.escalation_engine import DeltaEscalationEngine
from src.evaluator import DeltaEvaluator

app = Flask(__name__, template_folder="templates", static_folder="static")

# Global Agent Instances
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "intent_classifier.pkl")
REPORT_PATH = os.path.join(os.path.dirname(__file__), "reports", "benchmark_results.json")

def init_agent():
    golden_path = os.path.join(DATA_DIR, "golden_benchmark.json")
    if not os.path.exists(golden_path):
        save_datasets(DATA_DIR)
        
    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    clf = DeltaIntentClassifier()
    if os.path.exists(MODEL_PATH):
        clf.load_model(MODEL_PATH)
    else:
        clf.train(dataset)
        clf.save_model(MODEL_PATH)
        
    gen = DeltaResponseGenerator()
    esc = DeltaEscalationEngine()
    evaluator = DeltaEvaluator()
    return dataset, clf, gen, esc, evaluator

dataset, classifier, generator, escalation_engine, evaluator = init_agent()

@app.route("/")
def index():
    """Render Web Application Dashboard."""
    return render_template("index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    """API Endpoint for real-time customer query processing."""
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query text is required"}), 400
        
    intent, confidence = classifier.predict(query)
    all_probs = classifier.predict_all_probs(query)
    entities = generator.extract_context_entities(query)
    
    esc_res = escalation_engine.evaluate_query(
        query=query,
        intent=intent,
        confidence=confidence,
        complexity=data.get("complexity", "low"),
        urgency=data.get("urgency", "low"),
        sentiment=data.get("sentiment", "neutral")
    )
    
    reply = generator.generate_reply(
        query=query,
        intent=intent,
        sentiment=data.get("sentiment", "neutral"),
        escalation_status=esc_res["decision"],
        escalation_reason="; ".join(esc_res["reasons"])
    )
    
    judge_eval = evaluator.evaluate_llm_judge(reply, query, intent, reply)
    
    return jsonify({
        "query": query,
        "intent": intent,
        "confidence": confidence,
        "probabilities": all_probs,
        "entities": entities,
        "escalation": esc_res,
        "generated_reply": reply,
        "judge_eval": judge_eval
    })

@app.route("/api/benchmark", methods=["GET"])
def benchmark():
    """Return stored benchmark results or execute fresh benchmark."""
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
    else:
        report = evaluator.run_full_benchmark(dataset, classifier, generator, escalation_engine)
        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
    return jsonify(report)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "agent": "Delta Customer Support AI Agent v1.0.0"})

if __name__ == "__main__":
    print("Starting Delta AI Customer Support Web Application Server on http://127.0.0.1:5000...")
    app.run(host="127.0.0.1", port=5000, debug=True)
