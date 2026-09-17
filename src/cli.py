import os
import sys
import json
import argparse
from tabulate import tabulate

from src.data_pipeline import build_golden_dataset, save_datasets
from src.intent_classifier import DeltaIntentClassifier
from src.response_generator import DeltaResponseGenerator
from src.escalation_engine import DeltaEscalationEngine
from src.evaluator import DeltaEvaluator

def run_single_query(query: str, classifier, generator, escalation_engine):
    """Process a single query through the end-to-end AI Agent pipeline."""
    intent, confidence = classifier.predict(query)
    esc_res = escalation_engine.evaluate_query(query, intent, confidence)
    
    reply = generator.generate_reply(
        query=query,
        intent=intent,
        sentiment="neutral",
        escalation_status=esc_res['decision'],
        escalation_reason="; ".join(esc_res['reasons'])
    )
    
    print("\n" + "="*70)
    print("DELTA AI CUSTOMER SUPPORT AGENT - SINGLE QUERY EVALUATION")
    print("="*70)
    print(f"Customer Query  : {query}")
    print(f"Predicted Intent: {intent} (Confidence: {confidence:.2%})")
    print(f"Escalation      : {esc_res['decision']} (Risk Score: {esc_res['risk_score']:.2f})")
    print("Escalation Log  :")
    for r in esc_res['reasons']:
        print(f"  - {r}")
    print("-" * 70)
    print("Generated Reply :")
    print(reply)
    print("="*70 + "\n")

def run_full_evaluation():
    """Run full pipeline evaluation against golden benchmark dataset."""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    golden_path = os.path.join(data_dir, "golden_benchmark.json")
    
    if not os.path.exists(golden_path):
        print("Generating benchmark dataset...")
        save_datasets(data_dir)
        
    with open(golden_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    print(f"Loaded {len(dataset)} examples from {golden_path}")
    print("Initializing components...")
    
    classifier = DeltaIntentClassifier()
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "intent_classifier.pkl")
    if os.path.exists(model_path):
        classifier.load_model(model_path)
    else:
        classifier.train(dataset)
        classifier.save_model(model_path)
        
    generator = DeltaResponseGenerator()
    escalation_engine = DeltaEscalationEngine()
    evaluator = DeltaEvaluator()
    
    print("Executing benchmark evaluation pipeline...")
    report = evaluator.run_full_benchmark(dataset, classifier, generator, escalation_engine)
    
    # Save Report Output
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_json_path = os.path.join(reports_dir, "benchmark_results.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*70)
    print("DELTA AI AGENT - BENCHMARK EVALUATION SUMMARY REPORT")
    print("="*70)
    
    intent_metrics = report['intent_classification']
    print(f"\n1. INTENT CLASSIFIER PERFORMANCE ({report['total_benchmark_samples']} SAMPLES)")
    print(f"   Accuracy           : {intent_metrics['accuracy']:.4f} ({intent_metrics['accuracy']*100:.1f}%)")
    print(f"   Macro Precision    : {intent_metrics['macro_precision']:.4f}")
    print(f"   Macro Recall       : {intent_metrics['macro_recall']:.4f}")
    print(f"   Macro F1-Score     : {intent_metrics['macro_f1']:.4f}")
    
    print("\n   PER-CLASS INTENT METRICS:")
    headers = ["Intent Category", "Precision", "Recall", "F1-Score", "Support"]
    table_data = []
    for cls, m in intent_metrics['per_class'].items():
        table_data.append([cls, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1']:.3f}", m['support']])
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n2. ESCALATION ROUTING ENGINE")
    print(f"   Escalation Accuracy: {report['escalation_accuracy']:.4f} ({report['escalation_accuracy']*100:.1f}%)")
    
    print("\n3. GENERATED RESPONSE QUALITY METRICS")
    print(f"   Mean ROUGE-L       : {report['text_quality_metrics']['mean_rouge_l']:.4f}")
    print(f"   Mean BLEU-1        : {report['text_quality_metrics']['mean_bleu_1']:.4f}")
    print(f"   Mean Jaccard Sim   : {report['text_quality_metrics']['mean_jaccard_similarity']:.4f}")
    print(f"   LLM Judge Score    : {report['llm_judge_metrics']['mean_overall_score']:.2f} / 5.00")
    
    print("\n4. INTER-RATER AGREEMENT (HUMAN VS LLM JUDGE)")
    agr = report['inter_rater_agreement']
    print(f"   Pearson Correlation (r): {agr['pearson_r']:.4f}")
    print(f"   Mean Absolute Error    : {agr['mean_absolute_error']:.4f}")
    print(f"   Exact Agreement Ratio  : {agr['exact_agreement_ratio']:.4f}")
    print(f"   Cohen's Kappa Proxy    : {agr['cohens_kappa_proxy']:.4f}")
    print("="*70)
    print(f"Full JSON metrics report saved to: {report_json_path}\n")

def run_interactive_shell(classifier, generator, escalation_engine):
    """Launch interactive shell for testing user queries in real-time."""
    print("\n" + "="*70)
    print("WELCOME TO DELTA AIR LINES (@Delta) AI CUSTOMER SUPPORT CLI")
    print("Type 'exit' or 'quit' to end session.")
    print("="*70 + "\n")
    
    while True:
        try:
            query = input("Customer Tweet > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Exiting Delta AI Agent CLI. Goodbye!")
                break
            run_single_query(query, classifier, generator, escalation_engine)
        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delta Air Lines AI Customer Support Agent CLI")
    parser.add_argument("--mode", choices=["evaluate", "interactive", "single"], default="evaluate", help="Execution mode")
    parser.add_argument("--query", type=str, help="Single query text to evaluate")
    args = parser.parse_args()
    
    if args.mode == "evaluate":
        run_full_evaluation()
    else:
        # Load models for interactive / single query
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        golden_path = os.path.join(data_dir, "golden_benchmark.json")
        if not os.path.exists(golden_path):
            save_datasets(data_dir)
        with open(golden_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)
            
        clf = DeltaIntentClassifier()
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "intent_classifier.pkl")
        if os.path.exists(model_path):
            clf.load_model(model_path)
        else:
            clf.train(dataset)
            clf.save_model(model_path)
            
        gen = DeltaResponseGenerator()
        esc = DeltaEscalationEngine()
        
        if args.mode == "single":
            q = args.query if args.query else "DL402 was delayed and I lost my baggage tag #DL98122!"
            run_single_query(q, clf, gen, esc)
        else:
            run_interactive_shell(clf, gen, esc)
