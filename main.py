import sys
import os

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Delta Air Lines AI Customer Support Agent Entrypoint")
    parser.add_argument("--mode", choices=["evaluate", "interactive", "single"], default="evaluate", help="Execution mode")
    parser.add_argument("--query", type=str, help="Single query text")
    args = parser.parse_args()
    
    # Forward args to cli
    sys.argv = [sys.argv[0], "--mode", args.mode]
    if args.query:
        sys.argv.extend(["--query", args.query])
        
    from src.cli import run_full_evaluation, run_single_query, run_interactive_shell
    from src.data_pipeline import save_datasets, build_golden_dataset
    from src.intent_classifier import DeltaIntentClassifier
    from src.response_generator import DeltaResponseGenerator
    from src.escalation_engine import DeltaEscalationEngine
    
    if args.mode == "evaluate":
        run_full_evaluation()
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        golden_path = os.path.join(data_dir, "golden_benchmark.json")
        if not os.path.exists(golden_path):
            save_datasets(data_dir)
        with open(golden_path, "r", encoding="utf-8") as f:
            import json
            dataset = json.load(f)
            
        clf = DeltaIntentClassifier()
        model_path = os.path.join(os.path.dirname(__file__), "models", "intent_classifier.pkl")
        if os.path.exists(model_path):
            clf.load_model(model_path)
        else:
            clf.train(dataset)
            clf.save_model(model_path)
            
        gen = DeltaResponseGenerator()
        esc = DeltaEscalationEngine()
        
        if args.mode == "single":
            q = args.query if args.query else "My flight DL1421 from ATL to JFK is delayed by 4 hours!"
            run_single_query(q, clf, gen, esc)
        else:
            run_interactive_shell(clf, gen, esc)
