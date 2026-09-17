# Delta Air Lines (@Delta) Customer Support AI Agent

An end-to-end, production-ready AI Customer Support Agent pipeline for Delta Air Lines (`@Delta`), built on Twitter customer support conversation paradigms.

## Features

- **Intent Classifier**: Multi-class intent prediction (`flight_delay_cancellation`, `baggage_issue`, `booking_seat_change`, `refund_compensation`, `checkin_boarding`, `general_inquiry`) with full evaluation metrics (Accuracy, F1, Precision, Recall, Confusion Matrix).
- **Brand Reply Generator**: Intent-conditioned, tone-aligned response generation reflecting Delta's empathetic, professional, and actionable brand voice.
- **Rule-Based Escalation Engine**: Automated routing between `AUTO_REPLY` and `HUMAN_ESCALATION` based on query complexity, sentiment, and policy risk, with explicit structured audit logs.
- **Evaluation Harness & LLM-as-a-Judge**: Multi-metric evaluation framework computing lexical (BLEU, ROUGE), semantic similarity, LLM-as-a-judge scores across 4 key rubrics (Relevance, Tone, Actionability, Safety), and inter-rater agreement (Cohen's Kappa, Pearson, MAE).
- **Interactive CLI & Benchmark Runner**: Command-line application for live query simulation, batch evaluation, and benchmark generation.
- **Engineering Decision Logs & Technical Report**: In-depth architecture, trade-off analysis, failure mode breakdown, and misleading metrics post-mortem.

## Quick Start

```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline benchmark
python main.py --mode evaluate

# Launch interactive CLI session
python main.py --mode interactive

# Run test suite
pytest
```

## Project Structure

```
├── data/
│   ├── delta_tweets.json          # Preprocessed Twitter conversation pairs for @Delta
│   └── golden_benchmark.json      # Curated 200-example golden benchmark dataset
├── src/
│   ├── data_pipeline.py           # Preprocessing & golden dataset loader
│   ├── intent_classifier.py       # Multi-class intent classifier & trainer
│   ├── response_generator.py      # Delta brand reply generator
│   ├── escalation_engine.py       # Escalation matrix & routing engine
│   ├── evaluator.py               # Automated metrics & LLM-as-judge harness
│   └── cli.py                     # Interactive CLI app
├── tests/                         # Pytest test suite
├── reports/
│   ├── engineering_decision_log.md
│   └── technical_report.md
├── main.py                        # Main CLI entrypoint
└── requirements.txt               # Dependencies
```
