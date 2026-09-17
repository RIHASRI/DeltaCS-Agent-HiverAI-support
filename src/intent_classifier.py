import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold

from src.data_pipeline import INTENT_TAXONOMY, clean_tweet_text

class DeltaIntentClassifier:
    """
    Multi-class Intent Classifier for Delta Air Lines Customer Support queries.
    Uses an n-gram TF-IDF vectorizer paired with a calibrated Logistic Regression classifier.
    """
    def __init__(self):
        self.classes_ = list(INTENT_TAXONOMY.keys())
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=5000,
                sublinear_tf=True,
                strip_accents='unicode'
            )),
            ('clf', LogisticRegression(
                C=2.0,
                max_iter=500,
                class_weight='balanced',
                random_state=42
            ))
        ])
        self.is_trained = False

    def train(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train the classifier on a list of annotated query dictionaries."""
        texts = [clean_tweet_text(d['query']) for d in data]
        labels = [d['intent'] for d in data]
        
        self.pipeline.fit(texts, labels)
        self.is_trained = True
        
        # Evaluate on training dataset (self-consistency check)
        preds = self.pipeline.predict(texts)
        acc = accuracy_score(labels, preds)
        return {"training_accuracy": float(acc), "sample_count": len(data)}

    def cross_validate(self, data: List[Dict[str, Any]], n_splits: int = 5) -> Dict[str, Any]:
        """Perform stratified k-fold cross validation and return aggregated evaluation metrics."""
        texts = np.array([clean_tweet_text(d['query']) for d in data])
        labels = np.array([d['intent'] for d in data])
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_y_true = []
        all_y_pred = []
        
        for train_idx, test_idx in skf.split(texts, labels):
            X_tr, y_tr = texts[train_idx], labels[train_idx]
            X_te, y_te = texts[test_idx], labels[test_idx]
            
            fold_pipe = Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 3), max_features=5000, sublinear_tf=True)),
                ('clf', LogisticRegression(C=2.0, max_iter=500, class_weight='balanced', random_state=42))
            ])
            fold_pipe.fit(X_tr, y_tr)
            preds = fold_pipe.predict(X_te)
            
            all_y_true.extend(y_te)
            all_y_pred.extend(preds)
            
        metrics = self.compute_metrics(all_y_true, all_y_pred)
        return metrics

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict top intent and confidence score for a single customer query text.
        Returns (predicted_intent, confidence_score).
        """
        if not self.is_trained:
            # Fallback heuristic rule-based intent matching if not yet fit
            return self._heuristic_intent(text)
            
        clean_text = clean_tweet_text(text)
        probs = self.pipeline.predict_proba([clean_text])[0]
        top_idx = int(np.argmax(probs))
        predicted_intent = self.pipeline.classes_[top_idx]
        confidence = float(probs[top_idx])
        return predicted_intent, confidence

    def predict_all_probs(self, text: str) -> Dict[str, float]:
        """Return probability distribution over all intent classes."""
        if not self.is_trained:
            return {cls: 1.0 / len(self.classes_) for cls in self.classes_}
        clean_text = clean_tweet_text(text)
        probs = self.pipeline.predict_proba([clean_text])[0]
        return {cls: float(p) for cls, p in zip(self.pipeline.classes_, probs)}

    def compute_metrics(self, y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
        """Compute full classification metrics: Accuracy, Precision, Recall, F1, Confusion Matrix."""
        acc = accuracy_score(y_true, y_pred)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        labels = sorted(list(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        per_class = {}
        p_c, r_c, f1_c, sup_c = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
        for i, l in enumerate(labels):
            per_class[l] = {
                "precision": float(p_c[i]),
                "recall": float(r_c[i]),
                "f1": float(f1_c[i]),
                "support": int(sup_c[i])
            }
            
        return {
            "accuracy": float(acc),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(p_weighted),
            "weighted_recall": float(r_weighted),
            "weighted_f1": float(f1_weighted),
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "per_class": per_class
        }

    def _heuristic_intent(self, text: str) -> Tuple[str, float]:
        """Fallback keyword heuristic classifier."""
        txt = text.lower()
        if any(w in txt for w in ["delay", "cancel", "late", "reschedule", "tarmac", "stuck"]):
            return "flight_delay_cancellation", 0.85
        elif any(w in txt for w in ["bag", "luggage", "suitcase", "carousel", "tag", "damaged"]):
            return "baggage_issue", 0.85
        elif any(w in txt for w in ["seat", "upgrade", "first class", "comfort+", "typo", "ktn"]):
            return "booking_seat_change", 0.85
        elif any(w in txt for w in ["refund", "ecredit", "voucher", "charge", "reimburs", "downgrad"]):
            return "refund_compensation", 0.85
        elif any(w in txt for w in ["check", "boarding", "pass", "ssss", "kiosk", "gate"]):
            return "checkin_boarding", 0.85
        else:
            return "general_inquiry", 0.70

    def save_model(self, file_path: str):
        """Save trained model pipeline to pickle file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self.pipeline, f)

    def load_model(self, file_path: str):
        """Load trained model pipeline from pickle file."""
        with open(file_path, "rb") as f:
            self.pipeline = pickle.load(f)
        self.is_trained = True

if __name__ == "__main__":
    from src.data_pipeline import build_golden_dataset
    dataset = build_golden_dataset()
    classifier = DeltaIntentClassifier()
    cv_res = classifier.cross_validate(dataset, n_splits=5)
    print("Cross Validation Results on Golden Dataset:")
    print(f"Accuracy: {cv_res['accuracy']:.4f}")
    print(f"Macro F1: {cv_res['macro_f1']:.4f}")
    
    classifier.train(dataset)
    save_p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "intent_classifier.pkl")
    classifier.save_model(save_p)
    print(f"Model saved to {save_p}")
