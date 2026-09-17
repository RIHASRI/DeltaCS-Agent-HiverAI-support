import re
from typing import Dict, Any, List, Tuple

class DeltaEscalationEngine:
    """
    Rule-based escalation & safety decision engine for Delta Air Lines Customer Support.
    Determines whether a query can be handled via AUTO_REPLY or requires HUMAN_ESCALATION.
    """
    
    # Severe keywords triggering safety or human escalation
    DISTRESS_KEYWORDS = [
        "medical", "prescription", "medication", "doctor", "tarmac", "stuck on tarmac",
        "toddler", "infant", "wheelchair", "car seat", "stroller", "rude", "complaint",
        "lawyer", "legal", "dot complaint", "unacceptable", "starving", "emergency"
    ]
    
    COMPLEX_OPERATIONS = [
        "name change", "split reservation", "pnr split", "downgrade", "downgraded",
        "double charge", "charged twice", "reimbursement", "hotel cost", "out of pocket"
    ]

    def evaluate_query(
        self,
        query: str,
        intent: str,
        confidence: float,
        complexity: str = "low",
        urgency: str = "low",
        sentiment: str = "neutral"
    ) -> Dict[str, Any]:
        """
        Evaluate query against multi-dimensional risk matrix and return escalation decision.
        Returns:
            {
                "decision": "HUMAN_ESCALATION" | "AUTO_REPLY",
                "risk_score": float (0.0 to 1.0),
                "reasons": List[str],
                "flagged_keywords": List[str]
            }
        """
        reasons = []
        flagged_kw = []
        risk_score = 0.0
        txt = query.lower()
        
        # 1. Check Distress & Safety Keywords
        for kw in self.DISTRESS_KEYWORDS:
            if kw in txt:
                flagged_kw.append(kw)
        
        if flagged_kw:
            risk_score += 0.50
            reasons.append(f"Contains high-risk distress/safety keywords: {', '.join(flagged_kw)}")
            
        # 2. Check Complex Operational Actions
        found_ops = [op for op in self.COMPLEX_OPERATIONS if op in txt]
        if found_ops:
            risk_score += 0.35
            reasons.append(f"Requires complex manual GDS/billing action: {', '.join(found_ops)}")
            
        # 3. Intent Classifier Confidence Check
        if confidence < 0.60:
            risk_score += 0.30
            reasons.append(f"Low intent classification confidence ({confidence:.2f} < 0.60)")
            
        # 4. Sentiment & Frustration Check
        if sentiment == "frustrated":
            risk_score += 0.25
            reasons.append("Customer sentiment evaluated as highly frustrated")
            
        # 5. Complexity & Urgency Check
        if complexity == "high":
            risk_score += 0.25
            reasons.append("High query complexity annotation")
        if urgency == "high":
            risk_score += 0.20
            reasons.append("High urgency annotation")
            
        # Final Decision Threshold
        is_escalated = risk_score >= 0.45 or len(reasons) >= 2 or complexity == "high"
        decision = "HUMAN_ESCALATION" if is_escalated else "AUTO_REPLY"
        
        if not reasons:
            reasons.append("Query matches standard automated resolution workflow")

        return {
            "decision": decision,
            "risk_score": min(float(risk_score), 1.0),
            "reasons": reasons,
            "flagged_keywords": flagged_kw
        }

if __name__ == "__main__":
    engine = DeltaEscalationEngine()
    test_q = "My luggage has my prescription medication and was lost on DL502! I need it urgently!"
    result = engine.evaluate_query(test_q, "baggage_issue", 0.92, complexity="high", urgency="high", sentiment="frustrated")
    print("Escalation Evaluation Test:")
    print(f"Decision: {result['decision']}")
    print(f"Risk Score: {result['risk_score']}")
    print("Reasons:")
    for r in result['reasons']:
        print(f"- {r}")
