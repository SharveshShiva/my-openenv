from typing import Dict, Any, List
from .models import Action


def match_fraud_indicators(reasoning: str, true_indicators: List[str]) -> float:
    """Returns ratio (0.0 to 1.0) of matched fraud indicators."""
    if not true_indicators:
        return 1.0

    reasoning_lower = reasoning.lower()
    matches = 0

    for indicator in true_indicators:
        parts = indicator.split("_")
        if indicator in reasoning_lower or all(p in reasoning_lower for p in parts):
            matches += 1

    return matches / len(true_indicators)


def calculate_reward(action: Action, true_data: Dict[str, Any]) -> float:
    score = 0.0
    true_decision = true_data.get("true_decision", "")
    if action.decision == true_decision:
        score += 0.5
    true_fraud_score = true_data.get("fraud_score", 0.0)

    try:
        fraud_diff = abs(float(action.fraud_score) - float(true_fraud_score))
        fraud_reward = max(0.0, 0.2 * (1.0 - fraud_diff))
    except Exception:
        fraud_reward = 0.1 

    score += fraud_reward
    reasoning = str(action.reasoning).lower().strip()
    reasoning_score = 0.0

    if not reasoning or len(reasoning) < 20:
        reasoning_score -= 0.05

    elif len(reasoning) > 800:
        reasoning_score -= 0.05

    else:

        reasoning_score += 0.1

        fraud_indicators = true_data.get("fraud_indicators", [])
        spam_phrases = [
            "fallback",
            "error",
            "unable",
            "failed",
            "json"
        ]

        if any(s in reasoning for s in spam_phrases):
            reasoning_score -= 0.1
        else:
            if not fraud_indicators:
                good_keywords = ["normal", "valid", "routine", "consistent", "legitimate"]
                matches = sum(1 for k in good_keywords if k in reasoning)
                reasoning_score += min(0.2, 0.05 * matches)

            else:
                ratio = match_fraud_indicators(reasoning, fraud_indicators)
                reasoning_score += 0.2 * ratio

    score += reasoning_score
    final_score = max(0.01, min(0.99, score))

    return float(final_score)
