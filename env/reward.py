from typing import Dict, Any, List
from .models import Action

def match_fraud_indicators(reasoning: str, true_indicators: List[str]) -> float:
    """Returns a score ratio between 0.0 and 1.0 based on how many true indicators are mentioned."""
    if not true_indicators:
        return 1.0 # No fraud indicators to mention
        
    reasoning_lower = reasoning.lower()
    matches = 0
    for indicator in true_indicators:
        parts = indicator.split("_")
        if indicator in reasoning_lower or all(p in reasoning_lower for p in parts):
            matches += 1
            
    return matches / len(true_indicators)

def calculate_reward(action: Action, true_data: Dict[str, Any]) -> float:
    score = 0.0
    details = {}
    
    # 1. Decision Match (0.5 max)
    true_decision = true_data.get("true_decision", "")
    if action.decision == true_decision:
        score += 0.5
        details["decision_score"] = 0.5
    else:
        details["decision_score"] = 0.0
        
    # 2. Fraud Score Match (0.2 max)
    true_fraud_score = true_data.get("fraud_score", 0.0)
    fraud_diff = abs(action.fraud_score - true_fraud_score)
    fraud_reward = max(0.0, 0.2 * (1.0 - fraud_diff))
    score += fraud_reward
    details["fraud_score_score"] = fraud_reward
    
    # 3. Reasoning Match (up to 0.3)
    reasoning = action.reasoning.lower().strip()
    reasoning_score = 0.0
    
    # Penalties for poorly constructed reasoning
    if not reasoning or len(reasoning) < 20:
        # short reasoning
        reasoning_score -= 0.2
    elif len(reasoning) > 1000:
        # spam reasoning
        reasoning_score -= 0.2
    else:
        # Base credit for having substantial valid length text
        reasoning_score += 0.1
        
        fraud_indicators = true_data.get("fraud_indicators", [])
        
        # Check against irrelevant spamming of fallback phrases (Improved Spam Detection)
        spam_indicators = [
            "using fallback reasoning",
            "json parsing failed",
            "model error occurred",
            "unable to parse",
            "using fallback logic"
        ]
        
        if any(indicator in reasoning for indicator in spam_indicators):
            reasoning_score -= 0.3
        else:
            if not fraud_indicators:
                good_keywords = ["normal", "consistent", "routine", "valid", "legitimate", "low risk", "standard"]
                matches = sum(1 for k in good_keywords if k in reasoning)
                if matches > 0:
                    reasoning_score += min(0.2, 0.1 * matches)
            else:
                ratio = match_fraud_indicators(reasoning, fraud_indicators)
                if ratio > 0:
                    reasoning_score += 0.2 * ratio
                else:
                    reasoning_score -= 0.1

    score += reasoning_score
    details["reasoning_score"] = reasoning_score
    
    # Ensure score is strictly bounded [0.0, 1.0] absolutely
    final_score = max(0.0, min(1.0, score))
    return float(final_score)
