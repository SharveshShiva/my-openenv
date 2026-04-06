import json
import random
import os

def generate_claims():
    random.seed(42)
    claims = []
    
    first_names = ["John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Frank"]
    last_names = ["Smith", "Doe", "Johnson", "Brown", "Williams", "Jones", "Miller", "Davis"]
    
    treatments = {
        "easy": [("Routine Checkup", 150), ("Blood Test", 50), ("X-Ray", 200)],
        "medium": [("Physical Therapy", 1500), ("Dental Surgery", 2000), ("MRI Scan", 1200)],
        "hard": [("Cardiac Bypass", 50000), ("Spinal Fusion", 40000), ("Cancer Treatment", 80000)]
    }
    
    hospitals = ["City General", "Mercy Hospital", "St. Jude", "County Medical Center"]
    
    def get_noise(text, amount=0.1):
        if random.random() > amount: return text
        return text.replace("e", "3").replace("a", "@") if random.random() > 0.5 else text.lower()

    for i in range(100):
        # 30 easy, 40 medium, 30 hard
        if i < 30:
            difficulty = "easy"
        elif i < 70:
            difficulty = "medium"
        else:
            difficulty = "hard"
            
        is_fraud = random.random() > 0.7 # 30% fraud rate overall
            
        treatment_list = treatments[difficulty]
        treatment_name, base_cost = random.choice(treatment_list)
        
        # Calculate realistic cost with or without fraud
        if is_fraud:
            claim_amount = round(base_cost * random.uniform(1.8, 3.5), 2)
            fraud_score = random.uniform(0.7, 1.0)
            true_decision = "REJECT"
            fraud_indicators = ["inflated_cost"]
            if difficulty == "hard" and random.random() > 0.5:
                fraud_indicators.append("duplicate_billing")
            if difficulty == "medium" and random.random() > 0.5:
                fraud_indicators.append("unnecessary_treatment")
        else:
            claim_amount = round(base_cost * random.uniform(0.9, 1.1), 2)
            fraud_score = random.uniform(0.0, 0.3)
            true_decision = "APPROVE"
            fraud_indicators = []
            
        # Add edge cases for medium/hard
        if difficulty != "easy" and random.random() > 0.8 and not is_fraud:
            # Ambiguous but not fraud
            true_decision = "ESCALATE"
            fraud_score = random.uniform(0.4, 0.6)
            fraud_indicators = ["missing_documentation"]

        if difficulty == "easy" and not is_fraud and random.random() > 0.8:
            # Policy too short edge case
            policy_duration = random.randint(1, 2)
            true_decision = "REJECT"
            fraud_score = random.uniform(0.1, 0.3)
        else:
            policy_duration = random.randint(12, 120)

        history_claims = random.randint(0, 5)
        if is_fraud and random.random() > 0.5:
            history_claims = random.randint(10, 20)
            if "excessive_history" not in fraud_indicators:
                fraud_indicators.append("excessive_history")

        claim = {
            "claim_id": f"CLM-{1000 + i}",
            "difficulty": difficulty,
            "patient_name": f"{random.choice(first_names)} {random.choice(last_names)}",
            "age": random.randint(18, 85),
            "treatment": get_noise(treatment_name, 0.2),
            "hospital": get_noise(random.choice(hospitals), 0.1),
            "claim_amount": claim_amount,
            "policy_duration_months": policy_duration,
            "documents_submitted": ["invoice", "medical_report"] if "missing_documentation" not in fraud_indicators else ["invoice"],
            "history_claims_count": history_claims,
            "true_decision": true_decision,
            "fraud_score": round(fraud_score, 2),
            "fraud_indicators": fraud_indicators
        }
        claims.append(claim)
        
    os.makedirs("data", exist_ok=True)
    with open("data/claims.json", "w") as f:
        json.dump(claims, f, indent=4)

if __name__ == "__main__":
    generate_claims()
    print("Claims generated in data/claims.json")
