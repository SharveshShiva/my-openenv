from typing import Dict, Any, Tuple, Optional
from pydantic import ValidationError
from .models import ClaimData, Observation, Action
from .tasks import TaskManager
from .reward import calculate_reward

class InsuranceEnvironment:
    def __init__(self):
        self.task_manager = TaskManager()
        self.current_task_name: str = "easy"
        self.claims: list = []
        self.current_index: int = 0
        self.total_reward: float = 0.0

    def reset(self, task_name: str = "easy") -> Dict[str, Any]:
        """Resets the environment for a specific task difficulty."""
        if task_name not in ["easy", "medium", "hard"]:
            task_name = "easy"
            
        self.current_task_name = task_name
        self.claims = self.task_manager.get_task(task_name)
        self.current_index = 0
        self.total_reward = 0.0
        
        return {
            "status": "reset_successful",
            "task_name": self.current_task_name,
            "total_claims": len(self.claims)
        }

    def state(self) -> Optional[Observation]:
        """Returns the current observation."""
        if self.current_index >= len(self.claims):
            return None
            
        current_claim_data = self.claims[self.current_index]
        
        # Remove hidden fields
        safe_claim_data = {k: v for k, v in current_claim_data.items() 
                           if k not in ["true_decision", "fraud_score", "fraud_indicators", "difficulty"]}
                           
        try:
            claim_obj = ClaimData(**safe_claim_data)
        except ValidationError as e:
            # Handle potential internal validation errors by converting to a generic layout if necessary
            # For simplicity, we assume data generation provides perfectly matching valid data
            raise e
            
        return Observation(claim=claim_obj, difficulty=self.current_task_name)

    def step(self, action: Action) -> Tuple[Optional[Dict[str, Any]], float, bool, Dict[str, Any]]:
        """Processes the agent's action and advances the environment."""
        if self.current_index >= len(self.claims):
            return None, 0.0, True, {"error": "Environment is already done."}
            
        current_claim_data = self.claims[self.current_index]
        
        # Calculate Reward
        reward_val = calculate_reward(action, current_claim_data)
        self.total_reward += reward_val
        
        info = {
            "claim_id": current_claim_data.get("claim_id"),
            "true_decision": current_claim_data.get("true_decision"),
            "predicted_decision": action.decision,
            "true_fraud_score": current_claim_data.get("fraud_score"),
            "predicted_fraud_score": action.fraud_score,
            "reward": reward_val
        }
        
        # Advance state
        self.current_index += 1
        done = self.current_index >= len(self.claims)
        
        if not done:
            next_obs = self.state()
            next_obs_dict = next_obs.model_dump() if next_obs else None
        else:
            next_obs_dict = None
            
        return next_obs_dict, reward_val, done, info
