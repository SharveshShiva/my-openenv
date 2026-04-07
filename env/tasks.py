import json
import os
import random
from typing import List, Dict, Any

class TaskManager:
    def __init__(self, data_path: str = "data/claims.json"):
        self.data_path = data_path
        self.claims = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    full_path = os.path.join(base_dir, self.data_path)

    if not os.path.exists(full_path):
        from data_gen import generate_claims
        generate_claims()
        
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Dataset not found even after generation: {full_path}")

    with open(full_path, "r") as f:
        return json.load(f)
            
    def get_task(self, difficulty: str = "easy") -> List[Dict[str, Any]]:
        """Returns all claims matching the given difficulty."""
        matching_claims = [c for c in self.claims if c.get("difficulty") == difficulty]
        # In a real evaluation, we might shuffle or subset this, but for now we return them as a list
        return matching_claims

    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        tasks = {"easy": [], "medium": [], "hard": []}
        for c in self.claims:
            diff = c.get("difficulty")
            if diff in tasks:
                tasks[diff].append(c)
        return tasks
