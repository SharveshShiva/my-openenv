import json
import os
from typing import List, Dict, Any

class TaskManager:
    def __init__(self, data_path: str = "data/claims.json"):
        self.data_path = data_path
        self.claims = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        full_path = os.path.join(base_dir, self.data_path)

        if not os.path.exists(full_path):
            try:
                import sys
                sys.path.append(base_dir)

                from data_gen import generate_claims
                generate_claims()
            except Exception as e:
                raise RuntimeError(f"Failed to generate dataset: {e}")

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Dataset not found: {full_path}")

        try:
            with open(full_path, "r") as f:
                return json.load(f)
        except Exception as e:
            raise RuntimeError(f"Error loading dataset: {e}")

    def get_task(self, difficulty: str = "easy") -> List[Dict[str, Any]]:
        """Return claims filtered by difficulty."""
        return [c for c in self.claims if c.get("difficulty") == difficulty]

    def get_all_tasks(self) -> Dict[str, List[Dict[str, Any]]]:
        tasks = {"easy": [], "medium": [], "hard": []}
        for c in self.claims:
            diff = c.get("difficulty")
            if diff in tasks:
                tasks[diff].append(c)
        return tasks
