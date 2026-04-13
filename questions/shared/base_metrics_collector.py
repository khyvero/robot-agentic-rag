"""Base class for all research question metrics collectors."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import json
from pathlib import Path


class BaseMetricsCollector(ABC):
    # base class for all rq metrics collectors

    def __init__(self, rq_name: str):
        self.rq_name = rq_name
        self.results: List[Dict[str, Any]] = []

    @abstractmethod
    def collect_metrics(self, mission, strategy: str, prompt: str) -> Dict[str, Any]:
        """
        Implement metric collection logic for specific RQ.

        Args:
            mission: RobotMission object to evaluate
            strategy: Strategy name (e.g., "zero_shot", "single_rag", "dual_rag")
            prompt: User prompt that generated the mission

        Returns:
            Dictionary of metrics specific to the research question
        """
        pass

    def add_result(self, metrics: Dict[str, Any]):
        """Add a single result to the collection."""
        self.results.append(metrics)

    def save_results(self, filepath: str):
        # save collected results to csv
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.results)
        df.to_csv(filepath, index=False)
        print(f"[{self.rq_name}] Results saved to {filepath}")
        return filepath

    def load_results(self, filepath: str) -> List[Dict[str, Any]]:
        # load results from csv
        self.results = pd.read_csv(filepath).to_dict('records')
        print(f"[{self.rq_name}] Loaded {len(self.results)} results from {filepath}")
        return self.results

    def clear_results(self):
        # clear all collected results
        self.results = []

    def get_results_count(self) -> int:
        # get the number of collected results
        return len(self.results)