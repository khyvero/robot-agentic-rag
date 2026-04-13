"""
RQ3 Metrics Collector: Conversational Refinement Efficiency

Measures token consumption and response time for full regeneration vs adaptive editing.
"""

import sys
from pathlib import Path
from typing import Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from questions.shared.base_metrics_collector import BaseMetricsCollector
from questions.RQ3_conversational_efficiency.ollama_tracker import OllamaSessionTracker


class RQ3MetricsCollector(BaseMetricsCollector):
    """Collect RQ3 metrics for conversational efficiency."""

    def __init__(self):
        super().__init__("RQ3")
        self.tracker = OllamaSessionTracker()

    def collect_metrics(self, scenario_id: int, scenario_name: str,
                       approach: str, prompt: str, response: str,
                       response_time_s: float,
                       input_tokens: int = None, output_tokens: int = None,
                       total_tokens: int = None) -> Dict[str, Any]:
        """
        Collect RQ3 metrics for a single approach.

        Args:
            scenario_id: Scenario ID
            scenario_name: Scenario name
            approach: "full_regeneration" or "adaptive_editor"
            prompt: Input prompt
            response: Model response
            response_time_s: Response time in seconds
            input_tokens: Actual input tokens from Ollama (optional, will estimate if not provided)
            output_tokens: Actual output tokens from Ollama (optional, will estimate if not provided)
            total_tokens: Actual total tokens from Ollama (optional, will calculate if not provided)

        Returns:
            Dictionary of metrics
        """
        # Use actual token counts if provided, otherwise set to 0
        if input_tokens is None:
            input_tokens = 0
        if output_tokens is None:
            output_tokens = 0
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens

        metrics = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "approach": approach,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "response_time_s": response_time_s,
            "prompt_length": len(prompt),
            "response_length": len(response)
        }

        return metrics

    def calculate_comparison_statistics(self) -> Dict[str, Any]:
        """
        Calculate comparison statistics between full regeneration and adaptive editor.

        Returns:
            Dictionary with comparison metrics
        """
        if not self.results:
            return {}

        # Separate by approach
        full_regen = [r for r in self.results if r["approach"] == "full_regeneration"]
        adaptive = [r for r in self.results if r["approach"] == "adaptive_editor"]

        if not full_regen or not adaptive:
            return {}

        # Calculate averages
        avg_tokens_full = sum(r["total_tokens"] for r in full_regen) / len(full_regen)
        avg_tokens_adaptive = sum(r["total_tokens"] for r in adaptive) / len(adaptive)

        avg_time_full = sum(r["response_time_s"] for r in full_regen) / len(full_regen)
        avg_time_adaptive = sum(r["response_time_s"] for r in adaptive) / len(adaptive)

        # Calculate reductions
        token_reduction_pct = ((avg_tokens_full - avg_tokens_adaptive) / avg_tokens_full) * 100
        time_reduction_pct = ((avg_time_full - avg_time_adaptive) / avg_time_full) * 100

        return {
            "full_regeneration": {
                "avg_tokens": round(avg_tokens_full, 1),
                "avg_time_s": round(avg_time_full, 2)
            },
            "adaptive_editor": {
                "avg_tokens": round(avg_tokens_adaptive, 1),
                "avg_time_s": round(avg_time_adaptive, 2)
            },
            "improvements": {
                "token_reduction_pct": round(token_reduction_pct, 2),
                "time_reduction_pct": round(time_reduction_pct, 2)
            }
        }


if __name__ == "__main__":
    collector = RQ3MetricsCollector()

    # Simulate full regeneration
    collector.add_result(collector.collect_metrics(
        scenario_id=1,
        scenario_name="Object Substitution",
        approach="full_regeneration",
        prompt="Pick up test_tube_blood and place at 100, 200",
        response='{"mission_name": "Pick and Place", "steps": [...], "settings": {...}}',
        response_time_s=3.5
    ))

    # Simulate adaptive editor
    collector.add_result(collector.collect_metrics(
        scenario_id=1,
        scenario_name="Object Substitution",
        approach="adaptive_editor",
        prompt="Change blood to DNA",
        response='{"mission_name": "Pick and Place", "steps": [...updated...], "settings": {...}}',
        response_time_s=0.8
    ))

    comparison = collector.calculate_comparison_statistics()
    print("\n[RQ3 Comparison]")
    print(f"Full Regeneration: {comparison['full_regeneration']['avg_tokens']} tokens, {comparison['full_regeneration']['avg_time_s']}s")
    print(f"Adaptive Editor: {comparison['adaptive_editor']['avg_tokens']} tokens, {comparison['adaptive_editor']['avg_time_s']}s")
    print(f"Token Reduction: {comparison['improvements']['token_reduction_pct']}%")
    print(f"Time Reduction: {comparison['improvements']['time_reduction_pct']}%")