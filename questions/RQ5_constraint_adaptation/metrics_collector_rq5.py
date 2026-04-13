"""RQ5 Metrics Collector: Constraint-Based Runtime Task Adaptation"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from questions.shared.base_metrics_collector import BaseMetricsCollector
from core.domain import RobotMission

class RQ5MetricsCollector(BaseMetricsCollector):
    def __init__(self):
        super().__init__("RQ5")
        self.constraints = {
            "speed_limits": (1, 5),
            "workspace_bounds": {"x": (-800, 800), "y": (-800, 800)},
        }

    def check_constraints(self, mission: RobotMission) -> dict:
        # check all constraint types for a mission
        violations = {
            "speed_limits": False,
            "workspace_boundaries": False,
            "collision_avoidance": False,
            "tool_constraints": False
        }

        if not mission or not mission.steps:
            return violations

        # Check speed limits
        speed = getattr(mission, 'simulation_speed', 1)
        if not (self.constraints["speed_limits"][0] <= speed <= self.constraints["speed_limits"][1]):
            violations["speed_limits"] = True

        # Check workspace boundaries
        for step in mission.steps:
            step_type = step.get("type", "")
            params = step.get("params", {})

            if step_type in ["place", "place_in_area"]:
                x = params.get("x", 0)
                y = params.get("y", 0)

                x_min, x_max = self.constraints["workspace_bounds"]["x"]
                y_min, y_max = self.constraints["workspace_bounds"]["y"]

                if not (x_min <= x <= x_max and y_min <= y <= y_max):
                    violations["workspace_boundaries"] = True

        # Check tool constraints (gripper must be empty before pick)
        gripper_state = "empty"
        for step in mission.steps:
            step_type = step.get("type", "")

            if step_type == "pick" and gripper_state != "empty":
                violations["tool_constraints"] = True
            elif step_type == "pick":
                gripper_state = "holding"
            elif step_type in ["place", "pour"]:
                gripper_state = "empty"
            elif step_type == "ensure_gripper_empty":
                gripper_state = "empty"

        # Collision avoidance check (no overlapping coordinates for multiple objects)
        placed_positions = []
        for step in mission.steps:
            if step.get("type") == "place":
                params = step.get("params", {})
                x, y = params.get("x", 0), params.get("y", 0)
                if (x, y) in placed_positions:
                    violations["collision_avoidance"] = True
                placed_positions.append((x, y))

        return violations

    def calculate_preservation_rate(self, violations: dict) -> float:
        # calculate constraint preservation rate
        preserved = sum([not v for v in violations.values()])
        total = len(violations)
        return (preserved / total * 100) if total > 0 else 0

    def collect_metrics(self, scenario_id: int, scenario_name: str, approach: str,
                       original_mission: RobotMission, modified_mission: RobotMission,
                       replanning_time_s: float):
        # collect rq5 metrics for constraint adaptation
        original_violations = self.check_constraints(original_mission)
        modified_violations = self.check_constraints(modified_mission)

        original_preservation = self.calculate_preservation_rate(original_violations)
        modified_preservation = self.calculate_preservation_rate(modified_violations)

        return {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "approach": approach,
            "original_preservation_rate": original_preservation,
            "modified_preservation_rate": modified_preservation,
            "speed_limit_violation": modified_violations["speed_limits"],
            "workspace_boundary_violation": modified_violations["workspace_boundaries"],
            "collision_avoidance_violation": modified_violations["collision_avoidance"],
            "tool_constraint_violation": modified_violations["tool_constraints"],
            "replanning_time_s": replanning_time_s
        }

    def add_result(self, metrics: dict):
        # add metrics result to collection
        self.results.append(metrics)

    def calculate_comparison_statistics(self):
        # calculate comparative statistics between approaches
        if not self.results:
            return None

        import pandas as pd
        df = pd.DataFrame(self.results)

        full_regen = df[df["approach"] == "full_regeneration"]
        constraint_based = df[df["approach"] == "constraint_based"]

        if len(full_regen) == 0 or len(constraint_based) == 0:
            return None

        overhead_reduction = ((full_regen["replanning_time_s"].mean() -
                              constraint_based["replanning_time_s"].mean()) /
                             full_regen["replanning_time_s"].mean() * 100)

        return {
            "full_regeneration": {
                "avg_preservation_rate": full_regen["modified_preservation_rate"].mean(),
                "avg_replanning_time_s": full_regen["replanning_time_s"].mean()
            },
            "constraint_based": {
                "avg_preservation_rate": constraint_based["modified_preservation_rate"].mean(),
                "avg_replanning_time_s": constraint_based["replanning_time_s"].mean()
            },
            "improvements": {
                "overhead_reduction_pct": overhead_reduction
            }
        }