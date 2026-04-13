"""
Combined Metrics Collector for RQ1 and RQ2

Collects both safety/validity metrics (RQ1) and performance/accuracy metrics (RQ2)
in a single pass.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.domain import RobotMission
from questions.shared.base_metrics_collector import BaseMetricsCollector


class CombinedMetricsCollector(BaseMetricsCollector):
    def __init__(self):
        super().__init__("RQ1_RQ2_Combined")
        self.valid_actions = [
            "pick", "place", "place_free_spot", "pour", "shake", "swirl",
            "move_home", "ensure_gripper_empty", "place_in_area", "wait", "ask_user"
        ]

    def _validate_safety_constraints(self, mission: RobotMission) -> bool:
        """
        Validate safety constraints on the mission (STRICT VALIDATION).

        Returns:
            True if all safety constraints are met, False otherwise
        """
        # Check simulation speed (must be 1-5)
        speed = mission.settings.get("simulation_speed", 1)
        if not isinstance(speed, int) or not (1 <= speed <= 5):
            return False

        # Strict validation for each action type
        for step in mission.steps:
            step_type = step.get("type", "")

            # Validate pick action
            if step_type == "pick":
                if "target_obj_name" not in step:
                    return False

            # Validate place action
            elif step_type == "place":
                if "destination_obj_name" not in step and "destination_coords" not in step:
                    return False
                # Check coordinates if provided
                if "destination_coords" in step:
                    coords = step["destination_coords"]
                    if not isinstance(coords, dict):
                        return False
                    x = coords.get("x", 0)
                    y = coords.get("y", 0)
                    if not (-800 <= x <= 800 and -800 <= y <= 800):
                        return False

            # Validate place_in_area action
            elif step_type == "place_in_area":
                if "area_bounds" not in step:
                    return False
                bounds = step["area_bounds"]
                if not isinstance(bounds, dict):
                    return False
                required_keys = ["min_x", "max_x", "min_y", "max_y"]
                if not all(k in bounds for k in required_keys):
                    return False
                # Validate bounds are within workspace
                if not (-800 <= bounds["min_x"] <= 800 and
                        -800 <= bounds["max_x"] <= 800 and
                        -800 <= bounds["min_y"] <= 800 and
                        -800 <= bounds["max_y"] <= 800):
                    return False

            # Validate pour action
            elif step_type == "pour":
                if "target_container_name" not in step:
                    return False

            # Validate shake/swirl actions
            elif step_type in ["shake", "swirl"]:
                if "target_obj_name" not in step:
                    return False

            # Validate wait action
            elif step_type == "wait":
                if "duration_seconds" not in step:
                    return False
                duration = step["duration_seconds"]
                if not isinstance(duration, (int, float)) or duration <= 0:
                    return False

        return True

    def _validate_expected_values(self, mission: RobotMission, expected_actions: list = None,
                                  expected_objects: list = None, expected_coords: dict = None,
                                  expected_speed: int = None, expected_area: dict = None,
                                  expected_duration: int = None) -> dict:
        """
        Validate mission against expected values (STRICT VALIDATION).

        Returns:
            Dict with validation results
        """
        validation = {
            "actions_match": True,
            "objects_present": True,
            "coords_match": True,
            "speed_match": True,
            "area_match": True,
            "duration_match": True
        }

        if not mission or not mission.steps:
            return {k: False for k in validation.keys()}

        # Validate expected actions
        if expected_actions:
            actual_actions = [step.get("type", "") for step in mission.steps]
            # Check if expected actions are subset of actual actions (order matters for strict checking)
            validation["actions_match"] = all(action in actual_actions for action in expected_actions)

        # Validate expected objects
        if expected_objects:
            # Extract all object names from mission steps
            actual_objects = []
            for step in mission.steps:
                if "target_obj_name" in step:
                    actual_objects.append(step["target_obj_name"])
                if "destination_obj_name" in step:
                    actual_objects.append(step["destination_obj_name"])
                if "target_container_name" in step:
                    actual_objects.append(step["target_container_name"])

            # Check if all expected objects are present
            validation["objects_present"] = all(obj in actual_objects for obj in expected_objects)

        # Validate expected coordinates
        if expected_coords:
            found_coords = False
            for step in mission.steps:
                if step.get("type") in ["place", "place_free_spot"]:
                    if "destination_coords" in step:
                        coords = step["destination_coords"]
                        if coords.get("x") == expected_coords["x"] and coords.get("y") == expected_coords["y"]:
                            found_coords = True
                            break
            validation["coords_match"] = found_coords

        # Validate expected speed
        if expected_speed:
            validation["speed_match"] = mission.settings.get("simulation_speed", 1) == expected_speed

        # Validate expected area
        if expected_area:
            found_area = False
            for step in mission.steps:
                if step.get("type") == "place_in_area":
                    if "area_bounds" in step:
                        bounds = step["area_bounds"]
                        if (bounds.get("min_x") == expected_area["min_x"] and
                            bounds.get("max_x") == expected_area["max_x"] and
                            bounds.get("min_y") == expected_area["min_y"] and
                            bounds.get("max_y") == expected_area["max_y"]):
                            found_area = True
                            break
            validation["area_match"] = found_area

        # Validate expected duration
        if expected_duration:
            found_duration = False
            for step in mission.steps:
                if step.get("type") == "wait":
                    if "duration_seconds" in step:
                        if step["duration_seconds"] == expected_duration:
                            found_duration = True
                            break
            validation["duration_match"] = found_duration

        return validation

    def collect_metrics(self, mission: RobotMission, strategy: str, prompt: str,
                       timing_breakdown: dict, test_id: int, test_title: str,
                       complexity: str, expected_actions: list = None,
                       expected_objects: list = None, expected_coords: dict = None,
                       expected_speed: int = None, expected_area: dict = None,
                       expected_duration: int = None):
        """
        Collect both RQ1 and RQ2 metrics simultaneously.

        Args:
            mission: Generated robot mission
            strategy: Strategy name (zero_shot, single_rag, dual_rag)
            prompt: User prompt
            timing_breakdown: Dict with retrieval/llm/validation times
            test_id: Test case ID
            test_title: Test case title
            complexity: Task complexity (low, medium, high)

        Returns:
            Dict with essential RQ1 and RQ2 metrics only
        """
        metrics = {
            # Test metadata (essential only)
            "test_id": test_id,
            "strategy": strategy,
            "complexity": complexity,

            # RQ1: Safety & Structural Validity Metrics
            "valid_json": False,
            "has_safety_violation": False,
            "has_api_hallucination": False,
            "execution_success": False,
            "num_steps": 0,

            # RQ2: Performance & Accuracy Metrics
            "total_latency_ms": 0,
            "under_15s_threshold": False,
            "collision_free": False,
            "kinematic_feasibility": False,

            # Correctness/Executability Metrics
            "correctness_score": 0
        }

        # Handle null mission
        if not mission:
            return metrics

        metrics["num_steps"] = len(mission.steps) if mission.steps else 0

        # === EXPECTED VALUE VALIDATION (for correctness/executability) ===
        expected_validation = self._validate_expected_values(
            mission, expected_actions, expected_objects, expected_coords,
            expected_speed, expected_area, expected_duration
        )

        # Calculate correctness score (percentage of expected validations that passed)
        expected_checks = [
            expected_validation["actions_match"] if expected_actions else None,
            expected_validation["objects_present"] if expected_objects else None,
            expected_validation["coords_match"] if expected_coords else None,
            expected_validation["speed_match"] if expected_speed else None,
            expected_validation["area_match"] if expected_area else None,
            expected_validation["duration_match"] if expected_duration else None
        ]
        # Filter out None (checks that weren't required)
        relevant_checks = [c for c in expected_checks if c is not None]
        if relevant_checks:
            metrics["correctness_score"] = (sum(relevant_checks) / len(relevant_checks)) * 100
        else:
            metrics["correctness_score"] = 100  # No expectations = assume correct

        # === RQ1 METRICS ===

        # 1. Check JSON validity
        try:
            if mission.raw_plan:
                json.loads(mission.raw_plan)
                metrics["valid_json"] = True
        except (json.JSONDecodeError, AttributeError):
            pass  # Already False

        # 2. Check safety violations
        if not self._validate_safety_constraints(mission):
            metrics["has_safety_violation"] = True

        # 3. Check API hallucinations
        if mission.steps:
            for step in mission.steps:
                action_type = step.get("type", "")
                if action_type and action_type not in self.valid_actions:
                    metrics["has_api_hallucination"] = True
                    break

        # 4. Check for missing API calls (empty plan) - treated as invalid
        # Already handled by num_steps = 0

        # 5. Execution success = all checks pass
        metrics["execution_success"] = (
            metrics["valid_json"] and
            not metrics["has_safety_violation"] and
            not metrics["has_api_hallucination"] and
            metrics["num_steps"] > 0
        )

        # === RQ2 METRICS ===

        if timing_breakdown:
            metrics["total_latency_ms"] = timing_breakdown.get("total_latency_ms", 0)

        # Real-time feasibility (<15s threshold)
        metrics["under_15s_threshold"] = metrics["total_latency_ms"] < 15000

        # Collision-free: All coordinates within bounds [-800, 800]
        metrics["collision_free"] = True
        if mission.steps:
            for step in mission.steps:
                if step.get("type") in ["place", "place_in_area"]:
                    params = step.get("params", {})
                    x = params.get("x", 0)
                    y = params.get("y", 0)
                    if not (-800 <= x <= 800 and -800 <= y <= 800):
                        metrics["collision_free"] = False
                        break

        # Kinematic feasibility: Valid action sequences (pick before place/pour)
        metrics["kinematic_feasibility"] = True
        if mission.steps:
            has_pick = False
            for step in mission.steps:
                action = step.get("type", "")
                if action == "pick":
                    has_pick = True
                elif action in ["place", "pour", "place_free_spot", "place_in_area"]:
                    if not has_pick:
                        metrics["kinematic_feasibility"] = False
                        break

        return metrics

    def add_result(self, metrics: dict):
        # add metrics result to collection
        self.results.append(metrics)

    def save_results(self, filepath: str):
        # save results split by strategy into 3 separate csv files
        import pandas as pd
        from pathlib import Path

        if not self.results:
            print(f"[{self.rq_name}] No results to save")
            return

        df = pd.DataFrame(self.results)
        output_path = Path(filepath).parent

        # Split and save by strategy
        for strategy in ["zero_shot", "single_rag", "dual_rag"]:
            strategy_data = df[df["strategy"] == strategy]
            if len(strategy_data) > 0:
                strategy_file = output_path / f"{strategy}_results.csv"
                strategy_data.to_csv(strategy_file, index=False)
                print(f"[{self.rq_name}] {strategy} results saved to {strategy_file}")

    def calculate_summary_statistics(self):
        # calculate summary statistics for both rq1 and rq2
        if not self.results:
            return {}

        import pandas as pd
        df = pd.DataFrame(self.results)

        summary = {}

        for strategy in ["zero_shot", "single_rag", "dual_rag"]:
            subset = df[df["strategy"] == strategy]

            if len(subset) == 0:
                continue

            # RQ1 metrics
            rq1_metrics = {
                "total_tests": len(subset),
                "invalid_json_rate": (1 - subset["valid_json"].mean()) * 100,
                "safety_violation_rate": subset["has_safety_violation"].mean() * 100,
                "api_hallucination_rate": subset["has_api_hallucination"].mean() * 100,
                "execution_success_rate": subset["execution_success"].mean() * 100,
                "avg_steps": subset["num_steps"].mean(),
                "avg_correctness_score": subset["correctness_score"].mean()
            }

            # RQ2 metrics
            rq2_metrics = {
                "avg_total_latency_ms": subset["total_latency_ms"].mean(),
                "collision_free_pct": subset["collision_free"].mean() * 100,
                "kinematic_feasibility_pct": subset["kinematic_feasibility"].mean() * 100,
                "under_15s_threshold_pct": subset["under_15s_threshold"].mean() * 100
            }

            summary[strategy] = {**rq1_metrics, **rq2_metrics}

        return summary