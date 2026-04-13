"""
RQ3 Test Suite: Conversational Refinement Efficiency (CORRECTED V2)

Proper implementation:
- Full Regeneration: Generate complete plan from scratch
- Conversational Refinement: Load existing plan from JSON → apply modification

This properly tests the efficiency of modifying existing plans vs regenerating.
"""

import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from questions.RQ3_conversational_efficiency.metrics_collector_rq3 import RQ3MetricsCollector
from questions.RQ3_conversational_efficiency.ollama_tracker import patch_ollama, unpatch_ollama, reset_tracker, get_tracker
from core.strategies.dual_rag.strategy import DualRAGStrategy, AgentState
from core.knowledge_base import KnowledgeBase
from core.domain import RobotMission

# Real mission modifications based on declarative_tasks.json
# Each test loads an EXISTING recipe and modifies it
REAL_MISSION_MODIFICATIONS = [
    {"id": 1, "name": "Neutralization - Different Final Location",
     "recipe_name": "Acid Base Mix",  # Load this existing recipe
     "modification": "Place the final mixture at coordinates 600, 600 instead"},

    {"id": 2, "name": "DNA Extraction - Skip Blood Disposal",
     "recipe_name": "DNA Extraction",
     "modification": "Keep the blood tube, don't place it"},

    {"id": 3, "name": "Waste Disposal - Partial Disposal",
     "recipe_name": "Waste Disposal",
     "modification": "Only dispose test_tube_phenol and test_tube_hydrochloric_acid"},

    {"id": 4, "name": "Sample Analysis - Add Wait Step",
     "recipe_name": "Sample Analysis",
     "modification": "Wait 5 seconds after shaking for observation"},

    {"id": 5, "name": "Water Transfer - Change Final Position",
     "recipe_name": "Water Transfer",
     "modification": "Place test_tube_empty at 400, 400 instead of default"},

    {"id": 6, "name": "Phenol Handling - Skip Swirl",
     "recipe_name": "Phenol Handling",
     "modification": "Skip the swirl step, just shake and pour"},

    {"id": 7, "name": "Lab Setup - Extended Setup",
     "recipe_name": "Lab Setup",
     "modification": "Also add test_tube_blood at 250, 300"},

    {"id": 8, "name": "Heat Reaction - Different Position",
     "recipe_name": "Heat Reaction",
     "modification": "Place DNA tube at 400, 350 instead of 300, 350"},

    {"id": 9, "name": "Serial Dilution - Add Shake Step",
     "recipe_name": "Serial Dilution",
     "modification": "Shake test_tube_empty after first pour"},

    {"id": 10, "name": "Acid Base Mix - Different Container",
     "recipe_name": "Acid Base Mix",
     "modification": "Use beaker_water as the container instead of test_tube_empty"},
]


def load_existing_recipe(recipe_name: str, kb: KnowledgeBase):
    # load an existing recipe plan from declarative_tasks.json
    # Read the declarative tasks
    tasks_file = Path(__file__).parent.parent.parent / "data" / "knowledge" / "declarative_tasks.json"
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)

    # Find the matching recipe
    for task in tasks:
        if task.get("mission_name") == recipe_name:
            # Convert logic_steps to the plan format
            steps = []
            for i, step in enumerate(task.get("logic_steps", []), 1):
                # Parse the step and create action
                step_lower = step.lower()
                if "pick up" in step_lower:
                    obj = step.split("'")[1]
                    steps.append({"action": "pick", "params": {"target_obj_name": obj}})
                elif "place" in step_lower and " on " in step_lower:
                    parts = step.split("'")
                    obj = parts[1]
                    target = parts[3] if len(parts) > 3 else "free_spot"
                    steps.append({"action": "place", "params": {"target_obj_name": obj, "container_name": target}})
                elif "pour into" in step_lower:
                    parts = step.split("'")
                    target = parts[1]
                    steps.append({"action": "pour", "params": {"target_container_name": target}})
                elif "shake" in step_lower:
                    obj = step.split("'")[1]
                    steps.append({"action": "shake", "params": {"target_obj_name": obj}})
                elif "swirl" in step_lower:
                    obj = step.split("'")[1]
                    steps.append({"action": "swirl", "params": {"target_obj_name": obj}})
                elif "wait" in step_lower:
                    seconds = int(''.join(filter(str.isdigit, step)))
                    steps.append({"action": "wait", "params": {"seconds": seconds}})
                elif "move home" in step_lower:
                    steps.append({"action": "move_home", "params": {}})
                elif "ensure gripper empty" in step_lower:
                    steps.append({"action": "ensure_gripper_empty", "params": {}})

            plan = {
                "mission_name": recipe_name,
                "tasks": steps,
                "settings": {"simulation_speed": 1}
            }
            return json.dumps(plan, indent=2)

    return None


def test_approach(scenario, strategy, collector, approach, kb):
    # test either full regeneration or conversational refinement
    reset_tracker()

    if hasattr(strategy, 'state'):
        strategy.state = AgentState.IDLE

    if approach == "full_regeneration":
        # Full regeneration: Generate complete plan from scratch
        # Combine recipe name and modification into a descriptive prompt
        full_prompt = f"Execute {scenario['recipe_name']}. {scenario['modification']}"

        start = time.time()
        mission = strategy.generate_mission(full_prompt)

        # Handle plan review
        if hasattr(strategy, 'state') and strategy.state == AgentState.PLAN_REVIEW:
            mission = strategy.generate_mission("yes")
        elif hasattr(strategy, 'state') and strategy.state == AgentState.AMBIGUITY_CHECK:
            mission = strategy.generate_mission("None of these, create new task")
            if strategy.state == AgentState.PLAN_REVIEW:
                mission = strategy.generate_mission("yes")

        elapsed = time.time() - start
        prompt_for_metrics = full_prompt

    else:  # conversational_refinement - Load existing plan then modify
        # Step 1: Load existing recipe plan from JSON (NO GENERATION!)
        existing_plan_json = load_existing_recipe(scenario['recipe_name'], kb)

        if not existing_plan_json:
            print(f"  ❌ Could not load recipe: {scenario['recipe_name']}")
            return None

        # Step 2: Put the strategy into PLAN_REVIEW state with the existing plan
        start = time.time()
        strategy.state = AgentState.PLAN_REVIEW
        strategy.pending_plan_json = existing_plan_json

        # Step 3: Apply modification (this calls modify_plan())
        mission = strategy.generate_mission(scenario['modification'])

        # Step 4: Confirm the modified plan
        if hasattr(strategy, 'state') and strategy.state == AgentState.PLAN_REVIEW:
            mission = strategy.generate_mission("yes")

        elapsed = time.time() - start
        prompt_for_metrics = f"Load {scenario['recipe_name']} → {scenario['modification']}"

    # Get actual token counts from tracker
    tracker = get_tracker()
    input_tokens = tracker.get_total_prompt_tokens()
    output_tokens = tracker.get_total_completion_tokens()
    total_tokens = tracker.get_total_tokens()

    if mission and mission.raw_plan:
        return collector.collect_metrics(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            approach=approach,
            prompt=prompt_for_metrics,
            response=mission.raw_plan,
            response_time_s=elapsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens
        )
    return None


def run_rq3_tests(output_dir: Path = None):
    # run rq3 test suite
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RQ3: Conversational Refinement Efficiency (CORRECTED V2)")
    print("=" * 70)
    print(f"\nTesting {len(REAL_MISSION_MODIFICATIONS)} modifications")
    print("Approach:")
    print("  - Full Regeneration: Generate complete plan from scratch")
    print("  - Conversational Refinement: Load existing plan → modify")
    print("\nTracking tokens from BOTH Ollama models:")
    print("  - mistral:7b-instruct (routing/extraction)")
    print("  - llama3:8b (plan generation/modification)")
    print()

    # Patch ollama to track tokens
    patch_ollama()

    kb = KnowledgeBase()
    strategy = DualRAGStrategy(kb)
    collector = RQ3MetricsCollector()

    for i, sc in enumerate(REAL_MISSION_MODIFICATIONS, 1):
        print(f"\n[{i}/{len(REAL_MISSION_MODIFICATIONS)}] {sc['name']}")
        print(f"  Recipe: {sc['recipe_name']}")
        print(f"  Modify: {sc['modification']}")

        try:
            # Test full regeneration
            print(f"  [Full Regen]", end=" ")
            m_full = test_approach(sc, strategy, collector, "full_regeneration", kb)
            if m_full:
                collector.add_result(m_full)
                print(f"✓ {m_full['total_tokens']} tok, {m_full['response_time_s']:.2f}s")
            else:
                print("✗")

            # Test conversational refinement
            print(f"  [Conv Refine]", end=" ")
            m_refine = test_approach(sc, strategy, collector, "conversational_refinement", kb)
            if m_refine:
                collector.add_result(m_refine)
                print(f"✓ {m_refine['total_tokens']} tok, {m_refine['response_time_s']:.2f}s")
            else:
                print("✗")

            # Show savings
            if m_full and m_refine:
                tok_save = ((m_full['total_tokens'] - m_refine['total_tokens']) / m_full['total_tokens'] * 100)
                time_save = ((m_full['response_time_s'] - m_refine['response_time_s']) / m_full['response_time_s'] * 100)
                print(f"  💰 {tok_save:.1f}% tokens, {time_save:.1f}% time")

        except Exception as e:
            print(f"  ❌ {str(e)}")
            import traceback
            traceback.print_exc()

    # Save results
    results_file = output_dir / "rq3_results_v2.csv"
    collector.save_results(str(results_file))

    # Summary
    comp = collector.calculate_comparison_statistics()
    if comp:
        print("\n" * 2 + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<30} {'Full Regen':>15} {'Conv Refine':>15} {'Savings':>15}")
        print("-" * 75)
        print(f"{'Avg Tokens':<30} {comp['full_regeneration']['avg_tokens']:>15.0f} {comp.get('conversational_refinement', comp.get('adaptive_editor', {})).get('avg_tokens', 0):>15.0f} {comp['improvements']['token_reduction_pct']:>14.1f}%")
        print(f"{'Avg Time (s)':<30} {comp['full_regeneration']['avg_time_s']:>15.2f} {comp.get('conversational_refinement', comp.get('adaptive_editor', {})).get('avg_time_s', 0):>15.2f} {comp['improvements']['time_reduction_pct']:>14.1f}%")
        print("-" * 75)
        print(f"\nResults saved to: {results_file}")

    # Unpatch ollama
    unpatch_ollama()


if __name__ == "__main__":
    run_rq3_tests()