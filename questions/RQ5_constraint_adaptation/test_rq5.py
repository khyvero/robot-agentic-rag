"""
RQ5 Test Suite: Constraint-Based Runtime Task Adaptation

Tests runtime modifications comparing full regeneration vs constraint-based adaptive approach.
Uses REAL missions from declarative_tasks.json with constraint-focused modifications.
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from questions.RQ5_constraint_adaptation.metrics_collector_rq5 import RQ5MetricsCollector
from core.strategies.dual_rag.strategy import DualRAGStrategy, AgentState
from core.knowledge_base import KnowledgeBase

# Real mission constraint modifications
CONSTRAINT_MODIFICATION_SCENARIOS = [
    {"id": 1, "name": "Neutralization - Coordinate Constraint Change",
     "initial": "Neutralize acid with base",
     "modification": "Change final placement to coordinates (400, 400)"},

    {"id": 2, "name": "DNA Extraction - Workspace Boundary Test",
     "initial": "Extract DNA from blood sample",
     "modification": "Place DNA tube at coordinates (700, 700)"},

    {"id": 3, "name": "Waste Disposal - Speed Constraint Modification",
     "initial": "Clean up and dispose all chemicals",
     "modification": "Use simulation speed of 3 for faster disposal"},

    {"id": 4, "name": "Sample Analysis - Coordinate Adjustment",
     "initial": "Analyze blood sample",
     "modification": "Place blood tube at (200, 200) after analysis"},

    {"id": 5, "name": "Water Transfer - Multiple Coordinate Changes",
     "initial": "Distribute water to empty tubes",
     "modification": "Place first tube at (100, 100), second at (200, 200)"},

    {"id": 6, "name": "Phenol Handling - Workspace Constraint",
     "initial": "Handle phenol safely",
     "modification": "Place phenol tube at coordinates (500, -500) after handling"},

    {"id": 7, "name": "Lab Setup - Position Constraints",
     "initial": "Setup lab bench",
     "modification": "Position all items within 300mm radius from center"},

    {"id": 8, "name": "Heat Reaction - Safe Distance Constraint",
     "initial": "Heat DNA sample with bunsen burner",
     "modification": "Move DNA tube to safe distance at (600, 600)"},

    {"id": 9, "name": "Serial Dilution - Sequential Position Constraints",
     "initial": "Perform serial dilution",
     "modification": "Place tubes in line: (100, 0), (200, 0), (300, 0)"},

    {"id": 10, "name": "Acid Base Mix - Container Position Change",
     "initial": "Mix hydrochloric acid with sodium hydroxide",
     "modification": "Move container to coordinates (350, 350) before mixing"},

    {"id": 11, "name": "pH Test - Multiple Constraint Changes",
     "initial": "Test pH with phenolphthalein",
     "modification": "Use speed 2, place result at (250, 250)"},

    {"id": 12, "name": "Heat Water - Speed and Position Constraints",
     "initial": "Boil water in beaker",
     "modification": "Use speed 4, move to cooling zone at (100, 600)"},

    {"id": 13, "name": "Pour Action - Collision Avoidance",
     "initial": "Pour test_tube_blood into beaker_water",
     "modification": "Place beaker at (400, 400) to avoid collision zone"},

    {"id": 14, "name": "Shake Action - Safe Distance Constraint",
     "initial": "Shake test_tube_DNA",
     "modification": "Move to shaking zone at coordinates (0, 700)"},

    {"id": 15, "name": "Disposal - Workspace Boundary Constraint",
     "initial": "Dispose test_tube_phenol",
     "modification": "Move bin to accessible position at (700, -700)"}
]

def test_approach(scenario, strategy, collector, approach):
    # test either full regeneration or constraint-based adaptive approach
    if hasattr(strategy, 'state'):
        strategy.state = AgentState.IDLE

    original_mission = None
    modified_mission = None
    replanning_time = 0

    try:
        if approach == "full_regeneration":
            # Full regeneration: Generate complete new plan
            prompt = f"{scenario['initial']}. {scenario['modification']}"
            start = time.time()
            modified_mission = strategy.generate_mission(prompt)

            if hasattr(strategy, 'state') and strategy.state == AgentState.PLAN_REVIEW:
                modified_mission = strategy.generate_mission("yes")
            elif hasattr(strategy, 'state') and strategy.state == AgentState.AMBIGUITY_CHECK:
                modified_mission = strategy.generate_mission("None of these, create new task")
                if strategy.state == AgentState.PLAN_REVIEW:
                    modified_mission = strategy.generate_mission("yes")

            replanning_time = time.time() - start

            # Generate original for comparison
            if hasattr(strategy, 'state'):
                strategy.state = AgentState.IDLE
            original_mission = strategy.generate_mission(scenario['initial'])
            if hasattr(strategy, 'state') and strategy.state == AgentState.PLAN_REVIEW:
                original_mission = strategy.generate_mission("yes")

        else:  # constraint_based
            # Constraint-based: Generate original, then adapt
            original_mission = strategy.generate_mission(scenario['initial'])
            if hasattr(strategy, 'state') and strategy.state == AgentState.PLAN_REVIEW:
                original_mission = strategy.generate_mission("yes")
            elif hasattr(strategy, 'state') and strategy.state == AgentState.AMBIGUITY_CHECK:
                original_mission = strategy.generate_mission("None of these, create new task")
                if strategy.state == AgentState.PLAN_REVIEW:
                    original_mission = strategy.generate_mission("yes")

            # Adaptive modification
            if hasattr(strategy, 'state'):
                strategy.state = AgentState.IDLE

            start = time.time()
            modified_mission = strategy.generate_mission(scenario['modification'])

            if hasattr(strategy, 'state') and strategy.state == AgentState.PLAN_REVIEW:
                modified_mission = strategy.generate_mission("yes")

            replanning_time = time.time() - start

        # Collect metrics
        if original_mission and modified_mission:
            return collector.collect_metrics(
                scenario_id=scenario["id"],
                scenario_name=scenario["name"],
                approach=approach,
                original_mission=original_mission,
                modified_mission=modified_mission,
                replanning_time_s=replanning_time
            )

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

    return None

def run_rq5_tests(output_dir: Path = None):
    # run rq5 test suite
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "raw_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RQ5: Constraint-Based Runtime Task Adaptation")
    print("=" * 70)
    print(f"\nTesting {len(CONSTRAINT_MODIFICATION_SCENARIOS)} constraint modification scenarios\n")

    kb = KnowledgeBase()
    strategy = DualRAGStrategy(kb)
    collector = RQ5MetricsCollector()

    for i, sc in enumerate(CONSTRAINT_MODIFICATION_SCENARIOS, 1):
        print(f"\n[{i}/{len(CONSTRAINT_MODIFICATION_SCENARIOS)}] {sc['name']}")
        print(f"  Initial: {sc['initial']}")
        print(f"  Modify: {sc['modification']}")

        try:
            # Test full regeneration
            print("  [Full Regen] ", end="")
            m_full = test_approach(sc, strategy, collector, "full_regeneration")
            if m_full:
                collector.add_result(m_full)
                print(f"✓ {m_full['modified_preservation_rate']:.1f}% preserved, {m_full['replanning_time_s']:.2f}s")
            else:
                print("✗")

            # Test constraint-based adaptation
            print("  [Constraint] ", end="")
            m_constraint = test_approach(sc, strategy, collector, "constraint_based")
            if m_constraint:
                collector.add_result(m_constraint)
                print(f"✓ {m_constraint['modified_preservation_rate']:.1f}% preserved, {m_constraint['replanning_time_s']:.2f}s")
            else:
                print("✗")

            # Show improvements
            if m_full and m_constraint:
                overhead_reduction = ((m_full['replanning_time_s'] - m_constraint['replanning_time_s']) /
                                     m_full['replanning_time_s'] * 100)
                print(f"  💰 {overhead_reduction:.1f}% faster replanning")

        except Exception as e:
            print(f"  ❌ {str(e)}")

    # Save results
    results_file = output_dir / "rq5_results.csv"
    collector.save_results(str(results_file))

    # Summary
    comp = collector.calculate_comparison_statistics()
    if comp:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"\n{'Metric':<35} {'Full Regen':>15} {'Constraint':>15} {'Improvement':>15}")
        print("-" * 80)
        print(f"{'Avg Preservation Rate (%)':<35} {comp['full_regeneration']['avg_preservation_rate']:>15.1f} {comp['constraint_based']['avg_preservation_rate']:>15.1f} {comp['constraint_based']['avg_preservation_rate'] - comp['full_regeneration']['avg_preservation_rate']:>14.1f}%")
        print(f"{'Avg Replanning Time (s)':<35} {comp['full_regeneration']['avg_replanning_time_s']:>15.2f} {comp['constraint_based']['avg_replanning_time_s']:>15.2f} {comp['improvements']['overhead_reduction_pct']:>14.1f}%")
        print("-" * 80)
        print(f"\nResults saved to: {results_file}")

if __name__ == "__main__": run_rq5_tests()