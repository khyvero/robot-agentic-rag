"""
Combined Test Suite for RQ1 and RQ2

Runs a single test execution that collects BOTH:
- RQ1 metrics: Safety, structural validity, execution reliability
- RQ2 metrics: Latency breakdown, execution accuracy, real-time feasibility
- STRICT VALIDATION: Expected actions, objects, constraints (coords, speed, area, duration)

40 prompts × 3 strategies = 120 tests total
- 10 simple tasks (single-action, no constraints)
- 10 compound tasks (multi-step sequences)
- 10 constraint tasks (explicit spatial/speed constraints)
- 10 novel tasks (NOT in knowledge base)
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from questions.RQ1_RQ2_combined.metrics_collector_combined import CombinedMetricsCollector
from core.strategies.zero_shot import ZeroShotStrategy
from core.strategies.single_rag import SingleRAGStrategy
from core.strategies.dual_rag.strategy import DualRAGStrategy, AgentState
from core.knowledge_base import KnowledgeBase


def load_test_prompts() -> List[Dict]:
    """Load all 40 test prompts (10 simple, 10 compound, 10 constraint, 10 novel)."""
    prompts_path = Path(__file__).parent.parent / "shared" / "test_prompts.json"

    with open(prompts_path, "r") as f:
        data = json.load(f)

    all_cases = data["test_cases"]

    # Use all 40 test cases (IDs 1-40)
    return all_cases


def measure_with_timing(strategy, strategy_name: str, prompt: str) -> tuple:
    """
    Execute mission generation with component-level timing.

    Returns:
        Tuple of (mission, timing_breakdown)
    """
    timing = {
        "retrieval_time_ms": 0,
        "llm_generation_time_ms": 0,
        "validation_time_ms": 0,
        "total_latency_ms": 0
    }

    total_start = time.time()

    if strategy_name == "zero_shot":
        # Zero-shot has no retrieval
        llm_start = time.time()
        mission = strategy.generate_mission(prompt)
        llm_end = time.time()

        timing["llm_generation_time_ms"] = (llm_end - llm_start) * 1000
        timing["retrieval_time_ms"] = 0

    elif strategy_name in ["single_rag", "dual_rag"]:
        # RAG strategies: approximate breakdown
        start = time.time()
        mission = strategy.generate_mission(prompt)

        # Handle plan review for Dual-RAG
        if strategy_name == "dual_rag" and hasattr(strategy, 'state'):
            if strategy.state == AgentState.PLAN_REVIEW:
                mission = strategy.generate_mission("yes")
            elif strategy.state == AgentState.AMBIGUITY_CHECK:
                mission = strategy.generate_mission("None of these, create new task")
                if strategy.state == AgentState.PLAN_REVIEW:
                    mission = strategy.generate_mission("yes")

        end = time.time()
        elapsed = (end - start) * 1000

        # Approximate breakdown
        timing["retrieval_time_ms"] = elapsed * 0.10  # ~10% retrieval
        timing["llm_generation_time_ms"] = elapsed * 0.85  # ~85% LLM
        timing["validation_time_ms"] = elapsed * 0.05  # ~5% validation

    total_end = time.time()
    timing["total_latency_ms"] = (total_end - total_start) * 1000

    return mission, timing


def test_strategy(strategy_name: str, strategy, test_cases: List[Dict],
                 collector: CombinedMetricsCollector):
    # test a single strategy with all test cases
    print(f"\n{'=' * 70}")
    print(f"Testing Strategy: {strategy_name.upper()}")
    print(f"{'=' * 70}\n")

    for i, test_case in enumerate(test_cases, 1):
        test_id = test_case["id"]
        test_title = test_case["title"]
        prompt = test_case["user_prompt"]
        complexity = test_case.get("complexity", "medium")

        print(f"[{i}/{len(test_cases)}] Test #{test_id}: {test_title}")
        print(f"  Prompt: {prompt[:60]}...")

        try:
            # Measure with timing
            mission, timing_breakdown = measure_with_timing(strategy, strategy_name, prompt)

            # Extract expected values from test case
            expected_actions = test_case.get("expected_actions")
            expected_objects = test_case.get("expected_objects")
            expected_coords = test_case.get("expected_coords")
            expected_speed = test_case.get("expected_speed")
            expected_area = test_case.get("expected_area")
            expected_duration = test_case.get("expected_duration")

            # Collect BOTH RQ1 and RQ2 metrics with expected value validation
            metrics = collector.collect_metrics(
                mission=mission,
                strategy=strategy_name,
                prompt=prompt,
                timing_breakdown=timing_breakdown,
                test_id=test_id,
                test_title=test_title,
                complexity=complexity,
                expected_actions=expected_actions,
                expected_objects=expected_objects,
                expected_coords=expected_coords,
                expected_speed=expected_speed,
                expected_area=expected_area,
                expected_duration=expected_duration
            )

            collector.add_result(metrics)

            # Print summary (both RQ1 and RQ2)
            print(f"  ✓ Valid JSON: {metrics['valid_json']}", flush=True)
            print(f"  ✓ Safe: {not metrics['has_safety_violation']}", flush=True)
            print(f"  ✓ No Hallucinations: {not metrics['has_api_hallucination']}", flush=True)
            print(f"  ✓ Execution Success: {metrics['execution_success']}", flush=True)
            print(f"  ✓ Correctness Score: {metrics['correctness_score']:.1f}%", flush=True)
            print(f"  ⏱  Total Latency: {metrics['total_latency_ms']:.0f} ms", flush=True)
            print(f"  ⏱  Under 15s: {'✓' if metrics['under_15s_threshold'] else '✗'}", flush=True)

        except Exception as e:
            print(f"  ✗ ERROR: {str(e)}")
            # Collect metrics for error case
            metrics = collector.collect_metrics(
                mission=None,
                strategy=strategy_name,
                prompt=prompt,
                timing_breakdown=None,
                test_id=test_id,
                test_title=test_title,
                complexity=complexity
            )
            metrics["error"] = str(e)
            collector.add_result(metrics)

        print()


def run_combined_tests(output_dir: Path = None):
    # run combined rq1+rq2 test suite
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "raw_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70, flush=True)
    print("RQ1 + RQ2 COMBINED TEST SUITE", flush=True)
    print("Safety, Validity, Performance & Accuracy", flush=True)
    print("=" * 70, flush=True)

    # Load test cases
    print("\nLoading test cases...", flush=True)
    test_cases = load_test_prompts()
    print(f"Loaded {len(test_cases)} test cases")

    # Category breakdown
    categories = {}
    for tc in test_cases:
        cat = tc["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nTest Case Categories:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} tests")

    # Initialize strategies
    print("\nInitializing strategies...")
    kb = KnowledgeBase()

    strategies = {
        "zero_shot": ZeroShotStrategy(),
        "single_rag": SingleRAGStrategy(kb),
        "dual_rag": DualRAGStrategy(kb)
    }
    print("Strategies initialized: Zero-Shot, Single RAG, Dual-RAG")

    # Initialize combined collector
    collector = CombinedMetricsCollector()

    # Test each strategy
    total_start_time = time.time()

    for strategy_name, strategy in strategies.items():
        test_strategy(strategy_name, strategy, test_cases, collector)

    total_elapsed = time.time() - total_start_time

    # Save results
    results_file = output_dir / "combined_results.csv"
    collector.save_results(str(results_file))

    # Calculate and print summary
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (RQ1 + RQ2)")
    print("=" * 70 + "\n")

    summary = collector.calculate_summary_statistics()

    # RQ1 Summary
    print("RQ1: SAFETY & STRUCTURAL VALIDITY")
    print("-" * 70)
    print(f"{'Metric':<30} {'Zero-Shot':>12} {'Single RAG':>12} {'Dual-RAG':>12}")
    print("-" * 70)

    rq1_metrics = [
        ("Invalid JSON (%)", "invalid_json_rate"),
        ("Safety Violations (%)", "safety_violation_rate"),
        ("API Hallucinations (%)", "api_hallucination_rate"),
        ("Execution Success (%)", "execution_success_rate"),
        ("Avg. Steps", "avg_steps"),
        ("Avg. Correctness Score", "avg_correctness_score")
    ]

    for display_name, metric_key in rq1_metrics:
        values = []
        for strat in ["zero_shot", "single_rag", "dual_rag"]:
            if strat in summary and metric_key in summary[strat]:
                val = summary[strat][metric_key]
                if "%" in display_name:
                    values.append(f"{val:.1f}%")
                else:
                    values.append(f"{val:.1f}")
            else:
                values.append("N/A")
        print(f"{display_name:<30} {values[0]:>12} {values[1]:>12} {values[2]:>12}")

    print("\n")

    # RQ2 Summary
    print("RQ2: PERFORMANCE & ACCURACY")
    print("-" * 70)
    print(f"{'Metric':<30} {'Zero-Shot':>12} {'Single RAG':>12} {'Dual-RAG':>12}")
    print("-" * 70)

    rq2_metrics = [
        ("Avg. Total Latency (ms)", "avg_total_latency_ms"),
        ("Under 15s Threshold (%)", "under_15s_threshold_pct"),
        ("Collision-Free (%)", "collision_free_pct"),
        ("Kinematic Feasibility (%)", "kinematic_feasibility_pct")
    ]

    for display_name, metric_key in rq2_metrics:
        values = []
        for strat in ["zero_shot", "single_rag", "dual_rag"]:
            if strat in summary and metric_key in summary[strat]:
                val = summary[strat][metric_key]
                if "ms" in display_name:
                    values.append(f"{val:.0f}")
                else:
                    values.append(f"{val:.1f}%")
            else:
                values.append("N/A")
        print(f"{display_name:<30} {values[0]:>12} {values[1]:>12} {values[2]:>12}")

    print("-" * 70)
    print(f"\n✓ Total execution time: {total_elapsed:.2f}s")
    print(f"✓ Results saved to: {results_file}")
    print("\n📊 Next steps:")
    print(f"  1. Run analysis: python {Path(__file__).parent}/analyze_combined.py")
    print(f"  2. Generate visualizations: python {Path(__file__).parent}/visualize_combined.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run combined RQ1+RQ2 test suite")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    run_combined_tests(output_dir=output_dir)