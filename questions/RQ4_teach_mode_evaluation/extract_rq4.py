"""
Extract RQ4 Teach Mode Metrics from RQ1_RQ2_combined Data (CORRECTED)

RQ4: To what degree can a generative teach mode compilation approach reduce expert
programming dependency by enabling non-expert operators to define new robotic
manipulation tasks that are immediately executable and reusable?

CORRECTED METRIC DEFINITION:
- Reusable Task = execution_success = True (can be safely stored and executed again)
- NOT just syntactically valid (which doesn't guarantee safety)
"""

import pandas as pd
from pathlib import Path

# Read data from RQ1_RQ2_combined
data_dir = Path(__file__).parent.parent.parent / "questions" / "RQ1_RQ2_combined" / "results" / "raw_data"

zero_shot_df = pd.read_csv(data_dir / "zero_shot_results.csv")
dual_rag_df = pd.read_csv(data_dir / "dual_rag_results.csv")

print("=" * 90)
print("RQ4: TEACH MODE EVALUATION - CORRECTED REUSABILITY METRIC")
print("=" * 90)
print()

# ============================================================================
# CALCULATE RQ4 METRICS (CORRECTED)
# ============================================================================

def calculate_rq4_metrics_corrected(df, strategy_name):
    # calculate rq4 metrics with corrected reusability definition
    total_tests = len(df)

    # Valid Task %: execution_success = True
    valid_tasks = df['execution_success'].sum()
    valid_task_pct = (valid_tasks / total_tests) * 100

    # CORRECTED Reusable Task %: execution_success = True
    # Only tasks that work can be safely reused
    reusable_tasks = valid_tasks  # Same as valid tasks!
    reusable_task_pct = valid_task_pct  # Same percentage

    # Expert Intervention %: execution_success = False
    expert_intervention = (total_tests - valid_tasks)
    expert_intervention_pct = (expert_intervention / total_tests) * 100

    # Error Distribution (among failures)
    failures = df[df['execution_success'] == False]
    total_failures = len(failures)

    if total_failures > 0:
        # API Hallucination
        api_hallucination_count = failures['has_api_hallucination'].sum()
        api_hallucination_pct = (api_hallucination_count / total_failures) * 100

        # Safety Violations
        safety_violation_count = failures['has_safety_violation'].sum()
        safety_violation_pct = (safety_violation_count / total_failures) * 100

        # Schema Violation
        schema_violation_count = (~failures['valid_json']).sum()
        schema_violation_pct = (schema_violation_count / total_failures) * 100

        # Logical Errors (failures with no other error category)
        logical_errors = failures[
            (failures['valid_json'] == True) &
            (failures['has_api_hallucination'] == False) &
            (failures['has_safety_violation'] == False)
        ].shape[0]
        logical_error_pct = (logical_errors / total_failures) * 100
    else:
        # No failures - all percentages are 0
        api_hallucination_pct = 0.0
        safety_violation_pct = 0.0
        schema_violation_pct = 0.0
        logical_error_pct = 0.0

    return {
        'strategy': strategy_name,
        'total_tests': total_tests,
        'valid_task_pct': valid_task_pct,
        'reusable_task_pct': reusable_task_pct,
        'expert_intervention_pct': expert_intervention_pct,
        'api_hallucination_pct': api_hallucination_pct,
        'safety_violation_pct': safety_violation_pct,
        'schema_violation_pct': schema_violation_pct,
        'logical_error_pct': logical_error_pct,
        'valid_tasks': valid_tasks,
        'reusable_tasks': reusable_tasks,
        'expert_interventions': expert_intervention,
        'total_failures': total_failures if 'total_failures' in locals() else 0
    }

# Calculate for both strategies
zero_shot_metrics = calculate_rq4_metrics_corrected(zero_shot_df, "Zero-Shot LLM")
dual_rag_metrics = calculate_rq4_metrics_corrected(dual_rag_df, "Teach Mode (Proposed)")

# ============================================================================
# DISPLAY RESULTS IN PRESENTATION FORMAT
# ============================================================================

print("📊 TABLE 1: Task Creation Success Rate (CORRECTED)")
print("-" * 90)
print(f"{'Approach':<30} {'Valid Task (%)':<20} {'Reusable Task (%)':<25} {'Expert Intervention (%)':<20}")
print("-" * 90)

print(f"{zero_shot_metrics['strategy']:<30} "
      f"{zero_shot_metrics['valid_task_pct']:>16.1f}%   "
      f"{zero_shot_metrics['reusable_task_pct']:>20.1f}%   "
      f"{zero_shot_metrics['expert_intervention_pct']:>18.1f}%")

print(f"{dual_rag_metrics['strategy']:<30} "
      f"{dual_rag_metrics['valid_task_pct']:>16.1f}%   "
      f"{dual_rag_metrics['reusable_task_pct']:>20.1f}%   "
      f"{dual_rag_metrics['expert_intervention_pct']:>18.1f}%")

print("-" * 90)
print()

# Calculate improvements
valid_improvement = dual_rag_metrics['valid_task_pct'] - zero_shot_metrics['valid_task_pct']
reusable_improvement = dual_rag_metrics['reusable_task_pct'] - zero_shot_metrics['reusable_task_pct']
intervention_reduction = zero_shot_metrics['expert_intervention_pct'] - dual_rag_metrics['expert_intervention_pct']

print(f"✅ Improvements:")
print(f"  • Valid Task: +{valid_improvement:.1f}% ({zero_shot_metrics['valid_task_pct']:.1f}% → {dual_rag_metrics['valid_task_pct']:.1f}%)")
print(f"  • Reusable Task: +{reusable_improvement:.1f}% ({zero_shot_metrics['reusable_task_pct']:.1f}% → {dual_rag_metrics['reusable_task_pct']:.1f}%)")
print(f"  • Expert Intervention Reduction: -{intervention_reduction:.1f}% ({zero_shot_metrics['expert_intervention_pct']:.1f}% → {dual_rag_metrics['expert_intervention_pct']:.1f}%)")
print()

print("=" * 90)
print("📊 TABLE 2: Error Distribution Among Failed Tasks (CORRECTED)")
print("-" * 90)
print(f"{'Error Type':<30} {'Zero-Shot LLM (%)':<25} {'Teach Mode (%)':<20}")
print("-" * 90)

error_types = [
    ('API Hallucination', 'api_hallucination_pct'),
    ('Schema Violation', 'schema_violation_pct'),
    ('Logical Errors', 'logical_error_pct'),
    ('Safety Violations', 'safety_violation_pct')
]

for error_name, error_key in error_types:
    zero_val = zero_shot_metrics[error_key]
    dual_val = dual_rag_metrics[error_key]
    print(f"{error_name:<30} {zero_val:>22.1f}%   {dual_val:>18.1f}%")

print("-" * 90)
print()
print(f"Note: Error percentages are calculated among FAILED tasks only")
print(f"  • Zero-Shot: {zero_shot_metrics['total_failures']} failures")
print(f"  • Teach Mode: {dual_rag_metrics.get('total_failures', 0)} failures")
print()

print("=" * 90)
print("🔍 KEY INSIGHT: Why Reusability = Valid Task %")
print("=" * 90)
print()
print("CORRECTED Definition: Reusable = Can be SAFELY stored and executed again")
print()
print("Why the old definition (97.5%) was WRONG:")
print("  ❌ Old: Valid JSON + No Hallucination = 'Reusable' (39/40 = 97.5%)")
print("  → But 35 of those 39 tasks have SAFETY VIOLATIONS!")
print("  → Can be stored in DB, but UNSAFE to reuse")
print()
print("Why the new definition (10%) is CORRECT:")
print("  ✅ New: Execution Success = 'Reusable' (4/40 = 10%)")
print("  → Only tasks that work can be safely reused")
print("  → If it fails once, it will fail again - not reusable!")
print()

# ============================================================================
# SAVE CORRECTED RESULTS TO CSV
# ============================================================================

output_dir = Path(__file__).parent / "results" / "raw_data"
output_dir.mkdir(parents=True, exist_ok=True)

# Create summary dataframe
summary_data = []
for metrics in [zero_shot_metrics, dual_rag_metrics]:
    summary_data.append({
        'strategy': metrics['strategy'],
        'valid_task_pct': metrics['valid_task_pct'],
        'reusable_task_pct': metrics['reusable_task_pct'],
        'expert_intervention_pct': metrics['expert_intervention_pct'],
        'api_hallucination_pct': metrics['api_hallucination_pct'],
        'schema_violation_pct': metrics['schema_violation_pct'],
        'logical_error_pct': metrics['logical_error_pct'],
        'safety_violation_pct': metrics['safety_violation_pct'],
        'total_tests': metrics['total_tests'],
        'valid_tasks': metrics['valid_tasks'],
        'reusable_tasks': metrics['reusable_tasks'],
        'expert_interventions': metrics['expert_interventions']
    })

summary_df = pd.DataFrame(summary_data)
output_file = output_dir / "rq4_summary_corrected.csv"
summary_df.to_csv(output_file, index=False)

print(f"✅ Corrected results saved to: {output_file}")
print()

print("=" * 90)
print("📊 READY FOR VISUALIZATION (with corrected metrics)")
print("=" * 90)
print("Next step: Run generate_visualizations_corrected.py to create updated plots")
print()