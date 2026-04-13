"""RQ5 Analysis: Constraint-Based Runtime Task Adaptation"""
import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from questions.shared.statistical_utils import independent_t_test

def run_rq5_analysis(results_dir=None, output_dir=None):
    if not results_dir: results_dir = Path(__file__).parent / "results" / "raw_data"
    if not output_dir: output_dir = Path(__file__).parent / "results" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70 + "\nRQ5 ANALYSIS\n" + "=" * 70)

    df = pd.read_csv(results_dir / "rq5_results.csv")
    full = df[df["approach"] == "full_regeneration"]
    constraint = df[df["approach"] == "constraint_based"]

    # Summary statistics
    summary = pd.DataFrame([
        {
            "Approach": "Full Regeneration",
            "Avg Preservation Rate (%)": full["modified_preservation_rate"].mean(),
            "Avg Replanning Time (s)": full["replanning_time_s"].mean()
        },
        {
            "Approach": "Constraint-Based Adaptive",
            "Avg Preservation Rate (%)": constraint["modified_preservation_rate"].mean(),
            "Avg Replanning Time (s)": constraint["replanning_time_s"].mean()
        }
    ])

    overhead_reduction = ((full["replanning_time_s"].mean() -
                          constraint["replanning_time_s"].mean()) /
                         full["replanning_time_s"].mean() * 100)

    print("\n" + summary.to_string(index=False))
    print(f"\nOverhead Reduction: {overhead_reduction:.1f}%")

    summary.to_csv(output_dir / "rq5_summary.csv", index=False)

    # Constraint violation breakdown
    print("\n" + "=" * 70)
    print("CONSTRAINT VIOLATION ANALYSIS")
    print("=" * 70)

    violation_types = [
        "speed_limit_violation",
        "workspace_boundary_violation",
        "collision_avoidance_violation",
        "tool_constraint_violation"
    ]

    violation_summary = []
    for approach_name in ["full_regeneration", "constraint_based"]:
        subset = df[df["approach"] == approach_name]
        violations = {"Approach": approach_name}

        for vtype in violation_types:
            pct = (subset[vtype].sum() / len(subset) * 100) if len(subset) > 0 else 0
            violations[vtype.replace("_violation", "").replace("_", " ").title()] = pct

        violation_summary.append(violations)

    violation_df = pd.DataFrame(violation_summary)
    print("\n" + violation_df.to_string(index=False))
    violation_df.to_csv(output_dir / "rq5_constraint_violations.csv", index=False)

    # Statistical significance
    test_result = independent_t_test(
        full["replanning_time_s"].tolist(),
        constraint["replanning_time_s"].tolist()
    )
    print(f"\nStatistical Significance (Replanning Time): p={test_result['p_value']:.4f} ({'Yes' if test_result['significant'] else 'No'})")

    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__": run_rq5_analysis()