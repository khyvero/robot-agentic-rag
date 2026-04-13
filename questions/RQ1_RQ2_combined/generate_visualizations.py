"""
Generate Visualizations from ACTUAL Test Results
Reads real data from CSV files instead of hardcoded numbers

UPDATED VERSION: B&W compatible, legends below plots, no RQ text
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Professional styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Colorful palette - grayscale-distinguishable (different luminance values)
# Deep Blue (dark), Coral Orange (light), Emerald Green (medium)
COLORS = ['#2563EB', '#F97316', '#059669']
PATTERNS = ['///', '...', '']  # diagonal lines, dots, solid


def load_data(data_dir):
    """Load actual test results from CSV files"""
    zero = pd.read_csv(data_dir / 'zero_shot_results.csv')
    single = pd.read_csv(data_dir / 'single_rag_results.csv')
    dual = pd.read_csv(data_dir / 'dual_rag_results.csv')

    # Add proper category based on test_id (CSV only has low/medium/high)
    def get_category(test_id):
        if 1 <= test_id <= 10:
            return 'simple'
        elif 11 <= test_id <= 20:
            return 'compound'
        elif 21 <= test_id <= 30:
            return 'constraint'
        elif 31 <= test_id <= 40:
            return 'novel'
        return 'unknown'

    for df in [zero, single, dual]:
        df['category'] = df['test_id'].apply(get_category)

    return zero, single, dual


def create_table_as_image(data, title, output_path, col_widths=None):
    # create a styled table as png image (b&w compatible)
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=data.values,
                     colLabels=data.columns,
                     cellLoc='center',
                     loc='center',
                     colWidths=col_widths)

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Header styling (grayscale)
    for i in range(len(data.columns)):
        cell = table[(0, i)]
        cell.set_facecolor('#333333')  # Dark gray
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors (light gray / white for B&W printing)
    for i in range(1, len(data) + 1):
        for j in range(len(data.columns)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E8E8E8')  # Light gray
            else:
                cell.set_facecolor('#FFFFFF')  # White

    # Add title
    plt.title(title, fontsize=13, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: {output_path}")

    # Also save as CSV
    csv_path = output_path.with_suffix('.csv')
    data.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path}")


def create_rq1_tables(zero, single, dual, output_dir):
    # create rq1 tables from actual data
    print("\nGenerating RQ1 Tables (from actual data)...")

    # Table 1: Structural Validity and Safety Comparison
    data1 = pd.DataFrame({
        'Architecture': ['Zero-Shot (Llama 3.8b)', 'Single-RAG (Llama 3.8b)', 'Dual-RAG (Llama 3.8b)'],
        'Invalid JSON (%)': [
            100 - zero['valid_json'].mean() * 100,
            100 - single['valid_json'].mean() * 100,
            100 - dual['valid_json'].mean() * 100
        ],
        'Safety Violation (%)': [
            zero['has_safety_violation'].mean() * 100,
            single['has_safety_violation'].mean() * 100,
            dual['has_safety_violation'].mean() * 100
        ],
        'API Hallucination (%)': [
            zero['has_api_hallucination'].mean() * 100,
            single['has_api_hallucination'].mean() * 100,
            dual['has_api_hallucination'].mean() * 100
        ],
        'Execution Success (%)': [
            zero['execution_success'].mean() * 100,
            single['execution_success'].mean() * 100,
            dual['execution_success'].mean() * 100
        ]
    })

    # Round to 1 decimal
    for col in data1.columns[1:]:
        data1[col] = data1[col].round(1)

    create_table_as_image(
        data1,
        'Structural Validity and Safety Comparison',
        output_dir / 'rq1_table1_structural_validity.png'
    )

    # Table 2: Failure Mode Distribution
    data2 = pd.DataFrame({
        'Architecture': ['Zero-Shot (Llama 3.8b)', 'Single-RAG (Llama 3.8b)', 'Dual-RAG (Llama 3.8b)'],
        'Syntax Errors (%)': [
            (100 - zero['valid_json'].mean() * 100),
            (100 - single['valid_json'].mean() * 100),
            (100 - dual['valid_json'].mean() * 100)
        ],
        'Safety Limit Violations (%)': [
            zero['has_safety_violation'].mean() * 100,
            single['has_safety_violation'].mean() * 100,
            dual['has_safety_violation'].mean() * 100
        ],
        'Missing API Calls (%)': [
            zero['has_api_hallucination'].mean() * 100,
            single['has_api_hallucination'].mean() * 100,
            dual['has_api_hallucination'].mean() * 100
        ],
        'Logical Sequence Errors (%)': [
            (100 - zero['execution_success'].mean() * 100) - zero['has_safety_violation'].mean() * 100,
            (100 - single['execution_success'].mean() * 100) - single['has_safety_violation'].mean() * 100,
            (100 - dual['execution_success'].mean() * 100) - dual['has_safety_violation'].mean() * 100
        ]
    })

    # Round and ensure non-negative
    for col in data2.columns[1:]:
        data2[col] = data2[col].round(0).clip(lower=0)

    create_table_as_image(
        data2,
        'Failure Mode Distribution',
        output_dir / 'rq1_table2_failure_distribution.png'
    )


def get_success_by_complexity(df):
    # calculate success rate by category (simple/compound/constraint/novel)
    category_map = {
        'simple': 'Simple',
        'compound': 'Compound',
        'constraint': 'Constrained',
        'novel': 'Novel'
    }

    results = {}
    for cat_key, cat_name in category_map.items():
        subset = df[df['category'] == cat_key]
        if len(subset) > 0:
            results[cat_name] = subset['execution_success'].mean() * 100
        else:
            results[cat_name] = 0.0

    return results


def create_rq1_plots(zero, single, dual, output_dir):
    # create rq1 plots from actual data (b&w compatible)
    print("\nGenerating RQ1 Plots (from actual data)...")

    # Plot 1: Execution Reliability Across Task Complexity
    fig, ax = plt.subplots(figsize=(14, 8))

    zero_by_comp = get_success_by_complexity(zero)
    single_by_comp = get_success_by_complexity(single)
    dual_by_comp = get_success_by_complexity(dual)

    categories = ['Simple\nTasks', 'Compound\nTasks', 'Constrained\nTasks', 'Novel\nTasks']
    zero_shot = [zero_by_comp.get('Simple', 0), zero_by_comp.get('Compound', 0),
                 zero_by_comp.get('Constrained', 0), zero_by_comp.get('Novel', 0)]
    single_rag = [single_by_comp.get('Simple', 0), single_by_comp.get('Compound', 0),
                  single_by_comp.get('Constrained', 0), single_by_comp.get('Novel', 0)]
    dual_rag = [dual_by_comp.get('Simple', 0), dual_by_comp.get('Compound', 0),
                dual_by_comp.get('Constrained', 0), dual_by_comp.get('Novel', 0)]

    x = np.arange(len(categories))
    width = 0.25

    # Use colorful palette with hatching for grayscale distinction
    bars1 = ax.bar(x - width, zero_shot, width, label='Zero-Shot (Llama 3.8b)',
                   color=COLORS[0], edgecolor='black', linewidth=1.5, hatch='///')
    bars2 = ax.bar(x, single_rag, width, label='Single-RAG (Llama 3.8b)',
                   color=COLORS[1], edgecolor='black', linewidth=1.5, hatch='...')
    bars3 = ax.bar(x + width, dual_rag, width, label='Dual-RAG (Llama 3.8b)',
                   color=COLORS[2], edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{height:.1f}%', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    ax.set_ylabel('Execution Success Rate (%)', fontsize=13, fontweight='bold')
    ax.set_title('Execution Reliability Across Task Complexity',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim([0, 110])

    # Legend below plot
    ax.legend(title='Architectures', loc='upper center', bbox_to_anchor=(0.5, -0.15),
             frameon=True, shadow=False, fontsize=10, title_fontsize=11, ncol=3)

    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_dir / 'rq1_plot1_execution_reliability.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: rq1_plot1_execution_reliability.png")

    # Plot 2: Safety Filtering Funnel
    fig, ax = plt.subplots(figsize=(14, 9))

    stages = ['Generated', 'Valid', 'Safe', 'Executable']
    zero_shot = [
        100.0,
        zero['valid_json'].mean() * 100,
        (zero['valid_json'] & ~zero['has_safety_violation']).mean() * 100,
        zero['execution_success'].mean() * 100
    ]
    single_rag = [
        100.0,
        single['valid_json'].mean() * 100,
        (single['valid_json'] & ~single['has_safety_violation']).mean() * 100,
        single['execution_success'].mean() * 100
    ]
    dual_rag = [
        100.0,
        dual['valid_json'].mean() * 100,
        (dual['valid_json'] & ~dual['has_safety_violation']).mean() * 100,
        dual['execution_success'].mean() * 100
    ]

    y = np.arange(len(stages))
    height = 0.25

    # Use colorful palette with patterns for grayscale distinction
    bars1 = ax.barh(y + height, zero_shot, height, label='Zero-Shot (Llama 3.8b)',
                    color=COLORS[0], edgecolor='black', linewidth=1.5, hatch='///')
    bars2 = ax.barh(y, single_rag, height, label='Single-RAG (Llama 3.8b)',
                    color=COLORS[1], edgecolor='black', linewidth=1.5, hatch='...')
    bars3 = ax.barh(y - height, dual_rag, height, label='Dual-RAG (Llama 3.8b)',
                    color=COLORS[2], edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            width_val = bar.get_width()
            ax.text(width_val + 1.5, bar.get_y() + bar.get_height()/2.,
                   f'{width_val:.1f}%', ha='left', va='center',
                   fontsize=10, fontweight='bold')

    ax.set_xlabel('Percentage of Plans (%)', fontsize=13, fontweight='bold')
    ax.set_title('Safety Filtering Funnel', fontsize=14, fontweight='bold', pad=15)
    ax.set_yticks(y)
    ax.set_yticklabels(stages, fontsize=12)
    ax.set_xlim([0, 115])

    # Legend below
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             frameon=True, shadow=False, fontsize=11, ncol=3)

    ax.grid(axis='x', alpha=0.3, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_dir / 'rq1_plot2_safety_funnel.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: rq1_plot2_safety_funnel.png")


def create_rq2_tables(dual, output_dir):
    # create rq2 tables from actual data - dual-rag only
    print("\nGenerating RQ2 Tables (Dual-RAG actual data)...")

    # Table 1: Latency by Task Category (use proper 'category' column)
    category_map = {
        'simple': 'Simple Task (Single-step)',
        'compound': 'Compound tasks (Multi-step)',
        'constraint': 'Constrained Task',
        'novel': 'Novel Task'
    }

    latency_data = []
    for cat_key, cat_name in category_map.items():
        subset = dual[dual['category'] == cat_key]
        if len(subset) > 0:
            avg_latency = subset['total_latency_ms'].mean() / 1000
            latency_data.append({
                'Task Category': cat_name,
                'Average Total Latency': f'{avg_latency:.2f}s',
                'Tests': len(subset)
            })

    data1 = pd.DataFrame(latency_data)

    create_table_as_image(
        data1,
        'Average Latency by Task Category (Dual-RAG Strategy)',
        output_dir / 'rq2_table1_latency_breakdown.png',
        col_widths=[0.40, 0.30, 0.30]
    )

    # Table 2: Execution Accuracy
    data2_rows = []
    for cat_key, cat_name in category_map.items():
        subset = dual[dual['category'] == cat_key]
        if len(subset) > 0:
            data2_rows.append({
                'Task Category': cat_name,
                'Success Rate (%)': round(subset['execution_success'].mean() * 100, 1),
                'Collision-Free (%)': round(subset['collision_free'].mean() * 100, 1),
                'Kinematic Feasibility (%)': round(subset['kinematic_feasibility'].mean() * 100, 1)
            })

    data2 = pd.DataFrame(data2_rows)

    create_table_as_image(
        data2,
        'Execution Accuracy in Simulation (Dual-RAG Strategy)',
        output_dir / 'rq2_table2_execution_accuracy.png'
    )


def create_rq2_plots(zero, single, dual, output_dir):
    # create rq2 plots from actual data (b&w compatible)
    print("\nGenerating RQ2 Plots (from actual data)...")

    # Plot 1: Real-Time Feasibility Threshold
    fig, ax = plt.subplots(figsize=(14, 8))

    complexity_map = {
        'low': 'Simple\nTask',
        'medium': 'Compound\nTask',
        'high': 'Constrained\nTask',
        'novel': 'Novel\nTask'
    }

    categories = []
    latencies = []
    for comp_key, comp_name in complexity_map.items():
        subset = dual[dual['complexity'] == comp_key]
        if len(subset) > 0:
            categories.append(comp_name)
            latencies.append(subset['total_latency_ms'].mean() / 1000)

    # Use emerald green for single series
    bars = ax.bar(categories, latencies, color=COLORS[2],
                  width=0.6, edgecolor='black', linewidth=1.5, alpha=0.85)

    # Add value labels
    for bar, val in zip(bars, latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.8,
               f'{val:.2f}s', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    # Add 30s threshold line (black dashed for B&W)
    ax.axhline(y=30.0, color='black', linestyle='--', linewidth=2.5,
              label='Real-Time Threshold (< 30s)', alpha=0.8)

    ax.set_ylabel('Total Planning Latency (s)', fontsize=13, fontweight='bold')
    ax.set_title('Real-Time Feasibility Threshold Plot (Dual-RAG)',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim([0, 35])

    # Legend below
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             frameon=True, shadow=False, fontsize=11)

    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_dir / 'rq2_plot1_realtime_threshold.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: rq2_plot1_realtime_threshold.png")

    # Plot 2: Latency Comparison Across Strategies
    fig, ax = plt.subplots(figsize=(14, 8))

    strategies = ['Zero-Shot', 'Single-RAG', 'Dual-RAG']
    avg_latencies = [
        zero['total_latency_ms'].mean() / 1000,
        single['total_latency_ms'].mean() / 1000,
        dual['total_latency_ms'].mean() / 1000
    ]

    # Use colorful palette with patterns for grayscale distinction
    bars = ax.bar(strategies, avg_latencies,
                  color=COLORS,
                  edgecolor='black', linewidth=1.5, alpha=0.85)
    bars[0].set_hatch('///')
    bars[1].set_hatch('...')

    # Add value labels
    for bar, val in zip(bars, avg_latencies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{val:.2f}s', ha='center', va='bottom',
               fontsize=11, fontweight='bold')

    # Add 30s threshold line
    ax.axhline(y=30.0, color='black', linestyle='--', linewidth=2.5,
              label='Max Latency Threshold (30s)', alpha=0.8)

    ax.set_ylabel('Average Latency (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Average Latency Comparison Across Strategies',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim([0, 35])

    # Legend below
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             frameon=True, shadow=False, fontsize=11)

    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_dir / 'rq2_plot2_latency_comparison.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: rq2_plot2_latency_comparison.png")

    # Plot 3: Latency Across Strategies and Task Types
    fig, ax = plt.subplots(figsize=(14, 8))

    categories = ['Simple\nTasks', 'Compound\nTasks', 'Constrained\nTasks', 'Novel\nTasks']
    category_keys = ['simple', 'compound', 'constraint', 'novel']

    # Calculate average latency by category for each strategy
    zero_latencies = []
    single_latencies = []
    dual_latencies = []

    for cat_key in category_keys:
        zero_subset = zero[zero['category'] == cat_key]
        single_subset = single[single['category'] == cat_key]
        dual_subset = dual[dual['category'] == cat_key]

        zero_latencies.append(zero_subset['total_latency_ms'].mean() / 1000 if len(zero_subset) > 0 else 0)
        single_latencies.append(single_subset['total_latency_ms'].mean() / 1000 if len(single_subset) > 0 else 0)
        dual_latencies.append(dual_subset['total_latency_ms'].mean() / 1000 if len(dual_subset) > 0 else 0)

    x = np.arange(len(categories))
    width = 0.25

    # Use colorful palette with hatching for grayscale distinction
    bars1 = ax.bar(x - width, zero_latencies, width, label='Zero-Shot (Llama 3.8b)',
                   color=COLORS[0], edgecolor='black', linewidth=1.5, hatch='///')
    bars2 = ax.bar(x, single_latencies, width, label='Single-RAG (Llama 3.8b)',
                   color=COLORS[1], edgecolor='black', linewidth=1.5, hatch='...')
    bars3 = ax.bar(x + width, dual_latencies, width, label='Dual-RAG (Llama 3.8b)',
                   color=COLORS[2], edgecolor='black', linewidth=1.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                   f'{height:.1f}s', ha='center', va='bottom',
                   fontsize=9, fontweight='bold')

    # Add 30s threshold line
    ax.axhline(y=30.0, color='black', linestyle='--', linewidth=2.5,
              label='Real-Time Threshold (30s)', alpha=0.8)

    ax.set_ylabel('Average Latency (seconds)', fontsize=13, fontweight='bold')
    ax.set_title('Latency Comparison Across Strategies and Task Types',
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim([0, max(max(zero_latencies), max(single_latencies), max(dual_latencies)) * 1.2 + 5])

    # Legend below plot
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
             frameon=True, shadow=False, fontsize=10, ncol=4)

    ax.grid(axis='y', alpha=0.3, linestyle='--', color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_dir / 'rq2_plot3_latency_by_task_type.png',
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Saved: rq2_plot3_latency_by_task_type.png")


def main():
    # generate all presentation visualizations from actual test data
    data_dir = Path(__file__).parent / "results" / "raw_data"
    output_dir = Path(__file__).parent / "results" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GENERATING VISUALIZATIONS FROM ACTUAL TEST DATA")
    print("=" * 70)
    print(f"\nReading data from: {data_dir}")

    # Load actual data
    zero, single, dual = load_data(data_dir)

    print(f"\nLoaded data:")
    print(f"  Zero-Shot: {len(zero)} tests")
    print(f"  Single-RAG: {len(single)} tests")
    print(f"  Dual-RAG: {len(dual)} tests")

    # Generate RQ1 outputs
    create_rq1_tables(zero, single, dual, output_dir)
    create_rq1_plots(zero, single, dual, output_dir)

    # Generate RQ2 outputs
    create_rq2_tables(dual, output_dir)
    create_rq2_plots(zero, single, dual, output_dir)

    print("\n" + "=" * 70)
    print("✓ All visualizations generated from ACTUAL test data!")
    print(f"✓ Saved to: {output_dir}")
    print("\n📊 Generated files:")
    for file in sorted(output_dir.glob("*.png")):
        print(f"  - {file.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()