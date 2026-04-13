"""
Generate RQ5 visualizations - ONE PLOT PER PNG FILE

Each visualization is a separate file with a single plot.
Improved version: Uses both strategies, plots instead of tables where possible.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from matplotlib.patches import Patch

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Read results
df = pd.read_csv('questions/RQ5_constraint_adaptation/results/raw_data/rq5_results.csv')

# Separate by approach
full_regen = df[df['approach'] == 'full_regeneration']
constraint_based = df[df['approach'] == 'constraint_based']

# Calculate statistics
avg_preservation_full = full_regen['modified_preservation_rate'].mean()
avg_preservation_constraint = constraint_based['modified_preservation_rate'].mean()

avg_time_full = full_regen['replanning_time_s'].mean()
avg_time_constraint = constraint_based['replanning_time_s'].mean()
overhead_reduction = ((avg_time_full - avg_time_constraint) / avg_time_full) * 100

# Constraint-specific violation rates (percentage of tests with violations)
constraint_violations_full = {
    'Speed Limits': (full_regen['speed_limit_violation'].sum() / len(full_regen)) * 100,
    'Workspace Boundaries': (full_regen['workspace_boundary_violation'].sum() / len(full_regen)) * 100,
    'Collision Avoidance': (full_regen['collision_avoidance_violation'].sum() / len(full_regen)) * 100,
    'Tool Constraints': (full_regen['tool_constraint_violation'].sum() / len(full_regen)) * 100
}

constraint_violations_adapt = {
    'Speed Limits': (constraint_based['speed_limit_violation'].sum() / len(constraint_based)) * 100,
    'Workspace Boundaries': (constraint_based['workspace_boundary_violation'].sum() / len(constraint_based)) * 100,
    'Collision Avoidance': (constraint_based['collision_avoidance_violation'].sum() / len(constraint_based)) * 100,
    'Tool Constraints': (constraint_based['tool_constraint_violation'].sum() / len(constraint_based)) * 100
}

# Calculate preservation rates for BOTH strategies
preservation_full = {k: 100 - v for k, v in constraint_violations_full.items()}
preservation_adapt = {k: 100 - v for k, v in constraint_violations_adapt.items()}

print("=" * 70)
print("RQ5 FINAL VISUALIZATIONS - ONE PLOT PER FILE")
print("=" * 70)

# Create output directory
output_dir = Path('questions/RQ5_constraint_adaptation/results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Colorful palette - grayscale-distinguishable (different luminance values)
COLOR_PRIMARY = '#2563EB'    # Deep Blue - for "Full Regeneration"
COLOR_SECONDARY = '#F97316'  # Coral Orange - for "Constraint-Based"
COLOR_ACCENT = '#059669'     # Emerald Green - for improvements
COLOR_TABLE_HEADER = '#1E3A5F'
COLOR_TABLE_ROW1 = '#E8F4FD'
COLOR_TABLE_ROW2 = '#FFFFFF'

# Hatch patterns for additional grayscale distinction
HATCH_DIAGONAL = '///'
HATCH_DOTS = '...'

# ===================================================================
# PLOT 1: Constraint Preservation Rate Comparison (Bar Chart)
# ===================================================================
fig, ax = plt.subplots(figsize=(14, 8))

constraints = list(preservation_full.keys())
full_rates = list(preservation_full.values())
adapt_rates = list(preservation_adapt.values())

x = np.arange(len(constraints))
width = 0.35

bars1 = ax.bar(x - width/2, full_rates, width,
               label='Full Re-generation',
               color=COLOR_PRIMARY, edgecolor='black', linewidth=2, hatch=HATCH_DIAGONAL)
bars2 = ax.bar(x + width/2, adapt_rates, width,
               label='Constraint-Based Adaptation',
               color=COLOR_SECONDARY, edgecolor='black', linewidth=2, hatch=HATCH_DOTS)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Preservation Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Constraint Preservation Rate Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(constraints, fontsize=12)
ax.set_ylim(0, 115)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq5_preservation_rate_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: rq5_preservation_rate_comparison.png")
plt.close()

# ===================================================================
# PLOT 2: Re-planning Time Comparison (Bar Chart)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 8))

methods = ['Full\nRe-generation', 'Constraint-Based\nAdaptation']
times = [avg_time_full, avg_time_constraint]

bars = ax.bar(methods, times, color=[COLOR_PRIMARY, COLOR_SECONDARY],
              edgecolor='black', linewidth=2, width=0.5,
              hatch=[HATCH_DIAGONAL, HATCH_DOTS])

# Add value labels
for bar, time in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{time:.2f}s', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotation
ax.annotate('', xy=(1, avg_time_constraint), xytext=(0, avg_time_full),
            arrowprops=dict(arrowstyle='->', color='black', lw=2,
                          connectionstyle='arc3,rad=-0.2'))
mid_y = (avg_time_full + avg_time_constraint) / 2
ax.text(0.5, mid_y + 2, f'-{overhead_reduction:.1f}%\nfaster', ha='center', va='center',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FEF3C7', edgecolor='black'))

ax.set_ylabel('Average Re-planning Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Re-planning Time Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, max(times) * 1.25)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig(output_dir / 'rq5_replanning_time_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq5_replanning_time_comparison.png")
plt.close()

# ===================================================================
# PLOT 3: Constraint Violation Comparison Bar Chart
# ===================================================================
fig, ax = plt.subplots(figsize=(14, 8))

constraints = list(constraint_violations_full.keys())
full_violations = list(constraint_violations_full.values())
adapt_violations = list(constraint_violations_adapt.values())

x = np.arange(len(constraints))
width = 0.35

bars1 = ax.bar(x - width/2, full_violations, width,
               label='Full Re-generation',
               color=COLOR_PRIMARY, edgecolor='black', linewidth=2, hatch=HATCH_DIAGONAL)
bars2 = ax.bar(x + width/2, adapt_violations, width,
               label='Constraint-Based Adaptation',
               color=COLOR_SECONDARY, edgecolor='black', linewidth=2, hatch=HATCH_DOTS)

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 1,
                '0%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='gray')

for bar in bars2:
    height = bar.get_height()
    if height > 0:
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    else:
        ax.text(bar.get_x() + bar.get_width()/2., 1,
                '0%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='gray')

ax.set_ylabel('Violation Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Constraint Violation Rate Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(constraints, fontsize=12)
ax.set_ylim(0, max(max(full_violations), max(adapt_violations)) * 1.3 + 5)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq5_violation_rate_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq5_violation_rate_comparison.png")
plt.close()

# ===================================================================
# PLOT 4: Per-Scenario Re-planning Time Comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(16, 8))

scenarios = full_regen['scenario_id'].values
scenario_names = [f"S{i}" for i in scenarios]

full_times = full_regen.sort_values('scenario_id')['replanning_time_s'].values
adapt_times = constraint_based.sort_values('scenario_id')['replanning_time_s'].values

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x - width/2, full_times, width,
               label='Full Re-generation',
               color=COLOR_PRIMARY, edgecolor='black', linewidth=1.5, hatch=HATCH_DIAGONAL)
bars2 = ax.bar(x + width/2, adapt_times, width,
               label='Constraint-Based Adaptation',
               color=COLOR_SECONDARY, edgecolor='black', linewidth=1.5, hatch=HATCH_DOTS)

ax.set_xlabel('Test Scenarios', fontsize=14, fontweight='bold')
ax.set_ylabel('Re-planning Time (seconds)', fontsize=14, fontweight='bold')
ax.set_title('Re-planning Time per Scenario', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(scenario_names, fontsize=10)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq5_per_scenario_time.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq5_per_scenario_time.png")
plt.close()

# ===================================================================
# PLOT 5: Dumbbell Chart - Before/After Comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 8))

metrics = ['Avg. Preservation\nRate (%)', 'Avg. Re-planning\nTime (s)', 'Total Violations\n(count)']

# Calculate total violations
total_violations_full = sum([full_regen[col].sum() for col in
    ['speed_limit_violation', 'workspace_boundary_violation',
     'collision_avoidance_violation', 'tool_constraint_violation']])
total_violations_adapt = sum([constraint_based[col].sum() for col in
    ['speed_limit_violation', 'workspace_boundary_violation',
     'collision_avoidance_violation', 'tool_constraint_violation']])

# Normalize values to 0-100 scale for visualization
full_vals = [avg_preservation_full, avg_time_full, total_violations_full]
adapt_vals = [avg_preservation_constraint, avg_time_constraint, total_violations_adapt]

# For display, we'll show actual values but plot normalized
y_positions = np.arange(len(metrics))

# Draw connecting lines
for i in range(len(metrics)):
    ax.plot([full_vals[i], adapt_vals[i]], [i, i], color='#374151', linewidth=3, zorder=1)

# Draw dots for Full Re-generation
ax.scatter(full_vals, y_positions, s=400, color=COLOR_PRIMARY, edgecolor='black',
           linewidth=2, zorder=2, label='Full Re-generation')

# Draw dots for Constraint-Based
ax.scatter(adapt_vals, y_positions, s=400, color=COLOR_SECONDARY, edgecolor='black',
           linewidth=2, zorder=2, label='Constraint-Based Adaptation')

# Add value labels
labels_full = [f'{avg_preservation_full:.1f}%', f'{avg_time_full:.1f}s', f'{int(total_violations_full)}']
labels_adapt = [f'{avg_preservation_constraint:.1f}%', f'{avg_time_constraint:.1f}s', f'{int(total_violations_adapt)}']

for i in range(len(metrics)):
    # Full regen label
    ax.annotate(labels_full[i], (full_vals[i], i), textcoords="offset points",
                xytext=(0, 18), ha='center', fontsize=11, fontweight='bold', color=COLOR_PRIMARY)
    # Constraint-based label
    ax.annotate(labels_adapt[i], (adapt_vals[i], i), textcoords="offset points",
                xytext=(0, 18), ha='center', fontsize=11, fontweight='bold', color=COLOR_SECONDARY)

ax.set_yticks(y_positions)
ax.set_yticklabels(metrics, fontsize=12)
ax.set_xlabel('Value', fontsize=14, fontweight='bold')
ax.set_title('Full Re-generation vs Constraint-Based Adaptation', fontsize=16, fontweight='bold', pad=15)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq5_dumbbell_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq5_dumbbell_comparison.png")
plt.close()

# ===================================================================
# PLOT 6: Summary Metrics Bar Chart
# ===================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Subplot 1: Preservation Rate
ax1 = axes[0]
methods = ['Full\nRe-gen', 'Constraint\nBased']
preservation_vals = [avg_preservation_full, avg_preservation_constraint]
bars1 = ax1.bar(methods, preservation_vals, color=[COLOR_PRIMARY, COLOR_SECONDARY],
                edgecolor='black', linewidth=2, hatch=[HATCH_DIAGONAL, HATCH_DOTS])
for bar, val in zip(bars1, preservation_vals):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax1.set_ylabel('Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Preservation Rate', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 110)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 2: Re-planning Time
ax2 = axes[1]
time_vals = [avg_time_full, avg_time_constraint]
bars2 = ax2.bar(methods, time_vals, color=[COLOR_PRIMARY, COLOR_SECONDARY],
                edgecolor='black', linewidth=2, hatch=[HATCH_DIAGONAL, HATCH_DOTS])
for bar, val in zip(bars2, time_vals):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{val:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax2.set_ylabel('Time (s)', fontsize=12, fontweight='bold')
ax2.set_title('Re-planning Time', fontsize=14, fontweight='bold')
ax2.set_ylim(0, max(time_vals) * 1.2)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Subplot 3: Total Violations
ax3 = axes[2]
violation_vals = [total_violations_full, total_violations_adapt]
bars3 = ax3.bar(methods, violation_vals, color=[COLOR_PRIMARY, COLOR_SECONDARY],
                edgecolor='black', linewidth=2, hatch=[HATCH_DIAGONAL, HATCH_DOTS])
for bar, val in zip(bars3, violation_vals):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
             f'{int(val)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title('Total Violations', fontsize=14, fontweight='bold')
ax3.set_ylim(0, max(violation_vals) * 1.3 + 1)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Add overall title
fig.suptitle('Constraint Adaptation Summary', fontsize=16, fontweight='bold', y=1.02)

# Add legend
legend_elements = [
    Patch(facecolor=COLOR_PRIMARY, edgecolor='black', hatch=HATCH_DIAGONAL, label='Full Re-generation'),
    Patch(facecolor=COLOR_SECONDARY, edgecolor='black', hatch=HATCH_DOTS, label='Constraint-Based Adaptation')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
           ncol=2, fontsize=11, framealpha=0.95, edgecolor='black')

plt.tight_layout()
plt.savefig(output_dir / 'rq5_summary_metrics.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq5_summary_metrics.png")
plt.close()

# ===================================================================
# Clean up old table files
# ===================================================================
old_files = [
    'rq5_constraint_preservation_table.png',
    'rq5_constraint_preservation_table.csv',
    'rq5_replanning_overhead_table.png',
    'rq5_replanning_overhead_table.csv',
    'rq5_constraint_violation_comparison.png',
    'rq5_runtime_adaptation_timeline.png'
]
for old_file in old_files:
    old_path = output_dir / old_file
    if old_path.exists():
        old_path.unlink()
        print(f"✓ Removed old file: {old_file}")

print("\n" + "=" * 70)
print("✅ ALL VISUALIZATIONS GENERATED - ONE PLOT PER FILE")
print("=" * 70)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files (6 plots):")
print("  1. rq5_preservation_rate_comparison.png - Bar chart (both strategies)")
print("  2. rq5_replanning_time_comparison.png - Bar chart with improvement")
print("  3. rq5_violation_rate_comparison.png - Bar chart (both strategies)")
print("  4. rq5_per_scenario_time.png - Per-scenario comparison")
print("  5. rq5_dumbbell_comparison.png - Dumbbell chart")
print("  6. rq5_summary_metrics.png - 3-panel summary")

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
print(f"\nConstraint Preservation:")
print(f"  Full Regeneration:        {avg_preservation_full:.1f}%")
print(f"  Constraint-Based:         {avg_preservation_constraint:.1f}%")
print(f"  Difference:               {avg_preservation_constraint - avg_preservation_full:+.1f}%")

print(f"\nRe-planning Time:")
print(f"  Full Re-planning:         {avg_time_full:.2f}s")
print(f"  Constraint-Based:         {avg_time_constraint:.2f}s")
print(f"  Overhead Reduction:       {overhead_reduction:.1f}%")

print(f"\nTotal Violations:")
print(f"  Full Regeneration:        {int(total_violations_full)}")
print(f"  Constraint-Based:         {int(total_violations_adapt)}")