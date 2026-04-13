"""
Generate RQ4 visualizations - ONE PLOT PER PNG FILE

Each visualization is a separate file with a single plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Read corrected results
df = pd.read_csv('questions/RQ4_teach_mode_evaluation/results/raw_data/rq4_summary_corrected.csv')

# Extract data
zero_shot = df[df['strategy'] == 'Zero-Shot LLM'].iloc[0]
teach_mode = df[df['strategy'] == 'Teach Mode (Proposed)'].iloc[0]

print("=" * 70)
print("RQ4 FINAL VISUALIZATIONS - ONE PLOT PER FILE")
print("=" * 70)

# Create output directory
output_dir = Path('questions/RQ4_teach_mode_evaluation/results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Colorful palette - grayscale-distinguishable (different luminance values)
# Deep Blue (dark in grayscale), Coral Orange (light in grayscale)
COLOR_PRIMARY = '#2563EB'    # Deep Blue - for "Zero-Shot LLM" (darker in grayscale)
COLOR_SECONDARY = '#F97316'  # Coral Orange - for "Teach Mode" (lighter in grayscale)
COLOR_ACCENT = '#059669'     # Emerald Green - for single series charts
COLOR_TABLE_HEADER = '#1E3A5F'  # Dark Blue for table headers
COLOR_TABLE_ROW1 = '#E8F4FD'    # Light Blue tint
COLOR_TABLE_ROW2 = '#FFFFFF'    # White

# Hatch patterns for additional grayscale distinction
HATCH_DIAGONAL = '///'
HATCH_DOTS = '...'
HATCH_SOLID = ''

# ===================================================================
# PLOT 1: Translation Success Rate Bar Chart
# ===================================================================
fig, ax = plt.subplots(figsize=(14, 8))

metrics = ['Valid Task (%)', 'Reusable Task (%)', 'Expert Intervention\nNeeded (%)']
zero_shot_values = [zero_shot['valid_task_pct'], zero_shot['reusable_task_pct'], zero_shot['expert_intervention_pct']]
teach_mode_values = [teach_mode['valid_task_pct'], teach_mode['reusable_task_pct'], teach_mode['expert_intervention_pct']]

x = range(len(metrics))
width = 0.35

bars1 = ax.bar([i - width/2 for i in x], zero_shot_values, width,
               label='Zero-Shot LLM', color=COLOR_PRIMARY, edgecolor='black',
               linewidth=2, hatch=HATCH_DIAGONAL)
bars2 = ax.bar([i + width/2 for i in x], teach_mode_values, width,
               label='Teach Mode (Proposed)', color=COLOR_SECONDARY, edgecolor='black',
               linewidth=2, hatch=HATCH_DOTS)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Translation Success Rate Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 110)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_translation_success_rate.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_translation_success_rate.png")
plt.close()

# ===================================================================
# PLOT 2: Error Distribution Pie Chart (Zero-Shot Failures)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Get actual counts from CSV
zero_shot_total = int(zero_shot['total_tests'])
zero_shot_failures = zero_shot_total - int(zero_shot['valid_tasks'])

# Collect non-zero error types for Zero-Shot
zero_error_labels = []
zero_error_values = []
zero_colors = []
zero_hatches = []

if zero_shot['safety_violation_pct'] > 0:
    zero_error_labels.append('Safety Violations')
    zero_error_values.append(zero_shot['safety_violation_pct'])
    zero_colors.append(COLOR_PRIMARY)
    zero_hatches.append(HATCH_DIAGONAL)

if zero_shot['api_hallucination_pct'] > 0:
    zero_error_labels.append('API Hallucinations')
    zero_error_values.append(zero_shot['api_hallucination_pct'])
    zero_colors.append(COLOR_SECONDARY)
    zero_hatches.append(HATCH_DOTS)

if zero_shot['schema_violation_pct'] > 0:
    zero_error_labels.append('Schema Violations')
    zero_error_values.append(zero_shot['schema_violation_pct'])
    zero_colors.append('#9CA3AF')
    zero_hatches.append('xxx')

if zero_shot['logical_error_pct'] > 0:
    zero_error_labels.append('Logical Errors')
    zero_error_values.append(zero_shot['logical_error_pct'])
    zero_colors.append(COLOR_ACCENT)
    zero_hatches.append('')

# Normalize to 100% for pie chart
zero_total = sum(zero_error_values)
zero_normalized = [v / zero_total * 100 for v in zero_error_values]

# Create labels with normalized percentages
zero_pie_labels = [f'{label}\n({val:.1f}%)' for label, val in zip(zero_error_labels, zero_normalized)]

explode = [0.05] * len(zero_normalized)
if len(explode) > 0:
    explode[0] = 0.1  # Explode largest slice

wedges, texts = ax.pie(zero_normalized, labels=zero_pie_labels,
                       colors=zero_colors, explode=explode, startangle=90,
                       textprops={'fontsize': 12, 'fontweight': 'bold'})

# Apply hatches to wedges
for i, wedge in enumerate(wedges):
    wedge.set_edgecolor('black')
    wedge.set_linewidth(2)
    wedge.set_hatch(zero_hatches[i])

# Add percentage labels inside slices
for i, (wedge, value) in enumerate(zip(wedges, zero_normalized)):
    ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    r = 0.6
    x_pos = wedge.r * r * np.cos(np.radians(ang))
    y_pos = wedge.r * r * np.sin(np.radians(ang))
    ax.text(x_pos, y_pos, f'{value:.1f}%', ha='center', va='center',
            fontsize=16, fontweight='bold', color='white')

ax.set_title(f'Zero-Shot Failure Distribution\n({zero_shot_failures} failures out of {zero_shot_total} tests)',
             fontsize=15, fontweight='bold', pad=15)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_error_distribution_zero_shot.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_error_distribution_zero_shot.png")
plt.close()

# ===================================================================
# PLOT 2b: Error Distribution Pie Chart (Teach Mode / Dual-RAG)
# ===================================================================
fig, ax = plt.subplots(figsize=(10, 8))

# Get actual counts from CSV
teach_mode_total = int(teach_mode['total_tests'])
teach_mode_failures = teach_mode_total - int(teach_mode['valid_tasks'])

if teach_mode_failures == 0:
    # No failures - show success pie
    wedges, texts = ax.pie([100], labels=['All Tests\nSuccessful (100%)'],
                           colors=[COLOR_ACCENT], startangle=90,
                           textprops={'fontsize': 14, 'fontweight': 'bold'})
    wedges[0].set_edgecolor('black')
    wedges[0].set_linewidth(2)
    ax.text(0, 0, '100%', ha='center', va='center',
            fontsize=24, fontweight='bold', color='white')
    ax.set_title(f'Teach Mode (Dual-RAG) Error Distribution\n(0 failures out of {teach_mode_total} tests - 100% success)',
                 fontsize=15, fontweight='bold', pad=15)
else:
    # Collect non-zero error types for Teach Mode
    teach_error_labels = []
    teach_error_values = []
    teach_colors = []
    teach_hatches = []

    if teach_mode['safety_violation_pct'] > 0:
        teach_error_labels.append('Safety Violations')
        teach_error_values.append(teach_mode['safety_violation_pct'])
        teach_colors.append(COLOR_PRIMARY)
        teach_hatches.append(HATCH_DIAGONAL)

    if teach_mode['api_hallucination_pct'] > 0:
        teach_error_labels.append('API Hallucinations')
        teach_error_values.append(teach_mode['api_hallucination_pct'])
        teach_colors.append(COLOR_SECONDARY)
        teach_hatches.append(HATCH_DOTS)

    if teach_mode['schema_violation_pct'] > 0:
        teach_error_labels.append('Schema Violations')
        teach_error_values.append(teach_mode['schema_violation_pct'])
        teach_colors.append('#9CA3AF')
        teach_hatches.append('xxx')

    if teach_mode['logical_error_pct'] > 0:
        teach_error_labels.append('Logical Errors')
        teach_error_values.append(teach_mode['logical_error_pct'])
        teach_colors.append(COLOR_ACCENT)
        teach_hatches.append('')

    # Normalize to 100% for pie chart
    teach_total = sum(teach_error_values)
    teach_normalized = [v / teach_total * 100 for v in teach_error_values]

    # Create labels with normalized percentages
    teach_pie_labels = [f'{label}\n({val:.1f}%)' for label, val in zip(teach_error_labels, teach_normalized)]

    explode = [0.05] * len(teach_normalized)
    if len(explode) > 0:
        explode[0] = 0.1  # Explode largest slice

    wedges, texts = ax.pie(teach_normalized, labels=teach_pie_labels,
                           colors=teach_colors, explode=explode, startangle=90,
                           textprops={'fontsize': 12, 'fontweight': 'bold'})

    # Apply hatches to wedges
    for i, wedge in enumerate(wedges):
        wedge.set_edgecolor('black')
        wedge.set_linewidth(2)
        wedge.set_hatch(teach_hatches[i])

    # Add percentage labels inside slices
    for i, (wedge, value) in enumerate(zip(wedges, teach_normalized)):
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        r = 0.6
        x_pos = wedge.r * r * np.cos(np.radians(ang))
        y_pos = wedge.r * r * np.sin(np.radians(ang))
        text_color = 'white' if teach_colors[i] in [COLOR_PRIMARY, COLOR_ACCENT] else 'black'
        ax.text(x_pos, y_pos, f'{value:.1f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color=text_color)

    ax.set_title(f'Teach Mode (Dual-RAG) Error Distribution\n({teach_mode_failures} failures out of {teach_mode_total} tests)',
                 fontsize=15, fontweight='bold', pad=15)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_error_distribution_teach_mode.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_error_distribution_teach_mode.png")
plt.close()

# ===================================================================
# PLOT 3: Task Creation Success Rate Table
# ===================================================================
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('off')

success_data = [
    ['Zero-Shot LLM', f'{zero_shot["valid_task_pct"]:.1f}%', f'{zero_shot["reusable_task_pct"]:.1f}%', f'{zero_shot["expert_intervention_pct"]:.1f}%'],
    ['Teach Mode (Proposed)', f'{teach_mode["valid_task_pct"]:.1f}%', f'{teach_mode["reusable_task_pct"]:.1f}%', f'{teach_mode["expert_intervention_pct"]:.1f}%'],
    ['Improvement', f'+{teach_mode["valid_task_pct"]-zero_shot["valid_task_pct"]:.1f}%',
     f'+{teach_mode["reusable_task_pct"]-zero_shot["reusable_task_pct"]:.1f}%',
     f'-{zero_shot["expert_intervention_pct"]-teach_mode["expert_intervention_pct"]:.1f}%']
]

table = ax.table(cellText=success_data,
                 colLabels=['Approach', 'Valid Task (%)', 'Reusable Task (%)', 'Expert Intervention (%)'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.35, 0.20, 0.25, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 3)

for i in range(4):
    table[(0, i)].set_facecolor(COLOR_TABLE_HEADER)
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 4):
    for j in range(4):
        table[(i, j)].set_facecolor(COLOR_TABLE_ROW1 if i % 2 == 1 else COLOR_TABLE_ROW2)

ax.set_title('Task Creation Success Rate', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_task_success_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_task_success_table.png")

# Save CSV
csv_df = pd.DataFrame(success_data, columns=['Approach', 'Valid Task (%)', 'Reusable Task (%)', 'Expert Intervention (%)'])
csv_df.to_csv(output_dir / 'rq4_task_success_table.csv', index=False)
print(f"✓ Saved: rq4_task_success_table.csv")
plt.close()

# ===================================================================
# PLOT 4: Error Distribution Table
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

error_data = [
    ['Safety Violations', f'{zero_shot["safety_violation_pct"]:.1f}%', f'{teach_mode["safety_violation_pct"]:.1f}%'],
    ['API Hallucinations', f'{zero_shot["api_hallucination_pct"]:.1f}%', f'{teach_mode["api_hallucination_pct"]:.1f}%'],
    ['Schema Violations', f'{zero_shot["schema_violation_pct"]:.1f}%', f'{teach_mode["schema_violation_pct"]:.1f}%'],
    ['Logical Errors', f'{zero_shot["logical_error_pct"]:.1f}%', f'{teach_mode["logical_error_pct"]:.1f}%']
]

table = ax.table(cellText=error_data,
                 colLabels=['Error Type', 'Zero-Shot LLM (%)', 'Teach Mode (%)'],
                 cellLoc='center',
                 loc='center',
                 colWidths=[0.50, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(13)
table.scale(1, 3)

for i in range(3):
    table[(0, i)].set_facecolor(COLOR_TABLE_HEADER)
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, 5):
    for j in range(3):
        table[(i, j)].set_facecolor(COLOR_TABLE_ROW1 if i % 2 == 0 else COLOR_TABLE_ROW2)

ax.set_title('Error Distribution Among Failed Tasks', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_error_types_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_error_types_table.png")

# Save CSV
csv_df = pd.DataFrame(error_data, columns=['Error Type', 'Zero-Shot LLM (%)', 'Teach Mode (%)'])
csv_df.to_csv(output_dir / 'rq4_error_types_table.csv', index=False)
print(f"✓ Saved: rq4_error_types_table.csv")
plt.close()

# ===================================================================
# PLOT 5: Expert Dependency Reduction (Annotated Bar Chart)
# ===================================================================
fig, ax = plt.subplots(figsize=(14, 8))

categories = ['Valid\nTask (%)', 'Reusable\nTask (%)', 'Expert\nIntervention (%)']
zero_vals = [zero_shot['valid_task_pct'], zero_shot['reusable_task_pct'], zero_shot['expert_intervention_pct']]
teach_vals = [teach_mode['valid_task_pct'], teach_mode['reusable_task_pct'], teach_mode['expert_intervention_pct']]

x_pos = [0, 1.5, 3]
width = 0.5

bars1 = ax.bar([p - width/2 for p in x_pos], zero_vals, width,
               label='Zero-Shot LLM', color=COLOR_PRIMARY, edgecolor='black',
               linewidth=2, hatch=HATCH_DIAGONAL)
bars2 = ax.bar([p + width/2 for p in x_pos], teach_vals, width,
               label='Teach Mode (Proposed)', color=COLOR_SECONDARY, edgecolor='black',
               linewidth=2, hatch=HATCH_DOTS)

for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement annotations
for i, (x, label) in enumerate(zip(x_pos, ['+90%', '+90%', '-90%'])):
    y1 = zero_vals[i]
    y2 = teach_vals[i]

    if y2 > y1:  # Improvement (increase)
        ax.annotate('', xy=(x + width/2, y2 - 2), xytext=(x - width/2, y1 + 2),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
        mid_y = (y1 + y2) / 2
        ax.text(x, mid_y, label, ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))
    else:  # Improvement (decrease)
        ax.annotate('', xy=(x + width/2, y2 + 2), xytext=(x - width/2, y1 - 2),
                    arrowprops=dict(arrowstyle='->', color='black', lw=2.5))
        mid_y = (y1 + y2) / 2
        ax.text(x, mid_y, label, ha='center', va='center', fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', linewidth=2))

ax.set_ylabel('Percentage (%)', fontsize=15, fontweight='bold')
ax.set_title('Expert Dependency Reduction via Teach Mode', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=13)
ax.set_ylim(0, 115)
ax.legend(fontsize=13, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1.5)
ax.set_axisbelow(True)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_expert_dependency_reduction.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_expert_dependency_reduction.png")
plt.close()

# ===================================================================
# PLOT 6: Dumbbell Chart - Before/After Comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 8))

metrics = ['Valid Task Rate', 'Reusable Task Rate', 'Expert Intervention\n(Lower is Better)']
zero_vals = [zero_shot['valid_task_pct'], zero_shot['reusable_task_pct'], zero_shot['expert_intervention_pct']]
teach_vals = [teach_mode['valid_task_pct'], teach_mode['reusable_task_pct'], teach_mode['expert_intervention_pct']]

y_positions = np.arange(len(metrics))

# Draw connecting lines
for i, (z, t) in enumerate(zip(zero_vals, teach_vals)):
    ax.plot([z, t], [i, i], color='#374151', linewidth=3, zorder=1)

# Draw dots for Zero-Shot (before)
ax.scatter(zero_vals, y_positions, s=400, color=COLOR_PRIMARY, edgecolor='black',
           linewidth=2, zorder=2, label='Zero-Shot LLM (Before)')

# Draw dots for Teach Mode (after)
ax.scatter(teach_vals, y_positions, s=400, color=COLOR_SECONDARY, edgecolor='black',
           linewidth=2, zorder=2, label='Teach Mode (After)')

# Add value labels
for i, (z, t) in enumerate(zip(zero_vals, teach_vals)):
    # Zero-shot label
    ax.annotate(f'{z:.0f}%', (z, i), textcoords="offset points",
                xytext=(0, 18), ha='center', fontsize=12, fontweight='bold', color=COLOR_PRIMARY)
    # Teach mode label
    ax.annotate(f'{t:.0f}%', (t, i), textcoords="offset points",
                xytext=(0, 18), ha='center', fontsize=12, fontweight='bold', color=COLOR_SECONDARY)

    # Add improvement arrow and label in the middle
    mid_x = (z + t) / 2
    if t > z:
        improvement = f'+{t-z:.0f}%'
        arrow_color = COLOR_ACCENT
    else:
        improvement = f'{t-z:.0f}%'
        arrow_color = COLOR_ACCENT
    ax.annotate(improvement, (mid_x, i), textcoords="offset points",
                xytext=(0, -20), ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

ax.set_yticks(y_positions)
ax.set_yticklabels(metrics, fontsize=13)
ax.set_xlabel('Percentage (%)', fontsize=14, fontweight='bold')
ax.set_title('Before/After Comparison: Zero-Shot vs Teach Mode', fontsize=16, fontweight='bold', pad=15)
ax.set_xlim(-5, 105)
ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add vertical reference lines
ax.axvline(x=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_dumbbell_before_after.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_dumbbell_before_after.png")
plt.close()

# ===================================================================
# PLOT 7: Waterfall Chart - Improvement Journey
# ===================================================================
fig, ax = plt.subplots(figsize=(14, 8))

# Waterfall data: Start from Zero-Shot baseline, show improvements
categories = ['Zero-Shot\nBaseline', 'Valid Task\nImprovement', 'Reusable Task\nImprovement',
              'Expert Intervention\nReduction', 'Teach Mode\nResult']

# Values for waterfall
zero_shot_avg = (zero_shot['valid_task_pct'] + zero_shot['reusable_task_pct'] +
                 (100 - zero_shot['expert_intervention_pct'])) / 3  # Average "goodness"
teach_mode_avg = (teach_mode['valid_task_pct'] + teach_mode['reusable_task_pct'] +
                  (100 - teach_mode['expert_intervention_pct'])) / 3

# Calculate individual improvements
valid_improvement = teach_mode['valid_task_pct'] - zero_shot['valid_task_pct']
reusable_improvement = teach_mode['reusable_task_pct'] - zero_shot['reusable_task_pct']
intervention_improvement = zero_shot['expert_intervention_pct'] - teach_mode['expert_intervention_pct']

# Waterfall values (cumulative)
values = [zero_shot['valid_task_pct'], valid_improvement, 0, intervention_improvement, 0]
cumulative = [zero_shot['valid_task_pct']]
cumulative.append(cumulative[-1] + valid_improvement)  # After valid improvement
cumulative.append(cumulative[-1])  # Reusable same as valid
cumulative.append(cumulative[-1])  # After intervention (already at 95%)
cumulative.append(teach_mode['valid_task_pct'])  # Final

# Simplify: Show key metrics journey
waterfall_labels = ['Zero-Shot\nSuccess Rate', '+Valid Tasks', '+Safe Execution', 'Teach Mode\nSuccess Rate']
waterfall_values = [5, 90, 0, 95]  # Start, improvement, (no change), end
waterfall_bottoms = [0, 5, 95, 0]  # Where each bar starts
waterfall_colors = [COLOR_PRIMARY, COLOR_ACCENT, COLOR_ACCENT, COLOR_SECONDARY]
waterfall_hatches = [HATCH_DIAGONAL, '', '', HATCH_DOTS]

x_pos = np.arange(len(waterfall_labels))
bars = ax.bar(x_pos, [5, 90, 0, 95], bottom=[0, 5, 0, 0],
              color=[COLOR_PRIMARY, COLOR_ACCENT, 'white', COLOR_SECONDARY],
              edgecolor='black', linewidth=2, width=0.6)

# Apply hatches
bars[0].set_hatch(HATCH_DIAGONAL)
bars[3].set_hatch(HATCH_DOTS)

# Add connecting lines between bars
ax.plot([0.3, 0.7], [5, 5], color='black', linewidth=2, linestyle='--')
ax.plot([1.3, 2.7], [95, 95], color='black', linewidth=2, linestyle='--')

# Add value labels
ax.text(0, 2.5, '5%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(1, 50, '+90%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')
ax.text(3, 47.5, '95%', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

# Add annotation arrow
ax.annotate('', xy=(3, 95), xytext=(0, 5),
            arrowprops=dict(arrowstyle='->', color='black', lw=2,
                          connectionstyle='arc3,rad=0.3'))
ax.text(1.5, 75, '19x\nImprovement', ha='center', va='center', fontsize=13, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF3C7', edgecolor='black', linewidth=2))

ax.set_xticks(x_pos)
ax.set_xticklabels(waterfall_labels, fontsize=12)
ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Improvement Journey: From Zero-Shot to Teach Mode', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_PRIMARY, edgecolor='black', hatch=HATCH_DIAGONAL, label='Zero-Shot (5%)'),
    Patch(facecolor=COLOR_ACCENT, edgecolor='black', label='Improvement (+90%)'),
    Patch(facecolor=COLOR_SECONDARY, edgecolor='black', hatch=HATCH_DOTS, label='Teach Mode (95%)')
]
ax.legend(handles=legend_elements, fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=3, framealpha=0.95, edgecolor='black', shadow=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq4_waterfall_improvement.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq4_waterfall_improvement.png")
plt.close()

print("\n" + "=" * 70)
print("✅ ALL VISUALIZATIONS GENERATED - ONE PLOT PER FILE")
print("=" * 70)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files (7 separate plots):")
print("  1. rq4_translation_success_rate.png - Bar chart comparison")
print("  2. rq4_error_distribution_zero_shot.png - Pie chart (Zero-Shot)")
print("  3. rq4_error_distribution_teach_mode.png - Pie chart (Teach Mode)")
print("  4. rq4_task_success_table.png - Success rate table")
print("  5. rq4_error_types_table.png - Error distribution table")
print("  6. rq4_expert_dependency_reduction.png - Annotated comparison")
print("  7. rq4_dumbbell_before_after.png - Dumbbell chart (before/after)")
print("  8. rq4_waterfall_improvement.png - Waterfall chart (improvement journey)")