"""
Generate RQ3 visualizations - ONE PLOT PER PNG FILE

Each visualization is a separate file with a single plot.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Read results
df = pd.read_csv('questions/RQ3_conversational_efficiency/results/raw_data/rq3_results_v2.csv')

# Calculate statistics
full_regen = df[df['approach'] == 'full_regeneration']
conv_refine = df[df['approach'] == 'conversational_refinement']

avg_tokens_full = full_regen['total_tokens'].mean()
avg_tokens_refine = conv_refine['total_tokens'].mean()
token_reduction = ((avg_tokens_full - avg_tokens_refine) / avg_tokens_full) * 100

avg_time_full = full_regen['response_time_s'].mean()
avg_time_refine = conv_refine['response_time_s'].mean()
time_reduction = ((avg_time_full - avg_time_refine) / avg_time_full) * 100

print("=" * 70)
print("RQ3 FINAL VISUALIZATIONS - ONE PLOT PER FILE")
print("=" * 70)
print(f"\nToken Consumption:")
print(f"  Full Regeneration: {avg_tokens_full:.0f} tokens")
print(f"  Conversational Refinement: {avg_tokens_refine:.0f} tokens")
print(f"  Token Reduction: {token_reduction:.1f}%")

print(f"\nResponse Time:")
print(f"  Full Regeneration: {avg_time_full:.2f}s")
print(f"  Conversational Refinement: {avg_time_refine:.2f}s")
print(f"  Time Reduction: {time_reduction:.1f}%")

# Create output directory
output_dir = Path('questions/RQ3_conversational_efficiency/results/visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

# Colorful palette - grayscale-distinguishable (different luminance values)
# Deep Blue (dark in grayscale), Coral Orange (light in grayscale)
COLOR_PRIMARY = '#2563EB'    # Deep Blue - for "Full Regeneration" (darker in grayscale)
COLOR_SECONDARY = '#F97316'  # Coral Orange - for "Conversational Refinement" (lighter in grayscale)
COLOR_ACCENT = '#059669'     # Emerald Green - for single series charts
COLOR_TABLE_HEADER = '#1E3A5F'  # Dark Blue for table headers
COLOR_TABLE_ROW1 = '#E8F4FD'    # Light Blue tint
COLOR_TABLE_ROW2 = '#FFFFFF'    # White

# Hatch patterns for additional grayscale distinction
HATCH_DIAGONAL = '///'
HATCH_DOTS = '...'
HATCH_SOLID = ''

# ===================================================================
# PLOT 1: Token Efficiency Bar Chart
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 7))

methods = ['Full Regeneration', 'Conversational\nRefinement']
tokens = [avg_tokens_full, avg_tokens_refine]

bars = ax.bar(methods, tokens, color=[COLOR_PRIMARY, COLOR_SECONDARY],
              width=0.6, edgecolor='black', linewidth=2,
              hatch=[HATCH_DIAGONAL, HATCH_DOTS])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=16, fontweight='bold')

# Styling
ax.set_ylabel('Avg. Tokens Used', fontsize=14, fontweight='bold')
ax.set_title('Token Efficiency Bar Chart', fontsize=16, fontweight='bold', pad=15)
ax.set_ylim(0, max(tokens) * 1.15)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.tick_params(axis='both', labelsize=12)

# Add legend below
legend_labels = [f'Full Regeneration', f'Conversational Refinement',
                 f'Token Reduction: {token_reduction:.1f}%']
legend_handles = [plt.Rectangle((0,0),1,1, facecolor=COLOR_PRIMARY, edgecolor='black', hatch=HATCH_DIAGONAL),
                  plt.Rectangle((0,0),1,1, facecolor=COLOR_SECONDARY, edgecolor='black', hatch=HATCH_DOTS),
                  plt.Line2D([0], [0], color='none')]
ax.legend(legend_handles, legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.12),
          ncol=3, framealpha=0.95, edgecolor='black', shadow=False, fontsize=12)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq3_token_efficiency_bar_chart.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: rq3_token_efficiency_bar_chart.png")
plt.close()

# ===================================================================
# PLOT 2: Token Consumption Comparison Table
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

token_data = [
    ['Full Regeneration', f'{int(avg_tokens_full)}', '-'],
    ['Conversational Refinement', f'{int(avg_tokens_refine)}', f'{token_reduction:.1f}%']
]
token_table = ax.table(cellText=token_data,
                       colLabels=['Method', 'Avg. Tokens Used', 'Token Reduction (%)'],
                       cellLoc='center',
                       loc='center',
                       colWidths=[0.40, 0.30, 0.30])
token_table.auto_set_font_size(False)
token_table.set_fontsize(13)
token_table.scale(1, 3)

# Style header
for i in range(3):
    token_table[(0, i)].set_facecolor(COLOR_TABLE_HEADER)
    token_table[(0, i)].set_text_props(weight='bold', color='white')

# Style cells
for i in range(1, 3):
    for j in range(3):
        token_table[(i, j)].set_facecolor(COLOR_TABLE_ROW1 if i % 2 == 0 else COLOR_TABLE_ROW2)

ax.set_title('Token Consumption Comparison', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq3_token_consumption_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq3_token_consumption_table.png")

# Save CSV
csv_df = pd.DataFrame(token_data, columns=['Method', 'Avg. Tokens Used', 'Token Reduction (%)'])
csv_df.to_csv(output_dir / 'rq3_token_consumption_table.csv', index=False)
print(f"✓ Saved: rq3_token_consumption_table.csv")
plt.close()

# ===================================================================
# PLOT 3: Response Time Comparison Table
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('off')

time_data = [
    ['Full Regeneration', f'{avg_time_full:.2f}s'],
    ['Conversational Refinement', f'{avg_time_refine:.2f}s']
]
time_table = ax.table(cellText=time_data,
                      colLabels=['Method', 'Avg. Response Time (s)'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.50, 0.50])
time_table.auto_set_font_size(False)
time_table.set_fontsize(13)
time_table.scale(1, 3)

# Style header
for i in range(2):
    time_table[(0, i)].set_facecolor(COLOR_TABLE_HEADER)
    time_table[(0, i)].set_text_props(weight='bold', color='white')

# Style cells
for i in range(1, 3):
    for j in range(2):
        time_table[(i, j)].set_facecolor(COLOR_TABLE_ROW1 if i % 2 == 0 else COLOR_TABLE_ROW2)

ax.set_title('Response Time Comparison', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq3_response_time_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq3_response_time_table.png")

# Save CSV
csv_df = pd.DataFrame(time_data, columns=['Method', 'Avg. Response Time (s)'])
csv_df.to_csv(output_dir / 'rq3_response_time_table.csv', index=False)
print(f"✓ Saved: rq3_response_time_table.csv")
plt.close()

# ===================================================================
# PLOT 4: Per-Scenario Token Comparison
# ===================================================================
fig, ax = plt.subplots(figsize=(16, 9))

scenarios = full_regen.sort_values('scenario_id')['scenario_name'].values
x = range(len(scenarios))
width = 0.35

full_tokens = full_regen.sort_values('scenario_id')['total_tokens'].values
refine_tokens = conv_refine.sort_values('scenario_id')['total_tokens'].values

bars1 = ax.bar([i - width/2 for i in x], full_tokens, width, label='Full Regeneration',
               color=COLOR_PRIMARY, edgecolor='black', linewidth=1.5, hatch=HATCH_DIAGONAL)
bars2 = ax.bar([i + width/2 for i in x], refine_tokens, width, label='Conversational Refinement',
               color=COLOR_SECONDARY, edgecolor='black', linewidth=1.5, hatch=HATCH_DOTS)

ax.set_xlabel('Test Scenarios', fontsize=14, fontweight='bold')
ax.set_ylabel('Total Tokens', fontsize=14, fontweight='bold')
ax.set_title('Token Usage per Scenario', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels([f"T{i+1}" for i in x], fontsize=11)

ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

max_val = max(max(full_tokens), max(refine_tokens))
ax.set_ylim(0, max_val * 1.15)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq3_per_scenario_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq3_per_scenario_comparison.png")
plt.close()

# ===================================================================
# PLOT 5: Token Savings Distribution
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate per-scenario savings
full_sorted = full_regen.sort_values('scenario_id')
refine_sorted = conv_refine.sort_values('scenario_id')

token_savings = ((full_sorted['total_tokens'].values - refine_sorted['total_tokens'].values) /
                 full_sorted['total_tokens'].values * 100)

# Use grayscale bars
bars = ax.barh(range(len(token_savings)), token_savings, color=COLOR_SECONDARY,
               edgecolor='black', linewidth=1.5, hatch=HATCH_DIAGONAL)
ax.set_yticks(range(len(token_savings)))
ax.set_yticklabels([f"Test {i+1}" for i in range(len(token_savings))], fontsize=10)
ax.set_xlabel('Token Savings (%)', fontsize=12, fontweight='bold')
ax.set_title('Token Savings per Scenario', fontsize=14, fontweight='bold', pad=15)

# Add reference lines in black with different styles
avg_token_save = token_savings.mean()
ax.axvline(x=70, color='black', linestyle='--', linewidth=2, label='70% target')
ax.axvline(x=avg_token_save, color='black', linestyle=':', linewidth=2.5, label=f'Avg: {avg_token_save:.1f}%')

# Add avg text annotation at top of plot, outside bars
ax.annotate(f'Avg: {avg_token_save:.1f}%',
            xy=(avg_token_save, len(token_savings) - 0.5),
            xytext=(avg_token_save + 3, len(token_savings) + 0.8),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_ylim(-0.5, len(token_savings) + 1.5)  # Extend y-axis to fit annotation
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq3_token_savings_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq3_token_savings_distribution.png")
plt.close()

# ===================================================================
# PLOT 6: Time Savings Distribution
# ===================================================================
fig, ax = plt.subplots(figsize=(12, 8))

time_savings = ((full_sorted['response_time_s'].values - refine_sorted['response_time_s'].values) /
                full_sorted['response_time_s'].values * 100)

# Use grayscale bars
bars = ax.barh(range(len(time_savings)), time_savings, color=COLOR_PRIMARY,
               edgecolor='black', linewidth=1.5, hatch=HATCH_DOTS)
ax.set_yticks(range(len(time_savings)))
ax.set_yticklabels([f"Test {i+1}" for i in range(len(time_savings))], fontsize=10)
ax.set_xlabel('Time Savings (%)', fontsize=12, fontweight='bold')
ax.set_title('Time Savings per Scenario', fontsize=14, fontweight='bold', pad=15)

# Add reference lines in black with different styles
avg_time_save = time_savings.mean()
ax.axvline(x=70, color='black', linestyle='--', linewidth=2, label='70% target')
ax.axvline(x=avg_time_save, color='black', linestyle=':', linewidth=2.5, label=f'Avg: {avg_time_save:.1f}%')

# Add avg text annotation at top of plot, outside bars
ax.annotate(f'Avg: {avg_time_save:.1f}%',
            xy=(avg_time_save, len(time_savings) - 0.5),
            xytext=(avg_time_save + 3, len(time_savings) + 0.8),
            fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_ylim(-0.5, len(time_savings) + 1.5)  # Extend y-axis to fit annotation
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.10),
          ncol=2, framealpha=0.95, edgecolor='black', shadow=False)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig(output_dir / 'rq3_time_savings_distribution.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: rq3_time_savings_distribution.png")
plt.close()

print("\n" + "=" * 70)
print("✅ ALL VISUALIZATIONS GENERATED - ONE PLOT PER FILE")
print("=" * 70)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files (6 separate plots):")
print("  1. rq3_token_efficiency_bar_chart.png - Bar chart with token reduction")
print("  2. rq3_token_consumption_table.png - Token consumption comparison table")
print("  3. rq3_response_time_table.png - Response time comparison table")
print("  4. rq3_per_scenario_comparison.png - Per-scenario token usage bar chart")
print("  5. rq3_token_savings_distribution.png - Token savings per scenario")
print("  6. rq3_time_savings_distribution.png - Time savings per scenario")