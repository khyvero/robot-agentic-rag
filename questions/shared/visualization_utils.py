"""Visualization utilities for research question results."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Set publication-ready style
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Color palette for strategies - Colorful but grayscale-distinguishable
# Colors chosen to have different luminance values for B&W printing
STRATEGY_COLORS = {
    'zero_shot': '#2563EB',   # Deep Blue (dark in grayscale)
    'single_rag': '#F97316',  # Coral Orange (medium-light in grayscale)
    'dual_rag': '#059669'     # Emerald Green (medium in grayscale)
}


def setup_plot_style():
    # set up consistent plot style
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def save_figure(fig, filepath: str, dpi: int = 300):
    # save figure to file with consistent settings
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Saved figure to {filepath}")
    plt.close(fig)


def plot_comparison_table(df: pd.DataFrame, title: str, filepath: str,
                          columns: Optional[List[str]] = None):
    """
    Create a formatted comparison table.

    Args:
        df: DataFrame with comparison data
        title: Table title
        filepath: Output file path
        columns: Optional list of columns to include
    """
    setup_plot_style()

    if columns:
        df = df[columns]

    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=df.values, colLabels=df.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')

    plt.title(title, fontsize=12, weight='bold', pad=20)
    save_figure(fig, filepath)


def plot_bar_comparison(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str,
                        title: str, xlabel: str, ylabel: str, filepath: str,
                        figsize: Tuple[int, int] = (10, 6)):
    """
    Create a grouped bar chart comparing strategies.

    Args:
        df: DataFrame with data
        x_col: Column for x-axis
        y_col: Column for y-axis (metric values)
        hue_col: Column for grouping (usually strategy)
        title: Chart title
        xlabel: X-axis label
        ylabel: Y-axis label
        filepath: Output file path
        figsize: Figure size
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Get unique strategies
    strategies = df[hue_col].unique()
    colors = [STRATEGY_COLORS.get(s, '#95a5a6') for s in strategies]

    sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette=colors)

    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.legend(title=hue_col.replace('_', ' ').title())

    plt.xticks(rotation=45, ha='right')
    save_figure(fig, filepath)


def plot_latency_breakdown(df: pd.DataFrame, title: str, filepath: str,
                          components: List[str] = None,
                          figsize: Tuple[int, int] = (10, 6)):
    """
    Create a stacked bar chart showing latency breakdown by component.

    Args:
        df: DataFrame with latency data
        title: Chart title
        filepath: Output file path
        components: List of component columns (e.g., ['retrieval_time', 'llm_time', 'validation_time'])
        figsize: Figure size
    """
    setup_plot_style()

    if components is None:
        components = ['retrieval_time_ms', 'llm_generation_time_ms', 'validation_time_ms']

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate means by strategy
    if 'strategy' in df.columns:
        data = df.groupby('strategy')[components].mean()
    else:
        data = df[components].mean().to_frame().T

    data.plot(kind='bar', stacked=True, ax=ax, color=['#3498db', '#e74c3c', '#f39c12'])

    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.legend(title='Component', labels=[c.replace('_', ' ').title() for c in components])

    plt.xticks(rotation=0)
    save_figure(fig, filepath)


def plot_failure_mode_distribution(df: pd.DataFrame, title: str, filepath: str,
                                   figsize: Tuple[int, int] = (10, 6)):
    """
    Create a grouped bar chart showing failure mode distribution.

    Args:
        df: DataFrame with failure modes by strategy
        title: Chart title
        filepath: Output file path
        figsize: Figure size
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    # Expect df to have columns: strategy, failure_mode, percentage
    pivot_df = df.pivot(index='failure_mode', columns='strategy', values='percentage')

    pivot_df.plot(kind='bar', ax=ax, color=[STRATEGY_COLORS.get(s, '#95a5a6') for s in pivot_df.columns])

    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Failure Mode', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.legend(title='Strategy')

    plt.xticks(rotation=45, ha='right')
    save_figure(fig, filepath)


def plot_token_comparison(baseline_tokens: float, improved_tokens: float,
                         baseline_time: float, improved_time: float,
                         title: str, filepath: str,
                         figsize: Tuple[int, int] = (10, 5)):
    """
    Create a side-by-side comparison of tokens and time.

    Args:
        baseline_tokens: Baseline token count
        improved_tokens: Improved token count
        baseline_time: Baseline time
        improved_time: Improved time
        title: Chart title
        filepath: Output file path
        figsize: Figure size
    """
    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Token comparison
    categories = ['Full Regeneration', 'Adaptive Editor']
    tokens = [baseline_tokens, improved_tokens]
    colors = ['#e74c3c', '#27ae60']

    ax1.bar(categories, tokens, color=colors)
    ax1.set_ylabel('Tokens Used', fontsize=11)
    ax1.set_title('Token Consumption', fontsize=11, weight='bold')
    reduction = ((baseline_tokens - improved_tokens) / baseline_tokens) * 100
    ax1.text(0.5, max(tokens) * 0.95, f'{reduction:.1f}% reduction',
             ha='center', fontsize=9, weight='bold')

    # Time comparison
    times = [baseline_time, improved_time]
    ax2.bar(categories, times, color=colors)
    ax2.set_ylabel('Response Time (s)', fontsize=11)
    ax2.set_title('Response Time', fontsize=11, weight='bold')
    time_reduction = ((baseline_time - improved_time) / baseline_time) * 100
    ax2.text(0.5, max(times) * 0.95, f'{time_reduction:.1f}% faster',
             ha='center', fontsize=9, weight='bold')

    fig.suptitle(title, fontsize=12, weight='bold')
    save_figure(fig, filepath)


def plot_constraint_preservation(df: pd.DataFrame, title: str, filepath: str,
                                 constraints: List[str] = None,
                                 figsize: Tuple[int, int] = (10, 6)):
    """
    Create a grouped bar chart showing constraint preservation rates.

    Args:
        df: DataFrame with constraint preservation data
        title: Chart title
        filepath: Output file path
        constraints: List of constraint types
        figsize: Figure size
    """
    setup_plot_style()

    if constraints is None:
        constraints = ['speed_limits', 'workspace_boundaries', 'collision_avoidance', 'tool_constraints']

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate preservation rates
    if 'strategy' in df.columns:
        data = df.groupby('strategy')[constraints].mean() * 100
    else:
        data = df[constraints].mean().to_frame().T * 100

    data.plot(kind='bar', ax=ax)

    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Strategy', fontsize=11)
    ax.set_ylabel('Preservation Rate (%)', fontsize=11)
    ax.legend(title='Constraint Type', labels=[c.replace('_', ' ').title() for c in constraints])
    ax.set_ylim([0, 105])

    # Add horizontal line at 100%
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100% Preserved')

    plt.xticks(rotation=0)
    save_figure(fig, filepath)


def create_summary_report_figure(metrics_dict: Dict[str, float], title: str, filepath: str,
                                 figsize: Tuple[int, int] = (10, 8)):
    """
    Create a comprehensive summary figure with multiple metrics.

    Args:
        metrics_dict: Dictionary of metric names and values
        title: Chart title
        filepath: Output file path
        figsize: Figure size
    """
    setup_plot_style()

    fig, ax = plt.subplots(figsize=figsize)

    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Create horizontal bar chart
    y_pos = np.arange(len(metrics))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(metrics)))

    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.invert_yaxis()
    ax.set_xlabel('Value', fontsize=11)
    ax.set_title(title, fontsize=12, weight='bold')

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)

    save_figure(fig, filepath)