"""Statistical analysis utilities for research question evaluation."""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def calculate_percentage(count: int, total: int) -> float:
    # calculate percentage with safety check for division by zero
    if total == 0:
        return 0.0
    return (count / total) * 100


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """
    Calculate Cohen's d effect size.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def paired_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """
    Perform paired t-test and return statistics.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    t_stat, p_value = stats.ttest_rel(group1, group2)
    effect_size = cohens_d(group1, group2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "significant": p_value < 0.05
    }


def independent_t_test(group1: List[float], group2: List[float]) -> Dict[str, float]:
    """
    Perform independent t-test and return statistics.

    Args:
        group1: First group of values
        group2: Second group of values

    Returns:
        Dictionary with t-statistic, p-value, and effect size
    """
    t_stat, p_value = stats.ttest_ind(group1, group2)
    effect_size = cohens_d(group1, group2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "significant": p_value < 0.05
    }


def confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval for the mean.

    Args:
        data: List of values
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)

    return (mean - margin, mean + margin)


def descriptive_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate descriptive statistics for a dataset.

    Args:
        data: List of values

    Returns:
        Dictionary with mean, median, std, min, max, and percentiles
    """
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data, ddof=1),
        "min": np.min(data),
        "max": np.max(data),
        "q25": np.percentile(data, 25),
        "q75": np.percentile(data, 75),
        "count": len(data)
    }


def compare_strategies(df: pd.DataFrame, metric_column: str, strategy_column: str = "strategy") -> pd.DataFrame:
    """
    Compare metrics across different strategies with statistical tests.

    Args:
        df: DataFrame with results
        metric_column: Name of the metric column to compare
        strategy_column: Name of the strategy column (default "strategy")

    Returns:
        DataFrame with comparison statistics
    """
    strategies = df[strategy_column].unique()
    comparisons = []

    for i, strategy1 in enumerate(strategies):
        for strategy2 in strategies[i+1:]:
            group1 = df[df[strategy_column] == strategy1][metric_column].tolist()
            group2 = df[df[strategy_column] == strategy2][metric_column].tolist()

            if len(group1) > 0 and len(group2) > 0:
                test_result = independent_t_test(group1, group2)

                comparisons.append({
                    "strategy_1": strategy1,
                    "strategy_2": strategy2,
                    "mean_1": np.mean(group1),
                    "mean_2": np.mean(group2),
                    "diff": np.mean(group1) - np.mean(group2),
                    "t_statistic": test_result["t_statistic"],
                    "p_value": test_result["p_value"],
                    "effect_size": test_result["effect_size"],
                    "significant": test_result["significant"]
                })

    return pd.DataFrame(comparisons)


def calculate_reduction_percentage(baseline: float, improved: float) -> float:
    """
    Calculate reduction percentage (for metrics where lower is better).

    Args:
        baseline: Baseline value
        improved: Improved value

    Returns:
        Percentage reduction
    """
    if baseline == 0:
        return 0.0
    return ((baseline - improved) / baseline) * 100


def calculate_improvement_percentage(baseline: float, improved: float) -> float:
    """
    Calculate improvement percentage (for metrics where higher is better).

    Args:
        baseline: Baseline value
        improved: Improved value

    Returns:
        Percentage improvement
    """
    if baseline == 0:
        return 0.0
    return ((improved - baseline) / baseline) * 100