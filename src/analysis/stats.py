"""Statistical tests: t-test, Wilcoxon, ICC, F-test, effect sizes."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class ComparisonResult:
    metric: str
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    p_value: float
    test_name: str
    effect_size: float  # Cohen's d
    significant: bool  # p < 0.05
    ci_95_diff: tuple[float, float]

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "group_a": self.group_a,
            "group_b": self.group_b,
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "p_value": self.p_value,
            "test_name": self.test_name,
            "effect_size": self.effect_size,
            "significant": self.significant,
            "ci_95_diff": list(self.ci_95_diff),
        }


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def compare_groups(
    a: list[float],
    b: list[float],
    metric: str,
    group_a_name: str = "A",
    group_b_name: str = "B",
    paired: bool = False,
) -> ComparisonResult:
    """Compare two groups using appropriate test (t-test or Wilcoxon)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    if len(a_arr) < 3 or len(b_arr) < 3:
        return ComparisonResult(
            metric=metric, group_a=group_a_name, group_b=group_b_name,
            mean_a=float(np.mean(a_arr)), mean_b=float(np.mean(b_arr)),
            std_a=float(np.std(a_arr, ddof=1)) if len(a_arr) > 1 else 0,
            std_b=float(np.std(b_arr, ddof=1)) if len(b_arr) > 1 else 0,
            p_value=1.0, test_name="insufficient_data", effect_size=0.0,
            significant=False, ci_95_diff=(0.0, 0.0),
        )

    # Check normality
    if len(a_arr) >= 8 and len(b_arr) >= 8:
        _, p_norm_a = stats.shapiro(a_arr)
        _, p_norm_b = stats.shapiro(b_arr)
        use_parametric = p_norm_a > 0.05 and p_norm_b > 0.05
    else:
        use_parametric = True  # Default to parametric for small samples

    if paired and len(a_arr) == len(b_arr):
        if use_parametric:
            stat_val, p_val = stats.ttest_rel(a_arr, b_arr)
            test_name = "paired_t_test"
        else:
            stat_val, p_val = stats.wilcoxon(a_arr - b_arr)
            test_name = "wilcoxon_signed_rank"
    else:
        if use_parametric:
            stat_val, p_val = stats.ttest_ind(a_arr, b_arr, equal_var=False)
            test_name = "welch_t_test"
        else:
            stat_val, p_val = stats.mannwhitneyu(a_arr, b_arr, alternative="two-sided")
            test_name = "mann_whitney_u"

    d = cohens_d(a_arr, b_arr)

    # 95% CI for the difference in means (bootstrap-free approximation)
    diff_mean = float(np.mean(a_arr) - np.mean(b_arr))
    se_diff = np.sqrt(np.var(a_arr, ddof=1)/len(a_arr) + np.var(b_arr, ddof=1)/len(b_arr))
    ci_low = diff_mean - 1.96 * se_diff
    ci_high = diff_mean + 1.96 * se_diff

    return ComparisonResult(
        metric=metric,
        group_a=group_a_name,
        group_b=group_b_name,
        mean_a=float(np.mean(a_arr)),
        mean_b=float(np.mean(b_arr)),
        std_a=float(np.std(a_arr, ddof=1)),
        std_b=float(np.std(b_arr, ddof=1)),
        p_value=float(p_val),
        test_name=test_name,
        effect_size=float(d),
        significant=p_val < 0.05,
        ci_95_diff=(float(ci_low), float(ci_high)),
    )


def f_test_variance(a: list[float], b: list[float]) -> tuple[float, float]:
    """F-test for equality of variances. Returns (F-statistic, p-value)."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    var_a = np.var(a_arr, ddof=1)
    var_b = np.var(b_arr, ddof=1)
    if var_b == 0:
        return float("inf"), 0.0
    f_stat = var_a / var_b
    df1 = len(a_arr) - 1
    df2 = len(b_arr) - 1
    p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
    return float(f_stat), float(p_val)


def coefficient_of_variation(values: list[float]) -> float:
    """CV = std/mean. Higher = more variable."""
    arr = np.array(values, dtype=float)
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    return float(np.std(arr, ddof=1) / abs(mean))


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def icc_oneway(data: list[list[float]]) -> float:
    """Intraclass Correlation Coefficient (one-way random, ICC(1,1)).
    data: list of groups, each group is a list of measurements."""
    if not data or all(len(g) < 2 for g in data):
        return 0.0

    # Flatten
    all_vals = []
    group_labels = []
    for i, group in enumerate(data):
        for v in group:
            all_vals.append(v)
            group_labels.append(i)

    n_groups = len(data)
    n_per_group = len(data[0])  # Assumes balanced

    grand_mean = np.mean(all_vals)

    # Between-group variance
    group_means = [np.mean(g) for g in data if g]
    ms_between = n_per_group * np.var(group_means, ddof=1) if len(group_means) > 1 else 0

    # Within-group variance
    ss_within = sum(sum((v - np.mean(g))**2 for v in g) for g in data if g)
    df_within = sum(len(g) - 1 for g in data if len(g) > 1)
    ms_within = ss_within / df_within if df_within > 0 else 0

    if ms_between + (n_per_group - 1) * ms_within == 0:
        return 0.0

    icc = (ms_between - ms_within) / (ms_between + (n_per_group - 1) * ms_within)
    return float(max(0, icc))
