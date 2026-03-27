"""Cross-model diff comparison: Jaccard overlap, consensus diff, divergence analysis."""

import re
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class DiffComparison:
    task_id: str
    tool_a: str
    mode_a: str
    tool_b: str
    mode_b: str
    file_jaccard: float  # Jaccard similarity of files modified
    shared_files: list[str]
    unique_to_a: list[str]
    unique_to_b: list[str]

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "tool_a": f"{self.tool_a}/{self.mode_a}",
            "tool_b": f"{self.tool_b}/{self.mode_b}",
            "file_jaccard": self.file_jaccard,
            "shared_files": self.shared_files,
            "unique_to_a": self.unique_to_a,
            "unique_to_b": self.unique_to_b,
        }


def extract_files_from_diff(diff_text: str) -> set[str]:
    """Extract the set of files modified in a diff."""
    files = set()
    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                # a/path/to/file b/path/to/file
                files.add(parts[2].lstrip("a/"))
        elif line.startswith("+++ b/"):
            files.add(line[6:])
    return files


def compare_diffs(
    task_id: str,
    diff_a: str, tool_a: str, mode_a: str,
    diff_b: str, tool_b: str, mode_b: str,
) -> DiffComparison:
    """Compare two diffs for the same task."""
    files_a = extract_files_from_diff(diff_a)
    files_b = extract_files_from_diff(diff_b)

    shared = files_a & files_b
    only_a = files_a - files_b
    only_b = files_b - files_a
    union = files_a | files_b

    jaccard = len(shared) / len(union) if union else 1.0

    return DiffComparison(
        task_id=task_id,
        tool_a=tool_a, mode_a=mode_a,
        tool_b=tool_b, mode_b=mode_b,
        file_jaccard=jaccard,
        shared_files=sorted(shared),
        unique_to_a=sorted(only_a),
        unique_to_b=sorted(only_b),
    )


def consensus_files(diffs: list[tuple[str, str, str]], min_agreement: int = 3) -> set[str]:
    """Find files modified by at least min_agreement out of N tools.
    diffs: list of (diff_text, tool_name, mode)"""
    file_counts: dict[str, int] = defaultdict(int)
    for diff_text, _, _ in diffs:
        for f in extract_files_from_diff(diff_text):
            file_counts[f] += 1
    return {f for f, count in file_counts.items() if count >= min_agreement}


def pairwise_comparisons(
    task_id: str,
    diffs: list[tuple[str, str, str]],
) -> list[DiffComparison]:
    """Compare all pairs of diffs for a task.
    diffs: list of (diff_text, tool_name, mode)"""
    comparisons = []
    for i in range(len(diffs)):
        for j in range(i + 1, len(diffs)):
            diff_a, tool_a, mode_a = diffs[i]
            diff_b, tool_b, mode_b = diffs[j]
            comparisons.append(compare_diffs(
                task_id, diff_a, tool_a, mode_a, diff_b, tool_b, mode_b,
            ))
    return comparisons
