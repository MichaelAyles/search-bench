"""Aggregate results into paper-ready tables and summary statistics."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_report(results_path: Path) -> str:
    """Generate a text summary report from benchmark results."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    lines = ["# Search Benchmark Report", ""]

    # Read-only summary
    ro_scores = data.get("read_only_scores", [])
    if ro_scores:
        lines.append("## Read-Only Benchmark")
        lines.append("")
        lines.append(_summarize_scores(ro_scores))

    # Author/reviewer summary
    author_results = data.get("author_results", [])
    review_results = data.get("review_results", [])
    if author_results:
        lines.append("## Author/Reviewer Benchmark")
        lines.append("")
        lines.append(_summarize_authors(author_results))
    if review_results:
        lines.append(_summarize_reviews(review_results))

    return "\n".join(lines)


def _summarize_scores(scores: list[dict]) -> str:
    """Summarize read-only scores by tool and mode."""
    groups = defaultdict(list)
    for s in scores:
        key = f"{s['tool_name']}/{s['mode']}"
        groups[key].append(s)

    lines = ["| Tool/Mode | Recall (mean±std) | Precision | Keyword Coverage | N |"]
    lines.append("|---|---|---|---|---|")

    for key in sorted(groups):
        entries = groups[key]
        recalls = [e["file_recall"] for e in entries]
        precisions = [e["file_precision"] for e in entries]
        keywords = [e["keyword_coverage"] for e in entries]

        lines.append(
            f"| {key} | {np.mean(recalls):.3f}±{np.std(recalls):.3f} "
            f"| {np.mean(precisions):.3f} | {np.mean(keywords):.3f} | {len(entries)} |"
        )

    return "\n".join(lines) + "\n"


def _summarize_authors(results: list[dict]) -> str:
    groups = defaultdict(list)
    for r in results:
        key = f"{r['tool_name']}/{r['mode']}"
        groups[key].append(r)

    lines = ["| Tool/Mode | Avg TTTC (s) | Avg Lines Changed | Success | N |"]
    lines.append("|---|---|---|---|---|")

    for key in sorted(groups):
        entries = groups[key]
        tttcs = [e.get("tttc_seconds", 0) for e in entries]
        lines_changed = [
            e.get("diff_stat", {}).get("lines_added", 0) + e.get("diff_stat", {}).get("lines_removed", 0)
            for e in entries
        ]
        successes = sum(1 for e in entries if not e.get("error"))

        lines.append(
            f"| {key} | {np.mean(tttcs):.1f} | {np.mean(lines_changed):.0f} "
            f"| {successes}/{len(entries)} | {len(entries)} |"
        )

    return "\n".join(lines) + "\n"


def _summarize_reviews(results: list[dict]) -> str:
    # Verdict distribution per reviewer
    groups = defaultdict(lambda: defaultdict(int))
    for r in results:
        key = f"{r.get('reviewer_tool', '?')}/{r.get('reviewer_mode', '?')}"
        verdict = r.get("verdict", "UNKNOWN")
        groups[key][verdict] += 1

    lines = ["### Review Verdicts", ""]
    lines.append("| Reviewer | APPROVE | REQUEST_CHANGES | REJECT | UNKNOWN |")
    lines.append("|---|---|---|---|---|")

    for key in sorted(groups):
        d = groups[key]
        lines.append(
            f"| {key} | {d.get('APPROVE',0)} | {d.get('REQUEST_CHANGES',0)} "
            f"| {d.get('REJECT',0)} | {d.get('UNKNOWN',0)} |"
        )

    return "\n".join(lines) + "\n"
