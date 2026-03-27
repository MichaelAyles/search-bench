"""Reliability profiles, failure distributions, uptime timelines."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def generate_reliability_report(results_path: Path) -> str:
    """Generate reliability report from benchmark results."""
    data = json.loads(results_path.read_text(encoding="utf-8"))

    # Collect all run_meta entries
    all_metas = []
    for r in data.get("read_only_results", []):
        if "run_meta" in r:
            all_metas.append(r["run_meta"])
    for r in data.get("author_results", []):
        if "run_meta" in r:
            all_metas.append(r["run_meta"])
    for r in data.get("review_results", []):
        if "run_meta" in r:
            all_metas.append(r["run_meta"])

    if not all_metas:
        return "No run metadata found."

    lines = ["# Reliability Report", ""]

    # Per-tool summary
    tool_runs = defaultdict(list)
    for m in all_metas:
        tool_runs[m.get("tool_name", "?")].append(m)

    lines.append("## Per-Tool Reliability")
    lines.append("")
    lines.append("| Tool | Total Runs | Success Rate | First-Try Rate | Failures | Rate Limit Wait (s) |")
    lines.append("|---|---|---|---|---|---|")

    for tool in sorted(tool_runs):
        runs = tool_runs[tool]
        total = len(runs)
        successes = sum(1 for r in runs if r.get("success", False))
        first_try = sum(1 for r in runs if r.get("success", False) and r.get("retry_count", 0) == 0)
        failures = total - successes
        rate_wait = sum(r.get("rate_limit_wait_seconds", 0) for r in runs)

        lines.append(
            f"| {tool} | {total} | {successes/total:.1%} | {first_try/total:.1%} "
            f"| {failures} | {rate_wait:.0f} |"
        )

    # Failure category breakdown
    lines.append("")
    lines.append("## Failure Categories")
    lines.append("")

    for tool in sorted(tool_runs):
        failures = [r for r in tool_runs[tool] if not r.get("success", False)]
        if not failures:
            continue

        lines.append(f"### {tool}")
        cats = defaultdict(int)
        for f in failures:
            cats[f.get("failure_reason", "other")] += 1

        for cat in sorted(cats, key=cats.get, reverse=True):
            lines.append(f"  - {cat}: {cats[cat]}")
        lines.append("")

    # Mode comparison
    lines.append("## Reliability by Mode")
    lines.append("")
    mode_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for m in all_metas:
        mode = m.get("mode", "?")
        mode_stats[mode]["total"] += 1
        if m.get("success", False):
            mode_stats[mode]["success"] += 1

    for mode in sorted(mode_stats):
        s = mode_stats[mode]
        rate = s["success"] / s["total"] if s["total"] else 0
        lines.append(f"- **{mode}**: {s['success']}/{s['total']} ({rate:.1%})")

    return "\n".join(lines)
