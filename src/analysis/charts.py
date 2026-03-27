"""Chart generation for benchmark results."""

import json
from pathlib import Path

import numpy as np


def generate_all_charts(results_path: Path, output_dir: Path):
    """Generate all charts from benchmark results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(results_path.read_text(encoding="utf-8"))

    ro_results = data.get("read_only_results", [])
    ro_scores = data.get("read_only_scores", [])

    if ro_results and ro_scores:
        _scatter_quality_cost(ro_results, ro_scores, output_dir)
        _category_bars(ro_results, ro_scores, output_dir)
        _tttc_boxplots(ro_results, output_dir)
        _efficiency_heatmap(ro_results, ro_scores, output_dir)
        _variance_violins(ro_results, ro_scores, output_dir)

    author_results = data.get("author_results", [])
    review_results = data.get("review_results", [])

    if review_results:
        _review_matrix(review_results, output_dir)

    print(f"Charts saved to {output_dir}")


def _scatter_quality_cost(results, scores, output_dir):
    """Hero scatter: tokens (x, log) vs recall (y), colored by tool, shaped by mode."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    score_map = {(s["query_id"], s["tool_name"], s["mode"], s["run_number"]): s for s in scores}

    tool_colors = {"claude": "#E04E39", "codex": "#10A37F", "gemini": "#4285F4", "copilot": "#6F42C1"}
    mode_markers = {"native": "o", "rag": "^"}

    fig, ax = plt.subplots(figsize=(12, 8))

    for r in results:
        key = (r["query_id"], r["tool_name"], r["mode"], r["run_number"])
        s = score_map.get(key)
        if not s:
            continue

        tokens = r.get("tokens_input", 0) + r.get("tokens_output", 0)
        if tokens <= 0:
            continue
        recall = s.get("file_recall", 0)
        tttc = r.get("tttc_seconds", 1)

        color = tool_colors.get(r["tool_name"], "#999")
        marker = mode_markers.get(r["mode"], "o")
        size = max(20, min(200, tttc * 3))

        ax.scatter(tokens, recall, c=color, marker=marker, s=size, alpha=0.6)

    ax.set_xscale("log")
    ax.set_xlabel("Total Tokens (log scale)")
    ax.set_ylabel("File Recall@k")
    ax.set_title("Quality vs Cost: RAG vs Native Search")
    ax.set_ylim(-0.05, 1.05)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = []
    for tool, color in tool_colors.items():
        legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color, label=tool, markersize=8))
    for mode, marker in mode_markers.items():
        legend_elements.append(Line2D([0], [0], marker=marker, color="w", markerfacecolor="gray", label=mode, markersize=8))
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_dir / "scatter_quality_cost.png", dpi=150)
    plt.close(fig)


def _category_bars(results, scores, output_dir):
    """Grouped bar charts: recall and tokens by category, tool, mode."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Group scores by category, tool, mode
    groups = {}
    for s in scores:
        key = (s.get("tool_name"), s.get("mode"))
        cat = None
        # Find the category from the query_id prefix
        qid = s.get("query_id", "")
        if qid.startswith("exact"):
            cat = "exact_symbol"
        elif qid.startswith("concept"):
            cat = "conceptual"
        elif qid.startswith("cross"):
            cat = "cross_cutting"
        elif qid.startswith("refactor"):
            cat = "refactoring"
        else:
            cat = "other"

        group_key = (cat, key[0], key[1])
        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(s.get("file_recall", 0))

    if not groups:
        return

    # Simple bar chart of mean recall per tool/mode
    fig, ax = plt.subplots(figsize=(14, 6))
    categories = sorted(set(k[0] for k in groups))
    tools_modes = sorted(set((k[1], k[2]) for k in groups))

    x = np.arange(len(categories))
    width = 0.8 / len(tools_modes)

    for i, (tool, mode) in enumerate(tools_modes):
        means = []
        for cat in categories:
            vals = groups.get((cat, tool, mode), [])
            means.append(np.mean(vals) if vals else 0)
        ax.bar(x + i * width, means, width, label=f"{tool}/{mode}", alpha=0.8)

    ax.set_xticks(x + width * len(tools_modes) / 2)
    ax.set_xticklabels(categories, rotation=15)
    ax.set_ylabel("Mean File Recall")
    ax.set_title("File Recall by Category")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "category_recall_bars.png", dpi=150)
    plt.close(fig)


def _tttc_boxplots(results, output_dir):
    """Box plots of TTTC per tool and mode."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    groups = {}
    for r in results:
        key = f"{r['tool_name']}/{r['mode']}"
        if key not in groups:
            groups[key] = []
        groups[key].append(r.get("tttc_seconds", 0))

    if not groups:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = sorted(groups.keys())
    data = [groups[k] for k in labels]

    ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_ylabel("Time to Task Completion (seconds)")
    ax.set_title("TTTC Distribution by Tool and Mode")
    plt.xticks(rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "tttc_boxplots.png", dpi=150)
    plt.close(fig)


def _efficiency_heatmap(results, scores, output_dir):
    """Heatmap of tokens-per-relevant-file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Placeholder - build when data available
    pass


def _variance_violins(results, scores, output_dir):
    """Violin plots of recall variance across runs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    # Placeholder - build when data available
    pass


def _review_matrix(review_results, output_dir):
    """Cross-evaluation matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    # Build agreement matrix
    verdicts = {}
    for r in review_results:
        key = (r.get("reviewer_tool", ""), r.get("author_tool", ""))
        if key not in verdicts:
            verdicts[key] = []
        verdicts[key].append(r.get("verdict", ""))

    if not verdicts:
        return

    tools = sorted(set(k[0] for k in verdicts) | set(k[1] for k in verdicts))
    n = len(tools)
    matrix = np.zeros((n, n))

    for i, reviewer in enumerate(tools):
        for j, author in enumerate(tools):
            vs = verdicts.get((reviewer, author), [])
            if vs:
                approve_rate = sum(1 for v in vs if v == "APPROVE") / len(vs)
                matrix[i, j] = approve_rate

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(tools, rotation=45)
    ax.set_yticks(range(n))
    ax.set_yticklabels(tools)
    ax.set_xlabel("Author Tool")
    ax.set_ylabel("Reviewer Tool")
    ax.set_title("Approval Rate Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(output_dir / "review_matrix.png", dpi=150)
    plt.close(fig)
