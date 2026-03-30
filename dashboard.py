"""Streamlit dashboard for search-bench results."""

import json
import glob
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
PRICING = {
    "claude": {"input": 15.0, "output": 75.0},
    "codex": {"input": 2.5, "output": 10.0},
    "gemini": {"input": 0.075, "output": 0.30},
    "copilot": {"input": 1.0, "output": 5.0},
}


@st.cache_data
def load_results() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(RESULTS_DIR / "*.json")):
        if "results.json" in f:
            continue
        d = json.load(open(f))
        score = d.get("score") or {}
        qid = d.get("query_id", "")
        cat = qid.split("_")[0] if "_" in qid else "unknown"
        p = PRICING.get(d.get("tool_name", ""), {"input": 0, "output": 0})
        tok_in = d.get("tokens_input", 0) or 0
        tok_out = d.get("tokens_output", 0) or 0
        cost = (tok_in * p["input"] + tok_out * p["output"]) / 1_000_000

        rows.append({
            "tool": d.get("tool_name", ""),
            "mode": d.get("mode", ""),
            "query_id": qid,
            "category": cat,
            "run_number": d.get("run_number", 1),
            "tttc": d.get("tttc_seconds", 0),
            "recall": score.get("file_recall", 0),
            "precision": score.get("file_precision", 0),
            "f1": score.get("f1", 0),
            "keyword_coverage": score.get("keyword_coverage", 0),
            "anti_file_hits": score.get("anti_file_hits", 0),
            "tokens_in": tok_in,
            "tokens_out": tok_out,
            "cost_usd": cost,
            "error": d.get("error"),
            "answer_len": len(d.get("answer", "") or ""),
            "files_returned": len(d.get("files_returned", []) or []),
            "rounds": d.get("rounds", 0) or 0,
            "file": Path(f).name,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Suspicious run detection
# ---------------------------------------------------------------------------

def detect_suspicious(df: pd.DataFrame) -> pd.DataFrame:
    """Flag runs that look wrong. Returns a dataframe of flagged runs with reasons."""
    flags = []

    for _, row in df.iterrows():
        reasons = []

        # 1. Impossibly fast (< 2s) with no error — likely auth failure or broken wrapper
        if row["tttc"] < 2.0 and not row["error"]:
            reasons.append("Suspiciously fast (<2s) — possible auth failure or empty response")

        # 2. Long answer but zero files extracted — parsing failure
        if row["answer_len"] > 500 and row["files_returned"] == 0 and not row["error"]:
            reasons.append(f"Long answer ({row['answer_len']} chars) but no files extracted — parsing issue")

        # 3. Empty answer with no error — tool ran but produced nothing
        if row["answer_len"] == 0 and not row["error"]:
            reasons.append("Empty answer with no error — tool may not have searched")

        # 4. Perfect recall with very short answer — possibly lucky filename match
        if row["recall"] == 1.0 and row["answer_len"] < 100 and row["answer_len"] > 0:
            reasons.append("Perfect recall with very short answer — may be a lucky match")

        # 5. Timeout
        if row["error"] and "timeout" in str(row["error"]).lower():
            reasons.append(f"Timeout: {row['error']}")

        # 6. High recall but zero precision — returning too many files
        if row["recall"] > 0.8 and row["precision"] < 0.1 and row["files_returned"] > 5:
            reasons.append(f"High recall ({row['recall']:.2f}) but low precision ({row['precision']:.2f}) — returning {row['files_returned']} files")

        # 7. Extreme outlier on time (> 3x median for that tool/mode)
        # Handled below after we compute medians

        # 8. Anti-file hits — returned files that should NOT appear
        if row["anti_file_hits"] > 0:
            reasons.append(f"Returned {row['anti_file_hits']} anti-files (files that should not appear)")

        if reasons:
            flags.append({
                "file": row["file"],
                "tool": row["tool"],
                "mode": row["mode"],
                "query_id": row["query_id"],
                "run": row["run_number"],
                "tttc": row["tttc"],
                "recall": row["recall"],
                "answer_len": row["answer_len"],
                "flags": "; ".join(reasons),
                "severity": "high" if any("auth" in r or "Timeout" in r or "Empty answer" in r for r in reasons) else "medium",
            })

    # Time outlier detection
    medians = df.groupby(["tool", "mode"])["tttc"].median()
    for _, row in df.iterrows():
        med = medians.get((row["tool"], row["mode"]), 0)
        if med > 2 and row["tttc"] > med * 3:
            flags.append({
                "file": row["file"],
                "tool": row["tool"],
                "mode": row["mode"],
                "query_id": row["query_id"],
                "run": row["run_number"],
                "tttc": row["tttc"],
                "recall": row["recall"],
                "answer_len": row["answer_len"],
                "flags": f"Time outlier: {row['tttc']:.1f}s vs median {med:.1f}s (>{3}x)",
                "severity": "low",
            })

    return pd.DataFrame(flags) if flags else pd.DataFrame()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

st.set_page_config(page_title="search-bench results", layout="wide")
st.title("search-bench dashboard")

df = load_results()

if df.empty:
    st.error("No results found in results/")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")

# Reliable-only preset
UNRELIABLE_TOOLS = {"gemini", "codex"}
reliable_only = st.sidebar.checkbox("Reliable data only (Claude + Copilot)", value=True)

available_tools = [t for t in sorted(df["tool"].unique()) if not reliable_only or t not in UNRELIABLE_TOOLS]
tools = st.sidebar.multiselect("Tools", sorted(df["tool"].unique()), default=available_tools)
modes = st.sidebar.multiselect("Modes", df["mode"].unique(), default=list(df["mode"].unique()))
categories = st.sidebar.multiselect("Categories", sorted(df["category"].unique()), default=sorted(df["category"].unique()))
exclude_broken = st.sidebar.checkbox("Exclude broken runs (<2s, likely auth failures)", value=True)
exclude_parsing_failures = st.sidebar.checkbox("Flag copilot parsing gaps", value=True)

fdf = df[df["tool"].isin(tools) & df["mode"].isin(modes) & df["category"].isin(categories)]
if exclude_broken:
    fdf = fdf[~((fdf["tttc"] < 2.0) & (fdf["error"].isna()))]

# Add a column flagging copilot results where answer exists but no files extracted
fdf = fdf.copy()
fdf["parsing_gap"] = (fdf["tool"] == "copilot") & (fdf["answer_len"] > 500) & (fdf["files_returned"] == 0) & (fdf["error"].isna())

st.sidebar.metric("Total results", len(fdf))
st.sidebar.metric("Reliable results", len(fdf[~fdf["parsing_gap"]]))
st.sidebar.metric("Copilot parsing gaps", fdf["parsing_gap"].sum())

if reliable_only:
    st.sidebar.caption("Gemini excluded (auth failures). Codex excluded (empty answers on non-exact queries).")

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_overview, tab_compare, tab_recall, tab_speed, tab_categories, tab_suspicious, tab_raw = st.tabs([
    "Overview", "Comparison", "Recall & Precision", "Speed", "By Category", "Suspicious Runs", "Raw Data"
])

# --- Overview ---
with tab_overview:
    st.header("Summary")

    summary = (
        fdf.groupby(["tool", "mode"])
        .agg(
            n=("tttc", "count"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            precision_mean=("precision", "mean"),
            f1_mean=("f1", "mean"),
            time_mean=("tttc", "mean"),
            time_median=("tttc", "median"),
            errors=("error", lambda x: x.notna().sum()),
            parsing_gaps=("parsing_gap", "sum"),
            cost_mean=("cost_usd", "mean"),
        )
        .round(3)
        .reset_index()
    )
    summary.columns = ["Tool", "Mode", "N", "Recall", "Recall SD", "Precision", "F1", "Avg Time (s)", "P50 Time (s)", "Errors", "Parsing Gaps", "Avg Cost ($)"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    if fdf["parsing_gap"].any():
        gap_count = fdf["parsing_gap"].sum()
        st.warning(
            f"**{gap_count} copilot results** have long answers but no files extracted. "
            "Copilot's recall is likely underreported — the model found files but "
            "`_extract_files` couldn't parse them from the output format. "
            "These runs are flagged in the Suspicious Runs tab."
        )

    # Copilot: show recall with and without parsing gaps
    if "copilot" in fdf["tool"].values:
        st.subheader("Copilot recall adjustment")
        cop = fdf[fdf["tool"] == "copilot"]
        cop_clean = cop[~cop["parsing_gap"]]
        if not cop_clean.empty:
            col1, col2 = st.columns(2)
            for mode in ["native", "rag"]:
                all_mode = cop[cop["mode"] == mode]
                clean_mode = cop_clean[cop_clean["mode"] == mode]
                if all_mode.empty:
                    continue
                with col1 if mode == "native" else col2:
                    st.metric(
                        f"copilot/{mode} recall (all)",
                        f"{all_mode['recall'].mean():.3f}",
                    )
                    st.metric(
                        f"copilot/{mode} recall (parsed only)",
                        f"{clean_mode['recall'].mean():.3f}" if not clean_mode.empty else "N/A",
                        delta=f"{clean_mode['recall'].mean() - all_mode['recall'].mean():.3f}" if not clean_mode.empty else None,
                    )
            st.caption("'Parsed only' excludes runs where copilot answered but file extraction failed. True recall is likely between these two values.")

    # Native vs RAG delta table
    st.subheader("Native vs RAG delta")
    deltas = []
    for tool in fdf["tool"].unique():
        nat = fdf[(fdf["tool"] == tool) & (fdf["mode"] == "native")]
        rag = fdf[(fdf["tool"] == tool) & (fdf["mode"] == "rag")]
        if nat.empty or rag.empty:
            continue
        deltas.append({
            "Tool": tool,
            "Δ Recall": round(rag["recall"].mean() - nat["recall"].mean(), 3),
            "Δ F1": round(rag["f1"].mean() - nat["f1"].mean(), 3),
            "Δ Time (s)": round(rag["tttc"].mean() - nat["tttc"].mean(), 1),
            "RAG faster?": "Yes" if rag["tttc"].mean() < nat["tttc"].mean() else "No",
            "RAG more accurate?": "Yes" if rag["recall"].mean() > nat["recall"].mean() else "No" if rag["recall"].mean() < nat["recall"].mean() else "Same",
        })
    if deltas:
        st.dataframe(pd.DataFrame(deltas), use_container_width=True, hide_index=True)

# --- Comparison (multi-axis) ---
with tab_compare:
    st.header("Speed vs Accuracy Comparison")

    # Build per tool/mode/category aggregates
    compare_df = (
        fdf[~fdf["parsing_gap"]]
        .groupby(["tool", "mode"])
        .agg(
            recall=("recall", "mean"),
            precision=("precision", "mean"),
            f1=("f1", "mean"),
            time_mean=("tttc", "mean"),
            time_p50=("tttc", "median"),
            n=("tttc", "count"),
            error_rate=("error", lambda x: x.notna().mean()),
        )
        .reset_index()
    )
    compare_df["label"] = compare_df["tool"] + "/" + compare_df["mode"]
    compare_df["reliability"] = 1 - compare_df["error_rate"]

    # Quadrant chart: speed vs recall
    st.subheader("Speed vs Recall (top-left is best)")
    st.caption("Each point is a tool/mode combination. Excludes copilot parsing gaps.")
    fig = px.scatter(compare_df, x="time_mean", y="recall", color="tool", symbol="mode",
                     text="label", size="n", size_max=25,
                     labels={"time_mean": "Mean response time (s)", "recall": "Mean file recall"})
    fig.update_traces(textposition="top center")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Radar-style comparison table
    st.subheader("Multi-axis scorecard")
    scorecard = compare_df[["label", "recall", "precision", "f1", "time_p50", "reliability", "n"]].copy()
    scorecard.columns = ["Tool/Mode", "Recall", "Precision", "F1", "P50 Time (s)", "Reliability", "N"]
    scorecard = scorecard.sort_values("Recall", ascending=False)
    st.dataframe(scorecard.round(3), use_container_width=True, hide_index=True)

    # Per-category speed vs recall
    st.subheader("Speed vs Recall by Category")
    cat_compare = (
        fdf[~fdf["parsing_gap"]]
        .groupby(["tool", "mode", "category"])
        .agg(recall=("recall", "mean"), time=("tttc", "mean"), n=("tttc", "count"))
        .reset_index()
    )
    cat_compare["label"] = cat_compare["tool"] + "/" + cat_compare["mode"]

    fig = px.scatter(cat_compare, x="time", y="recall", color="label",
                     facet_col="category", facet_col_wrap=2,
                     size="n", size_max=20,
                     title="Speed vs Recall by Category",
                     labels={"time": "Mean time (s)", "recall": "Recall"})
    fig.update_layout(yaxis_range=[0, 1], height=600)
    st.plotly_chart(fig, use_container_width=True)

    # Intra-model: native vs RAG side by side
    st.subheader("Native vs RAG per tool (intra-model)")
    for tool in sorted(fdf["tool"].unique()):
        tool_data = compare_df[compare_df["tool"] == tool]
        if len(tool_data) < 2:
            continue
        st.write(f"**{tool}**")
        col1, col2, col3, col4 = st.columns(4)
        nat = tool_data[tool_data["mode"] == "native"].iloc[0] if "native" in tool_data["mode"].values else None
        rag = tool_data[tool_data["mode"] == "rag"].iloc[0] if "rag" in tool_data["mode"].values else None
        if nat is not None and rag is not None:
            col1.metric("Recall (native)", f"{nat['recall']:.3f}")
            col1.metric("Recall (RAG)", f"{rag['recall']:.3f}", delta=f"{rag['recall'] - nat['recall']:.3f}")
            col2.metric("F1 (native)", f"{nat['f1']:.3f}")
            col2.metric("F1 (RAG)", f"{rag['f1']:.3f}", delta=f"{rag['f1'] - nat['f1']:.3f}")
            col3.metric("P50 Time (native)", f"{nat['time_p50']:.1f}s")
            col3.metric("P50 Time (RAG)", f"{rag['time_p50']:.1f}s", delta=f"{rag['time_p50'] - nat['time_p50']:.1f}s", delta_color="inverse")
            col4.metric("Reliability (native)", f"{nat['reliability']:.1%}")
            col4.metric("Reliability (RAG)", f"{rag['reliability']:.1%}")

# --- Recall & Precision ---
with tab_recall:
    st.header("Recall & Precision")

    # Grouped bar: recall
    agg = fdf.groupby(["tool", "mode"]).agg(
        recall=("recall", "mean"),
        precision=("precision", "mean"),
        f1=("f1", "mean"),
    ).reset_index()
    agg["tool_mode"] = agg["tool"] + " / " + agg["mode"]

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(agg, x="tool", y="recall", color="mode", barmode="group",
                     title="File Recall by Tool", text_auto=".3f")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(agg, x="tool", y="precision", color="mode", barmode="group",
                     title="File Precision by Tool", text_auto=".3f")
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(agg, x="tool", y="f1", color="mode", barmode="group",
                 title="F1 Score by Tool", text_auto=".3f")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Per-query recall heatmap
    st.subheader("Recall by Query (heatmap)")
    for tool in sorted(fdf["tool"].unique()):
        tool_df = fdf[fdf["tool"] == tool]
        if tool_df.empty:
            continue
        pivot = tool_df.pivot_table(index="query_id", columns="mode", values="recall", aggfunc="mean")
        fig = px.imshow(pivot.T, aspect="auto", color_continuous_scale="RdYlGn",
                        title=f"{tool} — recall per query", zmin=0, zmax=1,
                        labels=dict(x="Query", y="Mode", color="Recall"))
        st.plotly_chart(fig, use_container_width=True)

# --- Speed ---
with tab_speed:
    st.header("Response Time")

    time_agg = fdf.groupby(["tool", "mode"]).agg(
        mean=("tttc", "mean"),
        median=("tttc", "median"),
        p90=("tttc", lambda x: x.quantile(0.9)),
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(time_agg, x="tool", y="mean", color="mode", barmode="group",
                     title="Mean Response Time (s)", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(time_agg, x="tool", y="median", color="mode", barmode="group",
                     title="Median Response Time (s)", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)

    # Box plots — shows distribution properly
    st.subheader("Time Distribution")
    fig = px.box(fdf, x="tool", y="tttc", color="mode",
                 title="Response Time Distribution",
                 labels={"tttc": "Time (s)", "tool": "Tool"},
                 points=False)
    st.plotly_chart(fig, use_container_width=True)

    # Speed vs Recall scatter with labels
    st.subheader("Speed vs Recall tradeoff")
    scatter_df = fdf.groupby(["tool", "mode"]).agg(
        recall=("recall", "mean"),
        time=("tttc", "mean"),
        n=("tttc", "count"),
    ).reset_index()
    scatter_df["label"] = scatter_df["tool"] + "/" + scatter_df["mode"]
    fig = px.scatter(scatter_df, x="time", y="recall", color="tool", symbol="mode",
                     text="label", size="n", size_max=20,
                     title="Speed vs Recall (top-left is best)",
                     labels={"time": "Mean time (s)", "recall": "Mean recall"})
    fig.update_traces(textposition="top center")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

# --- By Category ---
with tab_categories:
    st.header("Performance by Query Category")

    cat_recall = fdf.pivot_table(index="category", columns=["tool", "mode"], values="recall", aggfunc="mean")
    st.subheader("Recall by Category")
    st.dataframe(cat_recall.round(3), use_container_width=True)

    cat_time = fdf.pivot_table(index="category", columns=["tool", "mode"], values="tttc", aggfunc="mean")
    st.subheader("Avg Time by Category")
    st.dataframe(cat_time.round(1), use_container_width=True)

    # Per-category grouped bars
    cat_agg = fdf.groupby(["category", "tool", "mode"]).agg(
        recall=("recall", "mean"),
        time=("tttc", "mean"),
    ).reset_index()
    cat_agg["tool_mode"] = cat_agg["tool"] + "/" + cat_agg["mode"]

    fig = px.bar(cat_agg, x="category", y="recall", color="tool_mode", barmode="group",
                 title="Recall by Category and Tool/Mode", text_auto=".2f")
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(cat_agg, x="category", y="time", color="tool_mode", barmode="group",
                 title="Time by Category and Tool/Mode", text_auto=".0f")
    st.plotly_chart(fig, use_container_width=True)

# --- Suspicious Runs ---
with tab_suspicious:
    st.header("Suspicious Run Detection")

    flags_df = detect_suspicious(df)  # Run on unfiltered data

    if flags_df.empty:
        st.success("No suspicious runs detected.")
    else:
        severity_counts = flags_df["severity"].value_counts()
        col1, col2, col3 = st.columns(3)
        col1.metric("High severity", severity_counts.get("high", 0))
        col2.metric("Medium severity", severity_counts.get("medium", 0))
        col3.metric("Low severity", severity_counts.get("low", 0))

        st.subheader("Flagged runs")
        severity_filter = st.multiselect("Severity", ["high", "medium", "low"], default=["high", "medium"])
        filtered_flags = flags_df[flags_df["severity"].isin(severity_filter)]

        # Summary by tool
        st.write("**Flags by tool:**")
        tool_flags = filtered_flags.groupby("tool").size().reset_index(name="count")
        st.dataframe(tool_flags, use_container_width=True, hide_index=True)

        # Summary by flag type
        st.write("**Most common flags:**")
        all_flags = []
        for flags_str in filtered_flags["flags"]:
            all_flags.extend(f.strip().split(" — ")[0] for f in flags_str.split(";"))
        if all_flags:
            flag_counts = pd.Series(all_flags).value_counts().head(10).reset_index()
            flag_counts.columns = ["Flag", "Count"]
            fig = px.bar(flag_counts, x="Count", y="Flag", orientation="h",
                         title="Most Common Flags")
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("All flagged runs")
        st.dataframe(
            filtered_flags.sort_values(["severity", "tool", "query_id"]),
            use_container_width=True,
            hide_index=True,
        )

# --- Raw Data ---
with tab_raw:
    st.header("Raw Results")
    st.dataframe(fdf.sort_values(["tool", "mode", "query_id", "run_number"]), use_container_width=True, hide_index=True)

    st.download_button(
        "Download filtered results as CSV",
        fdf.to_csv(index=False),
        "search_bench_results.csv",
        "text/csv",
    )
