# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (requires Python 3.11+)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # with test deps
pip install -e ".[gpu]"          # with FAISS GPU support

# Index a codebase (builds FAISS + SQLite FTS5 indices)
python scripts/index_codebase.py ./benchmark/circuitsnips
# or via entry point after install:
index-codebase ./benchmark/circuitsnips

# Run benchmark (entry point — src/benchmark/runner.py:main)
search-bench --codebase ./benchmark/circuitsnips --phase read_only --tools claude
search-bench --codebase ./benchmark/circuitsnips --phase all

# Check CLI tool availability
bash scripts/setup_tools.sh

# Run tests
pytest
pytest tests/path/to/test_file.py::test_name   # single test
```

## Architecture

The benchmark runs each of the four CLI tools (Claude Code, Codex CLI, Gemini CLI, GitHub Copilot) in two modes — **Native** (built-in grep/glob/read) and **RAG** (MCP server providing hybrid semantic + keyword search) — across 60 read-only queries and 20 modification tasks.

```
MCP Server (FAISS + SQLite FTS5)
    ↕ stdio
CLI Tools (Claude Code, Codex, Gemini, Copilot)
    ↕
Benchmark Runner (dispatch, retry, checkpoint)
    ↕
Analysis Pipeline (stats, charts, report)
```

### Key modules

**`src/mcp_server/`** — RAG MCP server exposed over stdio

- `server.py` — Exposes three MCP tools: `semantic_search`, `symbol_lookup`, `related_code`
- `indexer.py` — Walks a codebase, chunks files, builds FAISS + SQLite FTS5 indices; entry point for `index-codebase`
- `chunker.py` — Tree-sitter AST chunking for Python/TS/JS; sliding window (40 lines, 10 overlap) for everything else; skips build artifacts and files >512KB
- `search.py` — `HybridSearch` class combining FAISS (`all-MiniLM-L6-v2` embeddings) with SQLite FTS5 (Porter stemming)
- `store.py` — SQLite metadata store; `chunks` table + `chunks_fts` virtual table with automatic FTS5 triggers

**`src/wrappers/`** — Per-tool CLI wrappers

- `base.py` — Shared types (`SearchMode`, `Query`, `SearchOp`, `QueryResult`), `ToolWrapper` ABC, `_extract_files()` shared helper, NATIVE vs RAG prompt templates
- `claude.py` — Runs `claude --print --output-format json`; parses nested tool-use messages for files accessed; 120s timeout
- `codex.py` — Runs `codex exec`; parses JSON output for tokens/answer
- `gemini.py` — Runs `gemini --yolo`, prompt via stdin
- `copilot.py` — Runs `copilot -p --output-format json --allow-all-tools`; parses JSONL event stream for messages, tool uses, and file accesses
- `token_counter.py` — Token estimation with tiktoken; pricing table for all four tools

**`src/benchmark/`** — Orchestration

- `runner.py` — Main orchestrator and `search-bench` CLI entry point; handles all three phases, checkpoint/resume, per-tool semaphores, exponential backoff retry (`_run_with_retry`, `_tool_with_retry`), MCP config injection (`MCPConfigManager`), git worktree isolation for author tasks, ANSI progress display, and report generation
- `scorer.py` — File recall/precision/F1 scoring with fuzzy path matching (normalize, suffix, basename fallback)

**`src/analysis/`** — Post-run analysis

- `stats.py` — Parametric and non-parametric tests (Welch's t-test, Wilcoxon, Mann-Whitney), Cohen's d, 95% CI; auto-selects test by normality
- `report.py` — Aggregates results into markdown tables
- `charts.py` — Matplotlib + Plotly visualizations
- `diff_compare.py`, `code_quality.py`, `reliability_report.py` — Author/reviewer and reliability analysis

**`queries/`** and **`tasks/`** — Benchmark data

- `queries/queries.json` — 60 queries with `ground_truth` files, `keywords`, `anti_files`; categories: `exact_symbol`, `conceptual`, `cross_cutting`, `refactoring`
- `queries/smoke_queries.json` — 6-query subset for quick validation
- `tasks/tasks.json` — 20 modification tasks with `expected_scope` files
- `tasks/smoke_tasks.json` — Subset for quick validation

**`configs/`** — Tool configuration files (MCP server paths for Claude, settings for Codex/Gemini/Copilot)

## Key design details

- **Author phase uses git worktrees** for isolation — each tool runs in `results/worktrees/{tool}_{mode}_{task}`, preventing concurrent checkout races. Stale worktrees from crashed runs are cleaned up automatically.
- **MCP config injection** writes tool-specific config files before RAG runs: `.mcp.json` for Claude (in worktree dir for author), `~/.codex/config.toml`, `~/.gemini/settings.json`, `~/.copilot/mcp-config.json`. Originals are backed up and restored.
- **Retry logic** uses exponential backoff for transient errors (timeouts, 429s, rate limits, 503s). Controlled via `--max-retries`.
- **All four tools** participate in all three benchmark phases (read-only, author, review).
- The target codebase (`./benchmark/circuitsnips`) is not included; clone it separately before indexing.
- FAISS index and SQLite DB paths default to `./data/circuitsnips.{db,faiss}`.
