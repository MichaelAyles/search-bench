# search-bench

Benchmark comparing RAG-based semantic search vs native agentic search across four AI coding CLI tools: **Claude Code**, **Codex CLI**, **Gemini CLI**, and **GitHub Copilot**.

## What it measures

Each tool runs against the [kicad-library](https://github.com/michaelayles/kicad-library) codebase — a Next.js/TypeScript KiCad circuit-sharing platform — in two modes:

| Mode | How the tool searches |
|---|---|
| **Native** | Built-in search: grep, glob, file reads |
| **RAG** | MCP server injected at startup: hybrid FAISS semantic + SQLite FTS5 keyword search |

### Benchmark phases

**Read-only (60 queries, 4 categories)**

| Category | Description | Example |
|---|---|---|
| `exact_symbol` | Find a specific function or class | "Where is the KiCad S-expression parser?" |
| `conceptual` | Understand how something works | "How does auth work across the app?" |
| `cross_cutting` | Trace a feature end-to-end | "Trace the search flow from input to DB" |
| `refactoring` | Assess change impact | "What files change if we swap Supabase?" |

**Author (20 modification tasks, 3 difficulty levels)**

Each tool implements a code change in an isolated git branch. Diffs are captured and analysed.

| Difficulty | Count | Example |
|---|---|---|
| Simple | 5 | Add file-size validation to upload API |
| Medium | 10 | Add recently-viewed section to profile page |
| Hard | 5 | Implement Redis rate limiter |

**Review (cross-evaluation)**

Every tool reviews every other tool's diffs for each task. Verdicts: `APPROVE`, `REQUEST_CHANGES`, `REJECT`.


### Metrics collected

- File recall and precision against hand-labelled ground truth
- F1 score
- Keyword coverage in answer text
- Token consumption (input + output) and estimated USD cost
- Time to task completion (TTTC)
- Run-to-run variance (3 runs per query)
- Author: lines added/removed, files modified, import changes
- Review: verdict distribution across author×reviewer pairs
- Reliability: per-tool success rate, failure categories, rate-limit wait time

---

## Setup

### Requirements

- Python 3.11+
- macOS or Linux (Windows untested)
- CLI tools installed and authenticated (see below)

### Install

```bash
git clone https://github.com/michaelayles/search-bench
cd search-bench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### Install CLI tools

```bash
# Claude Code
npm install -g @anthropic-ai/claude-code

# Codex CLI
npm install -g @openai/codex

# Gemini CLI
npm install -g @google/gemini-cli

# GitHub Copilot CLI
npm install -g @github/copilot

# Verify all four are available
bash scripts/setup_tools.sh
```

Each tool needs to be authenticated before running the benchmark. Refer to each tool's documentation for `claude auth`, `codex login`, `gemini auth`, and `copilot login`.

### Index the target codebase

```bash
# Clone kicad-library
git clone https://github.com/michaelayles/kicad-library ./benchmark/circuitsnips

# Build FAISS + SQLite indices (required for RAG mode)
python scripts/index_codebase.py ./benchmark/circuitsnips
# or: index-codebase ./benchmark/circuitsnips

# Indices are written to ./data/circuitsnips.{db,faiss}
```

Indexing uses Tree-sitter AST chunking for Python/TypeScript/JavaScript and sliding window (40 lines, 10-line overlap) for everything else. Files larger than 512KB and lock files are skipped.

---

## Running

### Smoke test (fastest — one tool, five queries)

```bash
search-bench \
  --codebase ./benchmark/circuitsnips \
  --smoke \
  --tools claude \
  --modes native
```

`--smoke` automatically switches to `queries/smoke_queries.json` (5 queries) and `tasks/smoke_tasks.json` (2 tasks), sets `--runs 1` and `--concurrency 2`.

### Read-only benchmark

```bash
# All tools, both modes, 3 runs per query
search-bench --codebase ./benchmark/circuitsnips --phase read_only

# Single tool, native mode only
search-bench --codebase ./benchmark/circuitsnips --phase read_only --tools claude --modes native

# Multiple specific tools
search-bench --codebase ./benchmark/circuitsnips --phase read_only --tools claude,gemini
```

### Full benchmark (all three phases)

```bash
search-bench --codebase ./benchmark/circuitsnips --phase all
```

### Resume after interruption

Results are checkpointed individually as JSON files. If a run is interrupted, re-running the same command will skip completed results automatically (`--resume` is on by default).

```bash
# Force a full rerun, ignoring existing checkpoints
search-bench --codebase ./benchmark/circuitsnips --phase read_only --no-resume
```

### All CLI options

```
--codebase PATH          Path to target codebase (required)
--phase                  read_only | author | review | all  (default: read_only)
--tools                  Comma-separated: claude,codex,gemini,copilot or "all"  (default: all)
--modes                  Comma-separated: native,rag or "all"  (default: all)
--runs N                 Runs per query for variance measurement  (default: 3)
--concurrency N          Max concurrent tool invocations  (default: 4)
--output-dir PATH        Results directory  (default: ./results)
--queries PATH           Query JSON file  (default: queries/queries.json)
--tasks PATH             Task JSON file  (default: tasks/tasks.json)
--smoke                  Use smoke files, 1 run, concurrency 2
--resume / --no-resume   Skip or force-rerun checkpointed results  (default: resume)
--db PATH                SQLite DB path for MCP server  (default: ./data/circuitsnips.db)
--faiss PATH             FAISS index path for MCP server  (default: ./data/circuitsnips.faiss)
--max-retries N          Max retries on transient errors per invocation  (default: 3)
--save-transcripts       Include raw tool output in read-only checkpoint files
```

---

## Outputs

All outputs are written to `./results/` (configurable with `--output-dir`).

| File | Description |
|---|---|
| `{tool}_{mode}_{query_id}_run{n}.json` | Checkpoint per read-only invocation |
| `author_{tool}_{mode}_{task_id}.json` | Checkpoint per author task |
| `review_{reviewer}_{mode}_of_{author}_{mode}_{task_id}.json` | Checkpoint per review |
| `results.json` | Aggregated results in analysis-ready format |
| `report.md` | Markdown tables: recall, precision, TTTC, review verdicts |
| `reliability.md` | Per-tool reliability: success rate, failure categories |
| `charts/scatter_quality_cost.png` | Recall vs tokens scatter (hero chart) |
| `charts/category_recall_bars.png` | Recall by query category |
| `charts/tttc_boxplots.png` | TTTC distribution per tool/mode |
| `charts/review_matrix.png` | Approval rate heatmap (reviewer × author) |

The `results.json` structure:

```json
{
  "read_only_results": [...],
  "read_only_scores":  [...],
  "author_results":    [...],
  "review_results":    [...],
  "metadata": {
    "timestamp": "...",
    "codebase": "...",
    "tools": [...],
    "modes": [...],
    "runs_per_query": 3,
    "total_invocations": 1440,
    "total_cost_usd": 12.34,
    "total_duration_seconds": 3600
  }
}
```

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  search-bench runner                 │
│  (asyncio, per-tool semaphore, checkpoint/resume)    │
└───────────┬────────────────────────┬────────────────┘
            │                        │
   ┌────────▼────────┐    ┌──────────▼──────────┐
   │  Tool wrappers  │    │    MCP server (RAG)  │
   │  Claude Code    │    │  FAISS semantic      │
   │  Codex CLI      │◄───│  SQLite FTS5 keyword │
   │  Gemini CLI     │    │  Hybrid RRF fusion   │
   │  GitHub Copilot │    └──────────────────────┘
   └────────┬────────┘       started per-tool via
            │                stdio, config injected
            │                before each RAG run
   ┌────────▼────────────────────────────────────┐
   │              Analysis pipeline               │
   │  scorer.py    — recall, precision, F1        │
   │  stats.py     — t-test, Wilcoxon, Cohen's d  │
   │  report.py    — markdown tables              │
   │  charts.py    — matplotlib / plotly          │
   │  reliability_report.py — failure profiles    │
   └─────────────────────────────────────────────┘
```

### Concurrency and retry

- One asyncio semaphore per tool (`Semaphore(1)`) prevents concurrent calls to the same tool, avoiding rate limits
- One global semaphore (`Semaphore(--concurrency)`) caps total in-flight invocations
- Different tools run in parallel within those bounds
- Transient errors (timeouts, 429s, rate limits, 503s) trigger exponential backoff retry up to `--max-retries` times
- `asyncio.gather(..., return_exceptions=True)` in all phases ensures one failed task doesn't abort the rest

### MCP config injection (RAG mode)

Before RAG runs start, the runner writes tool-specific MCP config files pointing to the local indices with absolute paths:

| Tool | Config location |
|---|---|
| Claude Code | `{codebase}/.mcp.json` |
| Codex CLI | `~/.codex/config.toml` |
| Gemini CLI | `~/.gemini/settings.json` |
| GitHub Copilot | `~/.copilot/mcp-config.json` |

Existing configs are backed up and restored after the run.

### Checkpoint/resume

Each invocation is saved as an individual JSON file immediately on completion. On re-run, existing files are skipped. This means a failed 8-hour full run can be resumed from exactly where it stopped.

### Author phase isolation

Each author task runs in its own **git worktree** under `results/worktrees/{tool}_{mode}_{task_id}`, on a dedicated branch `bench/{tool}_{mode}_{task_id}`. Worktrees give each concurrent tool an isolated filesystem while sharing the `.git` directory, preventing the race condition where multiple tools doing `git checkout` in the same working copy would stomp each other's changes.

The worktree and branch are removed in a `finally` block regardless of success or error. Stale worktrees from prior crashed runs are cleaned up automatically at the start of the author phase.

---

## Project structure

```
search-bench/
├── src/
│   ├── mcp_server/
│   │   ├── server.py       # MCP stdio server (3 tools: semantic_search, symbol_lookup, related_code)
│   │   ├── indexer.py      # Indexing pipeline, entry point for index-codebase CLI
│   │   ├── chunker.py      # Tree-sitter AST + sliding window chunking
│   │   ├── search.py       # HybridSearch: FAISS + FTS5 with RRF fusion
│   │   └── store.py        # SQLite schema: chunks table + FTS5 virtual table
│   ├── wrappers/
│   │   ├── base.py         # ToolWrapper ABC, Query/QueryResult/SearchOp dataclasses, prompt templates
│   │   ├── claude.py       # Claude Code: claude --print --output-format json
│   │   ├── codex.py        # Codex CLI: codex exec
│   │   ├── gemini.py       # Gemini CLI: gemini --yolo, prompt via stdin
│   │   ├── copilot.py      # GitHub Copilot CLI: copilot -p --output-format json
│   │   └── token_counter.py# Token estimation + USD cost (pricing table)
│   ├── benchmark/
│   │   ├── runner.py       # Main orchestrator + CLI entry point (search-bench)
│   │   └── scorer.py       # Fuzzy file path matching, all scoring metrics
│   └── analysis/
│       ├── stats.py        # t-test, Wilcoxon, Mann-Whitney, Cohen's d, ICC, F-test
│       ├── report.py       # Markdown report generator
│       ├── charts.py       # Matplotlib + Plotly chart generation
│       ├── code_quality.py # Diff analysis, ESLint/TSC runners
│       ├── diff_compare.py # Pairwise diff comparison, Jaccard similarity, consensus files
│       └── reliability_report.py  # Per-tool reliability profiles
├── queries/
│   ├── queries.json        # 60 queries with ground_truth, keywords, anti_files
│   └── smoke_queries.json  # 5-query subset for quick testing
├── tasks/
│   ├── tasks.json          # 20 modification tasks (simple/medium/hard)
│   └── smoke_tasks.json    # 2-task subset for quick testing
├── configs/                # Pre-generated MCP config templates per tool
├── scripts/
│   ├── index_codebase.py   # Standalone indexing script
│   ├── setup_tools.sh      # Check CLI tool + Python dep availability
│   └── setup_configs.py    # Regenerate configs/ with custom DB/FAISS paths
└── data/                   # Generated indices (gitignored)
    ├── circuitsnips.db
    └── circuitsnips.faiss
```

---

## Adding queries or tasks

**Query** (`queries/queries.json`):

```json
{
  "id": "exact_16",
  "text": "Where is the email notification handler?",
  "category": "exact_symbol",
  "ground_truth": ["src/lib/notifications.ts"],
  "keywords": ["email", "notify", "sendEmail"],
  "optional_files": ["src/lib/mailer.ts"],
  "anti_files": ["src/lib/push-notifications.ts"]
}
```

**Task** (`tasks/tasks.json`):

```json
{
  "id": "mod_simple_06",
  "task": "Add an X-Request-ID header to all API responses for tracing",
  "type": "simple",
  "expected_scope": ["src/middleware.ts"]
}
```

`anti_files` are returned by the scorer as `anti_file_hits` — a penalty indicator, not automatically subtracted from recall.

---

## Pricing reference

Estimates computed per-invocation based on token counts reported by each tool (or estimated via tiktoken where unreported).

| Tool | Model | Input ($/1M) | Output ($/1M) |
|---|---|---|---|
| Claude Code | claude-opus-4-5 | $15.00 | $75.00 |
| Codex CLI | gpt-5.4 | $2.50 | $10.00 |
| Gemini CLI | gemini-3-flash-preview | $0.075 | $0.30 |
| GitHub Copilot | claude-haiku-4.5 | $1.00 | $5.00 |

Prices as of 2025. Update `src/wrappers/token_counter.py` to adjust.

---

## License

MIT
