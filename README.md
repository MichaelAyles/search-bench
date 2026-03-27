# search-bench

Benchmark comparing RAG-based semantic search vs native agentic search across four AI coding CLI tools.

## What

Tests **Claude Code**, **Codex CLI**, **Gemini CLI**, and **GitHub Copilot** on 60 read-only code search queries and 20 code modification tasks. Each tool runs in two modes:

- **Native**: The tool uses its built-in search (grep, glob, file reads)
- **RAG**: The tool uses an MCP server providing hybrid semantic + keyword search

## Metrics

- File recall/precision against hand-labelled ground truth
- Token consumption and USD cost
- Time to task completion (TTTC)
- Variance across repeated runs
- Author/reviewer cross-evaluation (all tools review all diffs)
- CLI reliability profiles across ~2,880 invocations

## Quick Start

```bash
# Clone and set up
git clone https://github.com/MichaelAyles/search-bench
cd search-bench
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"

# Clone the target codebase
git clone https://github.com/MichaelAyles/circuitsnips ./benchmark/circuitsnips

# Index the codebase (builds FAISS + SQLite indices)
python scripts/index_codebase.py ./benchmark/circuitsnips

# Check tool availability
bash scripts/setup_tools.sh

# Run the benchmark (read-only phase)
search-bench --codebase ./benchmark/circuitsnips --phase read_only --tools claude

# Run full benchmark (all tools, all phases)
search-bench --codebase ./benchmark/circuitsnips --phase all
```

## Architecture

```
MCP Server (FAISS + SQLite FTS5)
    ↕ stdio
CLI Tools (Claude Code, Codex, Gemini, Copilot)
    ↕
Benchmark Runner (dispatch, retry, checkpoint)
    ↕
Analysis Pipeline (stats, charts, report)
```

## Project Structure

- `src/mcp_server/` - RAG MCP server (sentence-transformers, FAISS, SQLite FTS5)
- `src/wrappers/` - CLI tool wrappers (Claude, Codex, Gemini, Copilot)
- `src/benchmark/` - Benchmark orchestration (runner, scorer, author/reviewer, reliability)
- `src/analysis/` - Analysis pipeline (stats, charts, report, diff comparison)
- `queries/` - 60 benchmark queries with ground truth
- `tasks/` - 20 modification tasks for author/reviewer benchmark

## License

MIT
