"""Main benchmark orchestrator for search-bench."""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..wrappers.base import Query, QueryResult, SearchMode, _resolve_cmd
from ..wrappers.claude import ClaudeWrapper
from ..wrappers.codex import CodexWrapper
from ..wrappers.gemini import GeminiWrapper
from ..wrappers.copilot import CopilotWrapper
from ..wrappers.token_counter import estimate_cost
from ..analysis.code_quality import analyze_diff
from .scorer import score_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/benchmark/runner.py → root

TOOL_CLASSES = {
    "claude": ClaudeWrapper,
    "codex": CodexWrapper,
    "gemini": GeminiWrapper,
    "copilot": CopilotWrapper,
}
ALL_TOOLS = list(TOOL_CLASSES.keys())
ALL_MODES = ["native", "rag"]

AUTHOR_NATIVE_PROMPT = """You are implementing a code change in the CircuitSnips codebase in the current directory.

Task: {task_text}

Instructions:
- Search the codebase to understand the relevant code first
- Make the required changes directly to the files
- Do not ask for clarification; implement it to the best of your ability
- When done, state what files you changed

This is an automated benchmark run. Implement the changes now."""

AUTHOR_RAG_PROMPT = """You are implementing a code change in the CircuitSnips codebase in the current directory.
You have access to MCP tools: semantic_search, symbol_lookup, related_code.

Task: {task_text}

Instructions:
- Use the search tools to find relevant code before editing
- Make the required changes directly to the files
- Do not ask for clarification; implement it to the best of your ability
- When done, state what files you changed

This is an automated benchmark run. Implement the changes now."""

REVIEW_PROMPT = """You are reviewing a code diff for the CircuitSnips codebase.

Task that was attempted:
{task_text}

Diff:
{diff_text}

Review this diff and respond with exactly:
VERDICT: [APPROVE|REQUEST_CHANGES|REJECT]
REASON: [one sentence]

APPROVE = correctly and completely solves the task
REQUEST_CHANGES = on the right track but has issues
REJECT = wrong, empty, or doesn't attempt the task"""


# ---------------------------------------------------------------------------
# MCP config manager
# ---------------------------------------------------------------------------

class MCPConfigManager:
    """Writes and cleans up per-tool MCP config files for RAG mode."""

    def __init__(self, codebase_dir: Path, db_path: Path, faiss_path: Path):
        self.codebase_dir = codebase_dir
        self.db_path = db_path.resolve()
        self.faiss_path = faiss_path.resolve()
        self._backups: dict[str, tuple[Path, bytes | None]] = {}

    def _server_entry(self) -> dict:
        return {
            "command": sys.executable,
            "args": [
                "-m", "src.mcp_server.server",
                "--db", str(self.db_path),
                "--faiss", str(self.faiss_path),
            ],
            "env": {"PYTHONPATH": str(PROJECT_ROOT)},
        }

    def setup(self, tool: str, target_dir: Path | None = None) -> None:
        """Write MCP config for a tool.

        Args:
            tool: Tool name (claude, codex, gemini, copilot).
            target_dir: Override directory for config files that live in the
                codebase (Claude's .mcp.json). If None, uses self.codebase_dir.
        """
        cwd = target_dir or self.codebase_dir
        if tool == "claude":
            target = cwd / ".mcp.json"
            self._backup(tool, target)
            target.write_text(json.dumps(
                {"mcpServers": {"codebase-rag": self._server_entry()}}, indent=2
            ))

        elif tool == "codex":
            codex_dir = Path.home() / ".codex"
            codex_dir.mkdir(exist_ok=True)
            target = codex_dir / "config.toml"
            self._backup(tool, target)
            srv = self._server_entry()
            args_str = ", ".join(f'"{a}"' for a in srv["args"])
            env_str = "\n".join(
                f'  {k} = "{v}"' for k, v in srv.get("env", {}).items()
            )
            env_block = f"\n[mcp.env]\n{env_str}" if env_str else ""
            target.write_text(
                f'[[mcp]]\nname = "codebase-rag"\ncommand = "{srv["command"]}"\n'
                f"args = [{args_str}]\ntransport = \"stdio\"{env_block}\n"
            )

        elif tool == "gemini":
            gemini_dir = Path.home() / ".gemini"
            gemini_dir.mkdir(exist_ok=True)
            target = gemini_dir / "settings.json"
            self._backup(tool, target)
            target.write_text(json.dumps(
                {"mcpServers": {"codebase-rag": self._server_entry()}}, indent=2
            ))

        elif tool == "copilot":
            copilot_dir = Path.home() / ".copilot"
            copilot_dir.mkdir(exist_ok=True)
            target = copilot_dir / "mcp-config.json"
            self._backup(tool, target)
            target.write_text(json.dumps(
                {"mcpServers": {"codebase-rag": self._server_entry()}}, indent=2
            ))

    def teardown(self, tool: str) -> None:
        if tool not in self._backups:
            return
        target, original = self._backups.pop(tool)
        if original is None:
            if target.exists():
                target.unlink()
        else:
            target.write_bytes(original)

    def teardown_all(self) -> None:
        for tool in list(self._backups.keys()):
            self.teardown(tool)

    def _backup(self, tool: str, path: Path) -> None:
        self._backups[tool] = (path, path.read_bytes() if path.exists() else None)


# ---------------------------------------------------------------------------
# Progress display
# ---------------------------------------------------------------------------

class _Progress:
    """Thread-safe counters with a 2-second periodic table printer."""

    def __init__(self, total: int, tools: list[str], modes: list[str], label: str):
        self.total = total
        self.label = label
        self._lock = asyncio.Lock()
        self._counts: dict[str, dict] = {
            f"{t}/{m}": {"done": 0, "err": 0, "tttc": 0.0}
            for t in tools for m in modes
        }
        self._done = 0
        self._task: asyncio.Task | None = None

    async def record(self, tool: str, mode: str, tttc: float, error: bool) -> None:
        async with self._lock:
            self._done += 1
            key = f"{tool}/{mode}"
            if key in self._counts:
                self._counts[key]["done"] += 1
                self._counts[key]["tttc"] += tttc
                if error:
                    self._counts[key]["err"] += 1

    def _render(self) -> str:
        lines = [f"  {self.label} — {self._done}/{self.total}"]
        for key, c in sorted(self._counts.items()):
            if c["done"] == 0:
                continue
            avg = c["tttc"] / c["done"]
            lines.append(
                f"    {key:<20} {c['done']:>4} done  {c['err']:>2} err  avg {avg:5.1f}s"
            )
        return "\n".join(lines)

    async def _printer(self) -> None:
        prev_lines = 0
        while True:
            await asyncio.sleep(2)
            rendered = self._render()
            if prev_lines:
                # Move cursor up and clear to end-of-screen, then redraw
                sys.stdout.write(f"\x1b[{prev_lines}A\x1b[J")
            sys.stdout.write(rendered + "\n")
            sys.stdout.flush()
            prev_lines = rendered.count("\n") + 1

    def start(self) -> None:
        self._task = asyncio.ensure_future(self._printer())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_has_error(cp: Path) -> bool:
    """Check if a checkpoint file contains an errored result."""
    try:
        data = json.loads(cp.read_text())
        return bool(data.get("error"))
    except (json.JSONDecodeError, OSError):
        return True


def _ro_checkpoint(output_dir: Path, tool: str, mode: str, query_id: str, run: int) -> Path:
    return output_dir / f"{tool}_{mode}_{query_id}_run{run}.json"


def _author_checkpoint(output_dir: Path, tool: str, mode: str, task_id: str) -> Path:
    return output_dir / f"author_{tool}_{mode}_{task_id}.json"


def _review_checkpoint(
    output_dir: Path,
    reviewer_tool: str, reviewer_mode: str,
    author_tool: str, author_mode: str,
    task_id: str,
) -> Path:
    return output_dir / f"review_{reviewer_tool}_{reviewer_mode}_of_{author_tool}_{author_mode}_{task_id}.json"


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Query / task loading
# ---------------------------------------------------------------------------

def _load_queries(path: Path) -> list[Query]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    queries = []
    for item in raw:
        queries.append(Query(
            id=item["id"],
            text=item["text"],
            category=item["category"],
            ground_truth=item.get("ground_truth", []),
            keywords=item.get("keywords", []),
            optional_files=item.get("optional_files", []),
            anti_files=item.get("anti_files", []),
        ))
    return queries


def _load_tasks(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Git helpers (for author phase)
# ---------------------------------------------------------------------------

async def _git(args: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Author task runner (per-tool subprocess)
# ---------------------------------------------------------------------------

async def _run_tool_for_task(
    tool_name: str,
    prompt: str,
    codebase_dir: Path,
    timeout: int = 180,
) -> tuple[str, str | None]:
    """Run tool with a modification prompt. Returns (stdout_text, error_or_None)."""
    try:
        if tool_name == "claude":
            import os
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)
            proc = await asyncio.create_subprocess_exec(
                "claude", "--print", "--output-format", "json", "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(codebase_dir),
                env=env,
            )
        elif tool_name == "codex":
            cmd = _resolve_cmd("codex")
            proc = await asyncio.create_subprocess_exec(
                cmd, "exec", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(codebase_dir),
            )
        elif tool_name == "gemini":
            cmd = _resolve_cmd("gemini")
            proc = await asyncio.create_subprocess_exec(
                cmd, "--yolo", "-p", "",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(codebase_dir),
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")), timeout=timeout
            )
            return stdout.decode("utf-8", errors="replace"), None
        elif tool_name == "copilot":
            cmd = _resolve_cmd("copilot")
            proc = await asyncio.create_subprocess_exec(
                cmd, "-p", prompt,
                "--allow-all-tools",
                "--no-auto-update",
                "-s",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(codebase_dir),
            )
        else:
            return "", f"unknown tool: {tool_name}"

        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return stdout.decode("utf-8", errors="replace"), None

    except asyncio.TimeoutError:
        return "", f"Timeout after {timeout}s"
    except Exception as exc:
        return "", str(exc)


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

_RETRYABLE = ("timeout", "429", "rate limit", "rate_limit", "too many", "503", "overloaded", "connection")
_RATE_LIMIT = ("429", "rate limit", "rate_limit", "too many")


async def _run_with_retry(
    wrapper,
    query: "Query",
    mode: "SearchMode",
    run_number: int,
    max_retries: int = 3,
) -> tuple["QueryResult", int, float]:
    """Run a read-only query with exponential backoff retry.

    Returns (result, retry_count, total_rate_limit_wait_seconds).
    Only retries on timeout / rate-limit / transient server errors.
    """
    delay = 5.0
    rate_limit_wait = 0.0

    for attempt in range(max_retries + 1):
        result = await wrapper.run_query(query, mode, run_number)

        if result.error is None:
            return result, attempt, rate_limit_wait

        if attempt >= max_retries:
            break

        err_lower = result.error.lower()
        is_rate_limit = any(x in err_lower for x in _RATE_LIMIT)
        is_retryable = any(x in err_lower for x in _RETRYABLE)

        if not is_retryable:
            break

        wait = delay * 2 if is_rate_limit else delay
        if is_rate_limit:
            rate_limit_wait += wait

        print(
            f"\n  [retry {attempt + 1}/{max_retries}] {wrapper.name()} "
            f"error: {result.error[:60]} — retrying in {wait:.0f}s"
        )
        await asyncio.sleep(wait)
        delay = min(delay * 2, 60.0)

    return result, attempt, rate_limit_wait


async def _tool_with_retry(
    tool_name: str,
    prompt: str,
    codebase_dir: "Path",
    max_retries: int = 3,
    timeout: int = 180,
) -> tuple[str, str | None, int, float]:
    """Run _run_tool_for_task with retries.

    Returns (output, error, retry_count, rate_limit_wait_seconds).
    """
    delay = 5.0
    rate_limit_wait = 0.0

    for attempt in range(max_retries + 1):
        output, error = await _run_tool_for_task(tool_name, prompt, codebase_dir, timeout)

        if error is None:
            return output, None, attempt, rate_limit_wait

        if attempt >= max_retries:
            break

        err_lower = error.lower()
        is_rate_limit = any(x in err_lower for x in _RATE_LIMIT)
        is_retryable = any(x in err_lower for x in _RETRYABLE)

        if not is_retryable:
            break

        wait = delay * 2 if is_rate_limit else delay
        if is_rate_limit:
            rate_limit_wait += wait

        print(
            f"\n  [retry {attempt + 1}/{max_retries}] {tool_name} "
            f"error: {error[:60]} — retrying in {wait:.0f}s"
        )
        await asyncio.sleep(wait)
        delay = min(delay * 2, 60.0)

    return output, error, attempt, rate_limit_wait


# ---------------------------------------------------------------------------
# Review verdict parser
# ---------------------------------------------------------------------------

def _parse_verdict(text: str) -> tuple[str, str]:
    """Extract VERDICT and REASON from review response."""
    verdict = "UNKNOWN"
    reason = ""
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("VERDICT:"):
            v = line.split(":", 1)[1].strip().upper()
            for candidate in ("APPROVE", "REQUEST_CHANGES", "REJECT"):
                if candidate in v:
                    verdict = candidate
                    break
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
    return verdict, reason


# ---------------------------------------------------------------------------
# Read-only phase
# ---------------------------------------------------------------------------

async def _run_read_only(
    queries: list[Query],
    tools: list[str],
    modes: list[str],
    runs: int,
    output_dir: Path,
    codebase_dir: Path,
    db_path: Path,
    faiss_path: Path,
    concurrency: int,
    resume: bool,
    max_retries: int = 3,
    save_transcripts: bool = False,
    retry_errors: bool = False,
) -> tuple[list[dict], list[dict]]:
    """Execute read-only phase. Returns (results, scores)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    mcp = MCPConfigManager(codebase_dir, db_path, faiss_path)
    if "rag" in modes:
        for tool in tools:
            mcp.setup(tool)

    wrappers = {name: TOOL_CLASSES[name](codebase_dir) for name in tools}
    tool_sems = {name: asyncio.Semaphore(1) for name in tools}
    global_sem = asyncio.Semaphore(concurrency)

    # Build work list
    work = []
    skipped = 0
    for query in queries:
        for tool_name in tools:
            for mode in modes:
                for run_num in range(1, runs + 1):
                    cp = _ro_checkpoint(output_dir, tool_name, mode, query.id, run_num)
                    if resume and cp.exists():
                        if retry_errors and _checkpoint_has_error(cp):
                            pass  # fall through to re-run
                        else:
                            skipped += 1
                        continue
                    work.append((query, tool_name, mode, run_num, cp))

    total_planned = len(queries) * len(tools) * len(modes) * runs
    print(f"\nRead-only: {total_planned} total ({skipped} cached, {len(work)} to run)")

    prog = _Progress(len(work), tools, modes, "read-only")
    prog.start()

    async def _run_one(query: Query, tool_name: str, mode_str: str, run_num: int, cp: Path) -> None:
        wrapper = wrappers[tool_name]
        mode = SearchMode(mode_str)

        async with global_sem:
            async with tool_sems[tool_name]:
                result, retry_count, rl_wait = await _run_with_retry(
                    wrapper, query, mode, run_num, max_retries=max_retries
                )

        result.run_meta = {
            "tool_name": tool_name,
            "mode": mode_str,
            "success": result.error is None,
            "retry_count": retry_count,
            "rate_limit_wait_seconds": rl_wait,
            "failure_reason": result.error or "",
        }

        score = score_query(result, query)
        data = result.to_dict()
        data["score"] = score
        if save_transcripts and result.raw_transcript:
            data["raw_transcript"] = result.raw_transcript
        _save_json(cp, data)

        await prog.record(tool_name, mode_str, result.tttc_seconds, error=result.error is not None)

    raw = await asyncio.gather(*[_run_one(*args) for args in work], return_exceptions=True)
    exc_count = sum(1 for r in raw if isinstance(r, BaseException))
    if exc_count:
        print(f"\n  [warn] {exc_count} tasks raised unhandled exceptions (check logs)")

    prog.stop()
    print(prog._render())
    mcp.teardown_all()

    total_errors = sum(c["err"] for c in prog._counts.values())
    if total_errors:
        print(f"\n  {total_errors}/{len(work)} runs had errors")

    return _collect_ro_checkpoints(output_dir, tools, modes, queries, runs)


def _collect_ro_checkpoints(
    output_dir: Path,
    tools: list[str],
    modes: list[str],
    queries: list[Query],
    runs: int,
) -> tuple[list[dict], list[dict]]:
    results, scores = [], []
    for query in queries:
        for tool in tools:
            for mode in modes:
                for run in range(1, runs + 1):
                    cp = _ro_checkpoint(output_dir, tool, mode, query.id, run)
                    if not cp.exists():
                        continue
                    data = json.loads(cp.read_text(encoding="utf-8"))
                    score = data.pop("score", None)
                    results.append(data)
                    if score:
                        scores.append(score)
    return results, scores


# ---------------------------------------------------------------------------
# Author phase
# ---------------------------------------------------------------------------

async def _cleanup_stale_worktrees(codebase_dir: Path, worktrees_dir: Path) -> None:
    """Remove leftover worktrees and branches from a prior crashed run."""
    if not worktrees_dir.exists():
        return
    for entry in worktrees_dir.iterdir():
        if entry.is_dir():
            await _git(["worktree", "remove", str(entry), "--force"], codebase_dir)
    # Prune any worktree metadata that points to now-deleted paths
    await _git(["worktree", "prune"], codebase_dir)
    # Clean up orphaned bench/* branches
    code, stdout, _ = await _git(["branch", "--list", "bench/*"], codebase_dir)
    if code == 0:
        for line in stdout.splitlines():
            branch = line.strip().lstrip("* ")
            if branch:
                await _git(["branch", "-D", branch], codebase_dir)


async def _run_author(
    tasks: list[dict],
    tools: list[str],
    modes: list[str],
    output_dir: Path,
    codebase_dir: Path,
    db_path: Path,
    faiss_path: Path,
    concurrency: int,
    resume: bool,
    max_retries: int = 3,
    retry_errors: bool = False,
) -> list[dict]:
    """Execute author (code-modification) phase. Returns author results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    worktrees_dir = output_dir / "worktrees"
    worktrees_dir.mkdir(exist_ok=True)

    # Verify codebase is a git repo
    code, _, err = await _git(["rev-parse", "--abbrev-ref", "HEAD"], codebase_dir)
    if code != 0:
        print(f"  [warn] author phase: codebase is not a git repo ({err.strip()}); skipping")
        return []

    # Clean up stale worktrees/branches from prior crashed runs
    await _cleanup_stale_worktrees(codebase_dir, worktrees_dir)

    # MCP configs for tools that use global config paths (codex, gemini)
    # are set up once here. Claude's .mcp.json goes in each worktree.
    mcp = MCPConfigManager(codebase_dir, db_path, faiss_path)
    if "rag" in modes:
        for tool in tools:
            if tool != "claude":
                mcp.setup(tool)

    tool_sems = {name: asyncio.Semaphore(1) for name in tools}
    global_sem = asyncio.Semaphore(concurrency)

    work = []
    skipped = 0
    for task in tasks:
        for tool_name in tools:
            for mode in modes:
                cp = _author_checkpoint(output_dir, tool_name, mode, task["id"])
                if resume and cp.exists():
                    if retry_errors and _checkpoint_has_error(cp):
                        pass  # fall through to re-run
                    else:
                        skipped += 1
                        continue
                work.append((task, tool_name, mode, cp))

    print(f"\nAuthor: {len(tasks)*len(tools)*len(modes)} total ({skipped} cached, {len(work)} to run)")

    prog = _Progress(len(work), tools, modes, "author")
    prog.start()

    async def _run_one(task: dict, tool_name: str, mode_str: str, cp: Path) -> None:
        task_id = task["id"]
        task_text = task["task"]

        prompt_tmpl = AUTHOR_RAG_PROMPT if mode_str == "rag" else AUTHOR_NATIVE_PROMPT
        prompt = prompt_tmpl.format(task_text=task_text)

        branch = f"bench/{tool_name}_{mode_str}_{task_id}"
        wt_path = worktrees_dir / f"{tool_name}_{mode_str}_{task_id}"

        t0 = time.monotonic()
        error: str | None = None
        diff = ""
        retry_count, rl_wait = 0, 0.0

        async with global_sem:
            async with tool_sems[tool_name]:
                # Clean up if this worktree/branch exists from a prior crash
                if wt_path.exists():
                    await _git(["worktree", "remove", str(wt_path), "--force"], codebase_dir)
                await _git(["branch", "-D", branch], codebase_dir)

                # Create isolated worktree
                code, _, err = await _git(
                    ["worktree", "add", str(wt_path), "-b", branch, "HEAD"],
                    codebase_dir,
                )

                if code != 0:
                    error = f"git worktree add failed: {err.strip()}"
                else:
                    try:
                        # Write Claude MCP config into the worktree
                        if mode_str == "rag" and tool_name == "claude":
                            mcp.setup("claude", target_dir=wt_path)

                        _output, error, retry_count, rl_wait = await _tool_with_retry(
                            tool_name, prompt, wt_path, max_retries=max_retries
                        )
                        # Diff against HEAD within the worktree
                        _, diff, _ = await _git(["diff", "HEAD"], wt_path)
                    finally:
                        await _git(["worktree", "remove", str(wt_path), "--force"], codebase_dir)
                        await _git(["branch", "-D", branch], codebase_dir)

        tttc = time.monotonic() - t0
        diff_analysis = analyze_diff(diff, tool_name, mode_str, task_id)

        result = {
            "task_id": task_id,
            "tool_name": tool_name,
            "mode": mode_str,
            "tttc_seconds": tttc,
            "diff": diff,
            "diff_stat": {
                "lines_added": diff_analysis.lines_added,
                "lines_removed": diff_analysis.lines_removed,
                "files_modified": diff_analysis.files_modified,
            },
            "error": error,
            "run_meta": {
                "tool_name": tool_name,
                "mode": mode_str,
                "success": error is None and bool(diff),
                "retry_count": retry_count,
                "rate_limit_wait_seconds": rl_wait,
                "failure_reason": error or ("empty diff" if not diff else ""),
            },
        }
        _save_json(cp, result)
        await prog.record(tool_name, mode_str, tttc, error=bool(error))

    raw = await asyncio.gather(*[_run_one(*args) for args in work], return_exceptions=True)
    exc_count = sum(1 for r in raw if isinstance(r, BaseException))
    if exc_count:
        print(f"\n  [warn] {exc_count} author tasks raised unhandled exceptions")

    prog.stop()
    print(prog._render())
    mcp.teardown_all()

    total_errors = sum(c["err"] for c in prog._counts.values())
    if total_errors:
        print(f"\n  {total_errors}/{len(work)} author runs had errors")

    return _collect_author_checkpoints(output_dir, tools, modes, tasks)


def _collect_author_checkpoints(
    output_dir: Path, tools: list[str], modes: list[str], tasks: list[dict]
) -> list[dict]:
    results = []
    for task in tasks:
        for tool in tools:
            for mode in modes:
                cp = _author_checkpoint(output_dir, tool, mode, task["id"])
                if cp.exists():
                    results.append(json.loads(cp.read_text(encoding="utf-8")))
    return results


# ---------------------------------------------------------------------------
# Review phase
# ---------------------------------------------------------------------------

async def _run_review(
    tasks: list[dict],
    tools: list[str],
    modes: list[str],
    output_dir: Path,
    codebase_dir: Path,
    db_path: Path,
    faiss_path: Path,
    concurrency: int,
    resume: bool,
    author_results: list[dict],
    max_retries: int = 3,
    retry_errors: bool = False,
) -> list[dict]:
    """Execute review phase (every reviewer reviews every author diff)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Index author results by (task_id, tool, mode)
    author_index: dict[tuple[str, str, str], dict] = {}
    for ar in author_results:
        key = (ar["task_id"], ar["tool_name"], ar["mode"])
        author_index[key] = ar

    # Review only sends text prompts; no MCP config needed.
    tool_sems = {name: asyncio.Semaphore(1) for name in tools}
    global_sem = asyncio.Semaphore(concurrency)

    work = []
    skipped = 0
    for task in tasks:
        task_id = task["id"]
        for author_tool in tools:
            for author_mode in modes:
                author_key = (task_id, author_tool, author_mode)
                if author_key not in author_index:
                    continue
                ar = author_index[author_key]
                if not ar.get("diff"):
                    continue  # nothing to review

                for reviewer_tool in tools:
                    for reviewer_mode in modes:
                        cp = _review_checkpoint(
                            output_dir,
                            reviewer_tool, reviewer_mode,
                            author_tool, author_mode,
                            task_id,
                        )
                        if resume and cp.exists():
                            if retry_errors and _checkpoint_has_error(cp):
                                pass  # fall through to re-run
                            else:
                                skipped += 1
                                continue
                        work.append((task, ar, reviewer_tool, reviewer_mode, cp))

    print(f"\nReview: {len(work) + skipped} total ({skipped} cached, {len(work)} to run)")

    prog = _Progress(len(work), tools, modes, "review")
    prog.start()

    async def _run_one(
        task: dict, ar: dict,
        reviewer_tool: str, reviewer_mode: str,
        cp: Path,
    ) -> None:
        diff_text = ar["diff"]
        if len(diff_text) > 12_000:
            diff_text = diff_text[:12_000] + "\n... (truncated)"

        prompt = REVIEW_PROMPT.format(
            task_text=task["task"],
            diff_text=diff_text,
        )

        t0 = time.monotonic()
        async with global_sem:
            async with tool_sems[reviewer_tool]:
                output, error, retry_count, rl_wait = await _tool_with_retry(
                    reviewer_tool, prompt, codebase_dir, max_retries=max_retries
                )

        tttc = time.monotonic() - t0

        # For Claude, unwrap the JSON envelope to get the text response
        answer = output
        if reviewer_tool == "claude":
            try:
                data = json.loads(output)
                if isinstance(data, dict):
                    answer = data.get("result", data.get("text", output))
            except (json.JSONDecodeError, TypeError):
                pass

        verdict, reason = _parse_verdict(answer)

        result = {
            "task_id": task["id"],
            "author_tool": ar["tool_name"],
            "author_mode": ar["mode"],
            "reviewer_tool": reviewer_tool,
            "reviewer_mode": reviewer_mode,
            "verdict": verdict,
            "reasoning": reason,
            "tttc_seconds": tttc,
            "error": error,
            "run_meta": {
                "tool_name": reviewer_tool,
                "mode": reviewer_mode,
                "success": error is None,
                "retry_count": retry_count,
                "rate_limit_wait_seconds": rl_wait,
                "failure_reason": error or "",
            },
        }
        _save_json(cp, result)
        await prog.record(reviewer_tool, reviewer_mode, tttc, error=bool(error))

    raw = await asyncio.gather(*[_run_one(*args) for args in work], return_exceptions=True)
    exc_count = sum(1 for r in raw if isinstance(r, BaseException))
    if exc_count:
        print(f"\n  [warn] {exc_count} review tasks raised unhandled exceptions")

    prog.stop()
    print(prog._render())

    total_errors = sum(c["err"] for c in prog._counts.values())
    if total_errors:
        print(f"\n  {total_errors}/{len(work)} review runs had errors")

    return _collect_review_checkpoints(output_dir, tools, modes, tasks)


def _collect_review_checkpoints(
    output_dir: Path,
    tools: list[str],
    modes: list[str],
    tasks: list[dict],
) -> list[dict]:
    results = []
    for task in tasks:
        task_id = task["id"]
        for author_tool in tools:
            for author_mode in modes:
                for reviewer_tool in tools:
                    for reviewer_mode in modes:
                        cp = _review_checkpoint(
                            output_dir,
                            reviewer_tool, reviewer_mode,
                            author_tool, author_mode,
                            task_id,
                        )
                        if cp.exists():
                            results.append(json.loads(cp.read_text(encoding="utf-8")))
    return results


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def _aggregate_results(
    output_dir: Path,
    ro_results: list[dict],
    ro_scores: list[dict],
    author_results: list[dict],
    review_results: list[dict],
    tools: list[str],
    modes: list[str],
    runs: int,
    codebase_dir: Path,
) -> Path:
    total_cost = sum(
        estimate_cost(
            r.get("tool_name", "claude"),
            r.get("tokens_input", 0),
            r.get("tokens_output", 0),
        ).cost_usd
        for r in ro_results
    )
    total_duration = sum(r.get("tttc_seconds", 0) for r in ro_results)
    total_duration += sum(r.get("tttc_seconds", 0) for r in author_results)
    total_duration += sum(r.get("tttc_seconds", 0) for r in review_results)

    data = {
        "read_only_results": ro_results,
        "read_only_scores": ro_scores,
        "author_results": author_results,
        "review_results": review_results,
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "codebase": str(codebase_dir),
            "tools": tools,
            "modes": modes,
            "runs_per_query": runs,
            "total_invocations": len(ro_results) + len(author_results) + len(review_results),
            "total_cost_usd": round(total_cost, 4),
            "total_duration_seconds": round(total_duration, 1),
        },
    }

    results_path = output_dir / "results.json"
    _save_json(results_path, data)
    return results_path


def _generate_reports(results_path: Path, output_dir: Path) -> None:
    try:
        from ..analysis.report import generate_report
        report = generate_report(results_path)
        report_path = output_dir / "report.md"
        report_path.write_text(report, encoding="utf-8")
        print(f"  Report: {report_path}")
    except Exception as exc:
        print(f"  [warn] report generation failed: {exc}")

    try:
        from ..analysis.charts import generate_all_charts
        charts_dir = output_dir / "charts"
        generate_all_charts(results_path, charts_dir)
    except Exception as exc:
        print(f"  [warn] chart generation failed: {exc}")

    try:
        from ..analysis.reliability_report import generate_reliability_report
        rel_report = generate_reliability_report(results_path)
        rel_path = output_dir / "reliability.md"
        rel_path.write_text(rel_report, encoding="utf-8")
        print(f"  Reliability: {rel_path}")
    except Exception as exc:
        print(f"  [warn] reliability report failed: {exc}")


# ---------------------------------------------------------------------------
# Statistical comparisons
# ---------------------------------------------------------------------------

def _run_stats(ro_scores: list[dict]) -> None:
    if not ro_scores:
        return
    try:
        from ..analysis.stats import compare_groups

        # Native vs RAG for each tool
        print("\n--- Statistical Comparisons (native vs rag, file_recall) ---")
        tools = sorted({s["tool_name"] for s in ro_scores})
        for tool in tools:
            native = [s["file_recall"] for s in ro_scores if s["tool_name"] == tool and s["mode"] == "native"]
            rag = [s["file_recall"] for s in ro_scores if s["tool_name"] == tool and s["mode"] == "rag"]
            if native and rag:
                result = compare_groups(native, rag, "file_recall", f"{tool}/native", f"{tool}/rag")
                sig = "*" if result.significant else ""
                print(
                    f"  {tool}: native={result.mean_a:.3f} rag={result.mean_b:.3f} "
                    f"p={result.p_value:.3f} d={result.effect_size:.2f}{sig}"
                )
    except Exception as exc:
        print(f"  [warn] stats failed: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="search-bench",
        description="Benchmark RAG vs native search across AI coding CLI tools",
    )
    parser.add_argument("--codebase", required=True, type=Path,
                        help="Path to target codebase (CircuitSnips)")
    parser.add_argument("--phase", default="read_only",
                        choices=["read_only", "author", "review", "all"],
                        help="Which benchmark phase(s) to run (default: read_only)")
    parser.add_argument("--tools", default="all",
                        help="Comma-separated tools or 'all' (default: all)")
    parser.add_argument("--modes", default="all",
                        help="Comma-separated modes or 'all' (default: all)")
    parser.add_argument("--runs", type=int, default=3,
                        help="Runs per query (default: 3)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent tool invocations (default: 4)")
    parser.add_argument("--output-dir", type=Path, default=Path("./results"),
                        help="Results directory (default: ./results)")
    parser.add_argument("--queries", type=Path, default=Path("queries/queries.json"),
                        help="Query file (default: queries/queries.json)")
    parser.add_argument("--tasks", type=Path, default=Path("tasks/tasks.json"),
                        help="Task file (default: tasks/tasks.json)")
    parser.add_argument("--smoke", action="store_true",
                        help="Quick run: smoke queries/tasks, 1 run, concurrency 2")
    parser.add_argument("--resume", default=True, action="store_true",
                        help="Skip already-checkpointed results (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Force rerun of all results")
    parser.add_argument("--retry-errors", action="store_true",
                        help="Re-run checkpointed results that contain errors")
    parser.add_argument("--db", type=Path, default=Path("./data/circuitsnips.db"),
                        help="SQLite DB path for MCP server")
    parser.add_argument("--faiss", type=Path, default=Path("./data/circuitsnips.faiss"),
                        help="FAISS index path for MCP server")
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Max retries on transient errors per invocation (default: 3)")
    parser.add_argument("--save-transcripts", action="store_true",
                        help="Include raw tool output in read-only checkpoint files")
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    # --- Apply --smoke overrides ---
    if args.smoke:
        args.queries = Path("queries/smoke_queries.json")
        args.tasks = Path("tasks/smoke_tasks.json")
        args.runs = 1
        args.concurrency = 2

    # --- Resolve tools and modes ---
    if args.tools == "all":
        tools = ALL_TOOLS
    else:
        tools = [t.strip() for t in args.tools.split(",")]
        unknown = [t for t in tools if t not in TOOL_CLASSES]
        if unknown:
            print(f"Unknown tools: {unknown}. Available: {ALL_TOOLS}", file=sys.stderr)
            sys.exit(1)

    if args.modes == "all":
        modes = ALL_MODES
    else:
        modes = [m.strip() for m in args.modes.split(",")]
        unknown = [m for m in modes if m not in ALL_MODES]
        if unknown:
            print(f"Unknown modes: {unknown}. Available: {ALL_MODES}", file=sys.stderr)
            sys.exit(1)

    codebase_dir = args.codebase.resolve()
    if not codebase_dir.exists():
        print(f"Codebase not found: {codebase_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.resolve()
    db_path = args.db.resolve() if args.db.is_absolute() else (PROJECT_ROOT / args.db).resolve()
    faiss_path = args.faiss.resolve() if args.faiss.is_absolute() else (PROJECT_ROOT / args.faiss).resolve()

    # --- Check RAG index if needed ---
    if "rag" in modes:
        if not db_path.exists() or not faiss_path.exists():
            print(
                f"[warn] RAG mode requested but index not found:\n"
                f"  DB:    {db_path} ({'OK' if db_path.exists() else 'MISSING'})\n"
                f"  FAISS: {faiss_path} ({'OK' if faiss_path.exists() else 'MISSING'})\n"
                f"  Run: index-codebase {codebase_dir}\n"
                f"  Or use --modes native to skip RAG.",
                file=sys.stderr,
            )
            sys.exit(1)

    # --- Check tool availability ---
    print("Checking tool availability...")
    wrappers = {name: TOOL_CLASSES[name](codebase_dir) for name in tools}
    unavailable = []
    for name, wrapper in wrappers.items():
        ok = await wrapper.check_available()
        print(f"  {'✓' if ok else '✗'} {name}")
        if not ok:
            unavailable.append(name)
    if unavailable:
        print(f"Unavailable tools: {unavailable}. Install them or remove from --tools.", file=sys.stderr)
        sys.exit(1)

    # --- Load queries and tasks ---
    queries_path = args.queries if args.queries.is_absolute() else PROJECT_ROOT / args.queries
    tasks_path = args.tasks if args.tasks.is_absolute() else PROJECT_ROOT / args.tasks

    queries = _load_queries(queries_path)
    tasks = _load_tasks(tasks_path) if tasks_path.exists() else []
    print(f"\nLoaded {len(queries)} queries, {len(tasks)} tasks")
    print(f"Tools: {tools}  Modes: {modes}  Runs: {args.runs}  Concurrency: {args.concurrency}")
    print(f"Output: {output_dir}")

    phases = ["read_only", "author", "review"] if args.phase == "all" else [args.phase]

    ro_results: list[dict] = []
    ro_scores: list[dict] = []
    author_results: list[dict] = []
    review_results: list[dict] = []

    t_start = time.monotonic()

    # --- Read-only phase ---
    if "read_only" in phases:
        ro_results, ro_scores = await _run_read_only(
            queries=queries,
            tools=tools,
            modes=modes,
            runs=args.runs,
            output_dir=output_dir,
            codebase_dir=codebase_dir,
            db_path=db_path,
            faiss_path=faiss_path,
            concurrency=args.concurrency,
            resume=args.resume,
            max_retries=args.max_retries,
            save_transcripts=args.save_transcripts,
            retry_errors=args.retry_errors,
        )
        if ro_scores:
            recalls = [s["file_recall"] for s in ro_scores]
            mean_recall = sum(recalls) / len(recalls)
            print(f"\nRead-only complete: mean recall={mean_recall:.3f} over {len(ro_scores)} runs")

    # --- Author phase ---
    if "author" in phases:
        if not tasks:
            print("\n[warn] No tasks found; skipping author phase")
        else:
            author_results = await _run_author(
                tasks=tasks,
                tools=tools,
                modes=modes,
                output_dir=output_dir,
                codebase_dir=codebase_dir,
                db_path=db_path,
                faiss_path=faiss_path,
                concurrency=args.concurrency,
                resume=args.resume,
                max_retries=args.max_retries,
                retry_errors=args.retry_errors,
            )

    # --- Review phase ---
    if "review" in phases:
        if not author_results:
            # Try loading from checkpoints (e.g. --phase review after prior author run)
            author_results = _collect_author_checkpoints(output_dir, tools, modes, tasks)
        if not author_results:
            print("\n[warn] No author results found; skipping review phase")
        else:
            review_results = await _run_review(
                tasks=tasks,
                tools=tools,
                modes=modes,
                output_dir=output_dir,
                codebase_dir=codebase_dir,
                db_path=db_path,
                faiss_path=faiss_path,
                concurrency=args.concurrency,
                resume=args.resume,
                author_results=author_results,
                max_retries=args.max_retries,
                retry_errors=args.retry_errors,
            )

    # --- Aggregate and report ---
    # Reload from disk so partial prior runs are included even if this run skipped them
    if not ro_results and not ro_scores:
        ro_results, ro_scores = _collect_ro_checkpoints(output_dir, tools, modes, queries, args.runs)
    if not author_results:
        author_results = _collect_author_checkpoints(output_dir, tools, modes, tasks)
    if not review_results:
        review_results = _collect_review_checkpoints(output_dir, tools, modes, tasks)

    print(f"\nAggregating {len(ro_results)} read-only, {len(author_results)} author, "
          f"{len(review_results)} review results...")

    results_path = _aggregate_results(
        output_dir=output_dir,
        ro_results=ro_results,
        ro_scores=ro_scores,
        author_results=author_results,
        review_results=review_results,
        tools=tools,
        modes=modes,
        runs=args.runs,
        codebase_dir=codebase_dir,
    )
    print(f"  Results: {results_path}")

    _run_stats(ro_scores)

    print("\nGenerating reports...")
    _generate_reports(results_path, output_dir)

    elapsed = time.monotonic() - t_start
    print(f"\nDone in {elapsed:.0f}s. Results in {output_dir}/")


def main() -> None:
    args = _parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
