"""Claude Code wrapper using `claude --print --output-format json`."""

import asyncio
import json
import os
import time
from pathlib import Path

from .base import ToolWrapper, Query, QueryResult, SearchMode, SearchOp, _extract_files


def _clean_env() -> dict[str, str]:
    """Return env dict without CLAUDECODE to allow nested invocations."""
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    return env


class ClaudeWrapper(ToolWrapper):
    def __init__(self, codebase_dir: str | Path):
        self.codebase_dir = Path(codebase_dir)

    def name(self) -> str:
        return "claude"

    async def check_available(self) -> bool:
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=_clean_env(),
            )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def run_query(self, query: Query, mode: SearchMode, run_number: int = 1) -> QueryResult:
        prompt = self.get_prompt(query, mode)
        t0 = time.monotonic()

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "--print", "--output-format", "json",
                "-p", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.codebase_dir),
                env=_clean_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            return QueryResult(
                tool_name=self.name(),
                mode=mode.value,
                query_id=query.id,
                run_number=run_number,
                answer="",
                tttc_seconds=time.monotonic() - t0,
                error="Timeout after 120s",
            )
        except Exception as e:
            return QueryResult(
                tool_name=self.name(),
                mode=mode.value,
                query_id=query.id,
                run_number=run_number,
                answer="",
                tttc_seconds=time.monotonic() - t0,
                error=str(e),
            )

        tttc = time.monotonic() - t0
        raw = stdout.decode("utf-8", errors="replace")

        return self._parse_output(raw, query, mode, run_number, tttc)

    def _parse_output(
        self, raw: str, query: Query, mode: SearchMode, run_number: int, tttc: float
    ) -> QueryResult:
        result = QueryResult(
            tool_name=self.name(),
            mode=mode.value,
            query_id=query.id,
            run_number=run_number,
            answer="",
            tttc_seconds=tttc,
            raw_transcript=raw,
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            result.answer = raw
            result.files_returned = _extract_files(raw)
            return result

        # Handle claude --print JSON output format
        if isinstance(data, dict):
            result.answer = data.get("result", data.get("text", raw))
            result.tokens_input = data.get("input_tokens", 0)
            result.tokens_output = data.get("output_tokens", 0)

            # Parse tool uses from the conversation
            messages = data.get("messages", [])
            search_ops, files_accessed, rounds = _parse_tool_uses(messages)
            result.search_ops = search_ops
            result.files_accessed = files_accessed
            result.rounds = rounds
            result.time_searching = sum(
                s.duration_seconds for s in search_ops if s.type in ("Grep", "Glob")
            )
            result.time_reading = sum(
                s.duration_seconds for s in search_ops if s.type == "Read"
            )

        elif isinstance(data, list):
            # Array of message objects
            texts = []
            search_ops = []
            files_accessed = []
            for msg in data:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                if block.get("type") == "text":
                                    texts.append(block.get("text", ""))
                                elif block.get("type") == "tool_use":
                                    op = SearchOp(
                                        type=block.get("name", "unknown"),
                                        query=json.dumps(block.get("input", {}))[:200],
                                        results=0,
                                        token_cost=0,
                                    )
                                    search_ops.append(op)
                                    inp = block.get("input", {})
                                    if "file_path" in inp:
                                        files_accessed.append(inp["file_path"])
                                    if "path" in inp:
                                        files_accessed.append(inp["path"])

            result.answer = "\n".join(texts)
            result.search_ops = search_ops
            result.files_accessed = list(set(files_accessed))
            result.rounds = len(search_ops)

        result.files_returned = _extract_files(result.answer)
        return result


def _parse_tool_uses(messages: list) -> tuple[list[SearchOp], list[str], int]:
    """Parse tool use blocks from Claude's message array."""
    ops = []
    files = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                inp = block.get("input", {})
                op = SearchOp(
                    type=block.get("name", "unknown"),
                    query=json.dumps(inp)[:200],
                    results=0,
                    token_cost=0,
                )
                ops.append(op)
                for key in ("file_path", "path", "file"):
                    if key in inp:
                        files.append(inp[key])
    return ops, list(set(files)), len(ops)


