"""Codex CLI wrapper using `codex exec`."""

import asyncio
import json
import re
import time
from pathlib import Path

from .base import ToolWrapper, Query, QueryResult, SearchMode, SearchOp, _needs_shell, _resolve_cmd


class CodexWrapper(ToolWrapper):
    def __init__(self, codebase_dir: str | Path):
        self.codebase_dir = Path(codebase_dir)

    def name(self) -> str:
        return "codex"

    async def check_available(self) -> bool:
        try:
            cmd = _resolve_cmd("codex")
            if _needs_shell():
                proc = await asyncio.create_subprocess_shell(
                    f'"{cmd}" --version',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    cmd, "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            await proc.wait()
            return proc.returncode == 0
        except FileNotFoundError:
            return False

    async def run_query(self, query: Query, mode: SearchMode, run_number: int = 1) -> QueryResult:
        prompt = self.get_prompt(query, mode)
        t0 = time.monotonic()

        try:
            cmd = _resolve_cmd("codex")
            if _needs_shell():
                proc = await asyncio.create_subprocess_shell(
                    f'"{cmd}" exec "{prompt}"',
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.codebase_dir),
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    cmd, "exec", prompt,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.codebase_dir),
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

        result = QueryResult(
            tool_name=self.name(),
            mode=mode.value,
            query_id=query.id,
            run_number=run_number,
            answer=raw,
            tttc_seconds=tttc,
            raw_transcript=raw,
        )

        # Try to parse as JSON
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                result.answer = data.get("result", data.get("output", raw))
                result.tokens_input = data.get("input_tokens", 0)
                result.tokens_output = data.get("output_tokens", 0)
        except json.JSONDecodeError:
            pass

        result.files_returned = _extract_files(result.answer)
        return result


def _extract_files(text: str) -> list[str]:
    files = set()
    m = re.search(r"FILES:\s*\[?([^\]\n]+)\]?", text)
    if m:
        for f in m.group(1).split(","):
            f = f.strip().strip("'\"")
            if f and "/" in f:
                files.add(f)
    for m in re.finditer(r"(?:src|lib|app|pages|components)/[\w/.-]+\.\w+", text):
        files.add(m.group(0))
    return list(files)
