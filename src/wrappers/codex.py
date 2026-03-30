"""Codex CLI wrapper using `codex exec`."""

import asyncio
import json
import shlex
import time
from pathlib import Path

from .base import ToolWrapper, Query, QueryResult, SearchMode, SearchOp, _needs_shell, _resolve_cmd, _extract_files


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
                    f'{shlex.quote(cmd)} --version',
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

    async def _exec(self, prompt: str, cwd: Path, timeout: int = 120) -> tuple[bytes, bytes]:
        """Shared subprocess invocation for Codex CLI."""
        cmd = _resolve_cmd("codex")
        if _needs_shell():
            proc = await asyncio.create_subprocess_shell(
                f'{shlex.quote(cmd)} exec {shlex.quote(prompt)}',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                cmd, "exec", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
            )
        return await asyncio.wait_for(proc.communicate(), timeout=timeout)

    async def run_query(self, query: Query, mode: SearchMode, run_number: int = 1) -> QueryResult:
        prompt = self.get_prompt(query, mode)
        t0 = time.monotonic()

        try:
            stdout, stderr = await self._exec(prompt, self.codebase_dir, timeout=120)
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

    async def run_task(self, prompt: str, cwd: Path, timeout: int = 180) -> tuple[str, str | None]:
        try:
            stdout, stderr = await self._exec(prompt, cwd, timeout=timeout)
            return stdout.decode("utf-8", errors="replace"), None
        except asyncio.TimeoutError:
            return "", f"Timeout after {timeout}s"
        except Exception as exc:
            return "", str(exc)


