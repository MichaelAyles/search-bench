"""Gemini CLI wrapper."""

import asyncio
import json
import time
from pathlib import Path

from .base import ToolWrapper, Query, QueryResult, SearchMode, SearchOp, _needs_shell, _resolve_cmd, _extract_files


class GeminiWrapper(ToolWrapper):
    def __init__(self, codebase_dir: str | Path):
        self.codebase_dir = Path(codebase_dir)

    def name(self) -> str:
        return "gemini"

    async def check_available(self) -> bool:
        try:
            cmd = _resolve_cmd("gemini")
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
            cmd = _resolve_cmd("gemini")
            # Pass prompt via stdin, -p "" enables headless mode
            if _needs_shell():
                proc = await asyncio.create_subprocess_shell(
                    f'"{cmd}" --yolo -p ""',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.codebase_dir),
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    cmd, "--yolo", "-p", "",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.codebase_dir),
                )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=120,
            )
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
        result.files_returned = _extract_files(raw)
        return result


