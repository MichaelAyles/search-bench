"""Gemini CLI wrapper."""

import asyncio
import json
import os
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
            # Pass prompt via stdin with -p "" to enable headless mode; JSON for structured output
            env = os.environ.copy()
            if _needs_shell():
                proc = await asyncio.create_subprocess_shell(
                    f'"{cmd}" --yolo --output-format json -p ""',
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.codebase_dir),
                    env=env,
                )
            else:
                proc = await asyncio.create_subprocess_exec(
                    cmd, "--yolo", "--output-format", "json", "-p", "",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.codebase_dir),
                    env=env,
                )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=prompt.encode("utf-8")),
                timeout=300,
            )
        except asyncio.TimeoutError:
            return QueryResult(
                tool_name=self.name(),
                mode=mode.value,
                query_id=query.id,
                run_number=run_number,
                answer="",
                tttc_seconds=time.monotonic() - t0,
                error="Timeout after 300s",
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

        # Parse JSON output format
        answer = raw
        tokens_in = 0
        tokens_out = 0
        try:
            data = json.loads(raw)
            answer = data.get("response", raw)
            # Sum tokens across all models
            for model_stats in data.get("stats", {}).get("models", {}).values():
                toks = model_stats.get("tokens", {})
                tokens_in += toks.get("input", 0)
                tokens_out += toks.get("candidates", 0)
        except (json.JSONDecodeError, KeyError):
            pass

        result = QueryResult(
            tool_name=self.name(),
            mode=mode.value,
            query_id=query.id,
            run_number=run_number,
            answer=answer,
            tttc_seconds=tttc,
            raw_transcript=raw,
        )
        result.tokens_input = tokens_in
        result.tokens_output = tokens_out
        result.files_returned = _extract_files(answer)
        return result


