"""GitHub Copilot CLI wrapper using `copilot -p --output-format json`."""

import asyncio
import json
import time
from pathlib import Path

from .base import ToolWrapper, Query, QueryResult, SearchMode, SearchOp, _resolve_cmd, _extract_files


DEFAULT_MODEL = "claude-haiku-4.5"


class CopilotWrapper(ToolWrapper):
    def __init__(self, codebase_dir: str | Path, model: str = DEFAULT_MODEL):
        self.codebase_dir = Path(codebase_dir)
        self.model = model

    def name(self) -> str:
        return "copilot"

    async def check_available(self) -> bool:
        try:
            cmd = _resolve_cmd("copilot")
            proc = await asyncio.create_subprocess_exec(
                cmd, "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            # New CLI reports "GitHub Copilot CLI x.y.z"
            return proc.returncode == 0 and b"Copilot" in stdout
        except FileNotFoundError:
            return False

    async def _exec(self, prompt: str, cwd: Path, timeout: int = 120) -> tuple[bytes, bytes]:
        """Shared subprocess invocation for Copilot CLI."""
        cmd = _resolve_cmd("copilot")
        proc = await asyncio.create_subprocess_exec(
            cmd, "-p", prompt,
            "--output-format", "json",
            "--allow-all-tools",
            "--no-auto-update",
            "--model", self.model,
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

        return self._parse_output(raw, query, mode, run_number, tttc)

    async def run_task(self, prompt: str, cwd: Path, timeout: int = 180) -> tuple[str, str | None]:
        try:
            stdout, stderr = await self._exec(prompt, cwd, timeout=timeout)
            return stdout.decode("utf-8", errors="replace"), None
        except asyncio.TimeoutError:
            return "", f"Timeout after {timeout}s"
        except Exception as exc:
            return "", str(exc)

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

        # Output is JSONL — one JSON object per line
        texts = []
        search_ops = []
        files_accessed = []
        output_tokens = 0

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            data = event.get("data", {})

            if event_type == "assistant.message":
                content = data.get("content", "")
                if content:
                    texts.append(content)
                output_tokens += data.get("outputTokens", 0)

                # Parse tool requests from the message
                # Copilot uses "name"/"arguments", not "toolName"/"input"
                for tr in data.get("toolRequests", []):
                    tool_name = tr.get("name", tr.get("toolName", "unknown"))
                    tool_args = tr.get("arguments", tr.get("input", {}))
                    op = SearchOp(
                        type=tool_name,
                        query=json.dumps(tool_args)[:200],
                        results=0,
                        token_cost=0,
                    )
                    search_ops.append(op)
                    for key in ("file_path", "path", "file", "filePath"):
                        if key in tool_args:
                            files_accessed.append(tool_args[key])

            elif event_type == "tool.execution_start":
                tool_name = data.get("toolName", "")
                args = data.get("arguments", {})
                # Extract file paths from tool arguments
                for key in ("file_path", "path", "file", "filePath"):
                    if key in args:
                        files_accessed.append(args[key])

            elif event_type == "tool.execution_complete":
                tool_name = data.get("toolName", "")
                # Track files from view/read results
                if tool_name in ("read", "Read", "view"):
                    res = data.get("result", {})
                    tel = data.get("toolTelemetry", {}).get("properties", {})
                    # toolTelemetry sometimes has the file extension/path info
                    for key in ("filePath", "path"):
                        if key in res:
                            files_accessed.append(res[key])

            elif event_type == "result":
                # Final summary event with usage stats
                usage = data.get("usage", {})
                if not output_tokens and "totalApiDurationMs" in usage:
                    pass  # no token count available in result

        result.answer = "\n".join(texts) if texts else raw
        result.search_ops = search_ops
        result.files_accessed = list(set(files_accessed))
        result.rounds = len(search_ops)
        result.tokens_output = output_tokens

        # Normalise tool names: copilot namespaces MCP tools as
        # "codebase-rag-semantic_search" etc. Strip the server prefix.
        _search_types = {"grep", "glob", "Grep", "Glob",
                         "semantic_search", "codebase-rag-semantic_search"}
        _read_types = {"read", "Read", "view",
                       "codebase-rag-symbol_lookup", "symbol_lookup",
                       "codebase-rag-related_code", "related_code"}
        result.time_searching = sum(
            s.duration_seconds for s in search_ops if s.type in _search_types
        )
        result.time_reading = sum(
            s.duration_seconds for s in search_ops if s.type in _read_types
        )
        result.files_returned = _extract_files(result.answer)
        return result
