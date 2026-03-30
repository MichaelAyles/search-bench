"""Tool wrapper ABC and shared data types."""

import os
import shutil
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class SearchMode(Enum):
    NATIVE = "native"
    RAG = "rag"


@dataclass
class Query:
    id: str
    text: str
    category: str  # exact_symbol, conceptual, cross_cutting, refactoring
    ground_truth: list[str]  # Expected file paths
    keywords: list[str]  # Key terms for answer
    optional_files: list[str] = field(default_factory=list)
    anti_files: list[str] = field(default_factory=list)


@dataclass
class SearchOp:
    type: str  # grep, glob, read, semantic_search, symbol_lookup, etc.
    query: str
    results: int
    token_cost: int
    duration_seconds: float = 0.0


@dataclass
class QueryResult:
    tool_name: str
    mode: str  # "native" or "rag"
    query_id: str
    run_number: int
    answer: str
    files_accessed: list[str] = field(default_factory=list)
    files_returned: list[str] = field(default_factory=list)
    search_ops: list[SearchOp] = field(default_factory=list)
    tokens_input: int = 0
    tokens_output: int = 0
    tttc_seconds: float = 0.0
    time_searching: float = 0.0
    time_reading: float = 0.0
    time_thinking: float = 0.0
    rounds: int = 0
    raw_transcript: str = ""
    error: str | None = None
    run_meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "mode": self.mode,
            "query_id": self.query_id,
            "run_number": self.run_number,
            "answer": self.answer,
            "files_accessed": self.files_accessed,
            "files_returned": self.files_returned,
            "search_ops": [
                {"type": s.type, "query": s.query, "results": s.results,
                 "token_cost": s.token_cost, "duration_seconds": s.duration_seconds}
                for s in self.search_ops
            ],
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tttc_seconds": self.tttc_seconds,
            "time_searching": self.time_searching,
            "time_reading": self.time_reading,
            "time_thinking": self.time_thinking,
            "rounds": self.rounds,
            "error": self.error,
            "run_meta": self.run_meta,
        }


NATIVE_PROMPT_TEMPLATE = """You are analysing the CircuitSnips codebase in the current directory.

Question: {query_text}

Instructions:
- Search the codebase to find the answer
- List the specific file paths you found relevant
- Explain what each relevant file does in relation to the question
- Be specific: reference function names, class names, and line numbers where possible
- If you're unsure, say so rather than guessing

Format your response as:
FILES: [comma-separated list of file paths]
ANSWER: [your detailed answer]"""


RAG_PROMPT_TEMPLATE = """You are analysing the CircuitSnips codebase in the current directory.
You have access to a semantic search MCP tool called "semantic_search" that can find code by meaning.
You also have "symbol_lookup" for finding specific functions/classes and "related_code" for finding similar code.

Question: {query_text}

Instructions:
- Use the semantic_search tool to find relevant code
- You may also use symbol_lookup and related_code for more targeted searches
- List the specific file paths you found relevant
- Explain what each relevant file does in relation to the question
- Be specific: reference function names, class names, and line numbers where possible

Format your response as:
FILES: [comma-separated list of file paths]
ANSWER: [your detailed answer]"""


def _extract_files(text: str) -> list[str]:
    """Extract file paths mentioned in answer text (FILES: line + path patterns).

    Works across arbitrary codebases by matching any path-like string that
    contains at least one ``/`` and ends with a file extension, while
    filtering out URLs and other false positives.
    """
    import re
    files: set[str] = set()

    # 1. Parse the explicit FILES: line the prompts request.
    m = re.search(r"FILES:\s*\[?([^\]\n]+)\]?", text)
    if m:
        for f in m.group(1).split(","):
            f = f.strip().strip("'\"` ")
            if f and "/" in f:
                files.add(f)

    # 2. General path pattern: at least one directory component followed by a
    #    filename with an extension.  Matches things like:
    #      django/core/handlers/base.py
    #      packages/react-dom/src/client/ReactDOM.js
    #      kernel/sched/core.c
    #      ./src/main.py
    #    The negative look-behind rejects :// and .com/ so URLs are excluded.
    path_re = re.compile(
        r"(?<![:/\w.])"                # not preceded by : / word-char or . (rejects URLs)
        r"(\.?/?(?:[\w@-]+/)+[\w.-]+\.\w{1,10})"  # optional ./ then dir/…/file.ext
    )
    for m in path_re.finditer(text):
        candidate = m.group(1)
        # Strip leading ./ for normalization
        candidate = re.sub(r"^\./", "", candidate)
        # Skip semver-like fragments (e.g. 2.0/rc1/notes.txt captured oddly)
        if re.match(r"^\d+\.\d+/", candidate):
            continue
        files.add(candidate)

    return list(files)


def _needs_shell() -> bool:
    """On Windows, .CMD/.BAT wrappers need shell=True for asyncio subprocess."""
    return sys.platform == "win32"


def _resolve_cmd(cmd: str) -> str:
    """Resolve a command to its full path (handles .CMD on Windows)."""
    resolved = shutil.which(cmd)
    return resolved if resolved else cmd


class ToolWrapper(ABC):
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def run_query(self, query: Query, mode: SearchMode, run_number: int = 1) -> QueryResult:
        ...

    @abstractmethod
    async def check_available(self) -> bool:
        """Check if this tool is installed and accessible."""
        ...

    @abstractmethod
    async def run_task(self, prompt: str, cwd: Path, timeout: int = 180) -> tuple[str, str | None]:
        """Run the tool with a free-form prompt for modification/review tasks.

        Returns (stdout_text, error_or_None).  Subclasses reuse the same
        subprocess flags they use in ``run_query`` so there is only one place
        to update when CLI flags change.
        """
        ...

    def get_prompt(self, query: Query, mode: SearchMode) -> str:
        template = RAG_PROMPT_TEMPLATE if mode == SearchMode.RAG else NATIVE_PROMPT_TEMPLATE
        return template.format(query_text=query.text)
