"""Static analysis runner, complexity metrics, style consistency for author diffs."""

import asyncio
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StaticAnalysisResult:
    tool_name: str
    mode: str
    task_id: str
    lint_errors: int = 0
    lint_warnings: int = 0
    type_errors: int = 0
    tests_pass: bool = True
    tests_added: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    files_modified: int = 0
    files_created: int = 0
    files_deleted: int = 0
    imports_added: list[str] = field(default_factory=list)
    imports_removed: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "mode": self.mode,
            "task_id": self.task_id,
            "lint_errors": self.lint_errors,
            "lint_warnings": self.lint_warnings,
            "type_errors": self.type_errors,
            "tests_pass": self.tests_pass,
            "tests_added": self.tests_added,
            "lines_added": self.lines_added,
            "lines_removed": self.lines_removed,
            "files_modified": self.files_modified,
            "files_created": self.files_created,
            "files_deleted": self.files_deleted,
            "imports_added": self.imports_added,
            "imports_removed": self.imports_removed,
        }


async def run_eslint(codebase_dir: Path) -> tuple[int, int]:
    """Run ESLint and return (errors, warnings)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "npx", "eslint", ".", "--format", "json", "--quiet",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(codebase_dir),
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
        import json
        results = json.loads(stdout.decode("utf-8", errors="replace"))
        errors = sum(r.get("errorCount", 0) for r in results)
        warnings = sum(r.get("warningCount", 0) for r in results)
        return errors, warnings
    except Exception:
        return -1, -1


async def run_tsc(codebase_dir: Path) -> int:
    """Run TypeScript compiler in check mode. Returns error count."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "npx", "tsc", "--noEmit",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(codebase_dir),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        output = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")
        # Count error lines
        return len([l for l in output.splitlines() if "error TS" in l])
    except Exception:
        return -1


def parse_diff_imports(diff_text: str) -> tuple[list[str], list[str]]:
    """Extract added and removed imports from a diff."""
    added = []
    removed = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            if re.match(r"\+\s*(import |from |require\(|const .* = require)", line):
                added.append(line[1:].strip())
        elif line.startswith("-") and not line.startswith("---"):
            if re.match(r"-\s*(import |from |require\(|const .* = require)", line):
                removed.append(line[1:].strip())
    return added, removed


def analyze_diff(diff_text: str, tool_name: str, mode: str, task_id: str) -> StaticAnalysisResult:
    """Analyze a diff for basic code metrics."""
    result = StaticAnalysisResult(tool_name=tool_name, mode=mode, task_id=task_id)

    lines_added = 0
    lines_removed = 0
    files_seen = set()

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            # Extract file path
            parts = line.split()
            if len(parts) >= 3:
                files_seen.add(parts[2].lstrip("a/"))
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    result.lines_added = lines_added
    result.lines_removed = lines_removed
    result.files_modified = len(files_seen)

    imports_added, imports_removed = parse_diff_imports(diff_text)
    result.imports_added = imports_added
    result.imports_removed = imports_removed

    # Count test additions
    result.tests_added = sum(
        1 for line in diff_text.splitlines()
        if line.startswith("+") and re.search(r"(it\(|test\(|describe\(|def test_)", line)
    )

    return result
