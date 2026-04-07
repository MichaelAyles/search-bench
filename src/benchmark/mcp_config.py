"""MCP configuration manager for per-tool RAG mode setup."""

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/benchmark/mcp_config.py -> root


class MCPConfigManager:
    """Writes and cleans up per-tool MCP config files for RAG mode."""

    def __init__(self, codebase_dir: Path, db_path: Path, faiss_path: Path):
        self.codebase_dir = codebase_dir
        self.db_path = db_path.resolve()
        self.faiss_path = faiss_path.resolve()
        self._backups: dict[str, tuple[Path, bytes | None]] = {}
        # Shared log file for all MCP server instances in this run
        self._log_file = tempfile.NamedTemporaryFile(
            prefix="mcp_log_", suffix=".jsonl", delete=False, mode="w",
        )
        self.log_path = Path(self._log_file.name)
        self._log_file.close()  # server processes will append to it

    def _server_entry(self) -> dict:
        # Prefer the venv python (has mcp package installed) over sys.executable
        venv_python = PROJECT_ROOT / ".venv" / "bin" / "python3"
        python_cmd = str(venv_python) if venv_python.exists() else sys.executable
        return {
            "command": python_cmd,
            "args": [
                "-m", "src.mcp_server.server",
                "--db", str(self.db_path),
                "--faiss", str(self.faiss_path),
                "--log", str(self.log_path),
            ],
            "env": {"PYTHONPATH": str(PROJECT_ROOT)},
        }

    def read_log(self) -> list[dict]:
        """Read and return all logged MCP calls, then clear the log."""
        entries = []
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def clear_log(self) -> None:
        """Clear the log file for the next query."""
        if self.log_path.exists():
            self.log_path.write_text("")

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
