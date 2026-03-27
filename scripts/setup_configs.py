"""Generate MCP configs for each CLI tool."""

import json
from pathlib import Path


def generate_configs(db_path: str = "./data/circuitsnips.db", faiss_path: str = "./data/circuitsnips.faiss"):
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Claude Code (.mcp.json)
    claude_config = {
        "mcpServers": {
            "codebase-rag": {
                "command": "python",
                "args": ["-m", "src.mcp_server.server", "--db", db_path, "--faiss", faiss_path]
            }
        }
    }
    (configs_dir / "claude-mcp.json").write_text(json.dumps(claude_config, indent=2))

    # Codex CLI (config.toml)
    codex_config = f"""[[mcp]]
name = "codebase-rag"
command = "python"
args = ["-m", "src.mcp_server.server", "--db", "{db_path}", "--faiss", "{faiss_path}"]
transport = "stdio"
"""
    (configs_dir / "codex-config.toml").write_text(codex_config)

    # Gemini CLI (settings.json)
    gemini_config = {
        "mcpServers": {
            "codebase-rag": {
                "command": "python",
                "args": ["-m", "src.mcp_server.server", "--db", db_path, "--faiss", faiss_path]
            }
        }
    }
    (configs_dir / "gemini-settings.json").write_text(json.dumps(gemini_config, indent=2))

    # GitHub Copilot (VS Code settings)
    copilot_config = {
        "github.copilot.chat.mcpServers": {
            "codebase-rag": {
                "command": "python",
                "args": ["-m", "src.mcp_server.server", "--db", db_path, "--faiss", faiss_path]
            }
        }
    }
    (configs_dir / "copilot-settings.json").write_text(json.dumps(copilot_config, indent=2))

    print("Generated configs in ./configs/")
    for f in configs_dir.iterdir():
        if f.suffix in (".json", ".toml"):
            print(f"  {f}")


if __name__ == "__main__":
    generate_configs()
