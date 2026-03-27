"""MCP server exposing RAG search tools over stdio transport."""

import argparse
import json
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .search import HybridSearch
from .store import Store


def create_server(db_path: str, faiss_path: str) -> Server:
    store = Store(db_path)
    search = HybridSearch(store, faiss_path=faiss_path)

    server = Server("codebase-rag")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="semantic_search",
                description=(
                    "Search the codebase by meaning. Use for conceptual queries like "
                    "'how does authentication work' or 'where is error handling done'. "
                    "Returns ranked code snippets with file paths and line numbers."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of what you're looking for",
                        },
                        "top_k": {
                            "type": "integer",
                            "default": 10,
                            "description": "Number of results to return",
                        },
                        "file_filter": {
                            "type": "string",
                            "description": "Optional glob pattern to filter files, e.g. '*.ts' or 'src/api/**'",
                        },
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="symbol_lookup",
                description=(
                    "Find a specific function, class, or variable by name. "
                    "Use for exact lookups like 'find the handleUpload function'."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "The symbol name to find",
                        },
                    },
                    "required": ["symbol"],
                },
            ),
            Tool(
                name="related_code",
                description=(
                    "Given a file path and optional line range, find semantically "
                    "related code elsewhere in the codebase."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "start_line": {"type": "integer"},
                        "end_line": {"type": "integer"},
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": ["file_path"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "semantic_search":
            results = search.hybrid_search(
                query=arguments["query"],
                top_k=arguments.get("top_k", 10),
                file_filter=arguments.get("file_filter"),
            )
            return [TextContent(
                type="text",
                text=_format_results(results),
            )]

        elif name == "symbol_lookup":
            results = search.symbol_lookup(
                symbol=arguments["symbol"],
                limit=10,
            )
            return [TextContent(
                type="text",
                text=_format_results(results),
            )]

        elif name == "related_code":
            results = search.related_code(
                file_path=arguments["file_path"],
                start_line=arguments.get("start_line"),
                end_line=arguments.get("end_line"),
                top_k=arguments.get("top_k", 5),
            )
            return [TextContent(
                type="text",
                text=_format_results(results),
            )]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


def _format_results(results) -> str:
    if not results:
        return "No results found."

    parts = []
    for i, r in enumerate(results, 1):
        c = r.chunk
        header = f"[{i}] {c.file_path}:{c.start_line}-{c.end_line}"
        if c.symbol_name:
            header += f" ({c.chunk_type}: {c.symbol_name})"
        header += f"  [score: {r.score:.3f}, source: {r.source}]"

        parts.append(f"{header}\n```{c.language}\n{c.content}\n```")

    return "\n\n".join(parts)


async def run_server(db_path: str, faiss_path: str):
    server = create_server(db_path, faiss_path)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser(description="RAG MCP Server")
    parser.add_argument("--db", required=True, help="SQLite database path")
    parser.add_argument("--faiss", required=True, help="FAISS index path")
    parser.add_argument("--index-only", action="store_true", help="Only index, don't start server")
    parser.add_argument("--codebase", help="Codebase path (for --index-only)")
    args = parser.parse_args()

    if args.index_only:
        if not args.codebase:
            print("--codebase required with --index-only", file=sys.stderr)
            sys.exit(1)
        from .indexer import index_codebase
        index_codebase(
            Path(args.codebase).resolve(),
            Path(args.db).resolve(),
            Path(args.faiss).resolve(),
        )
    else:
        import asyncio
        asyncio.run(run_server(args.db, args.faiss))


if __name__ == "__main__":
    main()
