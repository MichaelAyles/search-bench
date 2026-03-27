"""Code chunking: Tree-sitter AST chunking for supported languages, sliding window fallback."""

from dataclasses import dataclass
from pathlib import Path

LANG_EXTENSIONS = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
}

SLIDING_WINDOW_EXTENSIONS = {
    ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".cfg", ".ini",
    ".css", ".scss", ".html", ".xml", ".svg", ".env", ".sh", ".bash",
}

SKIP_DIRS = {
    "node_modules", ".next", "dist", ".git", "__pycache__", ".venv",
    "venv", ".tox", "build", ".eggs", "*.egg-info",
}

SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}

MAX_FILE_SIZE = 512 * 1024  # 512KB


@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str
    symbol_name: str | None
    language: str
    content: str


def should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    if path.name in SKIP_FILES:
        return True
    return False


def get_language(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in LANG_EXTENSIONS:
        return LANG_EXTENSIONS[ext]
    if ext in SLIDING_WINDOW_EXTENSIONS:
        return ext.lstrip(".")
    return None


def chunk_sliding_window(
    content: str,
    file_path: str,
    language: str,
    window_lines: int = 40,
    overlap_lines: int = 10,
) -> list[Chunk]:
    """Sliding window chunking with line-based windows."""
    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    # Small files: single chunk
    if len(lines) <= window_lines:
        return [Chunk(
            file_path=file_path,
            start_line=1,
            end_line=len(lines),
            chunk_type="block",
            symbol_name=None,
            language=language,
            content=content,
        )]

    chunks = []
    start = 0
    while start < len(lines):
        end = min(start + window_lines, len(lines))
        chunk_content = "".join(lines[start:end])
        chunks.append(Chunk(
            file_path=file_path,
            start_line=start + 1,
            end_line=end,
            chunk_type="block",
            symbol_name=None,
            language=language,
            content=chunk_content,
        ))
        if end >= len(lines):
            break
        start += window_lines - overlap_lines

    return chunks


def _try_treesitter_chunk(content: str, file_path: str, language: str) -> list[Chunk] | None:
    """Try to chunk using tree-sitter. Returns None if tree-sitter isn't available."""
    try:
        import tree_sitter_python
        import tree_sitter_javascript
        import tree_sitter_typescript
        import tree_sitter as ts
    except ImportError:
        return None

    lang_map = {
        "python": tree_sitter_python.language,
        "javascript": tree_sitter_javascript.language,
        "typescript": tree_sitter_typescript.language_typescript,
    }

    lang_fn = lang_map.get(language)
    if lang_fn is None:
        return None

    parser = ts.Parser(ts.Language(lang_fn()))
    tree = parser.parse(content.encode("utf-8"))

    # Node types that represent meaningful chunk boundaries
    if language == "python":
        target_types = {"function_definition", "class_definition", "decorated_definition"}
    else:
        target_types = {
            "function_declaration", "class_declaration", "method_definition",
            "arrow_function", "export_statement", "lexical_declaration",
        }

    chunks = []
    lines = content.splitlines(keepends=True)

    def extract_symbol(node) -> str | None:
        """Extract the symbol name from a function/class node."""
        for child in node.children:
            if child.type in ("identifier", "property_identifier"):
                return child.text.decode("utf-8")
            if child.type == "name":
                return child.text.decode("utf-8")
        # For decorated definitions, look inside the wrapped definition
        if node.type == "decorated_definition":
            for child in node.children:
                if child.type in target_types:
                    return extract_symbol(child)
        return None

    def get_chunk_type(node) -> str:
        if "class" in node.type:
            return "class"
        if "function" in node.type or "method" in node.type or "arrow" in node.type:
            return "function"
        if node.type == "decorated_definition":
            for child in node.children:
                return get_chunk_type(child)
        return "block"

    visited_ranges = set()

    def visit(node, depth=0):
        if node.type in target_types:
            start = node.start_point[0]
            end = node.end_point[0] + 1
            range_key = (start, end)
            if range_key not in visited_ranges:
                visited_ranges.add(range_key)
                chunk_content = "".join(lines[start:end])
                if chunk_content.strip():
                    chunks.append(Chunk(
                        file_path=file_path,
                        start_line=start + 1,
                        end_line=end,
                        chunk_type=get_chunk_type(node),
                        symbol_name=extract_symbol(node),
                        language=language,
                        content=chunk_content,
                    ))
            return  # Don't recurse into children of matched nodes

        for child in node.children:
            visit(child, depth + 1)

    visit(tree.root_node)

    if not chunks:
        return None  # Fall back to sliding window

    # Also capture any top-level code not covered by function/class chunks
    covered = set()
    for c in chunks:
        for i in range(c.start_line - 1, c.end_line):
            covered.add(i)

    uncovered_start = None
    for i in range(len(lines)):
        if i not in covered:
            if uncovered_start is None:
                uncovered_start = i
        else:
            if uncovered_start is not None:
                block = "".join(lines[uncovered_start:i])
                if block.strip() and len(block.strip()) > 20:
                    chunks.append(Chunk(
                        file_path=file_path,
                        start_line=uncovered_start + 1,
                        end_line=i,
                        chunk_type="block",
                        symbol_name=None,
                        language=language,
                        content=block,
                    ))
                uncovered_start = None

    if uncovered_start is not None:
        block = "".join(lines[uncovered_start:])
        if block.strip() and len(block.strip()) > 20:
            chunks.append(Chunk(
                file_path=file_path,
                start_line=uncovered_start + 1,
                end_line=len(lines),
                chunk_type="block",
                symbol_name=None,
                language=language,
                content=block,
            ))

    chunks.sort(key=lambda c: c.start_line)
    return chunks


def chunk_file(file_path: Path, base_dir: Path | None = None) -> list[Chunk]:
    """Chunk a single file using the best available strategy."""
    if file_path.stat().st_size > MAX_FILE_SIZE:
        return []

    language = get_language(file_path)
    if language is None:
        return []

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeDecodeError):
        return []

    if not content.strip():
        return []

    rel_path = str(file_path.relative_to(base_dir)) if base_dir else str(file_path)
    # Normalize to forward slashes
    rel_path = rel_path.replace("\\", "/")

    # Try tree-sitter for supported languages
    if language in LANG_EXTENSIONS.values():
        ts_chunks = _try_treesitter_chunk(content, rel_path, language)
        if ts_chunks:
            return ts_chunks

    # Fallback to sliding window
    return chunk_sliding_window(content, rel_path, language)


def walk_and_chunk(codebase_dir: Path) -> list[Chunk]:
    """Walk a codebase directory and chunk all supported files."""
    all_chunks = []
    for path in sorted(codebase_dir.rglob("*")):
        if not path.is_file():
            continue
        if should_skip(path):
            continue
        chunks = chunk_file(path, base_dir=codebase_dir)
        all_chunks.extend(chunks)
    return all_chunks
