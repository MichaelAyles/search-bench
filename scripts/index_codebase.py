"""One-shot indexing CLI - convenience wrapper around the indexer."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mcp_server.indexer import index_codebase


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Index a codebase for RAG search")
    parser.add_argument("codebase", help="Path to codebase directory")
    parser.add_argument("--db", default="./data/circuitsnips.db")
    parser.add_argument("--faiss", default="./data/circuitsnips.faiss")
    args = parser.parse_args()

    codebase = Path(args.codebase).resolve()
    if not codebase.is_dir():
        print(f"Error: {codebase} is not a directory")
        sys.exit(1)

    db_path = Path(args.db).resolve()
    faiss_path = Path(args.faiss).resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    result = index_codebase(codebase, db_path, faiss_path)
    print(f"\nDone: {result['chunks']} chunks indexed in {result['time']:.1f}s")


if __name__ == "__main__":
    main()
