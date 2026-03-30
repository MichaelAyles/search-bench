"""Indexing pipeline: walk codebase, chunk, embed, store."""

import argparse
import time
from pathlib import Path

from .chunker import walk_and_chunk
from .search import HybridSearch
from .store import Store


def index_codebase(
    codebase_dir: Path,
    db_path: Path,
    faiss_path: Path,
) -> dict:
    """Index a codebase: chunk files, build embeddings, create FAISS + SQLite indices."""
    print(f"Indexing codebase: {codebase_dir}")
    t0 = time.monotonic()

    # Chunk all files
    print("  Chunking files...")
    chunks = walk_and_chunk(codebase_dir)
    t_chunk = time.monotonic()
    print(f"  Found {len(chunks)} chunks in {t_chunk - t0:.1f}s")

    if not chunks:
        print("  No chunks found. Check codebase path and file types.")
        return {"chunks": 0, "time": 0}

    # Store chunks in SQLite
    print("  Storing chunks in SQLite...")
    store = Store(db_path)
    store.clear()

    batch = [
        (c.file_path, c.start_line, c.end_line, c.chunk_type, c.symbol_name, c.language, c.content)
        for c in chunks
    ]
    chunk_ids = store.insert_chunks_batch(batch)
    t_store = time.monotonic()
    print(f"  Stored {len(chunks)} chunks in {t_store - t_chunk:.1f}s")

    # Build FAISS index using the actual SQLite IDs (not assumed 1..N)
    print("  Building FAISS index (embedding + indexing)...")
    search = HybridSearch(store)
    all_records = store.get_chunks_by_ids(chunk_ids)
    # Ensure the records are in the same order as chunk_ids so the FAISS
    # row-index-to-SQLite-ID mapping stays aligned.
    record_by_id = {r.id: r for r in all_records}
    ordered_records = [record_by_id[cid] for cid in chunk_ids]
    search.build_index(ordered_records, save_path=faiss_path, chunk_ids=chunk_ids)
    t_faiss = time.monotonic()
    print(f"  Built FAISS index in {t_faiss - t_store:.1f}s")

    # Store metadata
    store.set_meta("codebase_dir", str(codebase_dir))
    store.set_meta("chunk_count", str(len(chunks)))
    store.set_meta("index_time", str(time.monotonic() - t0))

    total = time.monotonic() - t0
    print(f"  Done! {len(chunks)} chunks indexed in {total:.1f}s")
    print(f"  DB: {db_path} ({db_path.stat().st_size / 1024:.0f} KB)")
    print(f"  FAISS: {faiss_path} ({faiss_path.stat().st_size / 1024:.0f} KB)")

    store.close()
    return {"chunks": len(chunks), "time": total}


def main():
    parser = argparse.ArgumentParser(description="Index a codebase for RAG search")
    parser.add_argument("--codebase", required=True, help="Path to codebase directory")
    parser.add_argument("--db", default="./data/circuitsnips.db", help="SQLite database path")
    parser.add_argument("--faiss", default="./data/circuitsnips.faiss", help="FAISS index path")
    args = parser.parse_args()

    codebase_dir = Path(args.codebase).resolve()
    db_path = Path(args.db).resolve()
    faiss_path = Path(args.faiss).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    faiss_path.parent.mkdir(parents=True, exist_ok=True)

    index_codebase(codebase_dir, db_path, faiss_path)


if __name__ == "__main__":
    main()
