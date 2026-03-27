"""Hybrid search: FAISS semantic + SQLite FTS5 keyword search."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .store import ChunkRecord, Store


@dataclass
class SearchResult:
    chunk: ChunkRecord
    score: float
    source: str  # "semantic", "keyword", or "hybrid"


class HybridSearch:
    def __init__(self, store: Store, faiss_path: str | Path | None = None):
        self.store = store
        self.faiss_index = None
        self.model = None
        self._faiss_path = Path(faiss_path) if faiss_path else None

        if self._faiss_path and self._faiss_path.exists():
            self._load_faiss()

    def _load_faiss(self):
        import faiss
        self.faiss_index = faiss.read_index(str(self._faiss_path))

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self.model

    def _embed(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def build_index(self, chunks: list[ChunkRecord], save_path: str | Path | None = None):
        """Build FAISS index from chunk records."""
        import faiss

        texts = [c.content for c in chunks]
        embeddings = self._embed(texts)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine
        index.add(embeddings.astype(np.float32))

        self.faiss_index = index
        if save_path:
            faiss.write_index(index, str(save_path))
            self._faiss_path = Path(save_path)

    def semantic_search(
        self, query: str, top_k: int = 10, file_filter: str | None = None
    ) -> list[SearchResult]:
        """Search by embedding similarity."""
        if self.faiss_index is None:
            return []

        query_vec = self._embed([query])
        # Fetch more results if we need to filter
        fetch_k = top_k * 3 if file_filter else top_k
        scores, indices = self.faiss_index.search(query_vec.astype(np.float32), fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.store.get_chunk(int(idx) + 1)  # SQLite IDs are 1-based
            if chunk is None:
                continue
            if file_filter and not _glob_match(chunk.file_path, file_filter):
                continue
            results.append(SearchResult(chunk=chunk, score=float(score), source="semantic"))
            if len(results) >= top_k:
                break

        return results

    def keyword_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Search by FTS5 keyword matching."""
        chunks = self.store.keyword_search(query, limit=limit)
        return [
            SearchResult(chunk=c, score=1.0 / (i + 1), source="keyword")
            for i, c in enumerate(chunks)
        ]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        file_filter: str | None = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[SearchResult]:
        """Combined semantic + keyword search with score fusion."""
        sem_results = self.semantic_search(query, top_k=top_k * 2, file_filter=file_filter)
        kw_results = self.keyword_search(query, limit=top_k * 2)

        # Reciprocal rank fusion
        chunk_scores: dict[int, float] = {}
        chunk_map: dict[int, ChunkRecord] = {}

        for rank, r in enumerate(sem_results):
            cid = r.chunk.id
            chunk_scores[cid] = chunk_scores.get(cid, 0) + semantic_weight / (rank + 1)
            chunk_map[cid] = r.chunk

        for rank, r in enumerate(kw_results):
            cid = r.chunk.id
            chunk_scores[cid] = chunk_scores.get(cid, 0) + keyword_weight / (rank + 1)
            chunk_map[cid] = r.chunk

        sorted_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:top_k]
        return [
            SearchResult(chunk=chunk_map[cid], score=chunk_scores[cid], source="hybrid")
            for cid in sorted_ids
        ]

    def symbol_lookup(self, symbol: str, limit: int = 10) -> list[SearchResult]:
        """Find chunks by symbol name."""
        chunks = self.store.symbol_search(symbol, limit=limit)
        return [
            SearchResult(chunk=c, score=1.0 / (i + 1), source="keyword")
            for i, c in enumerate(chunks)
        ]

    def related_code(
        self,
        file_path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Find code related to a given file/region by embedding similarity."""
        # Find the chunk(s) covering the specified region
        if start_line and end_line:
            # Build a query from the content of that region
            chunks = self.store.keyword_search(file_path, limit=50)
            matching = [
                c for c in chunks
                if c.file_path == file_path
                and c.start_line <= end_line
                and c.end_line >= start_line
            ]
        else:
            matching = [
                c for c in self.store.keyword_search(file_path, limit=50)
                if c.file_path == file_path
            ]

        if not matching:
            # Fall back to semantic search with the file path
            return self.semantic_search(file_path, top_k=top_k)

        # Use the content of matched chunks as the query
        query_text = "\n".join(c.content for c in matching[:3])
        results = self.semantic_search(query_text, top_k=top_k + len(matching))

        # Filter out the source chunks themselves
        source_ids = {c.id for c in matching}
        return [r for r in results if r.chunk.id not in source_ids][:top_k]


def _glob_match(path: str, pattern: str) -> bool:
    """Simple glob matching for file filters."""
    from fnmatch import fnmatch
    return fnmatch(path, pattern)
