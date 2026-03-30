"""SQLite metadata store with FTS5 keyword search."""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    id: int
    file_path: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, method, block, config
    symbol_name: str | None
    language: str
    content: str


class Store:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_type TEXT NOT NULL,
                symbol_name TEXT,
                language TEXT NOT NULL,
                content TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);
            CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON chunks(symbol_name);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                symbol_name,
                file_path,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, content, symbol_name, file_path)
                VALUES (new.id, new.content, new.symbol_name, new.file_path);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, content, symbol_name, file_path)
                VALUES ('delete', old.id, old.content, old.symbol_name, old.file_path);
            END;

            CREATE TABLE IF NOT EXISTS index_meta (
                key TEXT PRIMARY KEY,
                value TEXT
            );
        """)
        self.conn.commit()

    def clear(self):
        self.conn.executescript("""
            DELETE FROM chunks;
            DELETE FROM chunks_fts;
            DELETE FROM index_meta;
        """)
        self.conn.commit()

    def insert_chunk(
        self,
        file_path: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        symbol_name: str | None,
        language: str,
        content: str,
    ) -> int:
        cur = self.conn.execute(
            """INSERT INTO chunks (file_path, start_line, end_line, chunk_type, symbol_name, language, content)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_path, start_line, end_line, chunk_type, symbol_name, language, content),
        )
        self.conn.commit()
        return cur.lastrowid

    def insert_chunks_batch(self, chunks: list[tuple]) -> list[int]:
        """Insert multiple chunks atomically.

        Each tuple: (file_path, start_line, end_line, chunk_type, symbol_name, language, content).
        Returns a list of the actual SQLite rowids assigned to each chunk, in order.
        Uses an explicit transaction so a crash mid-batch cannot create partial data
        that would misalign with a FAISS index built afterwards.
        """
        ids: list[int] = []
        try:
            self.conn.execute("BEGIN")
            for chunk in chunks:
                cur = self.conn.execute(
                    """INSERT INTO chunks (file_path, start_line, end_line, chunk_type, symbol_name, language, content)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    chunk,
                )
                ids.append(cur.lastrowid)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return ids

    def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        row = self.conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if row is None:
            return None
        return ChunkRecord(**dict(row))

    def get_chunks_by_ids(self, ids: list[int]) -> list[ChunkRecord]:
        if not ids:
            return []
        placeholders = ",".join("?" * len(ids))
        rows = self.conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})", ids
        ).fetchall()
        return [ChunkRecord(**dict(r)) for r in rows]

    def keyword_search(self, query: str, limit: int = 20) -> list[ChunkRecord]:
        # Sanitize query for FTS5: remove special chars, quote each term
        import re
        terms = re.findall(r'\w+', query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms)
        try:
            rows = self.conn.execute(
                """SELECT chunks.* FROM chunks_fts
                   JOIN chunks ON chunks_fts.rowid = chunks.id
                   WHERE chunks_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (fts_query, limit),
            ).fetchall()
            return [ChunkRecord(**dict(r)) for r in rows]
        except Exception:
            logger.warning("keyword_search failed for query %r: ", query, exc_info=True)
            return []

    def symbol_search(self, symbol: str, limit: int = 10) -> list[ChunkRecord]:
        # Try exact match first
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE symbol_name = ? LIMIT ?",
            (symbol, limit),
        ).fetchall()
        if rows:
            return [ChunkRecord(**dict(r)) for r in rows]

        # Fall back to LIKE match
        rows = self.conn.execute(
            "SELECT * FROM chunks WHERE symbol_name LIKE ? LIMIT ?",
            (f"%{symbol}%", limit),
        ).fetchall()
        if rows:
            return [ChunkRecord(**dict(r)) for r in rows]

        # Fall back to FTS
        return self.keyword_search(symbol, limit)

    def get_chunk_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0]

    def set_meta(self, key: str, value: str):
        self.conn.execute(
            "INSERT OR REPLACE INTO index_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM index_meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def close(self):
        self.conn.close()
