"""Remembers which quiz questions you got right or wrong, by page (SQLite file)."""

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass

WEAK_BELOW = 0.7  # a page with accuracy under this counts as a weak spot


@dataclass(frozen=True)
class PageStat:
    doc_id: str
    filename: str
    page: int
    attempts: int
    correct: int

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.attempts, 2)

    @property
    def weak(self) -> bool:
        return self.accuracy < WEAK_BELOW


class ProgressStore:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._path = path
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS attempts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, doc_id TEXT NOT NULL, "
                "filename TEXT NOT NULL, page INTEGER NOT NULL, correct INTEGER NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def record(self, doc_id: str, filename: str, results: list[tuple[int, bool]]) -> None:
        """results: (page, answered_correctly) pairs."""
        with closing(self._connect()) as db, db:
            db.executemany(
                "INSERT INTO attempts (doc_id, filename, page, correct) VALUES (?, ?, ?, ?)",
                [(doc_id, filename, page, int(ok)) for page, ok in results],
            )

    def page_stats(self) -> list[PageStat]:
        """Weakest pages first (lowest accuracy, then most attempts)."""
        with closing(self._connect()) as db:
            rows = db.execute(
                "SELECT doc_id, filename, page, COUNT(*), SUM(correct) FROM attempts "
                "GROUP BY doc_id, page"
            ).fetchall()
        stats = [PageStat(d, f, p, n, int(c)) for d, f, p, n, c in rows]
        return sorted(stats, key=lambda s: (s.accuracy, -s.attempts, s.filename.lower(), s.page))

    def delete_document(self, doc_id: str) -> None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM attempts WHERE doc_id = ?", (doc_id,))
