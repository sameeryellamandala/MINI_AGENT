"""SQLite persistence for LangGraph checkpoints and the chat picker."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


def open_checkpointer(database_path: str | Path) -> tuple[SqliteSaver, sqlite3.Connection]:
    """Return a persistent LangGraph checkpointer and its owned connection."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(connection), connection


class ThreadRegistry:
    """Tiny thread index; conversation contents remain in LangGraph state."""

    def __init__(self, database_path: str | Path):
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_threads (
                    thread_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def touch(self, *, thread_id: str, title: str, timestamp: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO app_threads(thread_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(thread_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    title = CASE
                        WHEN app_threads.title = 'New conversation' THEN excluded.title
                        ELSE app_threads.title
                    END
                """,
                (thread_id, title, timestamp, timestamp),
            )

    def list_threads(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT thread_id, title, created_at, updated_at
                FROM app_threads
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

