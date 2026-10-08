"""Tiny async-friendly SQLite wrapper (stdlib sqlite3 in a worker thread).

Holds long-term memory, feedback and the audit log. SQLite keeps the setup zero-dependency; the
interface (execute / fetchall) is what a Postgres implementation would replace.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, kind TEXT NOT NULL,
  content TEXT NOT NULL, keywords TEXT NOT NULL, access_level TEXT NOT NULL DEFAULT 'internal',
  session_id TEXT, message_id TEXT, weight REAL NOT NULL DEFAULT 1.0, created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, user_id TEXT NOT NULL, session_id TEXT,
  message_id TEXT, run_id TEXT, rating INTEGER NOT NULL, comment TEXT, question TEXT);
CREATE TABLE IF NOT EXISTS audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, user_id TEXT, role TEXT, action TEXT NOT NULL,
  resource TEXT, outcome TEXT NOT NULL, detail TEXT);
"""


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA)
        return self._conn

    def _run(self, sql: str, params: tuple, fetch: bool):
        with self._lock:
            conn = self._connect()
            cur = conn.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()] if fetch else cur.lastrowid
            conn.commit()
            return rows

    async def execute(self, sql: str, params: tuple = ()) -> int:
        return await asyncio.to_thread(self._run, sql, params, False)

    async def fetchall(self, sql: str, params: tuple = ()) -> list[dict]:
        return await asyncio.to_thread(self._run, sql, params, True)
