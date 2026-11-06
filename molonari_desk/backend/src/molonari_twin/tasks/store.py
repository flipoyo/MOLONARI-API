"""Magasin SQLite des tâches, partagé par le runner et ses workers (un processus chacun).

Une connexion par opération, en mode WAL : lectures et écritures concurrentes sans verrou applicatif. Les
changements d'état sont conditionnels (`expect`) : le premier qui conclut une tâche l'emporte, et un worker
qui finit pendant qu'on le termine ne remplace pas « cancelled » par « succeeded ».
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from contextlib import closing
from pathlib import Path

from .model import TERMINAL, TaskEvent, TaskRecord, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    state TEXT NOT NULL,
    params TEXT NOT NULL,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    duration_s REAL,
    timeout_s REAL,
    pid INTEGER,
    backend_version TEXT,
    platform TEXT,
    error TEXT,
    result TEXT,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS tasks_key_state ON tasks (key, state);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_task ON events (task_id, id);
"""

FIELDS = ("id", "kind", "key", "state", "params", "created_at", "started_at", "finished_at", "duration_s",
          "timeout_s", "pid", "backend_version", "platform", "error", "result")
JSON_FIELDS = ("params", "error", "result")


class TaskStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        return db

    # Lecture ---------------------------------------------------------------------------------------

    def get(self, task_id: str) -> TaskRecord | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return _record(row) if row else None

    def with_state(self, state: str) -> list[TaskRecord]:
        """Tâches dans cet état, dans l'ordre de soumission."""
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM tasks WHERE state = ? ORDER BY seq", (state,)).fetchall()
        return [_record(row) for row in rows]

    def events(self, task_id: str, after: int | None = None) -> list[TaskEvent]:
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM events WHERE task_id = ? AND id > ? ORDER BY id",
                              (task_id, -1 if after is None else after)).fetchall()
        return [TaskEvent(row["id"], row["task_id"], row["type"], json.loads(row["data"]), row["at"]) for row in rows]

    def cancel_requested(self, task_id: str) -> bool:
        with closing(self._connect()) as db:
            row = db.execute("SELECT cancel_requested FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return bool(row and row[0])

    # Écriture --------------------------------------------------------------------------------------

    def submit(self, record: TaskRecord) -> TaskRecord:
        """Enregistre la tâche, sauf si une tâche de même clé est en cours : celle-ci est alors renvoyée,
        marquée `reused`. La vérification et l'insertion forment une seule transaction."""
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute("SELECT * FROM tasks WHERE key = ? AND state IN ('queued', 'running') ORDER BY seq LIMIT 1",
                                 (record.key,)).fetchone()
                if row is not None:
                    db.execute("COMMIT")
                    existing = _record(row)
                    existing.reused = True
                    return existing
                values = [_column(name, getattr(record, name)) for name in FIELDS]
                db.execute(f"INSERT INTO tasks ({', '.join(FIELDS)}) VALUES ({', '.join('?' * len(FIELDS))})", values)
                _insert_event(db, record.id, "state", {"state": record.state})
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return record

    def set_state(self, task_id: str, state: str, expect: Iterable[str], **fields) -> bool:
        """Passe la tâche à `state` si son état courant est dans `expect`, avec les autres champs donnés, et
        ajoute l'événement « state ». Un état terminal fixe `finished_at`. Renvoie False si l'état a changé
        entre-temps (rien n'est écrit)."""
        expect = tuple(expect)
        if state in TERMINAL:
            fields.setdefault("finished_at", now())
        columns = {"state": state, **fields}
        assignments = ", ".join(f"{name} = ?" for name in columns)
        values = [_column(name, value) for name, value in columns.items()]
        with closing(self._connect()) as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                cursor = db.execute(
                    f"UPDATE tasks SET {assignments} WHERE id = ? AND state IN ({', '.join('?' * len(expect))})",
                    [*values, task_id, *expect])
                changed = cursor.rowcount == 1
                if changed:
                    event = {"state": state}
                    if fields.get("error") is not None:
                        event["error"] = fields["error"]
                    _insert_event(db, task_id, "state", event)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        return changed

    def update(self, task_id: str, **fields) -> None:
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with closing(self._connect()) as db:
            db.execute(f"UPDATE tasks SET {assignments} WHERE id = ?",
                       [*(_column(name, value) for name, value in fields.items()), task_id])

    def request_cancel(self, task_id: str) -> None:
        with closing(self._connect()) as db:
            db.execute("UPDATE tasks SET cancel_requested = 1 WHERE id = ?", (task_id,))

    def add_event(self, task_id: str, type_: str, data: dict) -> int:
        with closing(self._connect()) as db:
            return _insert_event(db, task_id, type_, data)


def _column(name: str, value):
    return json.dumps(value, ensure_ascii=False) if name in JSON_FIELDS and value is not None else value


def _record(row: sqlite3.Row) -> TaskRecord:
    values = {name: row[name] for name in FIELDS}
    for name in JSON_FIELDS:
        if values[name] is not None:
            values[name] = json.loads(values[name])
    return TaskRecord(**values)


def _insert_event(db: sqlite3.Connection, task_id: str, type_: str, data: dict) -> int:
    cursor = db.execute("INSERT INTO events (task_id, type, data, at) VALUES (?, ?, ?, ?)",
                        (task_id, type_, json.dumps(data, ensure_ascii=False), now()))
    return cursor.lastrowid
