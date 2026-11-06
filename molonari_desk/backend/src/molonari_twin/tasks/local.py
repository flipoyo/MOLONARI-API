"""LocalTaskRunner : les tâches tournent sur le poste, un sous-processus par tâche (phases 3–4).

Un thread de distribution démarre les tâches « queued » dans l'ordre de soumission, dans la limite de
`max_workers`, et surveille celles qui tournent :
- un worker terminé sans conclure sa tâche la laisse « failed » (E_TASK_INTERRUPTED) ;
- au-delà de `timeout_s`, le worker est terminé et la tâche « failed » (E_TASK_TIMEOUT) ;
- une annulation que le job n'a pas honorée après `cancel_grace_s` termine le worker (« cancelled »).
Toute tâche qui ne réussit pas est nettoyée (fichiers partiels) avant de passer à son état terminal.

Les workers démarrent en mode « spawn » : un processus neuf, qui n'hérite ni des threads du serveur ni de
son état, et qu'on peut terminer sans arrêter l'API.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..errors import UserError
from . import worker
from .jobs import cleanup, get_job
from .model import TaskEvent, TaskRecord, error_dict, now, platform_info
from .runner import default_environment, task_key
from .store import TaskStore

log = logging.getLogger(__name__)


def default_db_path() -> Path:
    """Base des tâches par défaut : $XDG_CACHE_HOME/molonari-twin/tasks.sqlite (~/.cache sinon)."""
    cache = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(cache) / "molonari-twin" / "tasks.sqlite"


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class _Running:
    process: multiprocessing.process.BaseProcess
    started: float  # time.monotonic()
    timeout_s: float | None
    cancel_seen: float | None = None  # première fois que le distributeur a vu l'annulation demandée


class LocalTaskRunner:
    name = "local"

    def __init__(self, db_path: str | Path, max_workers: int = 1, cancel_grace_s: float = 10.0,
                 default_timeout_s: float | None = None, poll_s: float = 0.05, environment: dict | None = None):
        self.store = TaskStore(db_path)
        self.max_workers = max_workers
        self.cancel_grace_s = cancel_grace_s
        self.default_timeout_s = default_timeout_s
        self.poll_s = poll_s
        self.environment = default_environment() if environment is None else environment
        self._context = multiprocessing.get_context("spawn")
        self._running: dict[str, _Running] = {}
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._recover()
        self._thread = threading.Thread(target=self._loop, name="molonari-tasks", daemon=True)
        self._thread.start()

    # Protocole TaskRunner --------------------------------------------------------------------------

    def submit(self, kind: str, params: dict, key: str | None = None, timeout_s: float | None = None) -> TaskRecord:
        get_job(kind)
        record = TaskRecord(
            id=uuid.uuid4().hex, kind=kind, key=key or task_key(kind, params, environment=self.environment),
            state="queued", params=params, created_at=now(),
            timeout_s=self.default_timeout_s if timeout_s is None else timeout_s)
        record = self.store.submit(record)
        self._wake.set()
        return record

    def get(self, task_id: str) -> TaskRecord:
        record = self.store.get(task_id)
        if record is None:
            raise UserError("E_TASK_NOT_FOUND", task_id=task_id)
        return record

    def events(self, task_id: str, after: int | None = None) -> list[TaskEvent]:
        self.get(task_id)
        return self.store.events(task_id, after)

    def cancel(self, task_id: str) -> TaskRecord:
        record = self.get(task_id)
        if record.finished:
            raise UserError("E_TASK_FINISHED", task_id=task_id, state=record.state)
        self.store.request_cancel(task_id)
        # Une tâche encore en file n'a pas de worker : elle est annulée tout de suite.
        if not self.store.set_state(task_id, "cancelled", ["queued"]):
            self._wake.set()
        return self.get(task_id)

    def result(self, task_id: str) -> TaskRecord:
        record = self.get(task_id)
        if not record.finished:
            raise UserError("E_TASK_NOT_FINISHED", task_id=task_id, state=record.state)
        return record

    def wait(self, task_id: str, timeout_s: float | None = None) -> TaskRecord:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            record = self.get(task_id)
            if record.finished or (deadline is not None and time.monotonic() >= deadline):
                return record
            time.sleep(self.poll_s)

    def close(self) -> None:
        """Arrête le distributeur et termine les workers ; leurs tâches passent « failed » (E_TASK_INTERRUPTED)."""
        self._stop.set()
        self._wake.set()
        self._thread.join()
        for task_id, running in list(self._running.items()):
            _terminate(running.process)
            self._conclude(task_id, "failed", error=error_dict(UserError("E_TASK_INTERRUPTED")))
        self._running.clear()

    # Distribution et surveillance ------------------------------------------------------------------

    def _recover(self) -> None:
        """Une tâche « running » sans worker vivant (runner arrêté brutalement) passe « failed »."""
        for record in self.store.with_state("running"):
            if not pid_alive(record.pid):
                self._conclude(record.id, "failed", error=error_dict(UserError("E_TASK_INTERRUPTED")))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._supervise()
                self._start_queued()
            except Exception:  # le distributeur ne doit jamais s'arrêter sur une erreur
                log.exception("Task dispatcher error")
            self._wake.wait(self.poll_s)
            self._wake.clear()

    def _start_queued(self) -> None:
        for record in self.store.with_state("queued"):
            if len(self._running) >= self.max_workers:
                return
            if not self.store.set_state(record.id, "running", ["queued"], started_at=now(), platform=platform_info(),
                                        backend_version=self.environment.get("backend")):
                continue  # annulée entre-temps
            process = self._context.Process(target=worker.main, args=(str(self.store.path), record.id),
                                            name=f"molonari-task-{record.id[:8]}", daemon=True)
            process.start()
            self.store.update(record.id, pid=process.pid)
            self._running[record.id] = _Running(process, time.monotonic(), record.timeout_s)

    def _supervise(self) -> None:
        for task_id, running in list(self._running.items()):
            process = running.process
            if not process.is_alive():
                process.join()
                del self._running[task_id]
                self._conclude(task_id, "failed", error=error_dict(UserError("E_TASK_INTERRUPTED")))
                continue
            elapsed = time.monotonic() - running.started
            if running.timeout_s is not None and elapsed > running.timeout_s:
                _terminate(process)
                del self._running[task_id]
                self._conclude(task_id, "failed", error=error_dict(UserError("E_TASK_TIMEOUT", timeout_s=running.timeout_s)))
                continue
            if self.store.cancel_requested(task_id):
                running.cancel_seen = running.cancel_seen or time.monotonic()
                if time.monotonic() - running.cancel_seen > self.cancel_grace_s:
                    _terminate(process)
                    del self._running[task_id]
                    self._conclude(task_id, "cancelled")

    def _conclude(self, task_id: str, state: str, **fields) -> None:
        """Conclut une tâche que son worker n'a pas conclue (terminé de force, ou mort) : le nettoyage d'abord,
        l'état terminal ensuite. Une tâche déjà conclue par son worker y a déjà été nettoyée."""
        record = self.store.get(task_id)
        if record is None or record.state != "running":
            return
        cleanup(record.kind, record.params)
        self.store.set_state(task_id, state, ["running"], **fields)


def _terminate(process) -> None:
    process.terminate()
    process.join(5)
    if process.is_alive():
        process.kill()
        process.join()
