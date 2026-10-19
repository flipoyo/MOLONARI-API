"""Exécution des calculs longs hors du thread de l'interface.

Une tâche (`Task`) exécute une fonction dans le QThreadPool et communique avec
l'interface uniquement par signaux Qt (thread-safe) :
- `progress(object)` : rapport de progression ;
- `succeeded(object)` : résultat ;
- `failed(object)` : exception (affichée par l'interface, l'application continue) ;
- `cancelled()` : l'utilisateur a annulé.

La fonction reçoit un `TaskContext` : `context.report(p)` publie une progression,
`context.cancelled()` indique si l'annulation a été demandée. Les calculs JAX
libèrent le GIL : l'interface reste réactive pendant le calcul.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from PySide6 import QtCore

from ..engine.jax.dream import Cancelled
from ..logs import log


class TaskSignals(QtCore.QObject):
    progress = QtCore.Signal(object)
    succeeded = QtCore.Signal(object)
    failed = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    finished = QtCore.Signal()  # toujours émis en dernier


class TaskContext:
    def __init__(self, signals: TaskSignals, cancel_event: threading.Event):
        self._signals = signals
        self._cancel = cancel_event

    def report(self, progress) -> None:
        self._signals.progress.emit(progress)

    def cancelled(self) -> bool:
        return self._cancel.is_set()


class Task(QtCore.QRunnable):
    def __init__(self, name: str, function: Callable[[TaskContext], object]):
        super().__init__()
        self.setAutoDelete(False)
        self.name = name
        self.signals = TaskSignals()
        self._function = function
        self._cancel = threading.Event()
        self.started_at: float | None = None
        self.duration: float = 0.0

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:  # exécuté dans un thread du pool
        self.started_at = time.perf_counter()
        try:
            result = self._function(TaskContext(self.signals, self._cancel))
        except Cancelled:
            self.duration = time.perf_counter() - self.started_at
            log.info("Task cancelled: %s", self.name)
            self.signals.cancelled.emit()
        except BaseException as exc:  # noqa: BLE001 — transmis à l'interface
            self.duration = time.perf_counter() - self.started_at
            self.signals.failed.emit(exc)
        else:
            self.duration = time.perf_counter() - self.started_at
            if self._cancel.is_set():
                self.signals.cancelled.emit()
            else:
                self.signals.succeeded.emit(result)
        finally:
            self.signals.finished.emit()


class TaskManager(QtCore.QObject):
    """Une seule tâche de calcul à la fois (un calcul JAX occupe déjà tous les cœurs)."""

    task_started = QtCore.Signal(object)
    task_finished = QtCore.Signal(object)

    def __init__(self, parent: QtCore.QObject | None = None):
        super().__init__(parent)
        self._pool = QtCore.QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self.current: Task | None = None

    @property
    def busy(self) -> bool:
        return self.current is not None

    def start(self, task: Task) -> None:
        if self.current is not None:
            raise RuntimeError("A task is already running")
        self.current = task
        task.signals.finished.connect(lambda: self._on_finished(task), QtCore.Qt.ConnectionType.QueuedConnection)
        log.info("Task started: %s", task.name)
        self.task_started.emit(task)
        self._pool.start(task)

    def _on_finished(self, task: Task) -> None:
        if self.current is task:
            self.current = None
        self.task_finished.emit(task)

    def cancel(self) -> None:
        if self.current is not None:
            self.current.cancel()

    def wait(self, msecs: int = -1) -> bool:
        return self._pool.waitForDone(msecs)
