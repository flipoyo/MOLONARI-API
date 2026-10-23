"""Exécution d'une tâche dans le sous-processus d'un worker.

Le runner a déjà passé la tâche en « running ». Le worker exécute le job, publie sa progression dans le
magasin, lit le drapeau d'annulation à chaque `should_cancel()`, puis conclut la tâche : « succeeded »,
« cancelled » (Cancelled) ou « failed » (UserError avec son code, toute autre exception en E_SYSTEM_ERROR).
"""

from __future__ import annotations

import logging
import time

from ..errors import Cancelled, UserError
from .jobs import get_job
from .model import error_dict, jsonable
from .store import TaskStore

log = logging.getLogger(__name__)


class WorkerContext:
    def __init__(self, store: TaskStore, task_id: str):
        self.store, self.task_id = store, task_id

    def report(self, data: dict) -> None:
        self.store.add_event(self.task_id, "progress", jsonable(data))

    def log(self, message: str, **data) -> None:
        self.store.add_event(self.task_id, "log", {"message": message, **jsonable(data)})

    def should_cancel(self) -> bool:
        return self.store.cancel_requested(self.task_id)


def main(db_path: str, task_id: str) -> None:
    store = TaskStore(db_path)
    record = store.get(task_id)
    if record is None or record.state != "running":
        return
    started = time.perf_counter()
    try:
        result = get_job(record.kind).run(record.params, WorkerContext(store, task_id))
    except Cancelled:
        store.set_state(task_id, "cancelled", ["running"], duration_s=time.perf_counter() - started)
    except UserError as exc:
        store.set_state(task_id, "failed", ["running"], error=error_dict(exc), duration_s=time.perf_counter() - started)
    except Exception as exc:  # toute erreur imprévue conclut la tâche, avec son type et son texte
        log.exception("Task %s failed", task_id)
        error = UserError("E_SYSTEM_ERROR", detail=f"{type(exc).__name__}: {exc}")
        store.set_state(task_id, "failed", ["running"], error=error_dict(error), duration_s=time.perf_counter() - started)
    else:
        store.set_state(task_id, "succeeded", ["running"], result=jsonable(result), duration_s=time.perf_counter() - started)
