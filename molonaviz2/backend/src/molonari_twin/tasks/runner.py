"""Protocole des runners de tâches, et clé d'idempotence.

L'API ne connaît que ce protocole. LocalTaskRunner l'implémente en phases 3–4, CeleryTaskRunner en phase 5 :
le contrat HTTP, le client et le frontend ne changent pas d'une implémentation à l'autre.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from .model import TaskEvent, TaskRecord


class TaskRunner(Protocol):
    name: str
    max_workers: int

    def submit(self, kind: str, params: dict, key: str | None = None, timeout_s: float | None = None) -> TaskRecord:
        """Crée une tâche `queued`, ou renvoie la tâche en cours de même clé (`reused`). E_TASK_KIND_UNKNOWN."""

    def get(self, task_id: str) -> TaskRecord:
        """État de la tâche. E_TASK_NOT_FOUND."""

    def events(self, task_id: str, after: int | None = None) -> list[TaskEvent]:
        """Événements de la tâche d'identifiant supérieur à `after` (tous si None), dans l'ordre."""

    def cancel(self, task_id: str) -> TaskRecord:
        """Demande l'annulation. E_TASK_FINISHED si la tâche est terminée."""

    def result(self, task_id: str) -> TaskRecord:
        """Tâche terminée, avec son résultat ou son erreur. E_TASK_NOT_FINISHED sinon."""

    def wait(self, task_id: str, timeout_s: float | None = None) -> TaskRecord:
        """Attend la fin de la tâche (ou le délai) et renvoie son état."""

    def close(self) -> None:
        """Arrête le runner ; les tâches en cours sont interrompues."""


def default_environment() -> dict:
    from .. import __version__

    return {"backend": __version__}


def task_key(kind: str, params: dict, twin: str | None = None, environment: dict | None = None) -> str:
    """Empreinte SHA-256 du JSON canonique {twin, kind, params, environment} : indépendante de l'ordre des clés."""
    payload = {"twin": twin, "kind": kind, "params": params,
               "environment": default_environment() if environment is None else environment}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
