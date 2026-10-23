"""Une tâche, ses événements, et la forme sérialisable des erreurs."""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from ..errors import Issue, UserError

TaskState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
STATES: tuple[str, ...] = ("queued", "running", "succeeded", "failed", "cancelled")
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


def now() -> str:
    """Horodatage ISO 8601 avec fuseau (heure locale)."""
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def platform_info() -> str:
    return f"{platform.system()} {platform.release()} {platform.machine()} · Python {platform.python_version()}"


def jsonable(value):
    """Valeur JSON : un problème devient {code, field, params, message_en}, un chemin une chaîne."""
    if isinstance(value, Issue):
        return {"code": value.code, "field": value.field, "params": jsonable(value.params), "message_en": value.message}
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [jsonable(v) for v in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    return str(value)


def error_dict(error: UserError) -> dict:
    """Erreur attendue sous la forme du contrat : {code, field, params, message_en}."""
    return {"code": error.code, "field": error.field, "params": jsonable(error.params), "message_en": str(error)}


@dataclass
class TaskRecord:
    id: str
    kind: str
    key: str
    state: TaskState
    params: dict
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    timeout_s: float | None = None
    pid: int | None = None
    backend_version: str | None = None
    platform: str | None = None
    error: dict | None = None
    result: dict | None = None
    reused: bool = False  # soumission rattachée à une tâche identique déjà en cours

    @property
    def finished(self) -> bool:
        return self.state in TERMINAL

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class TaskEvent:
    id: int  # croissant pour toutes les tâches : sert de Last-Event-ID
    task_id: str
    type: str  # « state », « progress » ou « log »
    data: dict = field(default_factory=dict)
    at: str = ""
