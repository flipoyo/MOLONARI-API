"""Modèles des réponses de l'API /v1."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ..tasks.model import TaskRecord


class ErrorOut(BaseModel):
    code: str
    field: str | None = None
    params: dict[str, Any] = {}
    message_en: str


class InfoOut(BaseModel):
    api_version: str
    backend_version: str
    runner: str
    max_workers: int


class RunIn(BaseModel):
    strict: bool = False


class TaskOut(BaseModel):
    task_id: str
    kind: str
    state: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    backend_version: str | None = None
    platform: str | None = None
    error: ErrorOut | None = None
    reused: bool = False

    @classmethod
    def from_record(cls, record: TaskRecord) -> TaskOut:
        return cls(task_id=record.id, kind=record.kind, state=record.state, created_at=record.created_at,
                   started_at=record.started_at, finished_at=record.finished_at, duration_s=record.duration_s,
                   backend_version=record.backend_version, platform=record.platform, error=record.error,
                   reused=record.reused)


class TaskResultOut(BaseModel):
    task_id: str
    state: str
    result: dict[str, Any] | None = None
    error: ErrorOut | None = None

    @classmethod
    def from_record(cls, record: TaskRecord) -> TaskResultOut:
        return cls(task_id=record.id, state=record.state, result=record.result, error=record.error)
