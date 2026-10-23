"""Modèles pydantic de l'API /v1. Ce sont des données : ni calcul, ni accès aux fichiers."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Erreurs, informations ---------------------------------------------------------------------------


class ErrorOut(BaseModel):
    """Erreur attendue. Le client traduit à partir de `code` et de `params` ; `message_en` est le texte anglais."""

    code: str = Field(description="Stable code: E_… for an error, W_… for a warning.")
    field: str | None = Field(default=None, description="Input field concerned, if any.")
    params: dict[str, Any] = Field(default={}, description="Parameters of the message template.")
    message_en: str = Field(description="English message, formatted with params.")


class InfoOut(BaseModel):
    api_version: str
    backend_version: str
    runner: str
    max_workers: int


# Tâches ------------------------------------------------------------------------------------------


class RunIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strict: bool = Field(default=False, description="Refuse an unpublished code version or modified or untagged repositories.")


class TaskOut(BaseModel):
    task_id: str
    kind: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    backend_version: str | None = None
    platform: str | None = None
    error: ErrorOut | None = None
    reused: bool = Field(default=False, description="True when an identical task was already queued or running.")


class TaskResultOut(BaseModel):
    task_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    result: dict[str, Any] | None = None
    error: ErrorOut | None = None


# Réponses aux questions --------------------------------------------------------------------------


class ArrayOut(BaseModel):
    """Tableau en JSON : valeurs à plat, dans l'ordre C. NaN → null ; ±inf → "inf" / "-inf" ; dates en ISO 8601,
    sans fuseau (la convention de dates sera fixée avec le pipeline brut)."""

    dtype: str = Field(description='numpy dtype: "float64", "int64", "bool", "datetime64[s]"…')
    shape: list[int]
    data: list[Any]


class NpzRef(BaseModel):
    """Tableau rangé dans l'archive npz de la réponse, sous la clé `npz`."""

    npz: str


Value = ArrayOut | NpzRef | bool | int | float | str | None


class QuantityOut(BaseModel):
    value: Value
    unit: str = Field(description='SI unit, "1" for a dimensionless quantity.')
    interval: tuple[Value, Value] | None = Field(default=None, description="(low, high), shaped like value.")


class WarningOut(BaseModel):
    code: str
    field: str | None = None
    params: dict[str, Any] = {}
    message_en: str


class AnswerOut(BaseModel):
    data: dict[str, QuantityOut]
    provenance: dict[str, Any] = Field(description="Question, parameters, twin and results that produced the answer.")
    warnings: list[WarningOut] = []


# Catalogue des questions -------------------------------------------------------------------------


class QuestionOut(BaseModel):
    id: str = Field(description="Identifier used in the URL, e.g. exchanged_volume.")
    ref: str = Field(description="Reference in the plan, e.g. HN-12 (high level) or BN-34 (low level).")
    level: Literal["high", "low"]
    category: str
    label_en: str = Field(description="English label, also the translation key of the client.")
    description: str
    modes: list[Literal["D", "M"]] = Field(description="Twin modes: D (direct), M (mcmc).")
    since: str = Field(description="API version that introduced the question.")
    requires: Literal["config", "results"] = Field(description="What the twin must have: a valid configuration, or computed results.")
    params_schema: dict[str, Any] = Field(description="JSON Schema of the parameters (request body).")


class QuestionsOut(BaseModel):
    api_version: str
    questions: list[QuestionOut]
