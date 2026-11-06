"""Le catalogue des questions dans le contrat : `Question` → `QuestionOut`, une liste → `QuestionsOut`.

Le serveur (GET /v1/questions) et la ligne de commande (`molonari-twin questions --json`) donnent ainsi le même JSON,
et le catalogue se liste sans FastAPI.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from . import API_VERSION
from .models import QuestionOut, QuestionsOut

if TYPE_CHECKING:
    from ..api.catalog import Question


def question_out(question: Question) -> QuestionOut:
    return QuestionOut(id=question.id, ref=question.ref, level=question.level, category=question.category,
                       label_en=question.label, description=question.description, modes=list(question.modes),
                       since=question.since, requires=question.requires,
                       params_schema=question.params.model_json_schema())


def questions_out(questions: Iterable[Question]) -> QuestionsOut:
    return QuestionsOut(api_version=API_VERSION, questions=[question_out(q) for q in questions])
