"""Application FastAPI : questions posées à un twin, lancement d'un twin et suivi des tâches.

    GET    /v1/info                    version de l'API et du backend, runner
    GET    /v1/questions[?twin=ref]    catalogue, avec le schéma JSON des paramètres de chaque question
    POST   /v1/twins/{ref}/questions/{id}   réponse (200) ; JSON, ou npz avec Accept: application/x-npz
    POST   /v1/twins/{ref}/run         202 + tâche (la tâche en cours de même clé si elle existe)
    GET    /v1/tasks/{id}              état de la tâche
    GET    /v1/tasks/{id}/events       SSE : id, event, data ; reprise après Last-Event-ID ; fin sur un état terminal
    DELETE /v1/tasks/{id}              202 : annulation demandée ; 409 si la tâche est terminée
    GET    /v1/tasks/{id}/result       résultat ou erreur ; 409 tant que la tâche n'est pas terminée

Les questions lisent des résultats : elles répondent tout de suite, sans tâche ni JAX (TwinAPI). Calculer un
twin passe par une tâche. Erreurs : {code, field, params, message_en}. Les endpoints sont synchrones : FastAPI les exécute dans un pool de
threads, et l'API reste disponible pendant un calcul, qui tourne de toute façon dans un autre processus.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .. import __version__
from ..api import CATALOG, TwinAPI
from ..contract import API_VERSION
from ..contract.catalog import questions_out
from ..contract.codec import NPZ_MEDIA_TYPE, answer_to_json, answer_to_npz
from ..contract.models import (
    AnswerOut,
    ErrorOut,
    InfoOut,
    QuestionsOut,
    RunIn,
    TaskOut,
    TaskResultOut,
)
from ..errors import Issue, UserError
from ..questions import Answer
from ..tasks.model import TERMINAL, TaskRecord, error_dict
from ..tasks.runner import TaskRunner, task_key
from ..twin.config import SCHEMA_WRITTEN
from ..twin.results import RESULTS_FORMAT

HTTP_STATUS = {
    "E_TASK_NOT_FOUND": 404,
    "E_TWIN_NOT_FOUND": 404,
    "E_QUESTION_UNKNOWN": 404,
    "E_TASK_FINISHED": 409,
    "E_TASK_NOT_FINISHED": 409,
    "E_TWIN_NOT_COMPUTED": 409,
    "E_TWIN_FROZEN": 409,
    "E_QUESTION_NOT_AVAILABLE": 409,
    "E_QUESTION_NO_TWIN": 409,
    "E_QUESTION_TWIN_REQUIRED": 409,
    "E_TASK_KIND_UNKNOWN": 422,
}
ERRORS = {404: {"model": ErrorOut}, 409: {"model": ErrorOut}}


def http_status(code: str) -> int:
    """Statut HTTP d'une erreur attendue : 404 (inconnu), 409 (état du twin ou de la tâche), 422 (paramètres d'une
    question, ou réponse impossible avec ces paramètres), 400 sinon."""
    if code in HTTP_STATUS:
        return HTTP_STATUS[code]
    return 422 if code.startswith("E_QUESTION_") else 400


def questions_description() -> str:
    lines = ["Ask a question of the catalogue (GET /v1/questions) to a twin. The body is the question's parameters.", ""]
    lines += [f"- `{q.id}` ({q.ref}, {'/'.join(q.modes)}): {q.description}" for q in CATALOG if q.requires != "none"]
    return "\n".join(lines)


def global_questions_description() -> str:
    lines = ["Ask a question that is about no twin (`requires: none` in GET /v1/questions). The body is its parameters.", ""]
    lines += [f"- `{q.id}` ({q.ref}): {q.description}" for q in CATALOG if q.requires == "none"]
    return "\n".join(lines)


def task_out(record: TaskRecord) -> TaskOut:
    return TaskOut(task_id=record.id, kind=record.kind, state=record.state, created_at=record.created_at,
                   started_at=record.started_at, finished_at=record.finished_at, duration_s=record.duration_s,
                   backend_version=record.backend_version, platform=record.platform, error=record.error,
                   reused=record.reused)


def task_result_out(record: TaskRecord) -> TaskResultOut:
    return TaskResultOut(task_id=record.id, state=record.state, result=record.result, error=record.error)


def create_app(runner: TaskRunner, twins_folder: str | Path, events_poll_s: float = 0.1, device_task: str | None = None) -> FastAPI:
    """`device_task` : identifiant de la tâche `diag.device` lancée au démarrage (par `serve`) ; son résultat est l'appareil
    de /v1/info, null tant qu'elle n'a pas réussi."""
    twins_folder = Path(twins_folder)
    questions = TwinAPI(twins_folder)
    app = FastAPI(title="molonari-twin", version=API_VERSION,
                  responses={422: {"model": ErrorOut, "description": "Invalid request or parameters."}})

    @app.exception_handler(UserError)
    async def user_error(request: Request, exc: UserError) -> JSONResponse:
        return JSONResponse(error_dict(exc), status_code=http_status(exc.code))

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Requête mal formée (corps qui n'est pas un objet JSON, champ inconnu…) : même format que les autres
        erreurs, au lieu du {"detail": …} de FastAPI."""
        issues = [Issue(".".join(map(str, e["loc"])) or None, "E_PARAM_INVALID", prefixed=True, message=e["msg"])
                  for e in exc.errors()]
        return JSONResponse(error_dict(UserError("E_REQUEST_INVALID", issues=issues)), status_code=422)

    @app.get("/v1/info")
    def info() -> InfoOut:
        return InfoOut(api_version=API_VERSION, backend_version=__version__, runner=runner.name,
                       max_workers=runner.max_workers, twin_schema=SCHEMA_WRITTEN, results_format=RESULTS_FORMAT,
                       data_release=None, device=device())

    def device() -> str | None:
        if device_task is None:
            return None
        try:
            record = runner.get(device_task)
        except UserError:
            return None
        return record.result.get("device") if record.state == "succeeded" and record.result else None

    @app.get("/v1/questions", responses={404: {"model": ErrorOut}})
    def list_questions(twin: str | None = None) -> QuestionsOut:
        """The catalogue; with `twin`, only the questions that apply to its mode."""
        return questions_out(questions.questions(twin))

    def reply(answer: Answer, request: Request) -> Response:
        if NPZ_MEDIA_TYPE in request.headers.get("accept", ""):
            return Response(answer_to_npz(answer), media_type=NPZ_MEDIA_TYPE)
        return JSONResponse(answer_to_json(answer))

    answered = {**ERRORS, 200: {"content": {NPZ_MEDIA_TYPE: {}}, "description": "Answer, as JSON or as an npz archive."}}

    @app.post("/v1/questions/{question_id}", description=global_questions_description(), responses=answered)
    def ask_global(question_id: str, request: Request,
                   params: Annotated[dict[str, Any] | None, Body(description="Parameters of the question (see params_schema).")] = None,
                   ) -> AnswerOut:
        return reply(questions.ask_global(question_id, params), request)

    @app.post("/v1/twins/{ref}/questions/{question_id}", description=questions_description(), responses=answered)
    def ask(ref: str, question_id: str, request: Request,
            params: Annotated[dict[str, Any] | None, Body(description="Parameters of the question (see params_schema).")] = None,
            ) -> AnswerOut:
        return reply(questions.ask(ref, question_id, params), request)

    @app.post("/v1/twins/{ref}/run", status_code=202, responses=ERRORS)
    def run_twin(ref: str, body: RunIn | None = None) -> TaskOut:
        folder = questions.twin_folder(ref)
        params = {"path": str(folder.resolve()), "strict": (body or RunIn()).strict}
        record = runner.submit("twin.run", params, key=task_key("twin.run", params, twin=ref))
        return task_out(record)

    @app.get("/v1/tasks/{task_id}", responses=ERRORS)
    def get_task(task_id: str) -> TaskOut:
        return task_out(runner.get(task_id))

    @app.delete("/v1/tasks/{task_id}", status_code=202, responses=ERRORS)
    def cancel_task(task_id: str) -> TaskOut:
        return task_out(runner.cancel(task_id))

    @app.get("/v1/tasks/{task_id}/result", responses=ERRORS)
    def task_result(task_id: str) -> TaskResultOut:
        return task_result_out(runner.result(task_id))

    @app.get("/v1/tasks/{task_id}/events", responses={404: {"model": ErrorOut}})
    def task_events(task_id: str, request: Request,
                    last_event_id: str | None = Header(default=None, alias="Last-Event-ID")) -> StreamingResponse:
        runner.get(task_id)  # 404 avant d'ouvrir le flux
        after = int(last_event_id) if last_event_id and last_event_id.isdigit() else None

        async def stream():
            nonlocal after
            while not await request.is_disconnected():
                # L'état est lu avant les événements : s'il est terminal, son événement (écrit dans la même
                # transaction) est forcément dans la lecture qui suit, et aucun événement final n'est perdu.
                finished = (await run_in_threadpool(runner.get, task_id)).finished
                events = await run_in_threadpool(runner.events, task_id, after)
                for event in events:
                    after = event.id
                    yield f"id: {event.id}\nevent: {event.type}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"
                    if event.type == "state" and event.data.get("state") in TERMINAL:
                        return
                if finished:
                    return  # reprise après l'événement terminal : rien de plus à envoyer
                await asyncio.sleep(events_poll_s)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
