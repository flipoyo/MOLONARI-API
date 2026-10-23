"""Application FastAPI : lancement d'un twin et suivi des tâches.

    GET    /v1/info                    version de l'API et du backend, runner
    POST   /v1/twins/{ref}/run         202 + tâche (la tâche en cours de même clé si elle existe)
    GET    /v1/tasks/{id}              état de la tâche
    GET    /v1/tasks/{id}/events       SSE : id, event, data ; reprise après Last-Event-ID ; fin sur un état terminal
    DELETE /v1/tasks/{id}              202 : annulation demandée ; 409 si la tâche est terminée
    GET    /v1/tasks/{id}/result       résultat ou erreur ; 409 tant que la tâche n'est pas terminée

Erreurs : {code, field, params, message_en}. Les endpoints sont synchrones : FastAPI les exécute dans un pool de
threads, et l'API reste disponible pendant un calcul, qui tourne de toute façon dans un autre processus.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from .. import __version__
from ..contract import API_VERSION
from ..contract.models import ErrorOut, InfoOut, RunIn, TaskOut, TaskResultOut
from ..errors import UserError
from ..tasks.model import TERMINAL, TaskRecord, error_dict
from ..tasks.runner import TaskRunner, task_key

HTTP_STATUS = {
    "E_TASK_NOT_FOUND": 404,
    "E_TWIN_NOT_FOUND": 404,
    "E_TASK_FINISHED": 409,
    "E_TASK_NOT_FINISHED": 409,
    "E_TASK_KIND_UNKNOWN": 422,
}
TWIN_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ERRORS = {404: {"model": ErrorOut}, 409: {"model": ErrorOut}}


def task_out(record: TaskRecord) -> TaskOut:
    return TaskOut(task_id=record.id, kind=record.kind, state=record.state, created_at=record.created_at,
                   started_at=record.started_at, finished_at=record.finished_at, duration_s=record.duration_s,
                   backend_version=record.backend_version, platform=record.platform, error=record.error,
                   reused=record.reused)


def task_result_out(record: TaskRecord) -> TaskResultOut:
    return TaskResultOut(task_id=record.id, state=record.state, result=record.result, error=record.error)


def create_app(runner: TaskRunner, twins_folder: str | Path, events_poll_s: float = 0.1) -> FastAPI:
    twins_folder = Path(twins_folder)
    app = FastAPI(title="molonari-twin", version=API_VERSION)

    @app.exception_handler(UserError)
    async def user_error(request: Request, exc: UserError) -> JSONResponse:
        return JSONResponse(error_dict(exc), status_code=HTTP_STATUS.get(exc.code, 400))

    @app.get("/v1/info")
    def info() -> InfoOut:
        return InfoOut(api_version=API_VERSION, backend_version=__version__, runner=runner.name,
                       max_workers=runner.max_workers)

    @app.post("/v1/twins/{ref}/run", status_code=202, responses=ERRORS)
    def run_twin(ref: str, body: RunIn | None = None) -> TaskOut:
        folder = twins_folder / ref
        if not TWIN_REF.match(ref) or ".." in ref or not (folder / "twin.yaml").is_file():
            raise UserError("E_TWIN_NOT_FOUND", ref=ref, folder=str(twins_folder))
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
