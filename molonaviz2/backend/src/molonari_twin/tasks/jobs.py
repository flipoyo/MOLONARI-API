"""Types de tâches : une fonction exécutée par le worker, et un nettoyage exécuté par le runner quand la tâche
ne réussit pas (annulée, en échec, terminée de force, ou hors délai).

Une fonction de job reçoit ses paramètres (JSON) et un contexte : `context.report(dict)` publie une
progression, `context.should_cancel()` dit si l'annulation est demandée. Elle renvoie un résultat JSON léger ;
les résultats lourds restent dans des fichiers.
"""

from __future__ import annotations

import math
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..errors import Cancelled, UserError
from .model import jsonable


class JobContext(Protocol):
    def report(self, data: dict) -> None: ...

    def log(self, message: str, **data) -> None: ...

    def should_cancel(self) -> bool: ...


@dataclass(frozen=True)
class Job:
    run: Callable[[dict, JobContext], dict]
    cleanup: Callable[[dict], None] | None = None


# twin.run ----------------------------------------------------------------------------------------


def progress_dict(progress) -> dict:
    """Progression du DREAM (engine.jax.dream.Progress) en JSON."""
    rhat = float(progress.rhat_max)
    return {
        "phase": progress.phase,
        "iteration": int(progress.iteration),
        "total": int(progress.total),
        "elapsed_s": round(float(progress.elapsed), 3),
        "acceptance": float(progress.acceptance),
        "chain_energies": [float(e) for e in progress.chain_energies],
        "rhat_max": None if math.isnan(rhat) else rhat,
    }


def run_twin_job(params: dict, context: JobContext) -> dict:
    """Exécute le twin du dossier `params["path"]` (ou renvoie son résultat existant)."""
    from ..twin.run import run_twin

    run = run_twin(params["path"], on_progress=lambda p: context.report(progress_dict(p)),
                   should_cancel=context.should_cancel, strict=bool(params.get("strict", False)))
    status = run.status
    for warning in run.warnings:
        context.log(warning.message, code=warning.code)
    return {
        "twin": status.config.name,
        "twin_id": status.twin_id,
        "state": status.state,
        "computed": run.computed,
        "duration_s": round(run.duration_s, 3),
        "results_dir": str(status.results_dir),
        "warnings": jsonable(list(run.warnings)),
    }


def cleanup_twin_job(params: dict) -> None:
    """Supprime les résultats partiels (`results/*.partial`) laissés par un calcul interrompu."""
    from ..twin.registry import results_folder

    folder = results_folder(Path(params["path"]).parent)
    for partial in folder.glob("*.partial"):
        shutil.rmtree(partial, ignore_errors=True)


# diag.sleep --------------------------------------------------------------------------------------


def sleep_job(params: dict, context: JobContext) -> dict:
    """Tâche de diagnostic : `blocks` blocs de `block_s` secondes, avec progression et annulation coopérative.

    Options : `ignore_cancel` (ne lit jamais l'annulation, pour éprouver la terminaison forcée), `partial`
    (dossier créé au début et retiré à la fin, comme des résultats partiels), `fail` (« user » : UserError
    au dernier bloc ; « crash » : exception quelconque).
    """
    blocks, block_s = int(params.get("blocks", 5)), float(params.get("block_s", 0.1))
    partial = Path(params["partial"]) if params.get("partial") else None
    if partial is not None:
        partial.mkdir(parents=True, exist_ok=True)
    for block in range(1, blocks + 1):
        if not params.get("ignore_cancel") and context.should_cancel():
            raise Cancelled
        time.sleep(block_s)
        context.report({"phase": "diag", "iteration": block, "total": blocks})
        if block == blocks and params.get("fail") == "user":
            raise UserError("E_DIAG_FAILURE", block=block)
        if block == blocks and params.get("fail") == "crash":
            raise RuntimeError("diagnostic crash")
    if partial is not None:
        shutil.rmtree(partial)
    return {"blocks": blocks}


def cleanup_sleep_job(params: dict) -> None:
    if params.get("partial"):
        shutil.rmtree(params["partial"], ignore_errors=True)


JOBS: dict[str, Job] = {
    "twin.run": Job(run_twin_job, cleanup_twin_job),
    "diag.sleep": Job(sleep_job, cleanup_sleep_job),
}


def get_job(kind: str) -> Job:
    if kind not in JOBS:
        raise UserError("E_TASK_KIND_UNKNOWN", kind=kind, kinds=", ".join(sorted(JOBS)))
    return JOBS[kind]
