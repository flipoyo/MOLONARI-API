"""Tâches de calcul : cycle de vie, annulation coopérative puis forcée, timeout, idempotence, journal rejouable.

La suite est paramétrée par l'implémentation du runner (fixture `make_runner`). En phase 5, CeleryTaskRunner s'y
ajoute : la même suite vérifie alors la parité entre local et distant. Les tests propres au LocalTaskRunner
(reprise après un arrêt brutal, base par défaut) sont à part.
"""

import math
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
from molonari_twin.errors import UserError
from molonari_twin.tasks import jobs
from molonari_twin.tasks.local import LocalTaskRunner, default_db_path, pid_alive
from molonari_twin.tasks.model import TERMINAL, TaskRecord, now
from molonari_twin.tasks.runner import task_key
from molonari_twin.tasks.store import TaskStore

RUNNERS = {
    "local": lambda tmp_path, **options: LocalTaskRunner(tmp_path / "tasks.sqlite", **options),
}


@pytest.fixture(params=sorted(RUNNERS))
def make_runner(request, tmp_path):
    made = []

    def make(**options):
        runner = RUNNERS[request.param](tmp_path, **{"cancel_grace_s": 1.0, "poll_s": 0.02, **options})
        made.append(runner)
        return runner

    yield make
    for runner in made:
        runner.close()


@pytest.fixture
def runner(make_runner):
    return make_runner()


def sleep(blocks=3, block_s=0.05, **options):
    return {"blocks": blocks, "block_s": block_s, **options}


def wait_for(predicate, timeout_s=20.0):
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "délai dépassé"
        time.sleep(0.02)


def wait_progress(runner, task_id):
    wait_for(lambda: any(e.type == "progress" for e in runner.events(task_id)))


def wait_dead(pid):
    wait_for(lambda: not pid_alive(pid), timeout_s=10)


def states(runner, task_id):
    return [e.data["state"] for e in runner.events(task_id) if e.type == "state"]


def error_code(call):
    with pytest.raises(UserError) as info:
        call()
    return info.value.code


# Cycle de vie -------------------------------------------------------------------------------------


def test_successful_task_goes_through_every_state(runner):
    task = runner.submit("diag.sleep", sleep())
    assert task.state == "queued" and not task.reused
    done = runner.wait(task.id, 20)
    assert done.state == "succeeded"
    assert done.result == {"blocks": 3}
    assert done.error is None
    assert done.duration_s is not None and done.duration_s > 0
    assert done.started_at and done.finished_at and done.backend_version and done.platform
    assert states(runner, task.id) == ["queued", "running", "succeeded"]
    assert runner.result(task.id).result == {"blocks": 3}


def test_event_ids_increase_and_progress_is_published(runner):
    task = runner.submit("diag.sleep", sleep(blocks=4))
    runner.wait(task.id, 20)
    events = runner.events(task.id)
    ids = [e.id for e in events]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    progress = [e.data for e in events if e.type == "progress"]
    assert [p["iteration"] for p in progress] == [1, 2, 3, 4]
    assert all(p["total"] == 4 for p in progress)


def test_events_after_an_id_are_exactly_the_rest(runner):
    task = runner.submit("diag.sleep", sleep(blocks=4))
    runner.wait(task.id, 20)
    events = runner.events(task.id)
    for cut in range(len(events)):
        assert runner.events(task.id, after=events[cut].id) == events[cut + 1:]
    assert runner.events(task.id, after=None) == events


def test_user_error_keeps_its_code_and_parameters(runner):
    task = runner.submit("diag.sleep", sleep(blocks=2, fail="user"))
    done = runner.wait(task.id, 20)
    assert done.state == "failed"
    assert done.error == {"code": "E_DIAG_FAILURE", "field": None, "params": {"block": 2},
                          "message_en": "The diagnostic task failed on purpose at block 2."}
    assert runner.events(task.id)[-1].data == {"state": "failed", "error": done.error}


def test_unexpected_exception_becomes_a_system_error(runner):
    task = runner.submit("diag.sleep", sleep(blocks=1, fail="crash"))
    done = runner.wait(task.id, 20)
    assert done.state == "failed"
    assert done.error["code"] == "E_SYSTEM_ERROR"
    assert done.error["params"]["detail"] == "RuntimeError: diagnostic crash"


# Annulation et timeout ----------------------------------------------------------------------------


def test_cooperative_cancel_stops_within_one_block(make_runner):
    runner = make_runner(cancel_grace_s=10.0)  # un arrêt forcé prendrait au moins 10 s
    task = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    wait_progress(runner, task.id)
    asked = time.monotonic()
    runner.cancel(task.id)
    done = runner.wait(task.id, 20)
    assert done.state == "cancelled"
    assert time.monotonic() - asked < 2.0
    wait_dead(done.pid)
    assert states(runner, task.id) == ["queued", "running", "cancelled"]


def test_ignored_cancel_terminates_the_worker_and_cleans_up(make_runner, tmp_path):
    runner = make_runner(cancel_grace_s=0.5)
    partial = tmp_path / "results" / "x.partial"
    task = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05, ignore_cancel=True, partial=str(partial)))
    wait_progress(runner, task.id)
    assert partial.is_dir()
    asked = time.monotonic()
    runner.cancel(task.id)
    done = runner.wait(task.id, 20)
    assert not partial.exists()  # nettoyé avant l'état terminal, pas après
    assert done.state == "cancelled"
    assert time.monotonic() - asked >= 0.5  # le job n'a pas coopéré : arrêt après le délai de grâce
    wait_dead(done.pid)


@pytest.mark.parametrize("options, state", [
    ({"blocks": 1000}, "cancelled"),  # annulation coopérative
    ({"blocks": 2, "fail": "user"}, "failed"),
    ({"blocks": 2, "fail": "crash"}, "failed"),
])
def test_worker_cleans_up_before_concluding(runner, tmp_path, options, state):
    partial = tmp_path / "results" / "z.partial"
    task = runner.submit("diag.sleep", sleep(block_s=0.05, partial=str(partial), **options))
    wait_progress(runner, task.id)
    if state == "cancelled":
        runner.cancel(task.id)
    done = runner.wait(task.id, 20)
    assert done.state == state
    assert not partial.exists()


def test_timeout_fails_the_task_and_cleans_up(runner, tmp_path):
    partial = tmp_path / "results" / "y.partial"
    task = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05, ignore_cancel=True, partial=str(partial)),
                         timeout_s=0.5)
    done = runner.wait(task.id, 20)
    assert not partial.exists()  # nettoyé avant l'état terminal, pas après
    assert done.state == "failed"
    assert done.error["code"] == "E_TASK_TIMEOUT"
    assert done.error["params"] == {"timeout_s": 0.5}
    wait_dead(done.pid)


def test_queued_task_waits_for_a_free_worker_and_cancels_at_once(make_runner):
    runner = make_runner(max_workers=1)
    first = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    second = runner.submit("diag.sleep", sleep(blocks=1))
    wait_progress(runner, first.id)
    assert runner.get(second.id).state == "queued"
    assert runner.cancel(second.id).state == "cancelled"  # sans worker : annulée tout de suite
    assert states(runner, second.id) == ["queued", "cancelled"]
    runner.cancel(first.id)
    assert runner.wait(first.id, 20).state == "cancelled"


def test_cancel_or_result_at_the_wrong_moment(runner):
    task = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    assert error_code(lambda: runner.result(task.id)) == "E_TASK_NOT_FINISHED"
    runner.cancel(task.id)
    runner.wait(task.id, 20)
    assert error_code(lambda: runner.cancel(task.id)) == "E_TASK_FINISHED"


# Idempotence --------------------------------------------------------------------------------------


def test_same_key_while_running_returns_the_same_task(runner):
    first = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    again = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    assert again.id == first.id and again.reused
    other = runner.submit("diag.sleep", sleep(blocks=999, block_s=0.05))
    assert other.id != first.id and not other.reused
    for task in (first, other):
        runner.cancel(task.id)
        runner.wait(task.id, 20)
    after = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    assert after.id != first.id and not after.reused  # une tâche terminée n'est pas réutilisée
    runner.cancel(after.id)
    runner.wait(after.id, 20)


def test_task_key_ignores_key_order_but_not_the_environment():
    a = task_key("twin.run", {"path": "/t/twin_01", "strict": False}, twin="twin_01", environment={"backend": "1"})
    b = task_key("twin.run", {"strict": False, "path": "/t/twin_01"}, twin="twin_01", environment={"backend": "1"})
    assert a == b and len(a) == 64
    assert a != task_key("twin.run", {"path": "/t/twin_01", "strict": False}, twin="twin_01", environment={"backend": "2"})
    assert a != task_key("twin.run", {"path": "/t/twin_01", "strict": True}, twin="twin_01", environment={"backend": "1"})
    assert a != task_key("twin.run", {"path": "/t/twin_01", "strict": False}, twin="twin_02", environment={"backend": "1"})


# Erreurs d'usage ----------------------------------------------------------------------------------


def test_unknown_task_and_kind(runner):
    for call in (runner.get, runner.events, runner.cancel, runner.result):
        assert error_code(lambda call=call: call("nope")) == "E_TASK_NOT_FOUND"
    assert error_code(lambda: runner.submit("no.such.kind", {})) == "E_TASK_KIND_UNKNOWN"


# Propre au LocalTaskRunner ------------------------------------------------------------------------


def dead_pid():
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_local_runner_fails_running_tasks_whose_worker_died(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite")
    orphan = TaskRecord(id="orphan", kind="diag.sleep", key="k1", state="queued", params=sleep(), created_at=now())
    store.submit(orphan)
    store.set_state("orphan", "running", ["queued"], pid=dead_pid())
    alive = TaskRecord(id="alive", kind="diag.sleep", key="k2", state="queued", params=sleep(), created_at=now())
    store.submit(alive)
    store.set_state("alive", "running", ["queued"], pid=os.getpid())  # son runner est vivant : on n'y touche pas
    runner = LocalTaskRunner(tmp_path / "tasks.sqlite", poll_s=0.02)
    try:
        record = runner.get("orphan")
        assert record.state == "failed" and record.error["code"] == "E_TASK_INTERRUPTED"
        assert runner.get("alive").state == "running"
    finally:
        runner.close()


def test_local_runner_requeued_tasks_start_after_a_restart(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite")
    store.submit(TaskRecord(id="waiting", kind="diag.sleep", key="k", state="queued", params=sleep(blocks=1),
                            created_at=now()))
    runner = LocalTaskRunner(tmp_path / "tasks.sqlite", poll_s=0.02)
    try:
        assert runner.wait("waiting", 20).state == "succeeded"
    finally:
        runner.close()


def test_closing_the_local_runner_interrupts_running_tasks(tmp_path):
    runner = LocalTaskRunner(tmp_path / "tasks.sqlite", poll_s=0.02)
    task = runner.submit("diag.sleep", sleep(blocks=1000, block_s=0.05))
    wait_progress(runner, task.id)
    runner.close()
    record = TaskStore(tmp_path / "tasks.sqlite").get(task.id)
    assert record.state == "failed" and record.error["code"] == "E_TASK_INTERRUPTED"
    wait_dead(record.pid)


def test_default_database_follows_xdg_cache_home(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_db_path() == tmp_path / "molonari-twin" / "tasks.sqlite"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_db_path().parts[-3:] == (".cache", "molonari-twin", "tasks.sqlite")


def test_terminal_states():
    assert TERMINAL == {"succeeded", "failed", "cancelled"}


# Jobs ---------------------------------------------------------------------------------------------


def test_dream_progress_becomes_json():
    progress = SimpleNamespace(phase="sampling", iteration=np.int64(40), total=100, elapsed=1.23456,
                               acceptance=np.float64(0.25), chain_energies=np.array([1.5, 2.5]), rhat_max=math.nan)
    assert jobs.progress_dict(progress) == {"phase": "sampling", "iteration": 40, "total": 100, "elapsed_s": 1.235,
                                            "acceptance": 0.25, "chain_energies": [1.5, 2.5], "rhat_max": None}
    progress.rhat_max = 1.01
    assert jobs.progress_dict(progress)["rhat_max"] == 1.01


def test_twin_cleanup_removes_only_partial_results(tmp_path):
    twins = tmp_path / "twins"
    (twins / "twin_01").mkdir(parents=True)
    results = tmp_path / "results"
    (results / "abc.partial").mkdir(parents=True)
    (results / "abc").mkdir()
    jobs.cleanup_twin_job({"path": str(twins / "twin_01")})
    assert not (results / "abc.partial").exists()
    assert (results / "abc").is_dir()
