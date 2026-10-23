"""API HTTP /v1 : contrat des endpoints (faux runner en mémoire), intégration avec LocalTaskRunner, et calcul
jamais exécuté dans le processus de l'API."""

import json
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from molonari_twin import __version__
from molonari_twin.errors import UserError
from molonari_twin.server.app import API_VERSION, create_app
from molonari_twin.tasks.local import LocalTaskRunner
from molonari_twin.tasks.model import TaskEvent, TaskRecord, now

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"
ERROR_KEYS = {"code", "field", "params", "message_en"}


class FakeTaskRunner:
    """Runner en mémoire : il enregistre les soumissions, et le test fait avancer les tâches à la main."""

    name = "fake"
    max_workers = 2

    def __init__(self):
        self.records: dict[str, TaskRecord] = {}
        self.log: dict[str, list[TaskEvent]] = {}
        self.submitted: list[tuple[str, dict, str]] = []
        self._next_event = 0

    def submit(self, kind, params, key=None, timeout_s=None):
        self.submitted.append((kind, params, key))
        for record in self.records.values():
            if record.key == key and not record.finished:
                return TaskRecord(**{**record.to_dict(), "reused": True})
        record = TaskRecord(id=f"t{len(self.records) + 1}", kind=kind, key=key, state="queued", params=params,
                            created_at=now())
        self.records[record.id] = record
        self.log[record.id] = []
        self.emit(record.id, "state", {"state": "queued"})
        return record

    def emit(self, task_id, type_, data):
        self._next_event += 1
        self.log[task_id].append(TaskEvent(self._next_event, task_id, type_, data, now()))

    def finish(self, task_id, state, result=None, error=None):
        record = self.records[task_id]
        record.state, record.result, record.error = state, result, error
        self.emit(task_id, "state", {"state": state})

    def get(self, task_id):
        if task_id not in self.records:
            raise UserError("E_TASK_NOT_FOUND", task_id=task_id)
        return self.records[task_id]

    def events(self, task_id, after=None):
        self.get(task_id)
        return [e for e in self.log[task_id] if after is None or e.id > after]

    def cancel(self, task_id):
        record = self.get(task_id)
        if record.finished:
            raise UserError("E_TASK_FINISHED", task_id=task_id, state=record.state)
        self.finish(task_id, "cancelled")
        return record

    def result(self, task_id):
        record = self.get(task_id)
        if not record.finished:
            raise UserError("E_TASK_NOT_FINISHED", task_id=task_id, state=record.state)
        return record

    def wait(self, task_id, timeout_s=None):
        return self.get(task_id)

    def close(self):
        pass


@pytest.fixture
def twins(tmp_path):
    folder = tmp_path / "twins"
    folder.mkdir()
    shutil.copytree(EXAMPLES / "twin_01", folder / "twin_01")
    return folder


@pytest.fixture
def fake():
    return FakeTaskRunner()


@pytest.fixture
def client(fake, twins):
    return TestClient(create_app(fake, twins, events_poll_s=0.01))


def sse(response) -> list[dict]:
    """Événements SSE d'une réponse : [{id, event, data}]."""
    events, current = [], {}
    for line in response.iter_lines():
        if not line:
            if current:
                events.append(current)
            current = {}
            continue
        name, _, value = line.partition(": ")
        current[name] = int(value) if name == "id" else json.loads(value) if name == "data" else value
    if current:
        events.append(current)
    return events


def assert_error(response, status, code):
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == ERROR_KEYS
    assert body["code"] == code
    return body


# Contrat, avec un faux runner ---------------------------------------------------------------------


def test_info(client):
    assert client.get("/v1/info").json() == {"api_version": API_VERSION, "backend_version": __version__,
                                             "runner": "fake", "max_workers": 2}


def test_run_creates_a_task(client, fake, twins):
    response = client.post("/v1/twins/twin_01/run")
    assert response.status_code == 202
    body = response.json()
    assert body["task_id"] == "t1" and body["state"] == "queued" and body["kind"] == "twin.run"
    assert body["reused"] is False and body["error"] is None
    kind, params, key = fake.submitted[0]
    assert kind == "twin.run"
    assert params == {"path": str((twins / "twin_01").resolve()), "strict": False}
    assert len(key) == 64


def test_run_passes_strict_and_reuses_a_running_task(client, fake):
    first = client.post("/v1/twins/twin_01/run", json={"strict": True}).json()
    assert fake.submitted[0][1]["strict"] is True
    again = client.post("/v1/twins/twin_01/run", json={"strict": True}).json()
    assert again["task_id"] == first["task_id"] and again["reused"] is True
    other = client.post("/v1/twins/twin_01/run").json()
    assert other["task_id"] != first["task_id"]


@pytest.mark.parametrize("ref", ["twin_99", ".hidden", "twin_01..", "-x"])
def test_unknown_twin_is_404(client, fake, ref):
    body = assert_error(client.post(f"/v1/twins/{ref}/run"), 404, "E_TWIN_NOT_FOUND")
    assert body["params"]["ref"] == ref
    assert fake.submitted == []


def test_unknown_task_is_404(client):
    for method, url in (("GET", "/v1/tasks/nope"), ("GET", "/v1/tasks/nope/result"),
                        ("GET", "/v1/tasks/nope/events"), ("DELETE", "/v1/tasks/nope")):
        body = assert_error(client.request(method, url), 404, "E_TASK_NOT_FOUND")
        assert body["params"] == {"task_id": "nope"}
        assert body["message_en"] == "Task nope does not exist."


def test_task_state_and_result(client, fake):
    task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]
    assert client.get(f"/v1/tasks/{task_id}").json()["state"] == "queued"
    assert_error(client.get(f"/v1/tasks/{task_id}/result"), 409, "E_TASK_NOT_FINISHED")
    fake.finish(task_id, "succeeded", result={"twin_id": "abc"})
    response = client.get(f"/v1/tasks/{task_id}/result")
    assert response.status_code == 200
    assert response.json() == {"task_id": task_id, "state": "succeeded", "result": {"twin_id": "abc"}, "error": None}


def test_failed_task_result_carries_the_error(client, fake):
    task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]
    error = {"code": "E_TWIN_FROZEN", "field": None, "params": {"name": "twin_01", "previous": "x"}, "message_en": "…"}
    fake.finish(task_id, "failed", error=error)
    body = client.get(f"/v1/tasks/{task_id}/result").json()
    assert body["state"] == "failed" and body["result"] is None and body["error"] == error
    assert client.get(f"/v1/tasks/{task_id}").json()["error"] == error


def test_cancel(client, fake):
    task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]
    response = client.delete(f"/v1/tasks/{task_id}")
    assert response.status_code == 202 and response.json()["state"] == "cancelled"
    body = assert_error(client.delete(f"/v1/tasks/{task_id}"), 409, "E_TASK_FINISHED")
    assert body["params"] == {"task_id": task_id, "state": "cancelled"}


def test_events_stream_until_the_terminal_state(client, fake):
    task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]
    fake.emit(task_id, "progress", {"iteration": 1})
    fake.finish(task_id, "succeeded", result={})
    with client.stream("GET", f"/v1/tasks/{task_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = sse(response)
    assert [(e["event"], e["data"]) for e in events] == [
        ("state", {"state": "queued"}), ("progress", {"iteration": 1}), ("state", {"state": "succeeded"})]
    assert [e["id"] for e in events] == sorted(e["id"] for e in events)


def test_events_resume_after_last_event_id(client, fake):
    task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]
    for i in range(3):
        fake.emit(task_id, "progress", {"iteration": i})
    fake.finish(task_id, "succeeded", result={})
    with client.stream("GET", f"/v1/tasks/{task_id}/events") as response:
        everything = sse(response)
    for cut in range(len(everything)):
        headers = {"Last-Event-ID": str(everything[cut]["id"])}
        with client.stream("GET", f"/v1/tasks/{task_id}/events", headers=headers) as response:
            assert sse(response) == everything[cut + 1:]


def test_events_stream_waits_for_new_events(client, fake):
    task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]

    def later():
        time.sleep(0.2)
        fake.emit(task_id, "progress", {"iteration": 1})
        fake.finish(task_id, "succeeded", result={})

    threading.Thread(target=later).start()
    with client.stream("GET", f"/v1/tasks/{task_id}/events") as response:
        events = sse(response)
    assert [e["data"].get("state") for e in events] == ["queued", None, "succeeded"]


def test_openapi_lists_the_v1_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/v1/info", "/v1/twins/{ref}/run", "/v1/tasks/{task_id}", "/v1/tasks/{task_id}/result",
                          "/v1/tasks/{task_id}/events"}
    assert set(paths["/v1/tasks/{task_id}"]) == {"get", "delete"}


# Intégration avec LocalTaskRunner -----------------------------------------------------------------


@pytest.fixture
def local(tmp_path, twins):
    runner = LocalTaskRunner(tmp_path / "tasks.sqlite", cancel_grace_s=1.0, poll_s=0.02)
    yield runner, TestClient(create_app(runner, twins, events_poll_s=0.02))
    runner.close()


def test_api_answers_while_a_task_runs(local):
    runner, client = local
    task = runner.submit("diag.sleep", {"blocks": 1000, "block_s": 0.05})
    deadline = time.monotonic() + 20
    while client.get(f"/v1/tasks/{task.id}").json()["state"] != "running":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    started = time.monotonic()
    assert client.get("/v1/info").status_code == 200
    assert time.monotonic() - started < 1.0
    assert client.delete(f"/v1/tasks/{task.id}").status_code == 202
    assert runner.wait(task.id, 20).state == "cancelled"
    assert client.get(f"/v1/tasks/{task.id}").json()["state"] == "cancelled"


def test_sse_of_a_real_task_and_reconnection(local):
    runner, client = local
    task = runner.submit("diag.sleep", {"blocks": 5, "block_s": 0.05})
    with client.stream("GET", f"/v1/tasks/{task.id}/events") as response:
        events = sse(response)
    assert [e["data"]["state"] for e in events if e["event"] == "state"] == ["queued", "running", "succeeded"]
    assert [e["data"]["iteration"] for e in events if e["event"] == "progress"] == [1, 2, 3, 4, 5]
    middle = events[3]["id"]
    with client.stream("GET", f"/v1/tasks/{task.id}/events", headers={"Last-Event-ID": str(middle)}) as response:
        assert sse(response) == events[4:]


def test_api_process_does_not_load_the_engine(tmp_path, twins):
    """Créer l'app, lancer une tâche et la suivre ne charge ni JAX, ni ArviZ, ni Qt dans le processus de l'API."""
    code = textwrap.dedent(f"""
        import sys
        from fastapi.testclient import TestClient
        from molonari_twin.server.app import create_app
        from molonari_twin.tasks.local import LocalTaskRunner
        runner = LocalTaskRunner({str(tmp_path / "tasks.sqlite")!r}, poll_s=0.02)
        client = TestClient(create_app(runner, {str(twins)!r}))
        task = runner.submit("diag.sleep", {{"blocks": 2, "block_s": 0.01}})
        assert runner.wait(task.id, 30).state == "succeeded"
        assert client.get(f"/v1/tasks/{{task.id}}").json()["state"] == "succeeded"
        runner.close()
        print(sorted({{m.split(".")[0] for m in sys.modules}} & {{"jax", "jaxlib", "arviz", "PySide6"}}))
    """)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


@pytest.mark.slow
def test_running_a_real_twin_through_the_api(tmp_path, twins):
    """Un vrai twin (mode direct) calculé par un worker : même twin_id que la ligne de commande, et JAX n'est
    chargé que dans le worker."""
    expected = subprocess.run([sys.executable, "-m", "molonari_twin", "id", str(twins / "twin_01")],
                              capture_output=True, text=True, timeout=300, check=True).stdout.split()[0]
    code = textwrap.dedent(f"""
        import json, sys
        from fastapi.testclient import TestClient
        from molonari_twin.server.app import create_app
        from molonari_twin.tasks.local import LocalTaskRunner
        runner = LocalTaskRunner({str(tmp_path / "tasks.sqlite")!r}, poll_s=0.05)
        client = TestClient(create_app(runner, {str(twins)!r}))
        task_id = client.post("/v1/twins/twin_01/run").json()["task_id"]
        runner.wait(task_id, 600)
        body = client.get(f"/v1/tasks/{{task_id}}/result").json()
        runner.close()
        print(json.dumps({{"body": body, "jax": "jax" in sys.modules}}))
    """)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900, check=False)
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout.strip().splitlines()[-1])
    body = output["body"]
    assert body["state"] == "succeeded", body
    assert body["result"]["twin_id"] == expected
    assert body["result"]["computed"] is True
    assert Path(body["result"]["results_dir"], "metadata.json").is_file()
    assert output["jax"] is False
    assert not list((tmp_path / "results").glob("*.partial"))
