"""API HTTP /v1 : contrat des endpoints (faux runner en mémoire), intégration avec LocalTaskRunner, et calcul
jamais exécuté dans le processus de l'API."""

import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from molonari_twin import __version__
from molonari_twin.contract import API_VERSION
from molonari_twin.errors import UserError
from molonari_twin.server.app import create_app
from molonari_twin.tasks.local import LocalTaskRunner
from molonari_twin.tasks.model import TaskEvent, TaskRecord, now
from molonari_twin.twin.results import RESULTS_FORMAT

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
                                             "runner": "fake", "max_workers": 2, "twin_schema": "molonari-twin/1.1.0",
                                             "results_format": RESULTS_FORMAT, "data_release": None, "device": None}


def test_info_gives_the_device_once_the_diagnostic_task_succeeded(fake, twins):
    client = TestClient(create_app(fake, twins, device_task=fake.submit("diag.device", {}).id))
    assert client.get("/v1/info").json()["device"] is None  # en file
    fake.finish("t1", "succeeded", {"device": "CPU · float64"})
    assert client.get("/v1/info").json()["device"] == "CPU · float64"
    assert TestClient(create_app(fake, twins, device_task="t99")).get("/v1/info").json()["device"] is None  # tâche inconnue


def test_info_leaves_the_device_empty_when_the_diagnostic_task_failed(fake, twins):
    client = TestClient(create_app(fake, twins, device_task=fake.submit("diag.device", {}).id))
    fake.finish("t1", "failed")
    assert client.get("/v1/info").json()["device"] is None


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
    assert set(paths) == {"/v1/info", "/v1/questions", "/v1/questions/{question_id}", "/v1/twins/{ref}/questions/{question_id}", "/v1/twins/{ref}/run",
                          "/v1/tasks/{task_id}", "/v1/tasks/{task_id}/result", "/v1/tasks/{task_id}/events"}
    assert set(paths["/v1/tasks/{task_id}"]) == {"get", "delete"}


# Intégration avec LocalTaskRunner -----------------------------------------------------------------


@pytest.fixture
def local(tmp_path, twins):
    runner = LocalTaskRunner(tmp_path / "tasks.sqlite", cancel_grace_s=1.0, poll_s=0.02)
    yield runner, TestClient(create_app(runner, twins, events_poll_s=0.02))
    runner.close()


def test_device_is_filled_in_by_a_worker_that_loads_jax(tmp_path, twins):
    runner = LocalTaskRunner(tmp_path / "tasks.sqlite", poll_s=0.02)
    try:
        task = runner.submit("diag.device", {})
        client = TestClient(create_app(runner, twins, device_task=task.id))
        assert client.get("/v1/info").json()["device"] in (None, "CPU · float64")  # le worker démarre
        assert runner.wait(task.id, timeout_s=120).state == "succeeded"
        info = client.get("/v1/info").json()
        assert info["device"].endswith("float64") and info["twin_schema"] == "molonari-twin/1.1.0"
    finally:
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


# Questions (lot C2) -------------------------------------------------------------------------------

OPENAPI_SNAPSHOT = Path(__file__).parent / "fixtures" / "openapi_v1.json"


@pytest.fixture
def asking(computed_twins):
    return TestClient(create_app(FakeTaskRunner(), computed_twins))


def ask(client, ref, question, params=None, npz=False):
    headers = {"Accept": "application/x-npz"} if npz else {}
    return client.post(f"/v1/twins/{ref}/questions/{question}", json=params, headers=headers)


def test_question_catalog(asking):
    body = asking.get("/v1/questions").json()
    assert body["api_version"] == API_VERSION
    assert [q["id"] for q in body["questions"]] == ["describe", "water_flux", "exchange_direction", "exchanged_volume",
                                                     "heat_budget", "temperature", "sensor_intervals", "fit_quality",
                                                     "reliability", "public_convergence", "provenance", "residuals", "chain_data", "sampling_layout",
                                                     "soil_properties", "posterior_samples", "convergence",
                                                     "energy_breakdown", "parameters_catalog", "check", "fields", "raw_series",
                                                     "processing_preview"]
    volume = next(q for q in body["questions"] if q["id"] == "exchanged_volume")
    assert volume["ref"] == "HN-12" and volume["level"] == "high" and volume["modes"] == ["D", "M"]
    assert volume["label_en"] == "What volume of water crossed 1 m² of riverbed?"
    assert set(volume["params_schema"]["properties"]) == {"start", "end", "area"}
    assert volume["params_schema"]["additionalProperties"] is False
    assert len(asking.get("/v1/questions", params={"twin": "twin_01"}).json()["questions"]) == 15
    assert_error(asking.get("/v1/questions", params={"twin": "twin_99"}), 404, "E_TWIN_NOT_FOUND")


@pytest.mark.parametrize("ref", ["twin_01", "twin_03"])
def test_every_question_answers_in_json_and_npz(asking, ref):
    from molonari_twin.contract.codec import answer_from_json, answer_from_npz
    from molonari_twin.contract.models import AnswerOut

    for question in asking.get("/v1/questions", params={"twin": ref}).json()["questions"]:
        params = {"z": 0.2} if question["id"] == "temperature" else None
        response = ask(asking, ref, question["id"], params)
        assert response.status_code == 200, (question["id"], response.text)
        AnswerOut.model_validate(response.json())
        assert response.json()["provenance"]["twin"] == ref
        binary = ask(asking, ref, question["id"], params, npz=True)
        assert binary.status_code == 200 and binary.headers["content-type"] == "application/x-npz"
        from_json, from_npz = answer_from_json(response.content), answer_from_npz(binary.content)
        assert from_json.data.keys() == from_npz.data.keys()
        for name, quantity in from_json.data.items():
            np.testing.assert_array_equal(np.asarray(from_npz.data[name].value), np.asarray(quantity.value), err_msg=name)


def test_answer_values_through_http(asking):
    body = ask(asking, "twin_01", "HN-12", {"start": "2016-06-28T00:00:00", "area": 2.0}).json()
    assert body["data"]["net"]["unit"] == "m³"
    assert body["provenance"]["parameters"] == {"start": "2016-06-28T00:00:00", "end": None, "area": 2.0}
    assert body["provenance"]["question"] == "exchanged_volume" and body["provenance"]["ref"] == "HN-12"
    temperature = ask(asking, "twin_01", "temperature", {"z": 0.2}).json()["data"]["temperature"]["value"]
    assert temperature["dtype"] == "float64" and len(temperature["shape"]) == 1


@pytest.mark.parametrize("ref, question, params, status, code", [
    ("twin_01", "no_such_question", None, 404, "E_QUESTION_UNKNOWN"),
    ("twin_99", "describe", None, 404, "E_TWIN_NOT_FOUND"),
    ("twin_01", "water_flux", {"bogus": 1}, 422, "E_QUESTION_PARAMS"),
    ("twin_01", "sensor_intervals", {"level": 2}, 422, "E_QUESTION_PARAMS"),
    ("twin_01", "temperature", {"z": 9.0}, 422, "E_QUESTION_OUT_OF_RANGE"),
    ("twin_01", "water_flux", {"start": "2030-01-01T00:00:00"}, 422, "E_QUESTION_PERIOD_EMPTY"),
    ("twin_09", "water_flux", None, 409, "E_TWIN_NOT_COMPUTED"),
    ("twin_01", "chain_data", None, 409, "E_QUESTION_NOT_AVAILABLE"),
])
def test_question_errors(asking, ref, question, params, status, code):
    body = assert_error(ask(asking, ref, question, params), status, code)
    if code == "E_QUESTION_PARAMS":
        assert all(set(issue) == ERROR_KEYS and issue["code"] == "E_PARAM_INVALID" for issue in body["params"]["issues"])


def test_malformed_requests_use_the_error_format(asking, client):
    response = asking.post("/v1/twins/twin_01/questions/describe", content=b"[1, 2]", headers={"Content-Type": "application/json"})
    body = assert_error(response, 422, "E_REQUEST_INVALID")
    assert body["params"]["issues"][0]["code"] == "E_PARAM_INVALID"
    assert_error(client.post("/v1/twins/twin_01/run", json={"strict": True, "bogus": 1}), 422, "E_REQUEST_INVALID")


def test_openapi_matches_the_snapshot(asking):
    """Tout changement du contrat se voit ici. S'il est voulu : régénérer le fichier
    (MOLONARI_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_server.py -k snapshot) et monter API_VERSION."""
    schema = asking.get("/openapi.json").json()
    text = json.dumps(schema, indent=1, ensure_ascii=False, sort_keys=True) + "\n"
    if os.environ.get("MOLONARI_UPDATE_SNAPSHOTS") == "1":
        OPENAPI_SNAPSHOT.write_text(text, encoding="utf-8")
    assert schema["info"]["version"] == API_VERSION
    assert "HTTPValidationError" not in schema["components"]["schemas"]  # nos 422 ont le format ErrorOut
    assert text == OPENAPI_SNAPSHOT.read_text(encoding="utf-8"), "le contrat OpenAPI a changé : voir la docstring"


def test_api_process_answers_questions_without_the_engine(tmp_path, computed_twins):
    from conftest import GLOBAL_PARAMS

    code = textwrap.dedent(f"""
        import sys
        from fastapi.testclient import TestClient
        from molonari_twin.server.app import create_app
        from molonari_twin.tasks.local import LocalTaskRunner
        runner = LocalTaskRunner({str(tmp_path / "tasks.sqlite")!r}, poll_s=0.05)
        client = TestClient(create_app(runner, {str(computed_twins)!r}))
        for ref in ("twin_01", "twin_03"):
            for question in client.get("/v1/questions", params={{"twin": ref}}).json()["questions"]:
                body = {{"z": 0.2}} if question["id"] == "temperature" else None
                assert client.post(f"/v1/twins/{{ref}}/questions/{{question['id']}}", json=body).status_code == 200
        for question, body in {GLOBAL_PARAMS!r}.items():  # BN-03 et BN-04 : les fichiers d'un point, sans twin
            assert client.post(f"/v1/questions/{{question}}", json=body).status_code == 200, question
        runner.close()
        print(sorted({{m.split(".")[0] for m in sys.modules}} & {{"jax", "jaxlib", "arviz", "PySide6"}}))
    """)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() == "[]"


def test_questions_without_a_twin_have_their_own_endpoint(computed_twins, monkeypatch):
    import dataclasses

    from molonari_twin.api import CATALOG, service
    from molonari_twin.questions import Answer, Quantity

    ping = dataclasses.replace(CATALOG[0], id="ping", ref="BN-10", requires="none", modes=("D", "M"),
                               run=lambda twin, _: Answer(data={"pong": Quantity(1, "1")}, provenance={"parameters": {}}))
    monkeypatch.setattr(service, "CATALOG", (*CATALOG, ping))
    client = TestClient(create_app(FakeTaskRunner(), computed_twins))
    body = client.post("/v1/questions/ping").json()
    assert body["data"]["pong"]["value"] == 1 and body["provenance"] == {"parameters": {}, "question": "ping", "ref": "BN-10"}
    assert client.post("/v1/questions/ping", headers={"Accept": "application/x-npz"}).status_code == 200
    assert_error(ask(client, "twin_01", "ping"), 409, "E_QUESTION_NO_TWIN")
    assert_error(client.post("/v1/questions/describe"), 409, "E_QUESTION_TWIN_REQUIRED")
    assert_error(client.post("/v1/questions/nothing"), 404, "E_QUESTION_UNKNOWN")
    assert_error(client.post("/v1/questions/ping", json={"bogus": 1}), 422, "E_QUESTION_PARAMS")
    assert "ping" in [q["id"] for q in client.get("/v1/questions").json()["questions"]]
    assert "ping" not in [q["id"] for q in client.get("/v1/questions", params={"twin": "twin_01"}).json()["questions"]]


def test_parameters_catalog_has_its_endpoint(asking):
    response = asking.post("/v1/questions/parameters_catalog")
    assert response.status_code == 200
    body = response.json()
    assert body["provenance"]["question"] == "parameters_catalog" and "twin" not in body["provenance"]
    assert body["data"]["settings.name"]["value"]["data"][0] == "nb_cells"
    assert body["data"]["settings.maximum"]["value"]["data"].count(None) >= 1  # sans borne : null
    assert asking.post("/v1/questions/BN-10", json={}).status_code == 200
    binary = asking.post("/v1/questions/parameters_catalog", headers={"Accept": "application/x-npz"})
    assert binary.status_code == 200 and binary.headers["content-type"] == "application/x-npz"
    assert_error(ask(asking, "twin_01", "parameters_catalog"), 409, "E_QUESTION_NO_TWIN")
    parameters = next(q for q in asking.get("/v1/questions").json()["questions"] if q["id"] == "parameters_catalog")
    assert parameters["requires"] == "none" and parameters["ref"] == "BN-10" and parameters["level"] == "low"


def test_data_questions_have_their_endpoint(asking):
    from conftest import GLOBAL_PARAMS
    from molonari_twin.contract.codec import answer_from_json, answer_from_npz

    for question, params in GLOBAL_PARAMS.items():
        response = asking.post(f"/v1/questions/{question}", json=params)
        assert response.status_code == 200, response.text
        binary = asking.post(f"/v1/questions/{question}", json=params, headers={"Accept": "application/x-npz"})
        assert binary.status_code == 200 and binary.headers["content-type"] == "application/x-npz"
        from_json, from_npz = answer_from_json(response.content), answer_from_npz(binary.content)
        assert from_json.data.keys() == from_npz.data.keys() and "twin" not in response.json()["provenance"]
        for name, quantity in from_json.data.items():
            np.testing.assert_array_equal(np.asarray(from_npz.data[name].value), np.asarray(quantity.value), err_msg=name)
    assert asking.post("/v1/questions/BN-03", json=GLOBAL_PARAMS["raw_series"]).status_code == 200
    questions = {q["id"]: q for q in asking.get("/v1/questions").json()["questions"]}
    assert questions["processing_preview"]["requires"] == "none" and questions["raw_series"]["ref"] == "BN-03"
    assert "methods" in questions["processing_preview"]["params_schema"]["properties"]


def test_data_questions_refuse_with_structured_errors(asking, tmp_path):
    from conftest import GLOBAL_PARAMS

    (tmp_path / "t.xlsx").write_bytes(b"PK")
    body = assert_error(asking.post("/v1/questions/raw_series", json={"point_folder": str(tmp_path)}), 400, "E_PIPELINE_FORMAT_UNSUPPORTED")
    assert body["params"]["files"] == "t.xlsx"
    assert_error(asking.post("/v1/questions/raw_series", json={"point_folder": "/nowhere/at/all"}), 400, "E_FOLDER_NOT_FOUND")
    zone = {**GLOBAL_PARAMS["raw_series"], "start": "2016-06-27T12:00:00+01:00"}
    body = assert_error(asking.post("/v1/questions/raw_series", json=zone), 422, "E_QUESTION_PARAMS")
    assert [issue["field"] for issue in body["params"]["issues"]] == ["start"]
    assert_error(asking.post("/v1/questions/raw_series", json={}), 422, "E_QUESTION_PARAMS")


def test_check_of_an_invalid_twin_is_a_200(computed_twins, tmp_path):
    shutil.copytree(computed_twins, tmp_path / "twins")
    (tmp_path / "twins" / "twin_09" / "twin.yaml").write_text("schema: [", encoding="utf-8")
    client = TestClient(create_app(FakeTaskRunner(), tmp_path / "twins"))
    response = ask(client, "twin_09", "check")
    assert response.status_code == 200
    body = response.json()
    assert body["data"]["valid"]["value"] is False
    assert body["data"]["issue_codes"]["value"]["data"] == ["E_TWIN_INVALID_YAML"]
    assert [w["code"] for w in body["warnings"]] == ["E_TWIN_INVALID_YAML"]
    assert ask(client, "twin_99", "check").status_code == 404


def test_fields_as_npz_are_the_size_of_the_arrays(asking):
    from molonari_twin.contract.codec import answer_from_npz

    params = {"variables": ["temperature", "water_flux"]}
    binary = ask(asking, "twin_01", "fields", params, npz=True)
    assert binary.status_code == 200
    answer = answer_from_npz(binary.content)
    nz, nt = answer.data["temperature"].value.shape
    assert (nz, nt) == (answer.data["depth"].value.size, answer.data["time"].value.size)
    raw = 2 * nz * nt * 8  # deux champs en float64
    assert answer.data["temperature"].value.nbytes + answer.data["water_flux"].value.nbytes == raw
    assert 0.3 * raw < len(binary.content) <= raw * 1.1  # archive compressée : de l'ordre de la taille brute
    as_json = ask(asking, "twin_01", "fields", params)
    assert as_json.status_code == 200 and len(as_json.content) > len(binary.content)  # le JSON est plus lourd
    assert_error(ask(asking, "twin_01", "fields", {"variables": ["pressure"]}), 422, "E_QUESTION_VARIABLE")
