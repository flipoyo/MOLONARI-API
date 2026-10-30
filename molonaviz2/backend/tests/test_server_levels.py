"""Niveaux d'accès de l'API HTTP (T11) : `create_app(level=…)` et `serve --level`. Au niveau `user`, seules les cinq questions
de l'eau, de la chaleur et de la température répondent, par liste blanche ; `run` et `/v1/tasks/*` sont refusés (403).
Sans authentification : le niveau oriente l'usage, ce n'est pas une sécurité."""

import io
import json

import pytest

pytest.importorskip("fastapi")
import numpy as np
from fastapi.testclient import TestClient
from molonari_twin.api import CATALOG
from molonari_twin.contract.codec import answer_from_npz
from molonari_twin.server.app import create_app, http_status
from molonari_twin.twin.cli import main
from molonari_twin.twin.config import load_twin
from molonari_twin.twin.run import twin_status
from test_server import FakeTaskRunner, assert_error

FIVE = ["water_flux", "exchange_direction", "exchanged_volume", "heat_budget", "temperature"]
PARAMS = {"temperature": {"z": 0.2}}
FORBIDDEN_KEYS = {"seed", "samples", "twin", "twin_id", "results", "point", "name", "path", "folder", "repositories"}


@pytest.fixture
def fake():
    return FakeTaskRunner()


@pytest.fixture
def user(fake, computed_twins):
    return TestClient(create_app(fake, computed_twins, level="user"))


@pytest.fixture
def admin(fake, computed_twins):
    return TestClient(create_app(fake, computed_twins))  # le niveau par défaut est admin


@pytest.fixture
def cli(capsys, monkeypatch):
    def call(*args):
        monkeypatch.setenv("MOLONARI_LANG", "en")
        code = main([str(a) for a in args])
        return code, capsys.readouterr().out

    return call


def post(client, ref, question, params=None, npz=False):
    headers = {"Accept": "application/x-npz"} if npz else {}
    return client.post(f"/v1/twins/{ref}/questions/{question}", json=params, headers=headers)


def test_refusal_is_a_403():
    assert http_status("E_ACCESS_ADMIN_ONLY") == 403


# Même JSON que la ligne de commande, à chaque niveau ------------------------------------------------------------


@pytest.mark.parametrize("level", ["user", "admin"])
@pytest.mark.parametrize("ref", [None, "twin_01", "twin_03"])
def test_the_list_of_questions_is_the_json_of_the_command(fake, computed_twins, cli, level, ref):
    client = TestClient(create_app(fake, computed_twins, level=level))
    code, out = cli("questions", *([computed_twins / ref] if ref else []), "--level", level, "--json")
    assert code == 0
    response = client.get("/v1/questions", params={"twin": ref} if ref else None)
    assert json.loads(out) == response.json()
    if ref is None:
        assert len(response.json()["questions"]) == (5 if level == "user" else 23)


@pytest.mark.parametrize("level", ["user", "admin"])
@pytest.mark.parametrize("ref, question, params", [
    ("twin_01", "water_flux", None),
    ("twin_03", "water_flux", {"level": 0.8}),
    ("twin_03", "exchange_direction", None),
    ("twin_01", "exchanged_volume", {"area": 2.0}),
    ("twin_03", "heat_budget", {"z": 0.1}),
    ("twin_03", "temperature", {"z": 0.2, "t": "2016-07-01T12:00"}),
    ("twin_01", "temperature", {"t": "2016-07-01T12:00"}),
    ("twin_03", "describe", None),  # refusée au niveau user : même JSON d'erreur
    ("twin_03", "sensor_intervals", None),
    ("twin_03", "temperature", None),  # z ou t obligatoire, aux deux niveaux
    ("twin_09", "water_flux", None),
])
def test_an_answer_is_the_json_of_the_command(fake, computed_twins, cli, level, ref, question, params):
    client = TestClient(create_app(fake, computed_twins, level=level))
    code, out = cli("ask", computed_twins / ref, question, "--level", level, "--json", *(["--params", json.dumps(params)] if params else []))
    response = post(client, ref, question, params)
    assert json.loads(out) == response.json()
    assert code == (0 if response.status_code == 200 else 1)


# Niveau user : tout le catalogue --------------------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["twin_01", "twin_03"])
@pytest.mark.parametrize("question", [q.id for q in CATALOG])
def test_only_the_five_questions_answer_at_the_user_level(user, admin, computed_twins, ref, question):
    config, twin_id = load_twin(computed_twins / ref), twin_status(computed_twins / ref).twin_id
    if question not in FIVE:
        body = assert_error(post(user, ref, question, PARAMS.get(question)), 403, "E_ACCESS_ADMIN_ONLY")
        assert body["params"] == {"name": question}
        assert_error(user.post(f"/v1/questions/{question}"), 403, "E_ACCESS_ADMIN_ONLY")  # question globale
        assert_error(post(user, ref, question, PARAMS.get(question), npz=True), 403, "E_ACCESS_ADMIN_ONLY")
        return
    response = post(user, ref, question, PARAMS.get(question))
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["provenance"]) <= {"question", "ref", "parameters"} and body["provenance"]["question"] == question
    assert not FORBIDDEN_KEYS & set(body["data"]) and not FORBIDDEN_KEYS & set(body["provenance"]["parameters"])
    full = post(admin, ref, question, PARAMS.get(question)).json()
    assert set(body["data"]) <= set(full["data"])
    for forbidden in (str(computed_twins), twin_id, config.point):
        assert forbidden not in response.text
    # la négociation npz filtre de la même façon
    binary = post(user, ref, question, PARAMS.get(question), npz=True)
    assert binary.status_code == 200 and binary.headers["content-type"] == "application/x-npz"
    answer = answer_from_npz(binary.content)
    assert set(answer.data) == set(body["data"]) and set(answer.provenance) == set(body["provenance"])
    with np.load(io.BytesIO(binary.content), allow_pickle=False) as archive:
        content = str(archive["meta"]) + " ".join(archive.files)
    for forbidden in (str(computed_twins), twin_id, config.point, "samples", "seed"):
        assert forbidden not in content
    # posée au point d'entrée des questions globales, une de ces questions (qui porte sur un twin) reçoit la même erreur qu'un admin
    assert_error(user.post(f"/v1/questions/{question}", json=PARAMS.get(question)), 409, "E_QUESTION_TWIN_REQUIRED")


def test_global_questions_are_refused_at_the_user_level(user, admin):
    for question in ("parameters_catalog", "raw_series", "processing_preview"):
        assert_error(user.post(f"/v1/questions/{question}"), 403, "E_ACCESS_ADMIN_ONLY")
    assert admin.post("/v1/questions/parameters_catalog").status_code == 200
    assert_error(user.post("/v1/questions/nothing"), 404, "E_QUESTION_UNKNOWN")


def test_run_and_tasks_are_refused_at_the_user_level(user, fake):
    assert_error(user.post("/v1/twins/twin_01/run"), 403, "E_ACCESS_ADMIN_ONLY")
    assert_error(user.post("/v1/twins/twin_01/run", json={"strict": True}), 403, "E_ACCESS_ADMIN_ONLY")
    assert fake.submitted == [] and fake.records == {}
    for call in (user.get("/v1/tasks/t1"), user.get("/v1/tasks/t1/result"), user.get("/v1/tasks/t1/events"), user.delete("/v1/tasks/t1")):
        body = assert_error(call, 403, "E_ACCESS_ADMIN_ONLY")
        assert body["params"] == {"name": "tasks"}


def test_the_admin_level_runs_a_twin_and_follows_the_task(admin, fake):
    response = admin.post("/v1/twins/twin_01/run")
    assert response.status_code == 202 and fake.submitted
    assert admin.get(f"/v1/tasks/{response.json()['task_id']}").status_code == 200


def test_info_stays_available_at_the_user_level(user, admin):
    assert user.get("/v1/info").status_code == 200 and user.get("/v1/info").json() == admin.get("/v1/info").json()


def test_the_other_errors_are_unchanged_at_the_user_level(user):
    assert_error(post(user, "twin_01", "temperature"), 422, "E_QUESTION_PARAMS")
    assert_error(post(user, "twin_01", "no_such_question"), 404, "E_QUESTION_UNKNOWN")
    assert_error(post(user, "twin_99", "water_flux"), 404, "E_TWIN_NOT_FOUND")
    assert_error(post(user, "twin_09", "water_flux"), 409, "E_TWIN_NOT_COMPUTED")
    assert_error(user.get("/v1/questions", params={"twin": "twin_99"}), 404, "E_TWIN_NOT_FOUND")


# serve --level -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("flags, listed", [([], 5), (["--level", "user"], 5), (["--level", "admin"], 23)])
def test_serve_gives_its_level_to_the_api(monkeypatch, tmp_path, capsys, computed_twins, flags, listed):
    import uvicorn
    from molonari_twin.tasks.local import LocalTaskRunner

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **options: calls.append(app))
    monkeypatch.setattr(LocalTaskRunner, "close", lambda self: None)
    monkeypatch.setenv("MOLONARI_LANG", "en")
    assert main(["serve", "--local", "--twins-folder", str(computed_twins), "--tasks-db", str(tmp_path / "t.sqlite"), *flags]) == 0
    [app] = calls
    assert len(TestClient(app).get("/v1/questions").json()["questions"]) == listed
    captured = capsys.readouterr()
    assert captured.out.startswith("Local server: http://127.0.0.1:8765")
    assert f"Access level: {flags[-1] if flags else 'user'}" in captured.err
