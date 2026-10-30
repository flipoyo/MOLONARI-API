"""Commandes `molonari-twin questions` et `ask` : une couche mince sur TwinAPI. `--json` donne le JSON de /v1 (même
contenu que GET /v1/questions et POST /v1/twins/{ref}/questions/{question}), sans FastAPI."""

import json
import subprocess
import sys
import textwrap

import pytest
from molonari_twin.api import LEVELS
from molonari_twin.twin import cli
from molonari_twin.twin.cli import main

FIVE = ["water_flux", "exchange_direction", "exchanged_volume", "heat_budget", "temperature"]


@pytest.fixture
def run(capsys, monkeypatch):
    """run(*args, lang="en") → (code de sortie, sortie standard, sortie d'erreur) de la ligne de commande, dans ce processus."""

    def call(*args, lang="en"):
        monkeypatch.setenv("MOLONARI_LANG", lang)
        code = main([str(a) for a in args])
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return call


def test_the_levels_of_the_parser_are_those_of_the_api():
    assert cli.LEVELS == LEVELS


# Même JSON que l'API ------------------------------------------------------------------------------------------


@pytest.fixture
def client(computed_twins):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from molonari_twin.server.app import create_app
    from test_server import FakeTaskRunner

    return TestClient(create_app(FakeTaskRunner(), computed_twins))


@pytest.mark.parametrize("ref", [None, "twin_01", "twin_03"])
def test_questions_json_is_the_json_of_the_api(run, computed_twins, client, ref):
    args = ["questions", *([computed_twins / ref] if ref else []), "--level", "admin", "--json"]
    code, out, _ = run(*args)
    assert code == 0
    assert json.loads(out) == client.get("/v1/questions", params={"twin": ref} if ref else None).json()


@pytest.mark.parametrize("ref, question, params", [
    ("twin_01", "describe", None),
    ("twin_03", "water_flux", {"level": 0.8}),
    ("twin_03", "exchanged_volume", {"area": 2.0}),
    ("twin_01", "heat_budget", {"z": 0.2}),
    ("twin_03", "temperature", {"z": 0.2, "t": "2016-07-01T12:00"}),
    ("twin_03", "temperature", {"z": 0.2}),
    ("twin_03", "fit_quality", None),
    ("twin_03", "HN-10", None),  # la référence du plan vaut l'identifiant
])
def test_ask_json_is_the_json_of_the_api(run, computed_twins, client, ref, question, params):
    args = ["ask", computed_twins / ref, question, "--level", "admin", "--json"]
    code, out, _ = run(*args, *(["--params", json.dumps(params)] if params else []))
    assert code == 0
    assert json.loads(out) == client.post(f"/v1/twins/{ref}/questions/{question}", json=params).json()


def test_a_twin_yaml_path_is_accepted(run, computed_twins):
    code, out, _ = run("ask", computed_twins / "twin_01" / "twin.yaml", "exchanged_volume", "--json")
    assert code == 0 and json.loads(out)["data"]["net"]["unit"] == "m³/m²"


# Niveaux --------------------------------------------------------------------------------------------------------


def test_user_is_the_default_level(run, computed_twins):
    code, out, _ = run("questions", "--json")
    assert code == 0 and [q["id"] for q in json.loads(out)["questions"]] == FIVE
    code, out, _ = run("questions", computed_twins / "twin_03", "--level", "user", "--json")
    assert [q["id"] for q in json.loads(out)["questions"]] == FIVE
    code, out, _ = run("questions", "--level", "admin", "--json")
    assert len(json.loads(out)["questions"]) == 23


def test_the_user_level_refuses_the_other_questions_with_the_json_error_of_the_api(run, computed_twins):
    code, out, err = run("ask", computed_twins / "twin_03", "sensor_intervals", "--json")
    assert code == 1 and err == ""
    body = json.loads(out)
    assert set(body) == {"code", "field", "params", "message_en"}
    assert body["code"] == "E_ACCESS_ADMIN_ONLY" and body["params"] == {"name": "sensor_intervals"}
    code, out, _ = run("ask", computed_twins / "twin_03", "sensor_intervals", "--level", "admin", "--json")
    assert code == 0 and "measured" in json.loads(out)["data"]


def test_the_user_answer_has_neither_seed_nor_path_nor_twin(run, computed_twins):
    code, out, _ = run("ask", computed_twins / "twin_03", "water_flux", "--json")
    body = json.loads(out)
    assert code == 0 and set(body["provenance"]) == {"question", "ref", "parameters"} and "samples" not in body["data"]
    assert str(computed_twins) not in out


# Erreurs --------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("params, code_expected", [("{z", "E_REQUEST_INVALID"), ("[1, 2]", "E_REQUEST_INVALID"), ('{"bogus": 1}', "E_QUESTION_PARAMS"),
                                                   ('{"z": 9}', "E_QUESTION_OUT_OF_RANGE")])
def test_errors_with_json_are_the_json_error_of_the_api(run, computed_twins, params, code_expected):
    code, out, err = run("ask", computed_twins / "twin_03", "temperature", "--params", params, "--json")
    assert code == 1 and err == ""
    assert json.loads(out)["code"] == code_expected


def test_invalid_json_is_a_coded_error_without_a_traceback(run, computed_twins):
    code, out, err = run("ask", computed_twins / "twin_03", "temperature", "--params", "{z", lang="fr")
    assert code == 1 and out == "" and "Traceback" not in err
    assert err.startswith("Requête invalide")


@pytest.mark.parametrize("args, code_expected", [
    (["ask", "{twins}/twin_99", "water_flux"], "E_TWIN_NOT_FOUND"),
    (["ask", "{twins}/twin_09", "water_flux"], "E_TWIN_NOT_COMPUTED"),
    (["ask", "{twins}/twin_01", "no_such_question"], "E_QUESTION_UNKNOWN"),
    (["ask", "{twins}/twin_01", "temperature"], "E_QUESTION_PARAMS"),  # z ou t obligatoire : refus inchangé
    (["questions", "{twins}/twin_99"], "E_TWIN_NOT_FOUND"),
])
def test_expected_errors_exit_with_code_1(run, computed_twins, args, code_expected):
    args = [a.format(twins=computed_twins) for a in args]
    code, out, err = run(*args, "--json")
    assert code == 1 and json.loads(out)["code"] == code_expected
    code, out, err = run(*args)
    assert code == 1 and out == "" and err.strip() and "Traceback" not in err


# Texte lisible ----------------------------------------------------------------------------------------------------


def test_the_text_shows_label_values_units_and_warnings(run, computed_twins):
    code, out, _ = run("ask", computed_twins / "twin_01", "heat_budget", "--params", '{"z": 0.2}', lang="fr")
    assert code == 0
    assert out.splitlines()[0] == "Quels processus transportent la chaleur ? (heat_budget, HN-20)"
    assert "advective:" in out and "W/m²" in out and "dominant: " in out
    assert "Paramètres : z = 0.2" in out
    assert "Avertissements :" in out and "latéral" in out.lower()
    code, out, _ = run("ask", computed_twins / "twin_01", "heat_budget", "--params", '{"z": 0.2}', lang="en")
    assert out.splitlines()[0] == "Which processes carry the heat? (heat_budget, HN-20)" and "Warnings:" in out


def test_the_text_gives_the_interval_of_a_mcmc_flux(run, computed_twins):
    _, out, _ = run("ask", computed_twins / "twin_03", "water_flux", lang="fr")
    line = next(line for line in out.splitlines() if line.startswith("  mean:"))
    assert "m/s" in line and "intervalle" in line


def test_a_long_series_is_summarised(run, computed_twins):
    code, out, _ = run("ask", computed_twins / "twin_01", "temperature", "--params", '{"z": 0.2}', lang="fr")
    assert code == 0 and len(out.splitlines()) < 12
    line = next(line for line in out.splitlines() if line.startswith("  temperature:"))
    assert " valeurs, de " in line and line.endswith("°C")
    code, out, _ = run("ask", computed_twins / "twin_01", "temperature", "--params", '{"z": 0.2, "t": "2016-07-01T12:00"}')
    assert next(line for line in out.splitlines() if line.startswith("  temperature:")).endswith("°C")  # une valeur
    assert "values" not in next(line for line in out.splitlines() if line.startswith("  temperature:"))


def test_the_list_of_questions_is_readable(run, computed_twins):
    code, out, _ = run("questions", computed_twins / "twin_03", lang="fr")
    lines = out.splitlines()
    assert code == 0 and lines[0] == "Questions du niveau user :" and len(lines) == 6
    assert lines[1].split()[:3] == ["water_flux", "HN-10", "D"]


# Sans FastAPI ---------------------------------------------------------------------------------------------------

NO_WEB = textwrap.dedent(
    """
    import sys
    for name in ("fastapi", "starlette", "uvicorn"):
        sys.modules[name] = None
    from molonari_twin.twin.cli import main
    for args in (["questions"], ["questions", {twin!r}, "--level", "admin", "--json"],
                 ["ask", {twin!r}, "water_flux", "--json"], ["ask", {twin!r}, "temperature", "--params", '{{"z": 0.2}}']):
        assert main(args) == 0, args
    assert main(["ask", {twin!r}, "residuals", "--json"]) == 1  # refus du niveau user, aussi sans FastAPI
    print("OK", sorted({{m.split(".")[0] for m in sys.modules if sys.modules[m] is not None}} & {{"fastapi", "starlette", "uvicorn"}}))
    """
)


def test_questions_and_ask_work_without_fastapi(computed_twins):
    code = NO_WEB.format(twin=str(computed_twins / "twin_01"))
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600, check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip().endswith("OK []"), completed.stdout[-500:]
