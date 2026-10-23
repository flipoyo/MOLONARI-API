"""TwinAPI et catalogue des questions, sans HTTP : catalogue cohérent, paramètres validés, erreurs, réponses
sur de vrais twins calculés, cache des résultats."""

import re
import shutil
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from molonari_twin.api import CATALOG, TwinAPI
from molonari_twin.errors import UserError
from molonari_twin.i18n import load_catalog
from molonari_twin.questions import Answer
from molonari_twin.twin.results import load_results
from molonari_twin.twin.run import twin_status

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "twins"
SEMVER = re.compile(r"^\d+\.\d+\.\d+$")
# Listes 1.0.0 du plan (PLAN_TWIN_v2, section 2) : une question du catalogue a l'une de ces références.
PLAN_1_0_0 = {f"HN-{n:02d}" for n in (1, 2, 10, 11, 12, 20, 30, 33, 40, 41, 42, 43)} | {
    f"BN-{n:02d}" for n in (1, 2, 3, 4, 10, 11, 12, 13, 15, 20, 21, 22, 30, 31, 32, 34, 36, 40, 43, 60)}


@pytest.fixture
def twins(computed_twins):
    return computed_twins


@pytest.fixture
def api(twins):
    return TwinAPI(twins)


def code_of(call):
    with pytest.raises(UserError) as error:
        call()
    return error.value


# Catalogue ----------------------------------------------------------------------------------------


def test_catalog_is_consistent():
    ids, refs = [q.id for q in CATALOG], [q.ref for q in CATALOG]
    assert len(set(ids)) == len(ids) == 9 and len(set(refs)) == len(refs)
    french = load_catalog("fr")
    for question in CATALOG:
        assert re.fullmatch(r"[a-z_]+", question.id)
        assert question.ref in PLAN_1_0_0, question.ref
        assert SEMVER.match(question.since)
        assert set(question.modes) <= {"D", "M"} and question.modes
        assert question.requires in ("config", "results")
        assert question.label in french, question.label  # libellé traduit
        assert question.description.endswith("."), question.id
        assert question.level == ("high" if question.ref.startswith("HN-") else "low")


@pytest.mark.parametrize("question", CATALOG, ids=lambda q: q.id)
def test_every_parameter_schema_is_valid_json_schema(question):
    schema = question.params.model_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema.get("additionalProperties") is False
    for name, field in question.params.model_fields.items():
        assert field.description, f"{question.id}.{name} sans description"


@pytest.mark.parametrize("question", CATALOG, ids=lambda q: q.id)
def test_every_question_refuses_an_unknown_parameter(api, question):
    error = code_of(lambda: api.parameters(question, {"bogus": 1}))
    assert error.code == "E_QUESTION_PARAMS" and error.params["question"] == question.id
    assert [(i.code, i.field) for i in error.params["issues"]] == [("E_PARAM_INVALID", "bogus")]


@pytest.mark.parametrize("question_id, params, field", [
    ("water_flux", {"start": "2016-07-02T00:00:00", "end": "2016-07-01T00:00:00"}, None),
    ("water_flux", {"start": "2016-07-01T00:00:00+02:00"}, "start"),
    ("water_flux", {"start": "not a date"}, "start"),
    ("exchanged_volume", {"area": -1}, "area"),
    ("temperature", {}, None),
    ("temperature", {"z": -0.1}, "z"),
    ("sensor_intervals", {"level": 1.5}, "level"),
    ("sensor_intervals", {"sigma": -0.1}, "sigma"),
    ("residuals", {"level": 0}, "level"),
    ("chain_data", {"thin": 0}, "thin"),
    ("chain_data", {"chains": [0, -1]}, "chains"),
])
def test_values_outside_their_bounds_are_refused(api, question_id, params, field):
    error = code_of(lambda: api.parameters(api.question(question_id), params))
    assert error.code == "E_QUESTION_PARAMS"
    assert [i.field for i in error.params["issues"]] == [field]


def test_question_by_id_or_plan_reference(api):
    assert api.question("exchanged_volume") is api.question("HN-12")
    assert code_of(lambda: api.question("HN-99")).code == "E_QUESTION_UNKNOWN"


def test_questions_of_a_twin_follow_its_mode(api):
    direct = {q.id for q in api.questions("twin_01")}
    mcmc = {q.id for q in api.questions("twin_03")}
    assert mcmc == {q.id for q in CATALOG}
    assert direct == mcmc - {"chain_data", "sampling_layout"}


# Erreurs ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["twin_99", "../twin_01", ".hidden", "twin_01/"])
def test_unknown_twin(api, ref):
    error = code_of(lambda: api.ask(ref, "describe"))
    assert error.code == "E_TWIN_NOT_FOUND" and error.params["ref"] == ref


def test_question_of_the_wrong_mode(api):
    error = code_of(lambda: api.ask("twin_01", "chain_data"))
    assert error.code == "E_QUESTION_NOT_AVAILABLE" and error.params == {"question": "chain_data", "ref": "twin_01", "mode": "direct"}


def test_twin_without_results(api):
    assert api.ask("twin_09", "describe").data["name"].value == "twin_09"  # la configuration suffit
    assert code_of(lambda: api.ask("twin_09", "water_flux")).code == "E_TWIN_NOT_COMPUTED"


def test_twin_modified_after_its_run_is_frozen(tmp_path, twins):
    folder = tmp_path / "twins"
    shutil.copytree(twins, folder)
    shutil.copytree(twins.parent / "results", tmp_path / "results")
    yaml = folder / "twin_01" / "twin.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8").replace("nb_cells: 100", "nb_cells: 90"), encoding="utf-8")
    assert "nb_cells: 90" in yaml.read_text(encoding="utf-8")
    assert code_of(lambda: TwinAPI(folder).ask("twin_01", "water_flux")).code == "E_TWIN_FROZEN"


# Réponses -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("ref", ["twin_01", "twin_03"])
def test_every_applicable_question_answers(api, ref):
    status = twin_status(api.twins_folder / ref)
    for question in api.questions(ref):
        params = {"z": 0.2} if question.id == "temperature" else {}
        answer = api.ask(ref, question.id, params)
        assert isinstance(answer, Answer) and answer.data, question.id
        assert answer.provenance["question"] == question.id and answer.provenance["ref"] == question.ref
        assert answer.provenance["twin"] == ref
        if question.requires == "results":
            assert answer.provenance["twin_id"] == status.twin_id
            assert answer.provenance["results"] == str(status.results_dir)


def test_answers_match_the_question_functions(api):
    from molonari_twin.questions import twin_exchanged_volume, water_flux

    results = load_results(twin_status(api.twins_folder / "twin_01").results_dir)
    start = "2016-06-28T00:00:00"
    expected = water_flux(results, start=np.datetime64(start)).data["mean"].value
    assert api.ask("twin_01", "water_flux", {"start": start}).data["mean"].value == expected
    expected = twin_exchanged_volume(results, area=2.0).data["net"].value
    assert api.ask("twin_01", "HN-12", {"area": 2.0}).data["net"].value == expected


def test_measurement_interval_in_direct_mode_is_1_645_sigma(api):
    data = api.ask("twin_01", "sensor_intervals", {"level": 0.9}).data
    np.testing.assert_allclose(data["measured"].interval[1] - data["measured"].value, 1.645 * 0.1, rtol=1e-3)
    assert "predictive" not in data


def test_mcmc_sensor_intervals_have_both_bands(api):
    data = api.ask("twin_03", "sensor_intervals").data
    assert {"credible", "predictive", "coverage"} <= set(data)
    assert 0.0 <= data["coverage"].value <= 1.0


def test_sampling_layout_and_chain_data_count_the_same_samples(api):
    layout = api.ask("twin_03", "sampling_layout").data
    assert layout["samples"].value == layout["nb_chains"].value * layout["nb_iter"].value // layout["thin"].value == 8 * 40
    chains = api.ask("twin_03", "chain_data").data
    assert chains["energy"].value.size == layout["samples"].value
    assert api.ask("twin_03", "chain_data", {"chains": [0, 2], "thin": 2}).data["energy"].value.shape == (20, 2)


def test_results_are_cached_until_they_change(api, monkeypatch):
    import molonari_twin.twin.results as results_module

    calls = []
    original = results_module.load_results
    monkeypatch.setattr(results_module, "load_results", lambda folder: calls.append(folder) or original(folder))
    for _ in range(3):
        api.ask("twin_01", "water_flux")
    assert len(calls) == 1
    metadata = Path(api.ask("twin_01", "provenance").provenance["results"]) / "metadata.json"
    stat = metadata.stat()
    import os

    os.utime(metadata, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))  # résultats recalculés
    api.ask("twin_01", "water_flux")
    assert len(calls) == 2


def test_asking_every_question_loads_neither_jax_nor_arviz(twins):
    """Les questions lisent des résultats : le processus qui les pose (l'API) ne charge pas le moteur."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import sys
        from molonari_twin.api import TwinAPI
        api = TwinAPI({str(twins)!r})
        for ref in ("twin_01", "twin_03"):
            for question in api.questions(ref):
                api.ask(ref, question.id, {{"z": 0.2}} if question.id == "temperature" else {{}})
        print(sorted({{n.split(".")[0] for n in sys.modules}} & {{"jax", "jaxlib", "arviz", "PySide6", "fastapi"}}))
    """)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() == "[]"


def test_declared_outputs_are_the_answers(api):
    """La documentation (docs/API_V1.md) liste les sorties déclarées : elles doivent être celles des réponses.
    Toute sortie apparaît pour au moins un des twins D et M ; aucune réponse n'a de sortie non déclarée."""
    seen = {q.id: set() for q in CATALOG}
    for ref in ("twin_01", "twin_03"):
        for question in api.questions(ref):
            params = {"temperature": {"z": 0.2}, "residuals": {"intervals": True}}.get(question.id, {})
            answer = api.ask(ref, question.id, params)
            assert set(answer.data) <= set(question.outputs), (question.id, set(answer.data) - set(question.outputs))
            seen[question.id] |= set(answer.data)
    for question in CATALOG:
        assert seen[question.id] == set(question.outputs), (question.id, set(question.outputs) - seen[question.id])
