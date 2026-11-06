"""T4 : `generate` (état par le modèle direct, niveau user), `generation.yaml` écrit par `report`, `ask` sur un état généré.

- Les tests rapides construisent à la main un point (`chains.npz`, `generation.yaml`) : erreurs structurées, bornes, sortie.
- Les tests `slow` calculent un vrai point (twin direct et twin mcmc 2.0.0 du dépôt de données de test, `report` publié dans
  `public_reports/`) et vérifient l'état généré contre les champs conservés par le MCMC.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml
from conftest import FIXTURES
from molonari_twin.errors import UserError
from molonari_twin.report import generate_report
from molonari_twin.twin.cli import main as cli_main
from molonari_twin.twin.generate import (
    GENERATION_FILE,
    generate_state,
    is_state,
    lowest_energy_sample,
    read_generation,
)
from molonari_twin.twin.results import load_results
from molonari_twin.twin.run import run_twin

NB_ITER = 12
NB_CHAINS = 7
THIN = 3  # = publication_thin : tous les échantillons conservés sont publiés
PARAMETERS = ("neg_log10_k", "porosity", "lambda_s", "rhos_cs")

DIRECT = """schema: molonari-twin/2.0.0
name: twin_direct
mode: direct
data: {lab: Laboratory, point: SamplingPoints/Point034}
layers:
  - {name: haut, z_low: 0.2, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000}
  - {name: bas, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000}
numerics: {nb_cells: 40}
"""
MCMC = f"""schema: molonari-twin/2.0.0
name: twin_mcmc
mode: mcmc
data: {{lab: Laboratory, point: SamplingPoints/Point034}}
layers:
  - {{name: haut, z_low: 0.2}}
  - {{name: bas, z_low: 0.4}}
numerics: {{nb_cells: 20}}
mcmc:
  nb_chains: {NB_CHAINS}
  nb_iter: {NB_ITER}
  thin: {THIN}
  burn_in_max: 10
  n_field_samples: 7
report:
  models: {{forward: twin_direct, inversion: twin_mcmc}}
  publication_thin: {THIN}
  entries:
    - {{question: water_flux, parameters: {{level: 0.9}}}}
    - {{question: fit_quality}}
"""


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setenv("MOLONARI_LANG", "en")


def tree(folder):
    return {p.relative_to(folder).as_posix(): p.read_bytes() for p in sorted(folder.rglob("*")) if p.is_file()}


# Point à la main (rapide) --------------------------------------------------------------------------------------------

BOUNDS = {"neg_log10_k": [11.0, 15.0], "porosity": [0.01, 0.25], "lambda_s": [1.0, 5.0], "rhos_cs": [1e6, 1e7]}


def hand_point(root, nb_layers=2):
    """Point minimal : chains.npz (3 itérations publiées × 4 chaînes) et generation.yaml. Le twin direct est un dossier vide
    de contenu utile : les erreurs testées ici surviennent avant son chargement."""
    point = root / "Point999"
    twin = point / "twins" / "twin_direct"
    twin.mkdir(parents=True)
    (twin / "twin.yaml").write_text("schema: molonari-twin/2.0.0\n", encoding="utf-8")
    document = {"schema": "molonari-generation/1.0.0", "nb_cells": 20, "initial_condition": "linear",
                "layers": [{"name": f"c{i}", "z_low": 0.2 * (i + 1), "bounds": BOUNDS} for i in range(nb_layers)]}
    (twin / GENERATION_FILE).write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    rng = np.random.default_rng(0)
    energy = rng.uniform(10, 20, size=(3, 4))
    energy[1, 2] = 1.0  # le plus bas : itération 6, chaîne 2
    arrays = {"iteration": np.array([3, 6, 9]), "chain": np.arange(4), "energy": energy,
              "acceptance": np.full(4, 0.3), "resets": np.zeros(4, np.int64)}
    for k, name in enumerate(PARAMETERS):
        low, high = BOUNDS[name]
        arrays[f"samples.{name}"] = rng.uniform(low, high, size=(3, 4, nb_layers))
    (point / "public_reports").mkdir()
    np.savez(point / "public_reports" / "chains.npz", **arrays)
    return point


def write_params(path, rows):
    path.write_text(yaml.safe_dump({"layers": [dict(zip(PARAMETERS, row, strict=True)) for row in rows]}), encoding="utf-8")
    return path


def test_lowest_energy_sample_is_the_default_choice(tmp_path):
    point = hand_point(tmp_path)
    with np.load(point / "public_reports" / "chains.npz", allow_pickle=False) as data:
        assert lowest_energy_sample({k: data[k] for k in data.files}) == (2, 6)


def test_generation_yaml_is_read_back(tmp_path):
    generation = read_generation(hand_point(tmp_path) / "twins" / "twin_direct" / GENERATION_FILE)
    assert generation.nb_cells == 20 and [name for name, _ in generation.layers] == ["c0", "c1"]
    assert generation.bounds[0]["porosity"] == (0.01, 0.25)


def error_of(*args, **kwargs):
    with pytest.raises(UserError) as caught:
        generate_state(*args, **kwargs)
    return caught.value


def test_params_outside_the_bounds_are_refused_with_one_issue_per_value(tmp_path):
    point = hand_point(tmp_path)
    inside = [12.0, 0.15, 3.4, 5e6]
    params = write_params(tmp_path / "p.yaml", [inside, [16.0, 0.30, 3.4, 5e6]])
    error = error_of(point, params=params, output=tmp_path / "out")
    assert error.code == "E_GENERATE_PARAMS_OUT_OF_BOUNDS"
    assert sorted(issue.field for issue in error.params["issues"]) == ["layers[1].neg_log10_k", "layers[1].porosity"]
    assert {issue.code for issue in error.params["issues"]} == {"E_GENERATE_BOUND"}
    assert not (tmp_path / "out").exists()


def test_params_file_shape_is_checked(tmp_path):
    point = hand_point(tmp_path)
    one_layer = write_params(tmp_path / "one.yaml", [[12.0, 0.15, 3.4, 5e6]])
    assert error_of(point, params=one_layer).params["issues"][0].code == "E_GENERATE_PARAMS_LAYERS"
    missing = tmp_path / "missing.yaml"
    missing.write_text(yaml.safe_dump({"layers": [{"porosity": 0.1}, {"porosity": 0.1}]}), encoding="utf-8")
    assert {i.code for i in error_of(point, params=missing).params["issues"]} == {"E_GENERATE_PARAMS_NUMBER"}
    assert error_of(point, params=tmp_path / "absent.yaml").code == "E_GENERATE_PARAMS_FILE"


@pytest.mark.parametrize(("sample", "code"), [("3", "E_GENERATE_SAMPLE_FORMAT"), ("a:b", "E_GENERATE_SAMPLE_FORMAT"),
                                              ("9:6", "E_GENERATE_SAMPLE_UNKNOWN"), ("1:7", "E_GENERATE_SAMPLE_UNKNOWN")])
def test_sample_must_be_a_published_chain_and_iteration(tmp_path, sample, code):
    assert error_of(hand_point(tmp_path), sample=sample).code == code


def test_sample_and_params_exclude_each_other(tmp_path):
    assert error_of(hand_point(tmp_path), sample="0:3", params=tmp_path / "p.yaml").code == "E_GENERATE_SAMPLE_AND_PARAMS"


def test_missing_chains_or_settings_are_structured_errors(tmp_path):
    point = hand_point(tmp_path)
    (point / "public_reports" / "chains.npz").unlink()
    assert error_of(point).code == "E_GENERATE_NO_CHAINS"
    (point / "twins" / "twin_direct" / GENERATION_FILE).unlink()
    assert error_of(point).code == "E_GENERATE_NO_SETTINGS"
    assert error_of(tmp_path / "absent").code == "E_FOLDER_NOT_FOUND"


@pytest.mark.parametrize("where", ["public_reports", "results/abc", "public_reports/sub"])
def test_the_state_is_never_written_in_public_reports_or_results(tmp_path, where):
    point = hand_point(tmp_path)
    before = tree(point)
    assert error_of(point, output=point / where).code == "E_GENERATE_OUTPUT_FORBIDDEN"
    assert tree(point) == before and not (point / "results").exists()


def test_an_existing_folder_that_is_not_a_state_is_not_overwritten(tmp_path):
    point = hand_point(tmp_path)
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "keep.txt").write_text("x")
    assert error_of(point, output=tmp_path / "mine").code == "E_GENERATE_OUTPUT_EXISTS"
    assert (tmp_path / "mine" / "keep.txt").read_text() == "x"


def test_cli_reports_a_structured_error_and_exit_code_1(tmp_path, capsys):
    point = hand_point(tmp_path)
    params = write_params(tmp_path / "p.yaml", [[16.0, 0.15, 3.4, 5e6], [12.0, 0.15, 3.4, 5e6]])
    assert cli_main(["generate", str(point), "--params", str(params), "--output", str(tmp_path / "out")]) == 1
    assert "layers[0].neg_log10_k" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:  # --sample et --params s'excluent dès l'analyse des arguments
        cli_main(["generate", str(point), "--sample", "0:3", "--params", str(params)])
    assert usage.value.code == 2


def test_bounds_errors_come_before_the_engine_is_loaded(tmp_path):
    """Refuser des paramètres libres hors bornes ne charge pas JAX : le calcul seul le charge."""
    point = hand_point(tmp_path)
    params = write_params(tmp_path / "p.yaml", [[16.0, 0.15, 3.4, 5e6], [12.0, 0.15, 3.4, 5e6]])
    code = ("import sys\nfrom molonari_twin.twin.cli import main\n"
            f"status = main(['generate', {str(point)!r}, '--params', {str(params)!r}, '--output', {str(tmp_path / 'o')!r}])\n"
            "assert status == 1 and 'jax' not in sys.modules, sorted(m for m in sys.modules if m.startswith('jax'))")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120,
                               env={**os.environ, "MOLONARI_LANG": "en"})
    assert completed.returncode == 0, completed.stderr


# Point calculé (slow) ------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data_repo(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES / "lab", data / "Laboratory")
    shutil.copytree(FIXTURES / "point034", data / "SamplingPoints" / "Point034")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MOLONARI_DATA", str(data))
        yield data


@pytest.fixture(scope="module")
def point(tmp_path_factory, data_repo):
    """Point calculé : twin_direct et twin_mcmc, rapport public écrit dans `public_reports/` (donc `chains.npz`) et
    `generation.yaml` dans twin_direct."""
    root = tmp_path_factory.mktemp("molonari-twin") / "Point034"
    for name, text in (("twin_direct", DIRECT), ("twin_mcmc", MCMC)):
        (root / "twins" / name).mkdir(parents=True)
        (root / "twins" / name / "twin.yaml").write_text(text, encoding="utf-8")
    (root / "twins" / "twin_mcmc" / "orx.yaml").write_text("mcmc:\n  seed: 42\n", encoding="utf-8")
    run_twin(root / "twins" / "twin_direct")
    run_twin(root / "twins" / "twin_mcmc")
    generate_report(root / "twins" / "twin_mcmc", root / "public_reports")
    return root


def posterior_of(point):
    (folder,) = (point / "results").glob("*/posterior.npz")
    with np.load(folder, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}, folder.parent


def fields_of(results_folder):
    with np.load(results_folder / "fields_temperature.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def simulated(state):
    """Températures simulées (3, T) aux capteurs d'un état généré, relues de residuals.csv."""
    residuals = load_results(state).residuals
    return np.stack([residuals[f"t{i}_simulated"].to_numpy() for i in (1, 2, 3)])


@pytest.mark.slow
def test_report_writes_generation_yaml_in_twin_direct_and_not_in_the_public_report(point):
    path = point / "twins" / "twin_direct" / GENERATION_FILE
    generation = read_generation(path)
    assert generation.nb_cells == 20  # la discrétisation de l'inversion, pas les 40 cellules du twin direct
    assert generation.layers == (("haut", 0.2), ("bas", 0.4))
    assert generation.bounds[0] == {name: tuple(bound) for name, bound in BOUNDS.items()}  # les priors par défaut
    assert not (point / "twins" / "twin_mcmc" / GENERATION_FILE).exists()
    again = generate_report(point / "twins" / "twin_mcmc", point / "public_reports")
    assert again.folder == point / "public_reports" and path.read_bytes() == yaml.safe_dump(
        yaml.safe_load(path.read_text(encoding="utf-8")), sort_keys=False, allow_unicode=True).encode("utf-8")


@pytest.mark.slow
def test_the_bounds_are_absent_from_the_public_report(point):
    report = tree(point / "public_reports")
    assert "chains.npz" in report and not any("generation" in name for name in report)
    for name, content in report.items():
        assert b"bounds" not in content and b"initial_condition" not in content, name
        assert b"10000000" not in content and b"1000000.0" not in content, name  # la borne haute de ρs·cs, et la basse


@pytest.mark.slow
def test_default_state_is_the_lowest_energy_published_sample_and_deterministic(point, tmp_path):
    first = generate_state(point, output=tmp_path / "first")
    second = generate_state(point, output=tmp_path / "second")
    assert tree(first.folder) == tree(second.folder)  # au bit près, metadata.json compris : ni date ni chemin
    with np.load(point / "public_reports" / "chains.npz", allow_pickle=False) as data:
        chains = {k: data[k] for k in data.files}
    assert (first.chain, first.iteration) == lowest_energy_sample(chains)
    row, column = np.unravel_index(np.argmin(chains["energy"]), chains["energy"].shape)
    assert (first.chain, first.iteration) == (int(chains["chain"][column]), int(chains["iteration"][row]))
    generated = json.loads((first.folder / "metadata.json").read_text(encoding="utf-8"))["generated"]
    assert generated["free_parameters"] is False and generated["sample"] == {"chain": first.chain, "iteration": first.iteration}
    assert generated["nb_cells"] == 20 and is_state(first.folder)
    assert "twin_id" not in json.dumps(generated) and "seed" not in (first.folder / "metadata.json").read_text(encoding="utf-8")


@pytest.mark.slow
def test_a_generated_state_gives_back_the_field_kept_by_the_mcmc(point, tmp_path):
    """Même contrôle que test_each_stored_field_comes_from_the_parameters_of_its_chain_and_iteration (tolérance 1e-9)."""
    posterior, results = posterior_of(point)
    fields = fields_of(results)
    assert len(fields["chain"]) >= NB_CHAINS
    published = None
    for k in (0, len(fields["chain"]) - 1):
        chain, iteration = int(fields["chain"][k]), int(fields["iteration"][k])
        state = generate_state(point, sample=f"{chain}:{iteration}", output=tmp_path / f"s{k}")
        np.testing.assert_allclose(simulated(state.folder), fields["values"][k], rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(state.parameters, posterior["samples"][iteration // THIN - 1, chain], rtol=0, atol=0)
        # contrôle négatif : une autre chaîne, au même moment, donne un autre état
        other = generate_state(point, sample=f"{(chain + 1) % NB_CHAINS}:{iteration}", output=tmp_path / f"o{k}")
        assert np.abs(simulated(other.folder) - fields["values"][k]).max() > 1e-6
        published = state
    assert published is not None


@pytest.mark.slow
def test_free_parameters_are_marked_and_equal_the_sample_state_when_they_are_its_parameters(point, tmp_path):
    sample = generate_state(point, sample="3:6", output=tmp_path / "sample")
    free = generate_state(point, params=write_params(tmp_path / "p.yaml", sample.parameters), output=tmp_path / "free")
    generated = json.loads((free.folder / "metadata.json").read_text(encoding="utf-8"))["generated"]
    assert free.free_parameters and generated["free_parameters"] is True and generated["sample"] is None
    np.testing.assert_allclose(simulated(free.folder), simulated(sample.folder), rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_params_outside_the_bounds_of_the_real_point_are_refused(point, tmp_path):
    params = write_params(tmp_path / "p.yaml", [[12.0, 0.9, 3.4, 5e6], [12.0, 0.15, 3.4, 5e6]])
    error = error_of(point, params=params, output=tmp_path / "out")
    assert error.code == "E_GENERATE_PARAMS_OUT_OF_BOUNDS" and error.params["issues"][0].field == "layers[0].porosity"
    assert not (tmp_path / "out").exists()


@pytest.mark.slow
def test_generating_writes_nothing_in_public_reports_results_or_twins(point, tmp_path):
    before = tree(point)
    generate_state(point, output=tmp_path / "state")
    assert tree(point) == before  # ni public_reports/, ni results/, ni registry.csv, ni twins/
    assert (tmp_path / "state" / "metadata.json").is_file()


@pytest.mark.slow
def test_ask_water_flux_on_a_state_is_a_single_value_and_recalls_the_public_interval(point, tmp_path, capsys):
    state = tmp_path / "state"
    assert cli_main(["generate", str(point), "--output", str(state)]) == 0
    assert "published sample" in capsys.readouterr().out
    assert cli_main(["ask", str(state), "water_flux", "--level", "user", "--json"]) == 0
    answer = json.loads(capsys.readouterr().out)
    mean = answer["data"]["mean"]
    assert mean["interval"] is None and answer["data"]["uncertainty"]["value"] == "none"
    assert answer["data"]["model"]["value"] == "forward"
    codes = [w["code"] for w in answer["warnings"]]
    assert "W_GENERATED_STATE" in codes and "W_FREE_PARAMETERS" not in codes
    assert set(answer["provenance"]) <= {"question", "ref", "parameters"}  # niveau user : ni chemin ni état interne
    # en texte, la réponse rappelle que l'intervalle a posteriori est dans le rapport
    assert cli_main(["ask", str(state), "water_flux", "--level", "user"]) == 0
    out = capsys.readouterr().out
    assert "posterior interval is in the public report" in out and "uncertainty: none" in out
    assert np.isfinite(mean["value"])


@pytest.mark.slow
def test_ask_on_a_free_parameters_state_says_it_is_not_calibrated(point, tmp_path, capsys):
    state = generate_state(point, output=tmp_path / "s")
    free = generate_state(point, params=write_params(tmp_path / "p.yaml", state.parameters), output=tmp_path / "f")
    assert cli_main(["ask", str(free.folder), "temperature", "--params", '{"z": 0.2}', "--json"]) == 0
    codes = [w["code"] for w in json.loads(capsys.readouterr().out)["warnings"]]
    assert {"W_GENERATED_STATE", "W_FREE_PARAMETERS"} <= set(codes)


@pytest.mark.slow
def test_a_state_answers_only_the_user_questions(point, tmp_path, capsys):
    state = generate_state(point, output=tmp_path / "s").folder
    assert cli_main(["ask", str(state), "residuals", "--level", "user", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "E_ACCESS_USER_ONLY"
    assert cli_main(["ask", str(state), "residuals", "--level", "dev", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["code"] == "E_STATE_QUESTION_UNSUPPORTED"
    assert cli_main(["questions", str(state), "--level", "user", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert {q["id"] for q in listed["questions"]} == {"water_flux", "exchange_direction", "exchanged_volume", "heat_budget", "temperature"}
    assert cli_main(["ask", str(tmp_path / "nothing"), "water_flux", "--json"]) == 1  # ni état ni twin
    assert json.loads(capsys.readouterr().out)["code"] == "E_TWIN_NOT_FOUND"


@pytest.mark.slow
def test_asking_a_state_does_not_load_jax(point, tmp_path):
    state = generate_state(point, output=tmp_path / "s").folder
    code = ("import sys\nfrom molonari_twin.twin.cli import main\n"
            f"status = main(['ask', {str(state)!r}, 'water_flux', '--json'])\n"
            "assert status == 0 and 'jax' not in sys.modules")
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False, timeout=120,
                               env={**os.environ, "MOLONARI_LANG": "en"})
    assert completed.returncode == 0, completed.stderr
