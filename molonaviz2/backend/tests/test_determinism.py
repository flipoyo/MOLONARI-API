"""P2.4 : déterminisme du pipeline (du brut aux résultats) et mode --strict de bout en bout.

- Le twin part du brut (schéma 2.0.0) et passe par `run_twin`, donc par le pipeline, le moteur et l'écriture des
  fichiers. Les tests de l'échantillonneur seul sont dans test_engine_dream.py.
- La graine est dans `orx.yaml`, à côté de twin.yaml (P2.2c) : `orx_text()` la donne, `write_mcmc_twin()` écrit les deux.
- Les tests qui calculent un twin mcmc sont `slow` ; ceux du mode strict ne calculent rien (le refus précède le calcul).
"""

import json
import os
import shutil
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import FIXTURES, git_run
from molonari_twin.domain.compute import LayerSpec
from molonari_twin.errors import UserError
from molonari_twin.twin.config import load_twin
from molonari_twin.twin.run import run_twin

NB_ITER = 12
NB_CHAINS = 7
TWIN = """schema: molonari-twin/2.0.0
name: twin_01
mode: mcmc
data: {{lab: Laboratory, point: SamplingPoints/Point034}}
layers:
  - {{name: haut, z_low: 0.2}}
  - {{name: bas, z_low: 0.4}}
mcmc:
  nb_chains: {chains}
  nb_iter: {iterations}
  thin: {thin}
  burn_in_max: 10
  n_field_samples: 7
"""
DIRECT = """schema: molonari-twin/2.0.0
name: twin_01
mode: direct
data: {lab: Laboratory, point: SamplingPoints/Point034}
layers:
  - {name: couche 1, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000}
numerics: {nb_cells: 40}
"""


def orx_text(value: int) -> str:
    """Contenu de orx.yaml : la graine du MCMC, réservée à ORX depuis le schéma 2.0.0 (P2.2c)."""
    return f"mcmc:\n  seed: {value}\n"


def mcmc_text(thin: int = 3) -> str:
    return TWIN.format(thin=thin, chains=NB_CHAINS, iterations=NB_ITER)


@pytest.fixture(scope="module")
def data_repo(tmp_path_factory):
    """Dépôt de données minimal (Laboratory/ et SamplingPoints/Point034/) ; MOLONARI_DATA le désigne pendant le module."""
    data = tmp_path_factory.mktemp("data")
    shutil.copytree(FIXTURES / "lab", data / "Laboratory")
    shutil.copytree(FIXTURES / "point034", data / "SamplingPoints" / "Point034")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("MOLONARI_DATA", str(data))
        yield data


def write_mcmc_twin(root, seed_value: int = 42, thin: int = 3, name="twin_01"):
    """Twin mcmc avec sa couche orx : twin.yaml et, à côté, orx.yaml."""
    folder = write_twin(root, mcmc_text(thin), name)
    (folder / "orx.yaml").write_text(orx_text(seed_value), encoding="utf-8")
    return folder


def write_twin(root, text, name="twin_01"):
    folder = root / "twins" / name
    folder.mkdir(parents=True)
    (folder / "twin.yaml").write_text(text, encoding="utf-8")
    return folder


def load(folder, name="posterior.npz"):
    with np.load(folder / name, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


class Runs:
    """Twins mcmc calculés une fois pour tous les tests du module."""

    def __init__(self, root):
        self.root = root
        self.folders = {}

    def __call__(self, label, seed_value=42, thin=3):
        if label not in self.folders:
            folder = write_mcmc_twin(self.root / label, seed_value, thin)
            self.folders[label] = run_twin(folder).status.results_dir
        return self.folders[label]


@pytest.fixture(scope="module")
def runs(tmp_path_factory, data_repo):
    return Runs(tmp_path_factory.mktemp("runs"))


# P2.4a : thin = k -------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("thin", [2, 3, 5])
def test_thin_keeps_exactly_one_iteration_out_of_k_for_the_same_seed(runs, thin):
    full = load(runs("thin1", thin=1))
    thinned = load(runs(f"thin{thin}", thin=thin))
    kept = NB_ITER // thin  # 12 itérations : thin 5 garde les itérations 5 et 10, la 11e et la 12e sont perdues
    assert full["iteration"].tolist() == list(range(1, NB_ITER + 1))
    assert thinned["iteration"].tolist() == [thin * i for i in range(1, kept + 1)]
    assert thinned["samples"].shape[0] == thinned["energy"].shape[0] == kept
    # la même graine donne la même trajectoire : les itérations gardées sont celles de la trajectoire complète, au bit près
    np.testing.assert_array_equal(thinned["samples"], full["samples"][thin - 1::thin][:kept])
    np.testing.assert_array_equal(thinned["energy"], full["energy"][thin - 1::thin][:kept])
    # l'acceptation est comptée sur les itérations gardées seulement (dream.py) : elle change avec thin, sans cesser d'être déterministe
    assert thinned["acceptance"].shape == full["acceptance"].shape and ((0 <= thinned["acceptance"]) & (thinned["acceptance"] <= 1)).all()
    np.testing.assert_array_equal(thinned["resets"], full["resets"])


# P2.4b : indices de chaîne ------------------------------------------------------------------


@pytest.mark.slow
def test_chain_and_iteration_indices_of_the_fields_are_those_of_the_parameters(runs):
    results = runs("thin3")
    posterior, fields = load(results), load(results, "fields_temperature.npz")
    thin = 3
    assert posterior["samples"].shape == (NB_ITER // thin, NB_CHAINS, 2, 4)
    chains, iterations = fields["chain"], fields["iteration"]
    assert chains.shape == iterations.shape == (fields["values"].shape[0],)
    # chaque champ porte une chaîne et une itération qui existent dans posterior.npz
    assert set(chains.tolist()) <= set(range(NB_CHAINS))
    assert set(iterations.tolist()) <= set(posterior["iteration"].tolist())
    # q(0, t) est rangé dans le même ordre que les champs de température
    riverbed = load(results, "fields_riverbed_flux.npz")
    np.testing.assert_array_equal(riverbed["chain"], chains)
    np.testing.assert_array_equal(riverbed["iteration"], iterations)
    # les champs sont répartis également entre les chaînes
    assert len(set(np.bincount(chains, minlength=NB_CHAINS).tolist())) == 1


def recomputed_sensor_temperature(folder, x):
    """Température aux capteurs (°C) du modèle direct, pour des paramètres `x` (couches, 4) de l'espace des priors."""
    from molonari_twin.engine.jax.forward import build_geometry, make_forward

    config = load_twin(folder)
    mcmc = config.inversion_config()
    geom = build_geometry(config.column_inputs(), [layer.z_low for layer in mcmc.layers], mcmc.nb_cells)
    physical = [LayerSpec(layer.name, layer.z_low, *x[i]).physical() for i, layer in enumerate(mcmc.layers)]
    fields = make_forward(geom)(jnp.asarray(physical))
    return np.asarray(fields["temperature"])[geom.sensor_index][:, ::mcmc.time_step] - 273.15


@pytest.mark.slow
def test_each_stored_field_comes_from_the_parameters_of_its_chain_and_iteration(runs):
    results = runs("thin3")
    folder = runs.root / "thin3" / "twins" / "twin_01"
    posterior, fields = load(results), load(results, "fields_temperature.npz")
    thin = 3
    assert len(fields["chain"]) >= NB_CHAINS  # au moins un champ par chaîne
    differences = []
    for k, (chain, iteration) in enumerate(zip(fields["chain"], fields["iteration"], strict=True)):
        x = posterior["samples"][iteration // thin - 1, chain]
        np.testing.assert_allclose(fields["values"][k], recomputed_sensor_temperature(folder, x), rtol=1e-9, atol=1e-9)
        # contrôle négatif : les paramètres d'une autre chaîne donneraient un autre champ (le test discrimine)
        other = posterior["samples"][iteration // thin - 1, (chain + 1) % NB_CHAINS]
        differences.append(np.abs(fields["values"][k] - recomputed_sensor_temperature(folder, other)).max())
    assert min(differences) > 1e-6


# P2.4c : graine -------------------------------------------------------------------------------


def comparable_metadata(results):
    """metadata.json sans ce qui dépend de l'instant et de la machine (date, durée, provenance)."""
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    for key in ("created_at", "duration_s", "provenance"):
        metadata.pop(key)
    return metadata


@pytest.mark.slow
def test_same_seed_gives_identical_results_through_the_pipeline(runs, tmp_path):
    first = runs("thin3")
    second = runs("again", seed_value=42, thin=3)
    assert first != second and first.name == second.name  # deux exécutions, le même twin_id
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        if name != "metadata.json":
            assert (first / name).read_bytes() == (second / name).read_bytes(), name  # au bit près, fichiers entiers
    posterior_a, posterior_b = load(first), load(second)
    for key in posterior_a:
        assert posterior_a[key].tobytes() == posterior_b[key].tobytes(), key
    assert comparable_metadata(first) == comparable_metadata(second)  # empreintes des fichiers comprises


@pytest.mark.slow
def test_two_seeds_give_different_chains_and_different_twin_ids(runs):
    first, other = runs("thin3"), runs("seed43", seed_value=43, thin=3)
    assert first.name != other.name  # la graine entre dans le twin_id
    a, b = load(first), load(other)
    assert not np.array_equal(a["samples"], b["samples"])
    for chain in range(NB_CHAINS):  # chaque chaîne diffère, pas seulement l'une d'elles
        assert not np.array_equal(a["samples"][:, chain], b["samples"][:, chain]), chain
    assert "seed" not in json.dumps(comparable_metadata(first))  # la graine reste hors des sorties (P0.7)


@pytest.mark.slow
def test_same_seed_in_two_fresh_processes_gives_identical_posterior(tmp_path, data_repo):
    posteriors = []
    for label in ("a", "b"):
        folder = write_mcmc_twin(tmp_path / label)
        completed = subprocess.run([sys.executable, "-m", "molonari_twin", "run", str(folder)], capture_output=True, text=True,
                                   timeout=600, check=False, env={**os.environ, "MOLONARI_LANG": "en"})
        assert completed.returncode == 0, completed.stderr
        (results,) = (tmp_path / label / "results").iterdir()
        posteriors.append((results / "posterior.npz").read_bytes())
    assert posteriors[0] == posteriors[1]


# P2.4d : --strict ---------------------------------------------------------------------------

PUBLISHED_TAG = "molonari-twin-v1.0.0"


@pytest.fixture
def strict_setup(tmp_path, data_repo, monkeypatch):
    """Code publié (dépôt propre, tagué) et twin propre et tagué, tous deux dans de vrais dépôts git."""
    import molonari_twin
    from molonari_twin.engine import api
    from molonari_twin.twin import provenance

    code = tmp_path / "code"
    code.mkdir()
    git_run(code, "init", "-q")
    (code / "pyproject.toml").write_text("[project]\n")
    git_run(code, "add", ".")
    git_run(code, "commit", "-q", "-m", "code")
    git_run(code, "tag", PUBLISHED_TAG)
    monkeypatch.setattr(provenance, "code_path", lambda: code)
    monkeypatch.setattr(api, "engine_version", lambda: "molonari-twin 1.0.0 · jax 0.0.0")
    monkeypatch.setattr(molonari_twin, "__version__", "1.0.0")
    folder = write_twin(tmp_path, DIRECT)
    twins = folder.parent
    git_run(twins, "init", "-q")
    git_run(twins, "add", "twin_01")
    git_run(twins, "commit", "-q", "-m", "twin_01")
    git_run(twins, "tag", "campagne-2016")
    return type("Setup", (), {"root": tmp_path, "code": code, "twins": twins, "folder": folder, "monkeypatch": monkeypatch})


def refusal(folder):
    with pytest.raises(UserError) as refused:
        run_twin(folder, strict=True)
    assert refused.value.code == "E_STRICT_REFUSED"
    return [(issue.code, issue.params) for issue in refused.value.params["issues"]]


def nothing_written(setup):
    return not (setup.root / "results").exists() and not (setup.twins / "registry.csv").exists()


def test_strict_accepts_the_clean_published_setup(strict_setup):
    run = run_twin(strict_setup.folder, strict=True)
    assert run.computed


def test_strict_refuses_a_modified_twin_repository(strict_setup):
    (strict_setup.folder / "twin.yaml").write_text(DIRECT.replace("nb_cells: 40", "nb_cells: 50"), encoding="utf-8")
    assert refusal(strict_setup.folder) == [("E_STRICT_DIRTY", {"target": "twin"})]
    assert nothing_written(strict_setup)


def test_strict_refuses_an_untracked_file_in_the_twin_folder(strict_setup):
    (strict_setup.folder / "notes.txt").write_text("brouillon")
    assert refusal(strict_setup.folder) == [("E_STRICT_DIRTY", {"target": "twin"})]
    assert nothing_written(strict_setup)


def test_strict_refuses_a_modified_code_repository(strict_setup):
    (strict_setup.code / "pyproject.toml").write_text("[project]\nname = 'modifie'\n")
    assert refusal(strict_setup.folder) == [("E_STRICT_DIRTY", {"target": "code"})]
    assert nothing_written(strict_setup)


@pytest.mark.parametrize("dirty", [False, True])
def test_strict_refuses_a_local_build(strict_setup, dirty):
    """Un commit après le tag donne `+g<commit>` (et `.dirty` si le code est modifié), calculé par version_suffix."""
    from molonari_twin import gitinfo
    from molonari_twin.engine import api

    (strict_setup.code / "autre.py").write_text("x = 1\n")
    git_run(strict_setup.code, "add", ".")
    git_run(strict_setup.code, "commit", "-q", "-m", "après la publication")
    if dirty:
        (strict_setup.code / "autre.py").write_text("x = 2\n")
    suffix = gitinfo.version_suffix(strict_setup.code, PUBLISHED_TAG)
    assert suffix.startswith("+g") and suffix.endswith(".dirty") == dirty
    version = f"molonari-twin 1.0.0{suffix} · jax 0.0.0"
    strict_setup.monkeypatch.setattr(api, "engine_version", lambda: version)
    expected = [("E_STRICT_LOCAL_BUILD", {"version": version})]
    if dirty:
        expected.append(("E_STRICT_DIRTY", {"target": "code"}))
    assert refusal(strict_setup.folder) == expected
    assert nothing_written(strict_setup)


def test_strict_lists_every_reason_at_once(strict_setup):
    from molonari_twin.engine import api

    strict_setup.monkeypatch.setattr(api, "engine_version", lambda: "molonari-twin 1.0.0+g0123456789ab.dirty · jax 0.0.0")
    (strict_setup.folder / "twin.yaml").write_text(DIRECT.replace("nb_cells: 40", "nb_cells: 50"), encoding="utf-8")
    git_run(strict_setup.twins, "tag", "-d", "campagne-2016")
    assert [code for code, _ in refusal(strict_setup.folder)] == ["E_STRICT_LOCAL_BUILD", "E_STRICT_DIRTY", "E_STRICT_UNTAGGED"]


def test_a_modified_twin_runs_without_strict_and_is_refused_with_it(strict_setup):
    (strict_setup.folder / "notes.txt").write_text("brouillon")
    assert refusal(strict_setup.folder)
    assert run_twin(strict_setup.folder).computed  # sans --strict, le même dossier se calcule


def test_command_line_strict_refuses_a_modified_twin(strict_setup):
    (strict_setup.folder / "notes.txt").write_text("brouillon")
    completed = subprocess.run([sys.executable, "-m", "molonari_twin", "run", "--strict", str(strict_setup.folder)], capture_output=True,
                               text=True, timeout=300, check=False, env={**os.environ, "MOLONARI_LANG": "en"})
    assert completed.returncode == 1
    assert "Strict mode: the twin was not run." in completed.stderr
    assert "The twin repository has uncommitted changes." in completed.stderr
    assert nothing_written(strict_setup)
