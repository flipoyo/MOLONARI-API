"""Schéma molonari-twin/2.0.0 (P2.2) : le brut (data), le nettoyage (processing), la graine dans orx.yaml (couche orx,
P2.2c), les clés retirées, legacy_twin_id, et les empreintes par fichier des résultats (P2.2f).

Les twins 1.x sont couverts par test_twin.py (identifiants de référence inchangés, lecture sans changement)."""

import json
import shutil

import pytest
from conftest import FIXTURES
from molonari_twin.twin.config import TwinConfigError, load_twin
from molonari_twin.twin.run import run_twin

RAW_MCMC = """schema: molonari-twin/2.0.0
name: twin_01
mode: mcmc
data: {lab: Laboratory, point: SamplingPoints/Point034}
layers:
  - {name: couche 1, z_low: 0.4}
mcmc:
  nb_chains: 7
  nb_iter: 10
  burn_in_max: 10
  n_field_samples: 7
"""

ORX = """mcmc:
  seed: 42
"""

RAW_DIRECT = """schema: molonari-twin/2.0.0
name: twin_01
mode: direct
data: {lab: Laboratory, point: SamplingPoints/Point034}
processing:
  methods: {t1: iqr}
layers:
  - {name: couche 1, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000}
numerics: {nb_cells: 40}
"""


@pytest.fixture
def data_repo(tmp_path, monkeypatch):
    """Dépôt de données minimal : Laboratory/ et SamplingPoints/Point034/, copiés des fixtures."""
    data = tmp_path / "data"
    shutil.copytree(FIXTURES / "lab", data / "Laboratory")
    shutil.copytree(FIXTURES / "point034", data / "SamplingPoints" / "Point034")
    monkeypatch.setenv("MOLONARI_DATA", str(data))
    return data


def write_twin(tmp_path, text, name="twin_01"):
    folder = tmp_path / "twins" / name
    folder.mkdir(parents=True)
    (folder / "twin.yaml").write_text(text, encoding="utf-8")
    return folder


def write_orx(folder, text=ORX):
    (folder / "orx.yaml").write_text(text, encoding="utf-8")
    return folder


def seeded(tmp_path, seed=42, name="twin_01"):
    """Twin mcmc 2.0.0 avec sa couche orx (orx.yaml à côté de twin.yaml)."""
    return write_orx(write_twin(tmp_path, RAW_MCMC, name), ORX.replace("42", str(seed)))


def codes(error):
    return [(issue.field, issue.code) for issue in error.value.issues]


def test_raw_twin_is_read_from_the_data_repository(tmp_path, data_repo):
    config = load_twin(seeded(tmp_path))
    assert config.raw_data and config.schema_version == (2, 0, 0)
    assert config.point == "Point034" and config.mode == "mcmc"
    assert len(config.depths) == 4 and config.depths[-1] > 0
    assert config.measurements is None
    assert len(config.load_measurements()) > 3
    assert config.inversion_config().seed == 42 and config.seed == 42 and config.has_seed
    assert config.orx_path == tmp_path / "twins" / "twin_01" / "orx.yaml"


# P2.2c : la graine est dans orx.yaml ------------------------------------------------------------------------------


def test_raw_twin_without_orx_yaml_stays_readable(tmp_path, data_repo):
    from molonari_twin.questions.twin_info import describe

    config = load_twin(write_twin(tmp_path, RAW_MCMC))
    assert config.seed is None and not config.has_seed and config.orx_path is None
    assert config.mode == "mcmc" and len(config.load_measurements()) > 3
    assert describe(config).data["nb_chains"].value == 7
    assert config.canonical("v1", with_seed=False)["mode"] == "mcmc"
    with pytest.raises(TwinConfigError) as error:
        config.twin_id("v1")  # le twin_id contient la graine : sans elle, il n'existe pas
    assert codes(error) == [("orx.mcmc.seed", "E_TWIN_REQUIRED")]


def test_orx_yaml_without_seed_is_the_same_as_no_orx_yaml(tmp_path, data_repo):
    config = load_twin(write_orx(write_twin(tmp_path, RAW_MCMC), "platform: {device: cpu}\n"))
    assert config.seed is None and not config.has_seed
    assert config.reference_platform == {"device": "cpu"}


def test_direct_twin_needs_no_seed(tmp_path, data_repo):
    config = load_twin(write_twin(tmp_path, RAW_DIRECT))
    assert config.has_seed and config.require_seed() == 0
    assert config.twin_id("v1") == load_twin(write_orx(write_twin(tmp_path / "b", RAW_DIRECT))).twin_id("v1")  # sans effet


def test_run_is_refused_without_seed(tmp_path, data_repo):
    folder = write_twin(tmp_path, RAW_MCMC)
    with pytest.raises(TwinConfigError) as error:
        run_twin(folder)
    assert codes(error) == [("orx.mcmc.seed", "E_TWIN_REQUIRED")]
    assert str(error.value.issues[0]) == "orx.mcmc.seed: required."
    assert not (tmp_path / "results").exists()


def test_run_is_refused_without_seed_by_the_command_line(tmp_path, data_repo, capsys):
    from molonari_twin.twin.cli import main

    assert main(["run", str(write_twin(tmp_path, RAW_MCMC))]) == 1
    assert "orx.mcmc.seed" in capsys.readouterr().err


def test_run_task_is_refused_without_seed(tmp_path, data_repo):
    from types import SimpleNamespace

    from molonari_twin.tasks.jobs import run_twin_job
    from molonari_twin.tasks.model import error_dict

    context = SimpleNamespace(report=lambda *_: None, should_cancel=lambda: False, log=lambda *_, **__: None)
    with pytest.raises(TwinConfigError) as error:
        run_twin_job({"path": str(write_twin(tmp_path, RAW_MCMC))}, context)
    assert codes(error) == [("orx.mcmc.seed", "E_TWIN_REQUIRED")]
    assert error_dict(error.value)["code"] == "E_TWIN_CONFIG_INVALID"


@pytest.mark.parametrize("where", ["mcmc", "top"])
def test_seed_in_twin_yaml_is_refused(tmp_path, data_repo, where):
    text = RAW_MCMC.replace("mcmc:\n", "mcmc:\n  seed: 42\n") if where == "mcmc" else RAW_MCMC + "seed: 42\n"
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_orx(write_twin(tmp_path, text)))  # même avec la couche orx : la recette dev n'a pas de graine
    field = "mcmc.seed" if where == "mcmc" else "seed"
    assert codes(error) == [(field, "E_TWIN_SEED_ORX")]
    assert "orx.yaml" in str(error.value.issues[0])


def test_seed_in_twin_yaml_is_refused_in_direct_mode_too(tmp_path, data_repo):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, RAW_DIRECT + "seed: 1\n"))
    assert codes(error) == [("seed", "E_TWIN_SEED_ORX")]


@pytest.mark.parametrize("text, expected", [
    ("mcmc: {seed: 4.2}\n", [("orx.mcmc.seed", "E_TWIN_EXPECTED_INTEGER")]),
    ("mcmc: {seed: true}\n", [("orx.mcmc.seed", "E_TWIN_EXPECTED_INTEGER")]),
    ("mcmc: {seed: -1}\n", [("orx.mcmc.seed", "E_TWIN_MINIMUM")]),
    ("mcmc: {seed: 1, nb_iter: 3}\n", [("orx.mcmc.nb_iter", "E_TWIN_UNKNOWN_KEY")]),
    ("mcmc: 3\n", [("orx.mcmc", "E_TWIN_EXPECTED_MAPPING")]),
    ("seed: 1\n", [("orx.seed", "E_TWIN_UNKNOWN_KEY")]),
    ("schema: molonari-orx/9\nmcmc: {seed: 1}\n", [("orx.schema", "E_TWIN_SCHEMA")]),
    ("platform: {gpu: 1}\n", [("orx.platform.gpu", "E_TWIN_UNKNOWN_KEY")]),
    ("platform: {device: [cpu]}\n", [("orx.platform.device", "E_TWIN_EXPECTED_TEXT")]),
    ("mcmc: [\n", [("orx", "E_TWIN_INVALID_YAML")]),
])
def test_orx_yaml_is_checked(tmp_path, data_repo, text, expected):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_orx(write_twin(tmp_path, RAW_MCMC), text))
    assert codes(error) == expected


def test_orx_yaml_gives_the_seed_and_the_reference_platform(tmp_path, data_repo):
    text = "schema: molonari-orx/1.0.0\nmcmc:\n  seed: 7\nplatform:\n  python: 3.12\n  device: cpu\n"
    config = load_twin(write_orx(write_twin(tmp_path, RAW_MCMC), text))
    assert config.seed == 7 and config.inversion_config().seed == 7
    assert config.reference_platform == {"python": "3.12", "device": "cpu"}
    # La plateforme est une étiquette : elle n'entre pas dans le twin_id.
    assert config.twin_id("v1") == load_twin(seeded(tmp_path / "b", seed=7)).twin_id("v1")


def test_orx_yaml_is_ignored_by_a_1_x_twin(tmp_path):
    from conftest import EXAMPLE_TWINS
    from molonari_twin.engine.api import release_version

    plain = load_twin(shutil.copytree(EXAMPLE_TWINS / "twin_03", tmp_path / "a" / "twin_03"))
    with_orx = shutil.copytree(EXAMPLE_TWINS / "twin_03", tmp_path / "b" / "twin_03")
    write_orx(with_orx, "mcmc: {seed: 99}\n")
    config = load_twin(with_orx)
    assert config.seed == plain.seed and config.orx_path is None
    assert config.twin_id(release_version()) == plain.twin_id(release_version())


def test_1_x_twins_keep_their_seed_in_twin_yaml(tmp_path):
    from conftest import EXAMPLE_TWINS
    from molonari_twin.engine.api import release_version

    folder = shutil.copytree(EXAMPLE_TWINS / "twin_03", tmp_path / "twin_03")
    text = (folder / "twin.yaml").read_text(encoding="utf-8")
    (folder / "twin.yaml").write_text(text + "seed: 5\n", encoding="utf-8")
    config = load_twin(folder)
    assert config.seed == 5 and config.canonical(release_version())["seed"] == 5


def test_seed_is_part_of_the_twin_id(tmp_path, data_repo):
    ids = {seed: load_twin(seeded(tmp_path / str(seed), seed)).twin_id("v1") for seed in (42, 43)}
    assert ids[42] != ids[43]
    assert load_twin(seeded(tmp_path / "again")).twin_id("v1") == ids[42]


def test_twin_id_does_not_depend_on_where_the_data_is(tmp_path, data_repo, monkeypatch):
    reference = load_twin(seeded(tmp_path / "a")).twin_id("v1")
    moved = tmp_path / "elsewhere"
    shutil.copytree(data_repo, moved)
    monkeypatch.setenv("MOLONARI_DATA", str(moved))
    assert load_twin(seeded(tmp_path / "b")).twin_id("v1") == reference


def test_twin_id_follows_the_cleaning(tmp_path, data_repo):
    reference = load_twin(write_twin(tmp_path / "a", RAW_DIRECT)).twin_id("v1")
    changed = RAW_DIRECT.replace("{t1: iqr}", "{t1: zscore}")
    assert load_twin(write_twin(tmp_path / "b", changed)).twin_id("v1") != reference


def test_code_version_is_not_written_in_a_raw_twin(tmp_path, data_repo):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, RAW_DIRECT + "code_version: molonari-twin 0.0.1\n"))
    assert ("code_version", "E_TWIN_CODE_VERSION_REMOVED") in codes(error)


def test_keys_of_1_x_are_refused_with_their_new_place(tmp_path, data_repo):
    text = RAW_MCMC.replace("data: {lab: Laboratory, point: SamplingPoints/Point034}",
                            "data: {measurements: measurements.csv, start: 2016-07-12}") + "seed: 3\n"
    text = text.replace("layers:", "sensors: {depths: [0.1, 0.2, 0.3, 0.4]}\nlayers:")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text))
    found = codes(error)
    assert ("data.measurements", "E_TWIN_KEY_MOVED") in found
    assert ("data.start", "E_TWIN_KEY_MOVED") in found
    assert ("seed", "E_TWIN_SEED_ORX") in found
    assert ("sensors.depths", "E_TWIN_KEY_FROM_DATA") in found


def test_unknown_laboratory_or_point_is_reported(tmp_path, data_repo):
    text = RAW_DIRECT.replace("SamplingPoints/Point034", "SamplingPoints/Point999")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text))
    assert [f for f, _ in codes(error)] == ["data.point"]


@pytest.mark.parametrize("value", ["../../etc", "/etc"])
def test_paths_stay_inside_the_data_repository(tmp_path, data_repo, value):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, RAW_DIRECT.replace("SamplingPoints/Point034", value)))
    assert ("data.point", "E_TWIN_DATA_PATH") in codes(error)


def test_no_data_repository_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("MOLONARI_DATA", raising=False)
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, RAW_DIRECT))
    assert ("data", "E_TWIN_DATA_ROOT") in codes(error)


def test_processing_is_checked(tmp_path, data_repo):
    text = RAW_DIRECT.replace("{t1: iqr}", "{t1: moyenne}")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text))
    assert ("processing.methods.t1", "E_TWIN_METHOD") in codes(error)
    text = RAW_DIRECT.replace("processing:\n  methods: {t1: iqr}", "processing:\n  start: 2016-07-13\n  end: 2016-07-12")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path / "dates", text))
    assert ("processing.end", "E_TWIN_END_BEFORE_START") in codes(error)


def test_legacy_twin_id_is_a_label(tmp_path, data_repo):
    reference = load_twin(write_twin(tmp_path / "a", RAW_DIRECT)).twin_id("v1")
    labelled = load_twin(write_twin(tmp_path / "b", RAW_DIRECT + "legacy_twin_id: 7501fd11570bbe5c\n"))
    assert labelled.legacy_twin_id == "7501fd11570bbe5c"
    assert labelled.twin_id("v1") == reference
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path / "c", RAW_DIRECT + "legacy_twin_id: nope\n"))
    assert ("legacy_twin_id", "E_TWIN_LEGACY_ID") in codes(error)


def test_seed_is_hidden_from_the_description_and_from_the_recipe_in_metadata(tmp_path, data_repo):
    from molonari_twin.questions.twin_info import describe
    from molonari_twin.twin.config import without_seed

    config = load_twin(seeded(tmp_path))
    assert "seed" not in describe(config).data
    canonical = config.canonical("v1")
    assert canonical["mcmc"]["seed"] == 42
    assert "seed" not in json.dumps(without_seed(canonical))


def test_direct_raw_run_records_the_digest_of_every_result_file(tmp_path, data_repo):
    folder = write_twin(tmp_path, RAW_DIRECT)
    run = run_twin(folder)
    results = run.status.results_dir
    metadata = json.loads((results / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["data"]["lab"] == "Laboratory" and "cleaned_sha256" in metadata["data"]
    assert "seed" not in json.dumps(metadata["config"])
    files = metadata["files"]
    expected = {p.name for p in results.iterdir() if p.name != "metadata.json"}
    assert set(files) == expected and len(files) >= 4
    import hashlib

    for name, digest in files.items():
        assert hashlib.sha256((results / name).read_bytes()).hexdigest() == digest, name
    assert metadata["provenance"]["environment"]["platform"]



def test_orx_runs_and_the_dev_reads_the_results_without_the_seed(tmp_path, data_repo):
    """ORX (orx.yaml) calcule ; le dev, qui n'a pas la couche orx, lit : états, questions, API. Aucune graine nulle part."""
    from molonari_twin.api.service import TwinAPI
    from molonari_twin.engine.api import engine_version
    from molonari_twin.errors import UserError
    from molonari_twin.questions.check import check_twin
    from molonari_twin.twin.run import twin_status

    folder = seeded(tmp_path)
    run = run_twin(folder)
    assert run.computed and run.status.twin_id == load_twin(folder).twin_id(engine_version())  # la graine est dans l'id
    metadata = json.loads((run.status.results_dir / "metadata.json").read_text(encoding="utf-8"))
    assert "seed" not in json.dumps(metadata["config"]) and "seed" not in json.dumps(metadata["provenance"])
    assert run_twin(folder).computed is False  # ORX relance : résultat existant

    (folder / "orx.yaml").unlink()  # le dev : la couche orx n'est pas montée
    status = twin_status(folder)
    assert status.state == "computed" and status.twin_id == run.status.twin_id  # l'identifiant du registre
    api = TwinAPI(tmp_path / "twins")
    answers = [api.ask("twin_01", "describe"), api.ask("twin_01", "provenance"), check_twin(folder)]
    assert answers[0].data["mode"].value == "mcmc"
    assert answers[1].provenance["twin_id"] == run.status.twin_id
    checked = answers[2]
    assert checked.data["valid"].value and checked.data["twin_id"].value == "" and checked.data["state"].value == "computed"
    assert [w.code for w in checked.warnings] == ["W_TWIN_NO_SEED"]
    assert not [key for answer in answers for key in answer.data if "seed" in key]
    with pytest.raises(TwinConfigError):
        run_twin(folder)  # et le dev ne relance rien

    # Une recette modifiée après le calcul reste détectée sans la graine.
    edited = (folder / "twin.yaml").read_text(encoding="utf-8").replace("nb_iter: 10", "nb_iter: 11")
    (folder / "twin.yaml").write_text(edited, encoding="utf-8")
    assert twin_status(folder).state == "modified"
    with pytest.raises(UserError) as frozen:
        api.ask("twin_01", "provenance")
    assert frozen.value.code == "E_TWIN_FROZEN"
