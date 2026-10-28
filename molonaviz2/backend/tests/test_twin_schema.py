"""Schéma molonari-twin/2.0.0 (P2.2) : le brut (data), le nettoyage (processing), la graine obligatoire (mcmc.seed),
les clés retirées, legacy_twin_id, et les empreintes par fichier des résultats (P2.2f).

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
  seed: 42
  nb_chains: 7
  nb_iter: 10
  burn_in_max: 10
  n_field_samples: 7
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


def codes(error):
    return [(issue.field, issue.code) for issue in error.value.issues]


def test_raw_twin_is_read_from_the_data_repository(tmp_path, data_repo):
    config = load_twin(write_twin(tmp_path, RAW_MCMC))
    assert config.raw_data and config.schema_version == (2, 0, 0)
    assert config.point == "Point034" and config.mode == "mcmc"
    assert len(config.depths) == 4 and config.depths[-1] > 0
    assert config.measurements is None
    assert len(config.load_measurements()) > 3
    assert config.inversion_config().seed == 42 and config.seed == 42


def test_raw_twin_without_seed_is_refused(tmp_path, data_repo):
    text = RAW_MCMC.replace("  seed: 42\n", "")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text))
    assert ("mcmc.seed", "E_TWIN_REQUIRED") in codes(error)


def test_raw_twin_seed_must_be_a_whole_number(tmp_path, data_repo):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, RAW_MCMC.replace("seed: 42", "seed: 4.2")))
    assert ("mcmc.seed", "E_TWIN_EXPECTED_INTEGER") in codes(error)


def test_seed_is_part_of_the_twin_id(tmp_path, data_repo):
    ids = {seed: load_twin(write_twin(tmp_path / str(seed), RAW_MCMC.replace("seed: 42", f"seed: {seed}"))).twin_id("v1")
           for seed in (42, 43)}
    assert ids[42] != ids[43]
    again = load_twin(write_twin(tmp_path / "again", RAW_MCMC)).twin_id("v1")
    assert again == ids[42]


def test_twin_id_does_not_depend_on_where_the_data_is(tmp_path, data_repo, monkeypatch):
    reference = load_twin(write_twin(tmp_path / "a", RAW_MCMC)).twin_id("v1")
    moved = tmp_path / "elsewhere"
    shutil.copytree(data_repo, moved)
    monkeypatch.setenv("MOLONARI_DATA", str(moved))
    assert load_twin(write_twin(tmp_path / "b", RAW_MCMC)).twin_id("v1") == reference


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
    assert ("seed", "E_TWIN_KEY_MOVED") in found
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

    config = load_twin(write_twin(tmp_path, RAW_MCMC))
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

