"""Virtual lab (T5 à T7) : `lab.yaml`, `molonari-twin lab build`, et la lecture de `processed_data` par un twin 2.0.0.

Le test central : un twin depuis le brut et le même depuis `processed_data` ont le même `twin_id` et le même résultat."""

import json
import shutil

import numpy as np
import pytest
from conftest import FIXTURES
from molonari_twin.processing.cleanup import read_cleaned_csv
from molonari_twin.twin.config import TwinConfigError, frame_digest, load_twin
from molonari_twin.twin.lab import LabConfigError, build_point, read_lab
from molonari_twin.twin.processed import MANIFEST_NAME, MEASUREMENTS_NAME
from molonari_twin.twin.run import run_twin

LAB = """schema: molonari-lab/1.0.0
name: lab_point034
laboratory: Laboratory
code_version: {version}
points:
  Point034:
    folder: SamplingPoints/Point034
    processing:
      start: 2016-06-27 13:00
      end: 2016-07-01 10:15
      methods: {{t1: iqr}}
"""

LAYERS = """layers:
  - {name: couche 1, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5000000}
numerics: {nb_cells: 40}
"""

FROM_RAW = """schema: molonari-twin/2.0.0
name: twin_01
mode: direct
data: {lab: Laboratory, point: SamplingPoints/Point034}
processing:
  start: 2016-06-27 13:00
  end: 2016-07-01 10:15
  methods: {t1: iqr}
""" + LAYERS

FROM_PROCESSED = """schema: molonari-twin/2.0.0
name: twin_01
mode: direct
data: {processed: processed_data/Point034}
""" + LAYERS

MCMC = """mode: mcmc
layers:
  - {name: couche 1, z_low: 0.4}
mcmc: {nb_chains: 7, nb_iter: 10, burn_in_max: 10, n_field_samples: 7}
"""


def current_version():
    from molonari_twin import __version__

    return __version__


@pytest.fixture
def data_repo(tmp_path, monkeypatch):
    """Dépôt de données minimal (Laboratory/ et SamplingPoints/Point034/, copiés des fixtures) et le dossier du lab."""
    data = tmp_path / "data"
    shutil.copytree(FIXTURES / "lab", data / "Laboratory")
    shutil.copytree(FIXTURES / "point034", data / "SamplingPoints" / "Point034")
    monkeypatch.setenv("MOLONARI_DATA", str(data))
    return data


@pytest.fixture
def lab(tmp_path, data_repo):
    folder = tmp_path / "virtual_lab" / "lab_point034"
    folder.mkdir(parents=True)
    (folder / "lab.yaml").write_text(LAB.format(version=current_version()), encoding="utf-8")
    return folder


def write_twin(tmp_path, text, name="twin_01", group="twins"):
    folder = tmp_path / group / name
    folder.mkdir(parents=True)
    (folder / "twin.yaml").write_text(text, encoding="utf-8")
    return folder


def codes(error):
    return [(issue.field, issue.code) for issue in error.value.issues]


def processed_bytes(data):
    folder = data / "processed_data" / "Point034"
    return (folder / MEASUREMENTS_NAME).read_bytes(), (folder / MANIFEST_NAME).read_bytes()


# T5 : lab.yaml -------------------------------------------------------------------------------------------------


def test_lab_yaml_is_read(lab, data_repo):
    spec = read_lab(lab)
    assert (spec.name, spec.laboratory, spec.code_version) == ("lab_point034", "Laboratory", current_version())
    assert spec.root == data_repo.resolve() and spec.warnings == []
    point = spec.points["Point034"]
    assert point.folder == "SamplingPoints/Point034"
    assert point.processing.methods["t1"] == "iqr" and point.processing.methods["t2"] == "none"
    assert point.processing.start.isoformat() == "2016-06-27T13:00:00"
    assert read_lab(lab / "lab.yaml").name == "lab_point034"  # le fichier lui-même est admis


def test_lab_yaml_code_version_is_optional_and_a_difference_is_only_a_warning(lab, data_repo):
    (lab / "lab.yaml").write_text(LAB.format(version="0.0.1"), encoding="utf-8")
    spec = read_lab(lab)
    assert [w.code for w in spec.warnings] == ["W_LAB_CODE_VERSION"]
    assert spec.warnings[0].params["declared"] == "0.0.1" and spec.warnings[0].params["current"] == current_version()
    (lab / "lab.yaml").write_text(LAB.format(version="x").replace("code_version: x\n", ""), encoding="utf-8")
    assert read_lab(lab).code_version is None and read_lab(lab).warnings == []


def test_lab_yaml_refuses_unknown_keys_and_reports_every_problem(lab, data_repo):
    text = LAB.format(version="1").replace("name: lab_point034", "nam: lab_point034").replace("end: 2016-07-01 10:15", "end: 2016-06-01")
    (lab / "lab.yaml").write_text(text + "extra: 1\n", encoding="utf-8")
    with pytest.raises(LabConfigError) as error:
        read_lab(lab)
    found = codes(error)
    assert ("nam", "E_TWIN_UNKNOWN_KEY") in found and ("extra", "E_TWIN_UNKNOWN_KEY") in found
    assert ("name", "E_TWIN_REQUIRED") in found
    assert ("points.Point034.processing.end", "E_TWIN_END_BEFORE_START") in found
    assert "nam: unknown key (allowed: code_version, laboratory, name, points, schema)." in error.value.issues
    # une clé inconnue dans un point, dans son nettoyage, et un schéma d'un autre format
    text = LAB.format(version="1").replace("folder:", "dossier:").replace("methods: {t1: iqr}", "methodes: {}")
    (lab / "lab.yaml").write_text(text.replace("molonari-lab/1.0.0", "molonari-lab/2"), encoding="utf-8")
    with pytest.raises(LabConfigError) as error:
        read_lab(lab)
    found = codes(error)
    assert ("schema", "E_TWIN_SCHEMA") in found
    assert ("points.Point034.dossier", "E_TWIN_UNKNOWN_KEY") in found and ("points.Point034.folder", "E_TWIN_REQUIRED") in found
    assert ("points.Point034.processing.methodes", "E_TWIN_UNKNOWN_KEY") in found


def test_lab_yaml_refuses_missing_files_and_paths_outside_the_data(lab, data_repo):
    text = LAB.format(version="1").replace("laboratory: Laboratory", "laboratory: ../elsewhere").replace(
        "folder: SamplingPoints/Point034", "folder: SamplingPoints/Nope")
    (lab / "lab.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(LabConfigError) as error:
        read_lab(lab)
    assert codes(error) == [("laboratory", "E_TWIN_DATA_PATH"), ("points.Point034.folder", "E_FOLDER_NOT_FOUND")]
    with pytest.raises(LabConfigError) as error:
        read_lab(lab.parent / "absent")
    assert codes(error) == [(None, "E_FILE_NOT_FOUND")]
    (lab / "lab.yaml").write_text("points: [\n", encoding="utf-8")
    with pytest.raises(LabConfigError) as error:
        read_lab(lab)
    assert codes(error) == [(None, "E_TWIN_INVALID_YAML")]
    (lab / "lab.yaml").write_text(LAB.format(version="1").replace("Point034:\n", "../Point034:\n", 1), encoding="utf-8")
    with pytest.raises(LabConfigError) as error:
        read_lab(lab)
    assert ("points.../Point034", "E_TWIN_DATA_PATH") in codes(error)  # un nom de point est un nom de dossier


def test_data_root_comes_from_the_lab_folder_unless_molonari_data_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("MOLONARI_DATA", raising=False)
    workspace = tmp_path / "workspace"
    shutil.copytree(FIXTURES / "lab", workspace / "data" / "Laboratory")
    shutil.copytree(FIXTURES / "point034", workspace / "data" / "SamplingPoints" / "Point034")
    (workspace / "molonari.cgs").write_text("", encoding="utf-8")
    folder = workspace / "virtual_lab" / "lab_point034"
    folder.mkdir(parents=True)
    (folder / "lab.yaml").write_text(LAB.format(version=current_version()), encoding="utf-8")
    assert read_lab(folder).root == (workspace / "data").resolve()
    assert build_point(folder, "Point034").folder == (workspace / "data").resolve() / "processed_data" / "Point034"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setenv("MOLONARI_DATA", str(elsewhere))
    with pytest.raises(LabConfigError) as error:  # MOLONARI_DATA l'emporte
        read_lab(folder)
    assert ("laboratory", "E_FOLDER_NOT_FOUND") in codes(error)
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / "lab.yaml").write_text(LAB.format(version="1"), encoding="utf-8")
    monkeypatch.delenv("MOLONARI_DATA")
    with pytest.raises(LabConfigError) as error:
        read_lab(lonely)
    assert ("laboratory", "E_LAB_DATA_ROOT") in codes(error)


# T6 : lab build ------------------------------------------------------------------------------------------------


def test_build_writes_the_cleaned_measurements_and_the_manifest(lab, data_repo):
    result = build_point(lab, "Point034")
    folder = data_repo.resolve() / "processed_data" / "Point034"
    assert result.folder == folder and sorted(p.name for p in folder.iterdir()) == [MANIFEST_NAME, MEASUREMENTS_NAME]
    assert not list((data_repo / "processed_data").glob("*.partial")) and not list((data_repo / "processed_data").glob("*.old"))
    manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert result.manifest == manifest
    assert manifest["schema"] == "molonari-processed/1.0.0" and manifest["lab"] == "lab_point034" and manifest["point"] == "Point034"
    assert manifest["code_version"] == current_version()
    assert set(manifest["raw_files"]) == {"Point034_info.csv", "point034_P_measures.csv", "point034_T_measures.csv"}
    assert set(manifest["lab_files"]) == {"pressure_sensors/P508.csv", "shafts/S007.csv", "temperature_sensors/T001.csv"}
    import hashlib

    for name, digest in manifest["raw_files"].items():
        assert hashlib.sha256((data_repo / "SamplingPoints" / "Point034" / name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["lab_files"].items():
        assert hashlib.sha256((data_repo / "Laboratory" / name).read_bytes()).hexdigest() == digest
    assert len(manifest["depths"]) == 4 and manifest["sigma_pressure"] > 0 and manifest["sigma_temperature"] > 0
    assert manifest["utc_offset"] == "+02:00" and manifest["offset"] == 0.0
    assert manifest["processing"]["methods"]["t1"] == "iqr" and manifest["processing"]["start"] == "2016-06-27T13:00:00"
    assert manifest["rows"] > 3 and manifest["start"] >= "2016-06-27T13:00:00" and manifest["end"] <= "2016-07-01T10:15:00"
    # l'empreinte est celle des mesures nettoyées, et aussi celle du fichier (même texte que frame_digest)
    frame = read_cleaned_csv(folder / MEASUREMENTS_NAME, exact=True)
    assert len(frame) == manifest["rows"] and frame_digest(frame) == manifest["cleaned_sha256"]
    assert hashlib.sha256((folder / MEASUREMENTS_NAME).read_bytes()).hexdigest() == manifest["cleaned_sha256"]
    assert "created" not in json.dumps(manifest) and result.warnings  # fuseau provisoire des bruts : un avertissement


def test_build_runs_the_pipeline_and_nothing_else(lab, data_repo):
    from molonari_twin.twin import pipeline

    build_point(lab, "Point034")
    base = data_repo / "SamplingPoints" / "Point034"
    spec = read_lab(lab).points["Point034"]
    expected = pipeline.run_pipeline(base, data_repo / "Laboratory", spec.processing).cleaned
    frame = read_cleaned_csv(data_repo / "processed_data" / "Point034" / MEASUREMENTS_NAME, exact=True)
    assert len(frame) == len(expected) and frame_digest(frame) == frame_digest(expected)


def test_a_second_build_gives_the_same_bytes(lab, data_repo):
    build_point(lab, "Point034")
    first = processed_bytes(data_repo)
    build_point(lab, "Point034")
    assert processed_bytes(data_repo) == first
    assert sorted(p.name for p in (data_repo / "processed_data").iterdir()) == ["Point034"]


def test_a_changed_input_gives_other_bytes_and_replaces_the_folder(lab, data_repo):
    build_point(lab, "Point034")
    first = processed_bytes(data_repo)
    (lab / "lab.yaml").write_text(LAB.format(version=current_version()).replace("2016-07-01 10:15", "2016-06-30 10:15"), encoding="utf-8")
    build_point(lab, "Point034")
    second = processed_bytes(data_repo)
    assert second[0] != first[0] and second[1] != first[1]
    assert sorted(p.name for p in (data_repo / "processed_data" / "Point034").iterdir()) == [MANIFEST_NAME, MEASUREMENTS_NAME]


def test_a_failed_build_leaves_the_previous_folder_intact(lab, data_repo, monkeypatch):
    build_point(lab, "Point034")
    first = processed_bytes(data_repo)
    from molonari_twin.twin import processed

    def explode(_):
        raise OSError("disk full")

    monkeypatch.setattr(processed, "dump_manifest", explode)
    with pytest.raises(OSError):
        build_point(lab, "Point034")
    assert processed_bytes(data_repo) == first
    assert sorted(p.name for p in (data_repo / "processed_data").iterdir()) == ["Point034"]


def test_build_refuses_a_point_that_is_not_in_lab_yaml(lab, data_repo):
    from molonari_twin.errors import UserError

    with pytest.raises(UserError) as error:
        build_point(lab, "Point047")
    assert error.value.code == "E_LAB_POINT_UNKNOWN" and "Point034" in str(error.value)
    assert not (data_repo / "processed_data").exists()


def test_build_refuses_a_cleaning_that_leaves_nothing(lab, data_repo):
    from molonari_twin.errors import UserError

    (lab / "lab.yaml").write_text(LAB.format(version="1").replace("2016-06-27 13:00", "2020-01-01").replace("2016-07-01 10:15", "2020-02-01"),
                                  encoding="utf-8")
    with pytest.raises(UserError) as error:
        build_point(lab, "Point034")
    assert error.value.code == "E_LAB_BUILD_EMPTY"
    assert not (data_repo / "processed_data").exists()


def test_the_command_line_builds_in_the_user_language(lab, data_repo, capsys, monkeypatch):
    from molonari_twin.i18n import load_catalog
    from molonari_twin.twin.cli import main

    french = load_catalog("fr")
    monkeypatch.delenv("MOLONARI_LANG", raising=False)
    monkeypatch.delenv("MOLONAVIZ_LANG", raising=False)
    assert main(["lab", "build", str(lab), "--point", "Point034"]) == 0
    out = capsys.readouterr()
    assert out.out.startswith("Données traitées écrites : ") and str(data_repo.resolve() / "processed_data" / "Point034") in out.out
    assert out.err.startswith(french["Warning:"])
    first = processed_bytes(data_repo)
    monkeypatch.setenv("MOLONARI_LANG", "en")
    assert main(["lab", "build", str(lab / "lab.yaml"), "--point", "Point034"]) == 0
    assert capsys.readouterr().out.startswith("Processed data written: ")
    assert processed_bytes(data_repo) == first
    assert main(["lab", "build", str(lab), "--point", "Point047"]) == 1
    assert capsys.readouterr().err == 'Point "Point047" is not in lab.yaml (points: Point034).\n'
    (lab / "lab.yaml").write_text("nam: x\n", encoding="utf-8")
    assert main(["lab", "build", str(lab), "--point", "Point034"]) == 1
    assert capsys.readouterr().err.startswith("Invalid laboratory file:\n  - nam: unknown key")
    (lab / "lab.yaml").write_text(LAB.format(version="0.0.1"), encoding="utf-8")
    assert main(["lab", "build", str(lab), "--point", "Point034"]) == 0
    assert "Warning: lab.yaml was written for code 0.0.1" in capsys.readouterr().err


def test_the_command_line_has_a_help_for_lab_build(capsys):
    from molonari_twin.twin.cli import main

    with pytest.raises(SystemExit) as stop:
        main(["lab", "build", "--help"])
    assert stop.value.code == 0
    assert "--point" in capsys.readouterr().out


# T7 : data.processed -------------------------------------------------------------------------------------------


def test_a_twin_from_processed_data_has_the_same_twin_id_as_from_the_raw(lab, data_repo, tmp_path):
    build_point(lab, "Point034")
    raw = load_twin(write_twin(tmp_path, FROM_RAW, group="raw"))
    processed = load_twin(write_twin(tmp_path, FROM_PROCESSED, group="processed"))
    assert not raw.from_processed and processed.from_processed and processed.raw_data
    assert processed.point == raw.point == "Point034"
    assert processed.data_processed == "processed_data/Point034" and processed.processing is None
    assert processed.depths == raw.depths and processed.offset == raw.offset
    assert (processed.sigma_pressure, processed.sigma_temperature) == (raw.sigma_pressure, raw.sigma_temperature)
    assert (processed.start, processed.end) == (raw.start, raw.end)
    assert frame_digest(processed.load_measurements()) == frame_digest(raw.load_measurements())
    assert processed.twin_id("v1") == raw.twin_id("v1")
    assert processed.canonical("v1") == raw.canonical("v1")
    a, b = raw.column_inputs(), processed.column_inputs()
    for name in ("times", "head", "t_river", "t_sensors", "t_aquifer", "sensor_depths", "offset", "sigma_pressure", "sigma_temperature"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name  # bit à bit, pas à une tolérance près


def test_the_twin_id_is_the_same_in_mcmc_mode_and_with_sigma_overrides(lab, data_repo, tmp_path):
    build_point(lab, "Point034")
    sigmas = "sensors: {sigma_temperature: 0.05, sigma_pressure: 0.02}\n"
    head_raw = FROM_RAW.split("layers:")[0].replace("mode: direct\n", "")
    head_processed = FROM_PROCESSED.split("layers:")[0].replace("mode: direct\n", "")
    assert load_twin(write_twin(tmp_path, FROM_PROCESSED + sigmas, group="p")).twin_id("v1") == load_twin(
        write_twin(tmp_path, FROM_RAW + sigmas, group="r")).twin_id("v1")
    assert load_twin(write_twin(tmp_path, FROM_PROCESSED + sigmas, group="p2")).twin_id("v1") != load_twin(
        write_twin(tmp_path, FROM_PROCESSED, group="p3")).twin_id("v1")
    ids = []
    for group, head in (("pm", head_processed), ("rm", head_raw)):
        folder = write_twin(tmp_path, head + MCMC, group=group)
        (folder / "orx.yaml").write_text("mcmc:\n  seed: 42\n", encoding="utf-8")
        ids.append(load_twin(folder).twin_id("v1"))
    assert ids[0] == ids[1]


def test_both_twins_give_the_same_result_in_direct_mode(lab, data_repo, tmp_path):
    from molonari_twin.twin.results import load_results

    build_point(lab, "Point034")
    runs = {group: run_twin(write_twin(tmp_path / group, text)) for group, text in (("raw", FROM_RAW), ("processed", FROM_PROCESSED))}
    assert runs["raw"].computed and runs["processed"].computed
    assert runs["raw"].status.twin_id == runs["processed"].status.twin_id
    raw, processed = (load_results(runs[g].status.results_dir) for g in ("raw", "processed"))
    assert np.array_equal(raw.temperature, processed.temperature) and np.array_equal(raw.times, processed.times)
    assert raw.flux.equals(processed.flux) and raw.residuals.equals(processed.residuals)
    assert raw.metadata["rmse_celsius"] == processed.metadata["rmse_celsius"] and raw.metadata["config"] == processed.metadata["config"]
    data = processed.metadata["data"]
    manifest = json.loads((data_repo / "processed_data" / "Point034" / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert data["processed"] == "processed_data/Point034" and data["lab"] == "lab_point034"
    assert data["code_version"] == current_version() and data["cleaned_sha256"] == manifest["cleaned_sha256"]
    assert "point" not in data and raw.metadata["data"]["lab"] == "Laboratory"


def test_processed_data_is_exclusive_with_lab_point_and_processing(lab, data_repo, tmp_path):
    build_point(lab, "Point034")
    text = FROM_PROCESSED.replace("data: {processed: processed_data/Point034}",
                                  "data: {processed: processed_data/Point034, lab: Laboratory, point: SamplingPoints/Point034}")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, text + "processing: {start: 2016-06-28}\n", group="a"))
    assert codes(error) == [("data.lab", "E_TWIN_PROCESSED_EXCLUSIVE"), ("data.point", "E_TWIN_PROCESSED_EXCLUSIVE"),
                            ("processing", "E_TWIN_PROCESSED_EXCLUSIVE")]
    assert str(error.value.issues[0]) == "data.lab: cannot be combined with data.processed: the processed data are already cleaned."
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED + "processing: {methods: {t1: iqr}}\n", group="b"))
    assert codes(error) == [("processing", "E_TWIN_PROCESSED_EXCLUSIVE")]


def test_raw_twins_still_need_lab_and_point_and_1x_twins_do_not_know_processed(lab, data_repo, tmp_path):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_RAW.replace("data: {lab: Laboratory, point: SamplingPoints/Point034}", "data: {}"), group="a"))
    assert codes(error) == [("data.lab", "E_TWIN_REQUIRED"), ("data.point", "E_TWIN_REQUIRED")]
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_RAW.replace("point: SamplingPoints/Point034", "point: SamplingPoints/Point034, x: 1"), group="b"))
    assert codes(error) == [("data.x", "E_TWIN_UNKNOWN_KEY")]
    assert "allowed: lab, point, processed" in str(error.value.issues[0])
    one = """schema: molonari-twin/1.1.0
name: twin_01
mode: direct
data: {processed: processed_data/Point034}
sensors: {depths: [0.1, 0.2, 0.3, 0.4]}
""" + LAYERS
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, one, group="c"))
    assert ("data.processed", "E_TWIN_UNKNOWN_KEY") in codes(error) and ("data.measurements", "E_TWIN_REQUIRED") in codes(error)


def test_processed_data_must_exist_and_be_inside_the_data_repository(lab, data_repo, tmp_path):
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="a"))  # pas encore construit
    assert codes(error) == [("data.processed", "E_FOLDER_NOT_FOUND")]
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED.replace("processed_data/Point034", "../elsewhere"), group="b"))
    assert codes(error) == [("data.processed", "E_TWIN_DATA_PATH")]
    (data_repo / "processed_data" / "Point034").mkdir(parents=True)
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="c"))  # dossier sans manifeste
    assert codes(error) == [("data.processed", "E_FILE_NOT_FOUND")]


def test_the_fingerprint_of_the_measurements_is_checked_against_the_manifest(lab, data_repo, tmp_path):
    build_point(lab, "Point034")
    folder = data_repo / "processed_data" / "Point034"
    measurements = folder / MEASUREMENTS_NAME
    original = measurements.read_text(encoding="utf-8")
    lines = original.splitlines()
    cells = lines[3].split(",")
    cells[1] = repr(float(cells[1]) + 1e-12)  # une retouche minuscule
    lines[3] = ",".join(cells)
    measurements.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="a"))
    assert codes(error) == [("data.processed", "E_PROCESSED_DIGEST")]
    measurements.write_text(original, encoding="utf-8")  # rétabli : accepté
    assert load_twin(write_twin(tmp_path, FROM_PROCESSED, group="b")).from_processed
    measurements.write_text("\n".join(original.splitlines()[:-1]) + "\n", encoding="utf-8")  # une ligne de moins
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="c"))
    assert codes(error) == [("data.processed", "E_PROCESSED_DIGEST")]


def test_a_damaged_manifest_is_refused(lab, data_repo, tmp_path):
    build_point(lab, "Point034")
    path = data_repo / "processed_data" / "Point034" / MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))

    def refused(changed, group):
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(TwinConfigError) as error:
            load_twin(write_twin(tmp_path, FROM_PROCESSED, group=group))
        return error.value.issues

    assert [i.code for i in refused({**manifest, "schema": "molonari-processed/9"}, "a")] == ["E_PROCESSED_SCHEMA"]
    broken = {k: v for k, v in manifest.items() if k not in ("offset", "cleaned_sha256")}
    issues = refused({**broken, "depths": [0.1, 0.2], "sigma_pressure": 0, "processing": {"start": "pas une date"}}, "b")
    assert [i.code for i in issues] == ["E_PROCESSED_MANIFEST_KEY"]
    assert issues[0].params["keys"] == "offset, sigma_pressure, depths, cleaned_sha256, processing"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="c"))
    assert codes(error) == [("data.processed", "E_PROCESSED_MANIFEST_UNREADABLE")]
    path.write_text("{", encoding="utf-8")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="d"))
    assert codes(error) == [("data.processed", "E_PROCESSED_MANIFEST_UNREADABLE")]
    # un manifeste dont les profondeurs ne croissent pas est refusé comme celles d'un point
    path.write_text(json.dumps({**manifest, "depths": [0.1, 0.3, 0.2, 0.4]}), encoding="utf-8")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED, group="e"))
    assert codes(error) == [("data.processed", "E_TWIN_SENSOR_DEPTHS")]


def test_processed_data_needs_a_data_repository(lab, data_repo, tmp_path, monkeypatch):
    build_point(lab, "Point034")
    monkeypatch.delenv("MOLONARI_DATA")
    with pytest.raises(TwinConfigError) as error:
        load_twin(write_twin(tmp_path, FROM_PROCESSED))
    assert codes(error) == [("data", "E_TWIN_DATA_ROOT")]
    assert load_twin(tmp_path / "twins" / "twin_01", data_folder=data_repo).from_processed


def test_processed_data_is_used_without_the_raw_files(lab, data_repo, tmp_path):
    """Le point d'un `processed_data` : on peut retirer le brut, le twin se lit et se calcule quand même."""
    build_point(lab, "Point034")
    shutil.rmtree(data_repo / "SamplingPoints")
    shutil.rmtree(data_repo / "Laboratory")
    run = run_twin(write_twin(tmp_path, FROM_PROCESSED))
    assert run.computed and run.status.config.point == "Point034"
