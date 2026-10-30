"""Virtual lab (T5, T6) : `lab.yaml` et `molonari-twin lab build`."""

import json
import shutil

import pytest
from conftest import FIXTURES
from molonari_twin.processing.cleanup import read_cleaned_csv
from molonari_twin.twin.config import frame_digest
from molonari_twin.twin.lab import LabConfigError, build_point, read_lab
from molonari_twin.twin.processed import MANIFEST_NAME, MEASUREMENTS_NAME

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
