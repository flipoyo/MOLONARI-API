"""Virtual lab (T5) : lecture de `lab.yaml`."""

import shutil

import pytest
from conftest import FIXTURES
from molonari_twin.twin.lab import LabConfigError, read_lab

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
