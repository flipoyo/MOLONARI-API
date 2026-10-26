"""Pipeline du brut au résultat (P2.1) : du dossier d'un point et d'un laboratoire aux entrées d'une colonne,
et refus explicite de ce qui n'est pas pris en charge."""

import shutil
import subprocess
import sys
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from conftest import EXAMPLE_TWINS, FIXTURES
from molonari_twin.domain.models import CLEANED_COLUMNS
from molonari_twin.errors import Issue, UserError
from molonari_twin.processing.cleanup import CleanupSettings, read_cleaned_csv
from molonari_twin.twin.pipeline import (
    clean_point,
    column_inputs,
    load_point,
    read_raw_point,
    resolve_instruments,
    run_pipeline,
)

LAB = FIXTURES / "lab"
POINT034 = FIXTURES / "point034"
# Fenêtre de twin_01 : installation et retrait du dispositif exclus.
WINDOW = CleanupSettings(start=datetime(2016, 6, 27, 13, 0), end=datetime(2016, 7, 12, 10, 15))


@pytest.fixture
def point(tmp_path):
    """Copie modifiable du dossier de Point034 (mesures et fichier info seulement)."""
    folder = tmp_path / "Point034"
    folder.mkdir()
    for name in ("Point034_info.csv", "point034_P_measures.csv", "point034_T_measures.csv"):
        shutil.copy(POINT034 / name, folder / name)
    return folder


def refusal(call) -> UserError:
    with pytest.raises(UserError) as error:
        call()
    return error.value


# Du brut au résultat ---------------------------------------------------------------------------------------------


def test_raw_point034_gives_the_cleaned_measures_of_the_example_twin():
    """Critère de P2.1 : le brut de Point034 redonne les mesures nettoyées (twin_01/measurements.csv, exportées
    avec 6 chiffres significatifs, d'où la tolérance sur la charge)."""
    expected = read_cleaned_csv(EXAMPLE_TWINS / "twin_01" / "measurements.csv")
    result = run_pipeline(POINT034, LAB)
    assert len(result.cleaned) == len(expected) == 1436
    pd.testing.assert_frame_equal(result.cleaned, expected, check_exact=False, rtol=1e-5, atol=1e-7)
    assert list(result.cleaned.columns) == CLEANED_COLUMNS


def test_the_window_of_the_example_twin_keeps_the_same_measures():
    expected = read_cleaned_csv(EXAMPLE_TWINS / "twin_01" / "measurements.csv")
    expected = expected[(expected["date"] >= WINDOW.start) & (expected["date"] <= WINDOW.end)].reset_index(drop=True)
    result = run_pipeline(POINT034, LAB, WINDOW)
    assert len(result.cleaned) == len(expected) == 1430
    pd.testing.assert_frame_equal(result.cleaned, expected, check_exact=False, rtol=1e-5, atol=1e-7)
    assert result.cleanup.counts() == {"outside_window": 6}


def test_column_inputs_come_from_the_laboratory():
    """Profondeurs de la tige S007, σ de pression du capteur P508 (Sigma_Meas_P) et σ de température du
    thermomètre T001 (Sigma_Meas_T)."""
    result = run_pipeline(POINT034, LAB, WINDOW)
    inputs = result.inputs
    assert inputs.sensor_depths == (0.1, 0.2, 0.3, 0.4) and inputs.offset == 0.0
    assert (inputs.sigma_pressure, inputs.sigma_temperature) == (0.01, 0.03)
    assert inputs.nb_times == 1430 and inputs.t_sensors.shape == (1430, 3)
    instruments = result.point.instruments
    assert (instruments.pressure_sensor.name, instruments.shaft.name, instruments.thermometer.name) == ("P508", "S007", "T001")
    assert result.point.raw.name == "Point034" and result.point.lab_issues == []


def test_the_steps_chain_like_run_pipeline():
    loaded = load_point(POINT034, LAB)
    cleanup = clean_point(loaded, WINDOW)
    inputs = column_inputs(loaded, cleanup)
    whole = run_pipeline(POINT034, LAB, WINDOW).inputs
    np.testing.assert_array_equal(inputs.head, whole.head)
    np.testing.assert_array_equal(inputs.times, whole.times)


def test_cleanup_settings_are_applied():
    settings = CleanupSettings(methods={**CleanupSettings().methods, "t1": "iqr"}, excluded=[datetime(2016, 6, 28, 0, 0)])
    result = run_pipeline(POINT034, LAB, settings)
    counts = result.cleanup.counts()
    assert counts["manual"] == 1 and counts.get("iqr:t1", 0) > 0
    assert len(result.cleaned) == 1436 - sum(counts.values())


def test_sigmas_can_be_given_explicitly():
    inputs = run_pipeline(POINT034, LAB, sigma_temperature=0.2, sigma_pressure=0.02).inputs
    assert (inputs.sigma_pressure, inputs.sigma_temperature) == (0.02, 0.2)


@pytest.mark.parametrize("name, value", [("sigma_temperature", 0.0), ("sigma_pressure", -0.01)])
def test_a_sigma_that_is_not_positive_is_refused(name, value):
    error = refusal(lambda: run_pipeline(POINT034, LAB, **{name: value}))
    assert error.code == "E_PIPELINE_SIGMA_POSITIVE" and error.params == {"name": name, "sigma": value}


def test_a_laboratory_sigma_that_is_not_positive_is_refused(tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    path = lab / "pressure_sensors" / "P508.csv"
    path.write_text(path.read_text(encoding="utf-8").replace("Sigma_Meas_P,0.01", "Sigma_Meas_P,0"), encoding="utf-8")
    error = refusal(lambda: run_pipeline(POINT034, lab))
    assert (error.code, error.params["name"]) == ("E_PIPELINE_SIGMA_POSITIVE", "P508")


# Ce qui n'est pas pris en charge ---------------------------------------------------------------------------------


def test_a_time_zone_is_refused():
    """P2.1f n'est pas faite : les dates restent sans fuseau, et un fuseau en paramètre est refusé, pas ignoré."""
    for call in (lambda: run_pipeline(POINT034, LAB, timezone="Europe/Paris"), lambda: load_point(POINT034, LAB, timezone="UTC")):
        assert refusal(call).code == "E_PIPELINE_TIMEZONE"


def test_spreadsheet_formats_are_refused_as_unsupported(point):
    (point / "point034_P_measures.csv").unlink()
    (point / "point034_P_measures.xlsx").write_bytes(b"PK\x03\x04")
    (point / "point034_P_measures.hobo").write_bytes(b"\x00")
    error = refusal(lambda: read_raw_point(point))
    assert error.code == "E_PIPELINE_FORMAT_UNSUPPORTED"
    assert error.params["files"] == "point034_P_measures.hobo, point034_P_measures.xlsx"
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_PIPELINE_FORMAT_UNSUPPORTED"


@pytest.mark.parametrize("missing, code", [("point034_P_measures.csv", "E_PIPELINE_NO_PRESSURE_FILE"),
                                           ("point034_T_measures.csv", "E_PIPELINE_NO_TEMPERATURE_FILE")])
def test_a_missing_measure_file_is_refused(point, missing, code):
    (point / missing).unlink()
    error = refusal(lambda: read_raw_point(point))
    assert (error.code, error.params["folder"]) == (code, point)


def test_a_missing_measure_file_is_only_a_warning_when_not_required(point):
    (point / "point034_P_measures.csv").unlink()
    raw = read_raw_point(point, require_measures=False)
    assert raw.pressure is None and raw.temperature is not None
    assert [w.code for w in raw.warnings] == ["E_PIPELINE_NO_PRESSURE_FILE"] and isinstance(raw.warnings[0], Issue)
    (point / "point034_T_measures.csv").unlink()
    assert refusal(lambda: read_raw_point(point, require_measures=False)).code == "E_PIPELINE_NO_TEMPERATURE_FILE"


def test_a_folder_that_does_not_exist_is_refused(tmp_path):
    assert refusal(lambda: read_raw_point(tmp_path / "nowhere")).code == "E_FOLDER_NOT_FOUND"
    assert refusal(lambda: run_pipeline(POINT034, tmp_path / "nowhere")).code == "E_FOLDER_NOT_FOUND"


def test_pressure_in_kilopascals_is_refused(point):
    (point / "point034_P_measures.csv").write_text(
        '"Titre de tracé : 1"\n"#","Date Heure, GMT+01:00","Pression, kPa (LGR S/N: 1)","Temp., °C (LGR S/N: 1)"\n'
        "1,06/27/16 12:00:00 PM,101.3,18.3\n2,06/27/16 12:15:00 PM,101.4,18.1\n", encoding="utf-8")
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_MEASURES_UNIT_UNSUPPORTED"


def test_a_file_altered_by_a_spreadsheet_is_refused(point):
    shutil.copy(FIXTURES / "hobo_spreadsheet_altered_T.csv", point / "point034_T_measures.csv")
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_MEASURES_NOT_CHRONOLOGICAL"


def info_without(point, key):
    path = point / "Point034_info.csv"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(key)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_a_point_that_does_not_name_its_instruments_is_refused(point):
    info_without(point, "Shaft_Name")
    assert refusal(lambda: run_pipeline(point, LAB)).code == "E_PIPELINE_NO_SHAFT"


def test_a_point_without_info_is_refused(point):
    (point / "Point034_info.csv").unlink()
    error = refusal(lambda: run_pipeline(point, LAB))
    assert (error.code, error.params["folder"]) == ("E_PIPELINE_NO_INFO", point)


def test_a_pressure_sensor_missing_from_the_laboratory_is_refused(point, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    (lab / "pressure_sensors" / "P508.csv").unlink()
    error = refusal(lambda: run_pipeline(point, lab))
    assert (error.code, error.params["name"]) == ("E_POINT_PRESSURE_SENSOR_UNKNOWN", "P508")


def test_a_shaft_missing_from_the_laboratory_is_refused(point, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    (lab / "shafts" / "S007.csv").unlink()
    assert refusal(lambda: run_pipeline(point, lab)).code == "E_POINT_SHAFT_UNKNOWN"


def test_a_thermometer_missing_from_the_laboratory_is_refused_unless_sigma_is_given(point, tmp_path):
    lab = tmp_path / "lab"
    shutil.copytree(LAB, lab)
    (lab / "temperature_sensors" / "T001.csv").unlink()
    error = refusal(lambda: run_pipeline(point, lab))
    assert (error.code, error.params["name"], error.params["shaft"]) == ("E_PIPELINE_THERMOMETER_UNKNOWN", "T001", "S007")
    assert run_pipeline(point, lab, sigma_temperature=0.05).inputs.sigma_temperature == 0.05


def test_resolve_instruments_needs_the_info_file():
    raw = read_raw_point(POINT034)
    raw.info = None
    assert refusal(lambda: resolve_instruments(raw, load_point(POINT034, LAB).lab)).code == "E_PIPELINE_NO_INFO"


def test_every_pipeline_refusal_is_structured(point):
    """Un refus est une UserError à code stable E_… : le bord (CLI, API) la traduit à partir de son code."""
    (point / "point034_P_measures.csv").unlink()
    (point / "p.xlsx").write_bytes(b"PK")
    error = refusal(lambda: run_pipeline(point, LAB))
    assert error.code.startswith("E_PIPELINE_") and error.params["folder"] == point
    assert "Only the .csv exports" in str(error)


def test_the_pipeline_does_not_load_the_engine():
    code = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from molonari_twin.twin.pipeline import run_pipeline
        run_pipeline(Path({str(POINT034)!r}), Path({str(LAB)!r}))
        print(sorted({{n.split(".")[0] for n in sys.modules}} & {{"jax", "jaxlib", "arviz", "PySide6", "fastapi"}}))
    """)
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300, check=False)
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert completed.stdout.strip() == "[]"
