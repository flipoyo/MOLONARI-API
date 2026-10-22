"""Fichier projet .molo : contenu, intégrité, migrations ; import des anciennes bases Molonaviz."""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES, REPO_ROOT
from molonari_twin.domain.models import PointStage, SamplingPoint, Study, Thermometer
from molonari_twin.errors import UserError
from molonari_twin.io.lab_import import import_lab_directory
from molonari_twin.io.legacy_import import import_legacy_database
from molonari_twin.io.point_import import read_pressure_file, read_temperature_file
from molonari_twin.storage import project as project_module
from molonari_twin.storage.project import Project


@pytest.fixture
def project(tmp_path):
    p = Project.create(tmp_path / "test.molo")
    yield p
    p.close()


@pytest.fixture
def filled(project):
    project.add_lab(import_lab_directory(FIXTURES / "lab", name="Lab").lab)
    project.add_study(Study("Etude", "Lab"))
    point = SamplingPoint(
        name="Point034", study="Etude", pressure_sensor="P508", shaft="S007", setup_date=datetime(2016, 6, 27, 12),
        river_bed=0.505, offset=0.0, notice="Point 34", scheme=(FIXTURES / "point034" / "point34_schema.png").read_bytes(),
    )
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame
    project.add_point(point, raw_t, raw_p)
    return project, raw_t, raw_p


def test_round_trip_after_reopening(filled, tmp_path):
    project, raw_t, raw_p = filled
    project.close()
    reopened = Project.open(tmp_path / "test.molo")
    lab = reopened.get_lab("Lab")
    assert len(lab.thermometers) == 3 and len(lab.pressure_sensors) == 9 and len(lab.shafts) == 7
    assert reopened.studies() == [Study("Etude", "Lab")]
    point = reopened.get_point("Etude", "Point034")
    assert point.setup_date == datetime(2016, 6, 27, 12) and point.river_bed == 0.505
    assert point.scheme == (FIXTURES / "point034" / "point34_schema.png").read_bytes()
    pd.testing.assert_frame_equal(reopened.raw_temperature("Etude", "Point034"), raw_t.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(reopened.raw_pressure("Etude", "Point034"), raw_p.reset_index(drop=True), check_dtype=False)
    assert reopened.stage("Etude", "Point034") == PointStage.RAW
    reopened.close()


def test_duplicates_and_references_are_checked(filled):
    project, raw_t, raw_p = filled
    with pytest.raises(UserError, match="already exists"):
        project.add_study(Study("Etude", "Lab"))
    with pytest.raises(UserError, match="already exists"):
        project.add_lab_item("Lab", Thermometer("T001", "x", "y", 0.1))
    bad = SamplingPoint(name="Autre", study="Etude", pressure_sensor="P999", shaft="S007")
    with pytest.raises(UserError, match="P999"):
        project.add_point(bad, raw_t, raw_p)
    with pytest.raises(UserError, match="used by"):
        project.delete_lab("Lab")


def test_failed_write_leaves_project_unchanged(filled):
    project, raw_t, raw_p = filled
    broken = raw_p.copy()
    broken["date"] = "pas une date"  # échoue pendant l'insertion des mesures, après l'insertion du point
    point = SamplingPoint(name="Point035", study="Etude", pressure_sensor="P505", shaft="S003")
    with pytest.raises(ValueError):
        project.add_point(point, raw_t, broken)
    assert project.point_names("Etude") == ["Point034"]


def test_stages_cleaned_and_computations(filled):
    project, raw_t, raw_p = filled
    cleaned = raw_t.merge(raw_p, on="date")
    cleaned["pressure"] = 0.01
    cleaned = cleaned[["date", "temp_bed", "t1", "t2", "t3", "t4", "pressure"]]
    project.set_cleaned("Etude", "Point034", cleaned, {"methods": {}})
    assert project.stage("Etude", "Point034") == PointStage.CLEANED
    assert project.cleanup_settings("Etude", "Point034") == {"methods": {}}

    arrays = {"temperature": np.arange(6.0).reshape(2, 3), "times": np.array(["2016-06-27T12:00"], "datetime64[s]")}
    cid = project.add_computation("Etude", "Point034", "direct", {"nb_cells": 2}, arrays, 1.5, "engine", "CPU")
    assert project.stage("Etude", "Point034") == PointStage.DIRECT_MODEL
    loaded = project.load_result(cid)
    np.testing.assert_array_equal(loaded["temperature"], arrays["temperature"])
    np.testing.assert_array_equal(loaded["times"], arrays["times"])
    info = project.computations("Etude", "Point034")[0]
    assert (info.kind, info.config, info.engine, info.device, info.duration_s) == ("direct", {"nb_cells": 2}, "engine", "CPU", 1.5)

    # De nouvelles mesures nettoyées rendent les calculs obsolètes : ils sont supprimés.
    project.set_cleaned("Etude", "Point034", cleaned, {"methods": {}})
    assert project.computations("Etude", "Point034") == []
    project.reset_point("Etude", "Point034")
    assert project.stage("Etude", "Point034") == PointStage.RAW
    assert project.cleanup_settings("Etude", "Point034") is None


def test_deleting_a_study_removes_its_points(filled):
    project, _, _ = filled
    project.delete_study("Etude")
    assert project.studies() == []
    assert project._con.execute("SELECT COUNT(*) FROM raw_temperature").fetchone()[0] == 0


def test_opening_other_files(tmp_path):
    with pytest.raises(UserError, match="not found"):
        Project.open(tmp_path / "absent.molo")
    junk = tmp_path / "junk.molo"
    junk.write_text("not sqlite")
    with pytest.raises(UserError, match="not a Molonaviz project"):
        Project.open(junk)
    legacy = tmp_path / "Molonari.sqlite"
    con = sqlite3.connect(legacy)
    con.executescript((REPO_ROOT / "src/molonaviz/docs/ERD_structure.sql").read_text())
    con.close()
    with pytest.raises(UserError, match="Import Molonaviz database"):
        Project.open(legacy)


def test_newer_schema_is_refused(project, tmp_path):
    project._con.execute(f"PRAGMA user_version = {project_module.SCHEMA_VERSION + 1}")
    project.close()
    with pytest.raises(UserError, match="newer version"):
        Project.open(tmp_path / "test.molo")


def test_migration_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(project_module, "MIGRATIONS", {0: project_module._SCHEMA_V1 + " CREATE TABLE labs (x);"})
    with pytest.raises(sqlite3.OperationalError):
        Project.create(tmp_path / "broken.molo")
    con = sqlite3.connect(tmp_path / "broken.molo")
    assert con.execute("PRAGMA user_version").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] == 0


# Import des anciennes bases -------------------------------------------------------------


def make_legacy_database(directory):
    """Base Molonaviz minimale construite avec le schéma d'origine (ERD_structure.sql)."""
    (directory / "Notices").mkdir()
    (directory / "Notices" / "notice034.txt").write_text("Notice du point 34", encoding="utf-8")
    path = directory / "Molonari.sqlite"
    con = sqlite3.connect(path)
    con.executescript((REPO_ROOT / "src/molonaviz/docs/ERD_structure.sql").read_text())
    con.executescript(
        """
        INSERT INTO Labo (ID, Name) VALUES (1, 'Lab');
        INSERT INTO Thermometer (ID, Name, ManuName, ManuRef, Error, beta, V, Labo) VALUES (1, 'T001', 'Mines', 'Corefge', 0.03, 0, 0, 1);
        INSERT INTO PressureSensor (ID, Name, Datalogger, Calibration, Intercept, DuDH, DuDT, Error, ThermoModel, Labo)
            VALUES (1, 'P508', 'Hobo', '2016/08/12 10:10:10', 1.5, -8.9, -0.01, 0.01, 1, 1);
        INSERT INTO Shaft (ID, Name, Datalogger, Depth1, Depth2, Depth3, Depth4, ThermoModel, Labo)
            VALUES (1, 'S007', 'Hobo', 0.1, 0.2, 0.3, 0.4, 1, 1);
        INSERT INTO Study (ID, Name, Labo) VALUES (1, 'Etude', 1);
        INSERT INTO SamplingPoint (ID, Name, Notice, Setup, LastTransfer, "Offset", RiverBed, Shaft, PressureSensor, Study, Scheme, CleanupScript)
            VALUES (1, 'Point034', '/ancien/chemin/Notices/notice034.txt', '2016/06/27 12:00:00', '2016/07/12 10:30:00', 0.0, 0.505, 1, 1, 1, '', '');
        INSERT INTO RawMeasuresTemp (Date, Temp1, Temp2, Temp3, Temp4, SamplingPoint) VALUES
            ('2016/06/27 12:00:00', 13.8, 13.9, 13.9, 13.8, 1), ('2016/06/27 12:15:00', 13.9, 13.9, 14.0, 13.9, 1);
        INSERT INTO RawMeasuresPress (Date, TempBed, Voltage, SamplingPoint) VALUES
            ('2016/06/27 12:00:00', 18.3, 1.27, 1), ('2016/06/27 12:15:00', 13.9, 1.07, 1);
        INSERT INTO Point (ID, SamplingPoint) VALUES (1, 1);
        INSERT INTO Date (ID, Date, PointKey) VALUES (1, '2016/06/27 12:00:00', 1), (2, '2016/06/27 12:15:00', 1);
        INSERT INTO CleanedMeasures (Date, TempBed, Temp1, Temp2, Temp3, Temp4, Pressure, PointKey) VALUES
            (1, 18.3, 13.8, 13.9, 13.9, 13.8, 0.02, 1), (2, 13.9, 13.9, 13.9, 14.0, 13.9, 0.03, 1);
        """
    )
    con.close()
    return path


def test_legacy_import(project, tmp_path):
    legacy = make_legacy_database(tmp_path)
    report = import_legacy_database(legacy, project)
    assert (report.labs, report.studies, report.points, report.cleaned_points) == (1, 1, 1, 1)
    assert report.warnings == []
    lab = project.get_lab("Lab")
    assert lab.pressure_sensors[0].calibration_date == datetime(2016, 8, 12, 10, 10, 10)
    assert lab.shafts[0].depths == (0.1, 0.2, 0.3, 0.4)
    point = project.get_point("Etude", "Point034")
    assert point.notice == "Notice du point 34"  # retrouvée à côté de la base malgré le chemin absolu périmé
    assert point.river_bed == 0.505
    assert len(project.raw_temperature("Etude", "Point034")) == 2
    cleaned = project.cleaned("Etude", "Point034")
    assert cleaned["pressure"].tolist() == [0.02, 0.03]
    assert project.stage("Etude", "Point034") == PointStage.CLEANED


def test_legacy_import_twice_skips_existing(project, tmp_path):
    legacy = make_legacy_database(tmp_path)
    import_legacy_database(legacy, project)
    report = import_legacy_database(legacy, project)
    assert (report.labs, report.studies, report.points) == (0, 0, 0)
    assert len(report.warnings) == 2


def test_layers_of_older_projects_with_lateral_exchange_are_read():
    """Les projets plus anciens enregistrent q_s et t_lateral par couche : ces clés sont ignorées."""
    from molonari_twin.domain.compute import ForwardConfig, InversionConfig, LayerSpec

    stored = ForwardConfig([LayerSpec("a", 0.4)]).to_dict()
    stored["layers"][0].update(q_s=0.0, t_lateral=None)
    assert ForwardConfig.from_dict(stored).layers == [LayerSpec("a", 0.4)]
    stored = InversionConfig([LayerSpec("a", 0.4)]).to_dict()
    stored["layers"][0].update(q_s=0.0, t_lateral=None)
    assert InversionConfig.from_dict(stored).layers == [LayerSpec("a", 0.4)]


def small_inversion_result():
    from molonari_twin.engine.results import (
        ForwardResult,
        InversionResult,
        QuantileResult,
    )

    rng = np.random.default_rng(0)
    times = np.array(["2016-06-27T12:00", "2016-06-27T12:15"], "datetime64[s]")
    best = ForwardResult(
        depths=np.linspace(0.05, 0.35, 4), times=times, temperature=rng.normal(size=(4, 2)), water_flux=rng.normal(size=(4, 2)),
        advective_flux=rng.normal(size=(4, 2)), conductive_flux=rng.normal(size=(4, 2)), sensor_index=np.array([0, 1, 2]),
        observed=rng.normal(size=(2, 3)), rmse=rng.normal(size=4),
    )
    quantiles = [QuantileResult(q, rng.normal(size=(4, 2)), rng.normal(size=(4, 2)), rng.normal(size=4)) for q in (0.05, 0.5, 0.95)]
    return InversionResult(
        best=best, best_parameters=rng.normal(size=(1, 4)), samples=rng.normal(size=(5, 3, 1, 4)), energies=rng.normal(size=(5, 3)),
        acceptance=rng.uniform(size=3), burn_in_iterations=20, burn_in_converged=True, rhat=rng.normal(size=(1, 4)), quantiles=quantiles,
        field_depths=best.depths, field_times=times, thin=2, field_chain=np.tile(np.arange(3), 2),
        field_iteration=np.repeat(np.array([4, 8]), 3), resets=np.array([0, 1, 0]),
        sensor_temperature=rng.normal(size=(6, 3, 2)), field_temperature=rng.normal(size=(6, 4, 2)).astype(np.float32),
        field_water_flux=rng.normal(size=(6, 4, 2)).astype(np.float32),
    )


def test_inversion_result_round_trip_through_a_project(filled):
    from molonari_twin.engine.results import OPTIONAL_ARRAYS, InversionResult

    project = filled[0]
    result = small_inversion_result()
    cid = project.add_computation("Etude", "Point034", "inversion", {}, result.to_arrays(), 1.0, "engine", "CPU")
    loaded = InversionResult.from_arrays(project.load_result(cid))
    assert loaded.thin == 2
    for name in ("samples", "energies", "acceptance", *OPTIONAL_ARRAYS):
        expected, actual = getattr(result, name), getattr(loaded, name)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape, name
        np.testing.assert_array_equal(actual, expected)


def test_inversion_result_of_0_3_0_is_read_without_the_new_arrays(filled):
    from molonari_twin.engine.results import OPTIONAL_ARRAYS, InversionResult

    project = filled[0]
    arrays = small_inversion_result().to_arrays()
    for name in ("thin", *OPTIONAL_ARRAYS):
        del arrays[name]
    cid = project.add_computation("Etude", "Point034", "inversion", {}, arrays, 1.0, "engine", "CPU")
    loaded = InversionResult.from_arrays(project.load_result(cid))
    assert loaded.thin == 1 and all(getattr(loaded, name) is None for name in OPTIONAL_ARRAYS)
    np.testing.assert_array_equal(loaded.samples, arrays["samples"])
