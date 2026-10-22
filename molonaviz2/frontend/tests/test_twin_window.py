"""Fenêtre d'un twin et création d'un twin depuis l'application (pytest-qt, hors écran)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest
from conftest import FIXTURES
from molonari_twin.domain.compute import ForwardConfig, InversionConfig, LayerSpec
from molonari_twin.io.lab_import import import_lab_directory
from molonari_twin.io.point_import import read_pressure_file, read_temperature_file
from molonari_twin.processing.cleanup import CleanupSettings, clean
from molonari_twin.twin.config import load_twin
from molonari_twin.twin.create import create_twin
from molonari_twin.twin.results import load_results
from molonari_twin.twin.run import run_twin

DEPTHS = (0.1, 0.2, 0.3, 0.4)
# Champs du format 2.0.0, avec la première cellule (profondeur|unité).
FIELD_FILES = {"water_flux_field.csv": "depth_m|m_per_s", "heat_flux_advective.csv": "depth_m|W_per_m2", "heat_flux_conductive.csv": "depth_m|W_per_m2"}


@pytest.fixture(scope="module")
def point034():
    lab = import_lab_directory(FIXTURES / "lab").lab
    sensor = next(p for p in lab.pressure_sensors if p.name == "P508")
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame
    return clean(raw_t, raw_p, sensor, CleanupSettings()).cleaned.iloc[:400]


@pytest.fixture
def twins(tmp_path):
    folder = tmp_path / "demo" / "twins"
    folder.mkdir(parents=True)
    return folder


def make_twin(twins, measurements, config=None, sigma_t=0.05):
    config = config or ForwardConfig([LayerSpec("couche 1", 0.4)], nb_cells=40)
    return create_twin(twins, measurements, "Point034", "test", DEPTHS, 0.0, sigma_t, 0.01, config)


def edit(folder, old, new):
    path = folder / "twin.yaml"
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


def as_format_1(results_dir):
    """Ramène un dossier de résultats direct au format 1.x : sans les fichiers ni les clés de 2.0.0."""
    for name in FIELD_FILES:
        (results_dir / name).unlink()
    metadata = json.loads((results_dir / "metadata.json").read_text())
    for key in ("results_format", "instruments", "conventions"):
        del metadata[key]
    (results_dir / "metadata.json").write_text(json.dumps(metadata))
    return results_dir


def small_inversion_config(**overrides):
    values = {"layers": [LayerSpec("couche 1", 0.4)], "nb_cells": 20, "nb_chains": 7, "nb_iter": 10, "burn_in_max": 10, "n_field_samples": 7}
    return InversionConfig(**{**values, **overrides})


# Interface ---------------------------------------------------------------------------------


def test_welcome_page_lists_twins_and_window_runs_them(qtbot, tmp_path, twins, point034, monkeypatch):
    from molonaviz2.settings import Settings
    from molonaviz2.ui import errors as ui_errors
    from molonaviz2.ui.main_window import MainWindow
    from PySide6 import QtCore

    shown = []
    previous = ui_errors.set_presenter(lambda *args: shown.append(args))
    folder = make_twin(twins, point034)
    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    settings.twin_folder = twins
    window = MainWindow(settings)
    qtbot.addWidget(window)
    assert window.welcome.twins.count() == 1
    assert "not run yet" in window.welcome.twins.item(0).text()

    twin_window = window.open_twin(folder)
    assert twin_window.run_button.isEnabled() and not twin_window.tabs.isVisible()
    twin_window.run()
    qtbot.waitUntil(lambda: not window.tasks.busy, timeout=120_000)
    qtbot.waitUntil(lambda: twin_window.results is not None, timeout=10_000)
    assert twin_window.status.state == "computed"
    assert not twin_window.run_button.isEnabled() and "Already computed" in twin_window.run_button.toolTip()
    assert "Infiltration" in twin_window.flux_summary.text()
    from molonaviz2.ui.units import measurement_level

    sigma = twin_window.status.config.sigma_temperature
    dashed = {round(abs(float(line.get_ydata()[0])), 9) for line in twin_window.residuals.figure.axes[0].lines if line.get_linestyle() == "--"}
    assert dashed == {round(measurement_level.z * sigma, 9)}  # résidus : lignes à ± z·σT (90 % par défaut)
    window.welcome.refresh_twins(twins)
    assert "computed" in window.welcome.twins.item(0).text()

    copy = twins / "twin_02"
    shutil.copytree(folder, copy)
    edit(copy, "name: twin_01", "name: twin_02")
    duplicate_window = window.open_twin(copy)
    assert "Duplicate of twin_01" in duplicate_window.state.text() and duplicate_window.results is not None
    ui_errors.set_presenter(previous)
    assert shown == []


def test_twin_window_warns_about_results_of_format_1(qtbot, tmp_path, twins, point034):
    from molonaviz2.settings import Settings
    from molonaviz2.ui.main_window import MainWindow
    from PySide6 import QtCore

    folder = make_twin(twins, point034)
    as_format_1(run_twin(folder).status.results_dir)
    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    window = MainWindow(settings)
    qtbot.addWidget(window)
    twin_window = window.open_twin(folder)
    assert twin_window.results is not None and not twin_window.result_warnings.isHidden()
    assert "before format 2.0.0" in twin_window.result_warnings.text()
    run_twin_window = window.open_twin(run_twin(make_twin(twins, point034.iloc[:200])).status.config.folder)
    assert "before format 2.0.0" not in run_twin_window.result_warnings.text()


def test_create_twin_from_a_computation_in_the_app(qtbot, tmp_path, twins, point034, monkeypatch):
    from molonari_twin.domain.models import SamplingPoint, Study
    from molonari_twin.storage.project import Project
    from molonaviz2.settings import Settings
    from molonaviz2.ui.main_window import MainWindow
    from PySide6 import QtCore

    project = Project.create(tmp_path / "p.molo")
    project.add_lab(import_lab_directory(FIXTURES / "lab", name="Lab").lab)
    project.add_study(Study("E", "Lab"))
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame.iloc[:200]
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame.iloc[:200]
    project.add_point(SamplingPoint("Point034", "E", "P508", "S007", river_bed=0.505), raw_t, raw_p)
    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    settings.twin_folder = twins
    window = MainWindow(settings, project)
    qtbot.addWidget(window)
    window.open_point("E", "Point034")
    view = window.point_tabs.currentWidget()
    view.cleanup_tab.apply()
    view.forward_tab.nb_cells.set_value(30)
    view.forward_tab.run()
    qtbot.waitUntil(lambda: not window.tasks.busy, timeout=120_000)
    view.results_tab.refresh()
    monkeypatch.setattr("molonaviz2.ui.main_window.ask", lambda *a, **k: False)
    created = window.create_twin_from_computation(view.context, view.results_tab.info)
    assert created == twins / "twin_01"
    config = load_twin(created)
    assert config.point == "Point034" and config.nb_cells == 30 and config.depths == (0.1, 0.2, 0.3, 0.4)
    data = pd.read_csv(created / "measurements.csv")
    assert len(data) == len(project.cleaned("E", "Point034"))
    # Même calcul dans l'application et dans le twin.
    run = run_twin(created)
    app_result = view.results_tab.forward
    twin_temperature = load_results(run.status.results_dir).temperature
    assert np.max(np.abs(twin_temperature - app_result.temperature)) < 1e-4  # mesures relues depuis le CSV (6 chiffres)


def open_twin_window(qtbot, tmp_path, folder):
    from molonaviz2.settings import Settings
    from molonaviz2.ui.main_window import MainWindow
    from PySide6 import QtCore

    settings = Settings(QtCore.QSettings(str(tmp_path / "s.ini"), QtCore.QSettings.Format.IniFormat))
    window = MainWindow(settings)
    qtbot.addWidget(window)
    return window.open_twin(folder)


def predictive_in(twin_window):
    return [[c for c in ax.collections if c.get_gid() == "predictive"] for ax in twin_window.sensors.figure.axes]


def test_twin_window_draws_the_predictive_band_of_questions(qtbot, tmp_path, twins, point034):
    from molonari_twin.questions import (
        coverage,
        observed_at,
        predictive_band,
        sensor_samples,
    )
    from molonaviz2.ui.predictive import PREDICTIVE_SEED
    from molonaviz2.ui.units import measurement_level

    folder = make_twin(twins, point034.iloc[:150], small_inversion_config(store_fields="sensors", time_step=2), sigma_t=0.2)
    results = load_results(run_twin(folder).status.results_dir)
    twin_window = open_twin_window(qtbot, tmp_path, folder)
    assert twin_window.tabs.tabText(twin_window.sensors_index) == "Sensors t₁–t₃" and twin_window.predictive_note.isHidden()

    samples = sensor_samples(results).data
    band = predictive_band(samples["temperature"].value, 0.2, measurement_level.level, seed=PREDICTIVE_SEED).data["temperature"]
    n = len(samples["time"].value)
    for i, (found,) in enumerate(predictive_in(twin_window)):
        vertices = found.get_paths()[0].vertices[:, 1]
        np.testing.assert_array_equal(vertices[1:n + 1], band.interval[0][i])
        np.testing.assert_array_equal(vertices[n + 2:2 * n + 2][::-1], band.interval[1][i])
    observed = np.stack([results.residuals[f"t{i + 1}_measured"].to_numpy(float) for i in range(3)], axis=1)
    dates = pd.to_datetime(results.residuals["date"]).to_numpy()
    expected = coverage(observed_at(dates, observed, samples["time"].value).data["temperature"].value, band).data
    assert expected["points"].value == 3 * n  # dates des champs = une date sur 2 des mesures
    label = f"Predictive 90 % — {expected['coverage'].value * 100:.0f} % of measurements (n = {expected['points'].value})"
    assert label in [text.get_text() for text in twin_window.sensors.figure.axes[0].get_legend().get_texts()]


@pytest.mark.parametrize("kind, mention", [
    ("format_1", "Results written before format 2.0.0"),
    ("mcmc_without_fields", "No temperature fields were stored"),
    ("direct", None),
])
def test_twin_window_explains_a_missing_predictive_band(qtbot, tmp_path, twins, point034, kind, mention):
    if kind == "mcmc_without_fields":
        folder = make_twin(twins, point034.iloc[:150], small_inversion_config(store_fields="none"), sigma_t=0.2)
        run_twin(folder)
    else:
        folder = make_twin(twins, point034)
        results_dir = run_twin(folder).status.results_dir
        if kind == "format_1":
            as_format_1(results_dir)
    twin_window = open_twin_window(qtbot, tmp_path, folder)
    assert twin_window.results is not None and twin_window.predictive is None
    assert all(found == [] for found in predictive_in(twin_window))
    assert len(twin_window.sensors.figure.axes) == 3  # mesures et simulation restent dessinées
    if mention is None:
        assert twin_window.predictive_note.isHidden()
    else:
        assert not twin_window.predictive_note.isHidden() and mention in twin_window.predictive_note.text()
