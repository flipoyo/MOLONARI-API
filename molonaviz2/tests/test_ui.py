"""Interface (pytest-qt, hors écran) : flux de travail complet, garde-fous, erreurs, langue, unités."""

import html

import numpy as np
import pytest
from conftest import FIXTURES
from PySide6 import QtCore, QtWidgets

from molonaviz2.domain.models import PointStage, SamplingPoint, Study
from molonaviz2.io.lab_import import import_lab_directory
from molonaviz2.io.point_import import (
    read_point_info,
    read_pressure_file,
    read_temperature_file,
)
from molonaviz2.settings import Settings
from molonaviz2.storage.project import Project
from molonaviz2.ui import errors as ui_errors
from molonaviz2.ui.dialogs.lab_import import LabImportDialog
from molonaviz2.ui.dialogs.point_import import PointImportDialog
from molonaviz2.ui.main_window import MainWindow
from molonaviz2.ui.point_view.view import PointView
from molonaviz2.ui.units import (
    measurement_interval,
    measurement_level,
    temperature_unit,
    z_score,
)
from molonaviz2.ui.widgets import TypedConfirmation

TASK_TIMEOUT = 120_000


@pytest.fixture(autouse=True)
def isolated_settings(qapp, tmp_path):
    from molonaviz2.i18n import install_language

    install_language(qapp, "en")  # langue source : les tests vérifient les textes anglais
    QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
    QtCore.QSettings.setPath(QtCore.QSettings.Format.IniFormat, QtCore.QSettings.Scope.UserScope, str(tmp_path / "settings"))
    temperature_unit.set("°C")
    measurement_level.set(0.90)
    yield
    temperature_unit.set("°C")
    measurement_level.set(0.90)


@pytest.fixture
def shown_errors():
    """Erreurs présentées à l'utilisateur (au lieu d'ouvrir des boîtes modales)."""
    errors = []
    previous = ui_errors.set_presenter(lambda title, message, details: errors.append((title, message, details)))
    ui_errors.install_excepthook()
    yield errors
    ui_errors.set_presenter(previous)


@pytest.fixture
def settings(tmp_path):
    return Settings(QtCore.QSettings(str(tmp_path / "ui.ini"), QtCore.QSettings.Format.IniFormat))


def fill_project(project: Project) -> None:
    project.add_lab(import_lab_directory(FIXTURES / "lab", name="Lab").lab)
    project.add_study(Study("Etude", "Lab"))
    info = read_point_info(FIXTURES / "point034" / "Point034_info.csv")
    point = SamplingPoint("Point034", "Etude", "P508", "S007", info.setup_date, info.measure_date, 0.505, 0.0, "Point 34",
                          (FIXTURES / "point034" / "point34_schema.png").read_bytes())
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame.iloc[:300]
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame.iloc[:300]
    project.add_point(point, raw_t, raw_p)


@pytest.fixture
def window(qtbot, settings, tmp_path, shown_errors):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    project = Project.create(tmp_path / "ui.molo")
    fill_project(project)
    win._set_project(project)
    yield win
    win.tasks.cancel()
    win.tasks.wait()


def open_point(qtbot, window) -> PointView:
    window.open_point("Etude", "Point034")
    view = window.point_tabs.currentWidget()
    assert isinstance(view, PointView)
    return view


def wait_task(qtbot, window):
    qtbot.waitUntil(lambda: not window.tasks.busy, timeout=TASK_TIMEOUT)
    for _ in range(5):
        QtWidgets.QApplication.processEvents()


# Démarrage et garde-fous ------------------------------------------------------------------


def test_welcome_page_and_disabled_actions_explain_why(qtbot, settings):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    assert win.central.currentWidget() is win.welcome
    assert not win.project_dock.isVisible()
    for action in (win.import_lab_action, win.new_study_action, win.import_point_action):
        assert not action.isEnabled()
        assert action.toolTip()  # la raison est donnée


def test_actions_become_available_step_by_step(qtbot, settings, tmp_path):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    project = Project.create(tmp_path / "steps.molo")
    win._set_project(project)
    assert win.import_lab_action.isEnabled()
    assert not win.new_study_action.isEnabled() and "laboratory" in win.new_study_action.toolTip().lower()
    project.add_lab(import_lab_directory(FIXTURES / "lab", name="Lab").lab)
    win._project_changed()
    assert win.new_study_action.isEnabled() and not win.import_point_action.isEnabled()
    project.add_study(Study("E", "Lab"))
    win._project_changed()
    assert win.import_point_action.isEnabled()
    assert str(project.path) in settings.recent_projects()


def test_lab_import_dialog_previews_before_writing(qtbot):
    dialog = LabImportDialog(existing_names=["Lab"])
    qtbot.addWidget(dialog)
    dialog.load(FIXTURES / "lab")
    assert dialog.tree.topLevelItemCount() == 3
    assert dialog.tree.topLevelItem(1).childCount() == 9
    assert dialog.name.text() == "lab"  # nom proposé : celui du dossier
    dialog.name.setText("Lab")
    assert not dialog.import_button.isEnabled() and "already exists" in dialog.name_error.text()
    dialog.name.setText("Montberneux")
    assert dialog.import_button.isEnabled()
    assert dialog.lab().name == "Montberneux"


def test_point_import_dialog_fills_everything_from_the_folder(qtbot):
    lab = import_lab_directory(FIXTURES / "lab", name="Lab").lab
    dialog = PointImportDialog("Etude", lab, existing_points=[])
    qtbot.addWidget(dialog)
    dialog.fill_from_folder(FIXTURES / "point034")
    assert dialog.name.text() == "Point034"
    assert dialog.pressure_sensor.currentText() == "P508" and dialog.shaft.currentText() == "S007"
    assert dialog.river_bed.value() == 0.505
    assert "1460" in dialog.pressure_preview.text() and "1436" in dialog.temperature_preview.text()
    assert "Several date formats" in dialog.pressure_preview.text()
    assert dialog.import_button.isEnabled()
    point, raw_t, raw_p = dialog.point_data()
    assert point.notice.startswith("Point 34") and point.scheme is not None
    assert len(raw_t) == 1436 and len(raw_p) == 1460


def test_point_import_dialog_refuses_a_spreadsheet_altered_file(qtbot):
    lab = import_lab_directory(FIXTURES / "lab", name="Lab").lab
    dialog = PointImportDialog("Etude", lab, existing_points=["Point034"])
    qtbot.addWidget(dialog)
    dialog.fill_from_folder(FIXTURES / "point034")
    assert "already exists" in dialog.problems.text()
    dialog.name.setText("Autre")
    dialog.temperature_file.set_path(FIXTURES / "hobo_spreadsheet_altered_T.csv")
    assert "chronological" in dialog.temperature_preview.text()
    assert not dialog.import_button.isEnabled()


def test_computation_tabs_are_locked_until_measurements_are_cleaned(qtbot, window):
    view = open_point(qtbot, window)
    assert not view.tabs.isTabEnabled(view.FORWARD)
    assert "Clean the measurements first" in view.tabs.tabToolTip(view.FORWARD)
    assert not view.tabs.isTabEnabled(view.RESULTS)
    assert view.tabs.currentIndex() == view.CLEANUP  # le point s'ouvre sur l'étape à faire


# Flux de travail ---------------------------------------------------------------------------


def test_cleanup_preview_then_apply(qtbot, window):
    view = open_point(qtbot, window)
    tab = view.cleanup_tab
    tab.methods["t1"].setCurrentIndex(tab.methods["t1"].findData("zscore"))
    tab.preview()
    assert "dates kept out of" in tab.summary.text()
    assert window.project.stage("Etude", "Point034") == PointStage.RAW  # rien n'est écrit avant d'appliquer
    tab.apply()
    assert view.context.stage == PointStage.CLEANED
    assert view.tabs.isTabEnabled(view.FORWARD)
    assert window.project.cleanup_settings("Etude", "Point034")["methods"]["t1"] == "zscore"
    assert "Cleaned" in view.stage_bar.text()


def test_direct_model_then_inversion_then_results(qtbot, window, shown_errors, tmp_path):
    view = open_point(qtbot, window)
    view.cleanup_tab.apply()
    view.tabs.setCurrentIndex(view.FORWARD)
    view.forward_tab.nb_cells.set_value(30)
    view.run_current()  # F5
    assert window.tasks.busy
    assert not view.forward_tab.run_button.isEnabled()  # pas de second calcul en parallèle
    wait_task(qtbot, window)
    assert view.context.stage == PointStage.DIRECT_MODEL
    assert view.forward_tab.summary.text().startswith("RMSE")
    assert view.tabs.isTabEnabled(view.RESULTS)

    tab = view.inversion_tab
    view.tabs.setCurrentIndex(view.INVERSION)
    tab.advanced.setChecked(True)
    for field, value in ((tab.nb_cells, 20), (tab.nb_chains, 7), (tab.nb_iter, 20), (tab.burn_in_max, 20), (tab.n_field_samples, 14)):
        field.set_value(value)
    assert tab.run_button.isEnabled(), tab.issues.text()
    tab.run()
    wait_task(qtbot, window)
    assert view.context.stage == PointStage.INVERSION
    assert "Burn-in" in tab.status.text()

    tab.show_results.emit(tab._last_id)
    results = view.results_tab
    assert view.tabs.currentIndex() == view.RESULTS
    assert results.inversion is not None
    assert results.tabs.isTabEnabled(results.parameters_index)
    assert results.summary_model.rowCount() == 4
    assert "seed" in results.provenance.text()
    assert "20 iterations kept per chain" in results.provenance.text() and "fields per chain" in results.provenance.text()

    folder = tmp_path / "exports"
    results.export_csv(str(folder))
    assert {p.name for p in folder.iterdir()} >= {"temperature_C.csv", "sensors.csv", "posterior_summary.csv", "posterior_samples.csv"}
    netcdf = tmp_path / "posterior.nc"
    results.export_netcdf(str(netcdf))
    import arviz as az

    assert "neg_log10_k" in az.from_netcdf(str(netcdf)).posterior
    assert results.sigma_t() == (tab.sigma_t.value(), "σT of the inversion")
    legend = [text.get_text() for text in results.sensors.figure.axes[0].get_legend().get_texts()]
    assert "Credible interval" in legend and any("σT (90 %)" in label for label in legend)
    results.tabs.setCurrentIndex(results.parameters_index)
    figure = tmp_path / "histograms.svg"
    results.histograms.export(figure)
    assert figure.stat().st_size > 0
    assert shown_errors == []


def test_csv_export_writes_heat_fluxes_positive_downwards(tmp_path):
    import pandas as pd
    from conftest import synthetic_inputs

    from molonaviz2.domain.compute import ForwardConfig, LayerSpec
    from molonaviz2.engine import api
    from molonaviz2.ui.point_view.results_tab import write_csv_exports

    forward = api.run_forward(synthetic_inputs(nb_times=50), ForwardConfig([LayerSpec("a", 0.4)], nb_cells=20))
    names = {path.name for path in write_csv_exports(tmp_path, forward, None, ["a"])}
    assert "conductive_heat_flux_down_W_m2.csv" in names and "conductive_heat_flux_W_m2.csv" not in names
    exported = pd.read_csv(tmp_path / "conductive_heat_flux_down_W_m2.csv", index_col=0).to_numpy(float)
    np.testing.assert_allclose(exported, -forward.conductive_flux, rtol=1e-12)


def test_reusing_a_configuration_reloads_it(qtbot, window):
    view = open_point(qtbot, window)
    view.cleanup_tab.apply()
    view.forward_tab.layers.add_layer()
    view.forward_tab.nb_cells.set_value(24)
    view.forward_tab.run()
    wait_task(qtbot, window)
    view.forward_tab.load_config(type(view.forward_tab.config())(view.forward_tab.layers.layers()[:1], 50))  # on change tout
    view.results_tab.refresh()
    view.results_tab._reuse()
    assert view.tabs.currentIndex() == view.FORWARD
    assert view.forward_tab.nb_cells.value() == 24
    assert view.forward_tab.layers.rowCount() == 2


def test_measurement_interval_bounds():
    assert z_score(0.6827) == pytest.approx(1.0, abs=1e-3)
    assert z_score(0.90) == pytest.approx(1.6449, abs=1e-4)
    assert z_score(0.95) == pytest.approx(1.9600, abs=1e-4)
    observed = np.array([[10.0, 12.0, 14.0]])
    low, high = measurement_interval(observed, 0.2, 0.90)
    np.testing.assert_allclose((low + high) / 2, observed)
    np.testing.assert_allclose(high - low, 2 * z_score(0.90) * 0.2)
    from molonaviz2.ui.units import TemperatureUnit

    fahrenheit = TemperatureUnit("°F")  # instance locale : ne redessine pas les fenêtres des autres tests
    np.testing.assert_allclose(fahrenheit.convert(high) - fahrenheit.convert(low), 9 / 5 * (high - low))


def test_measurement_interval_around_measured_temperatures(qtbot, window):
    view = open_point(qtbot, window)
    view.cleanup_tab.apply()
    view.forward_tab.nb_cells.set_value(20)
    view.forward_tab.run()
    wait_task(qtbot, window)
    results = view.results_tab
    results.refresh()
    sigma, source = results.sigma_t()
    thermometer = view.context.sensors().thermometer
    assert sigma == pytest.approx(thermometer.sigma) and thermometer.name in source
    assert results.sigma_source.text().startswith("σT = ")

    def band_labels():
        return [text.get_text() for text in results.sensors.figure.axes[0].get_legend().get_texts()]

    assert "Measured ± 1.645 σT (90 %)" in band_labels()
    band = results.sensors.figure.axes[1].collections[0].get_paths()[0].vertices[:, 1]
    measured = results.forward.observed[:, 1]
    assert band.max() == pytest.approx(measured.max() + z_score(0.90) * sigma, abs=1e-9)
    assert band.min() == pytest.approx(measured.min() - z_score(0.90) * sigma, abs=1e-9)
    results.measurement_level.setCurrentIndex(2)  # 95 % : le niveau est partagé et le graphe suit
    assert measurement_level.level == 0.95
    assert "Measured ± 1.96 σT (95 %)" in band_labels()


def test_per_chain_box_follows_the_settings(qtbot, window):
    view = open_point(qtbot, window)
    tab = view.inversion_tab
    assert "Field grid" not in tab.per_chain.text()  # dates inconnues avant le nettoyage
    view.cleanup_tab.apply()
    tab.advanced.setChecked(True)
    for field, value in ((tab.nb_cells, 20), (tab.nb_chains, 7), (tab.nb_iter, 30), (tab.burn_in_max, 40), (tab.n_field_samples, 14)):
        field.set_value(value)
    text = html.unescape(tab.per_chain.text())
    assert "up to 40 iterations per chain (stops early once R-hat < 1.2)" in text and "30 iterations per chain, 210 parameter sets" in text
    assert "2 per chain, one every 15 iterations, 14 in all" in text and "requested" not in text
    assert "Field grid: 20 cells ×" in text and "at most 490" in text  # 7 × (40 + 30)
    tab.n_field_samples.set_value(15)
    assert "3 per chain, one every 10 iterations, 21 in all" in tab.per_chain.text() and "(15 requested" in tab.per_chain.text()
    tab.nb_chains.set_value(0)  # invalide
    assert "Fix the highlighted settings" in tab.per_chain.text()


def test_per_chain_box_shows_thinning_and_stored_fields(qtbot, window):
    from molonaviz2.ui.dialogs.notice import key_for

    view = open_point(qtbot, window)
    tab = view.inversion_tab
    assert key_for(tab.thin.edit) == "inversion.thin" and key_for(tab.store_fields) == "inversion.store_fields"
    view.cleanup_tab.apply()
    tab.advanced.setChecked(True)
    tab.nb_iter.set_value(1000)
    tab.thin.set_value(4)
    text = html.unescape(tab.per_chain.text())
    assert "250 iterations per chain out of 1000 computed (one in 4)" in text and "kept iterations" in text

    def stored_line():
        return next(line for line in html.unescape(tab.per_chain.text()).split("<br>") if line.startswith("Fields stored"))

    assert tab.store_fields.currentData() == "sensors" and stored_line().startswith("Fields stored in the result: At the sensors, about")
    sensors = stored_line()
    tab.store_fields.setCurrentIndex(tab.store_fields.findData("full"))
    assert stored_line() != sensors and stored_line().startswith("Fields stored in the result: Full fields, about")
    tab.store_fields.setCurrentIndex(tab.store_fields.findData("none"))
    assert stored_line() == "Fields stored in the result: none."

    config = tab.config()
    assert (config.thin, config.store_fields, config.fields_per_chain) == (4, "none", None)
    tab.restore_defaults()
    assert (tab.config().thin, tab.config().store_fields) == (1, "sensors")
    tab.load_config(config)
    assert (tab.config().thin, tab.config().store_fields) == (4, "none")


def test_every_setting_is_linked_to_the_parameter_guide(qtbot, window, monkeypatch):
    from molonaviz2.domain.compute import SETTING_LIMITS
    from molonaviz2.notice import BY_KEY
    from molonaviz2.ui.dialogs.notice import key_for

    view = open_point(qtbot, window)
    tab = view.inversion_tab
    for field in tab._fields():
        key = field.property("notice_key")
        assert key in BY_KEY and field.toolTip().startswith(BY_KEY[key].text("summary"))
        name = "nb_cells" if key == "numerics.nb_cells" else key.removeprefix("inversion.")
        assert (field._minimum, field._maximum) == SETTING_LIMITS[name]  # mêmes bornes que la notice
        assert key_for(field.edit) == key  # Maj+F1 depuis la zone de saisie
    assert view.forward_tab.nb_cells.property("notice_key") == "numerics.nb_cells"
    assert key_for(tab.quantiles) == "inversion.quantiles"
    cleanup = view.cleanup_tab
    assert key_for(cleanup.start) == "cleaning.period" and key_for(cleanup.exclude_button) == "cleaning.exclusions"
    assert all(combo.property("notice_key") == "cleaning.outliers" for combo in cleanup.methods.values())
    assert view.results_tab.measurement_level.property("notice_key") == "results.measurement_level"

    table = tab.layers  # bornes a priori : deux colonnes (min, max) par paramètre
    assert [table._column_key(c) for c in (0, 1, 2, 3, 4)] == ["layers.name", "layers.z_low", "layers.neg_log10_k", "layers.neg_log10_k", "layers.porosity"]
    table.setCurrentCell(0, 6)
    assert key_for(table) == "layers.lambda_s"
    assert view.forward_tab.layers._column_key(3) == "layers.porosity"  # valeurs : une colonne par paramètre

    action = next(a for a in window.menuBar().actions()[-1].menu().actions() if a.text() == "Parameter guide")
    assert action.shortcut().toString() == "Shift+F1"
    monkeypatch.setattr(QtWidgets.QApplication, "focusWidget", staticmethod(lambda: tab.nb_chains.edit))
    notice = window.open_notice()
    assert notice.isVisible() and notice.current_key() == "inversion.nb_chains"
    page = notice.page.toPlainText()
    assert "Number of chains" in page and "Default\n10" in page and "3 – 200" in page and "mcmc.nb_chains" in page
    notice.search.setText("gelman")
    assert "inversion.gr_threshold" in notice.visible_keys() and "layers.porosity" not in notice.visible_keys()
    assert window.open_notice("layers.porosity") is notice  # une seule fenêtre, rouverte sur la fiche demandée
    assert notice.current_key() == "layers.porosity" and notice.search.text() == ""


def test_inversion_can_be_cancelled_and_saves_nothing(qtbot, window):
    view = open_point(qtbot, window)
    view.cleanup_tab.apply()
    tab = view.inversion_tab
    tab.advanced.setChecked(True)
    tab.nb_cells.set_value(20)
    tab.burn_in_max.set_value(100_000)
    tab.nb_chains.set_value(7)
    tab.run()
    qtbot.waitUntil(lambda: "Burn-in" in tab.status.text(), timeout=TASK_TIMEOUT)
    tab.cancel()
    wait_task(qtbot, window)
    assert "cancelled" in tab.status.text()
    assert window.project.computations("Etude", "Point034") == []


def test_invalid_settings_disable_running_with_a_reason(qtbot, window):
    view = open_point(qtbot, window)
    view.cleanup_tab.apply()
    forward = view.forward_tab
    forward.layers.item(0, 3).setText("abc")  # porosité illisible
    assert not forward.run_button.isEnabled()
    assert "Enter a number" in forward.issues.text()
    inversion = view.inversion_tab
    inversion.advanced.setChecked(True)
    inversion.nb_chains.set_value(5)
    inversion.delta.set_value(3)
    assert not inversion.run_button.isEnabled()
    assert "2·δ + 1" in inversion.issues.text()
    inversion.layers.item(0, 4).setText("0.5")  # n min > n max
    assert "Minimum must be lower than maximum" in inversion.issues.text()


# Erreurs -----------------------------------------------------------------------------------


def test_error_inside_a_computation_is_shown_and_the_app_keeps_working(qtbot, window, shown_errors, monkeypatch):
    from molonaviz2.engine import api

    view = open_point(qtbot, window)
    view.cleanup_tab.apply()
    monkeypatch.setattr(api, "run_forward", lambda inputs, config: (_ for _ in ()).throw(RuntimeError("panne simulée")))
    view.forward_tab.run()
    wait_task(qtbot, window)
    assert len(shown_errors) == 1
    title, message, details = shown_errors[0]
    assert title == "Unexpected error" and "panne simulée" in message and "RuntimeError" in details
    assert view.context.stage == PointStage.CLEANED  # rien n'a été enregistré
    assert view.forward_tab.run_button.isEnabled()  # on peut relancer


@pytest.mark.qt_no_exception_capture
def test_exception_in_a_qt_slot_does_not_abort(qtbot, window, shown_errors):
    """Le défaut qui faisait planter l'ancien Molonaviz : une exception dans un slot Qt."""

    def broken_slot():
        raise ValueError("bogue dans un slot")

    QtCore.QTimer.singleShot(0, broken_slot)
    qtbot.waitUntil(lambda: bool(shown_errors), timeout=5000)
    assert "bogue dans un slot" in shown_errors[0][1]
    window._project_changed()  # l'application répond toujours
    assert window.explorer.topLevelItemCount() == 2


def test_user_errors_show_only_the_message(shown_errors):
    from molonaviz2.errors import UserError

    ui_errors.report_exception(UserError("Le fichier est vide."))
    assert shown_errors == [("Action not completed", "Le fichier est vide.", "")]


def test_destructive_confirmation_requires_typing_the_name(qtbot):
    dialog = TypedConfirmation("Supprimer", "…", "Point034", "Supprimer")
    qtbot.addWidget(dialog)
    assert not dialog.confirm.isEnabled()
    dialog.edit.setText("Point03")
    assert not dialog.confirm.isEnabled()
    dialog.edit.setText("Point034")
    assert dialog.confirm.isEnabled()


# Langue et unités --------------------------------------------------------------------------


def test_language_switch_keeps_the_project_open(qtbot, tmp_path, shown_errors):
    from molonaviz2.app import Application

    path = tmp_path / "lang.molo"
    project = Project.create(path)
    fill_project(project)
    project.close()
    controller = Application(QtWidgets.QApplication.instance(), str(path))
    first = controller.window
    qtbot.addWidget(first)
    assert first.menuBar().actions()[0].text() == "&Fichier"  # français par défaut
    controller.change_language("en")
    second = controller.window
    qtbot.addWidget(second)
    assert second is not first
    assert second.menuBar().actions()[0].text() == "&File"
    assert second.project is not None and second.project.path == path
    controller.change_language("fr")
    assert controller.window.menuBar().actions()[0].text() == "&Fichier"
    controller.window.close()


def test_temperature_unit_applies_everywhere(qtbot, window):
    view = open_point(qtbot, window)
    plot = view.measures_tab.temperature_plot
    window.set_temperature_unit("K")
    assert "K" in plot.getPlotItem().getAxis("left").labelText
    data = plot.getPlotItem().listDataItems()[0].getData()[1]
    assert np.nanmin(data) > 250  # valeurs en kelvins
    window.set_temperature_unit("°C")
    data = plot.getPlotItem().listDataItems()[0].getData()[1]
    assert np.nanmax(data) < 40


def test_deleting_a_point_closes_its_tab(qtbot, window, monkeypatch):
    open_point(qtbot, window)
    monkeypatch.setattr("molonaviz2.ui.main_window.confirm_destructive", lambda *a, **k: True)
    window._delete_point("Etude", "Point034")
    assert window.point_tabs.count() == 0
    assert window.project.point_names("Etude") == []


def test_window_state_is_remembered(qtbot, settings, tmp_path):
    win = MainWindow(settings)
    qtbot.addWidget(win)
    win.resize(900, 600)
    win.close()
    geometry, state = settings.window_state()
    assert geometry is not None and state is not None


def test_screenshot_is_not_blank(qtbot, window, tmp_path):
    view = open_point(qtbot, window)
    window.resize(1280, 800)
    view.tabs.setCurrentIndex(view.MEASURES)
    QtWidgets.QApplication.processEvents()
    image = window.grab().toImage()
    pixels = np.array([image.pixel(x, y) for x in range(0, image.width(), 37) for y in range(0, image.height(), 37)])
    assert len(np.unique(pixels)) > 20  # du contenu, pas un cadre vide


# Bande prédictive et vue Chaînes (A6) ------------------------------------------------------


def run_small_inversion(qtbot, window, view, store_fields="sensors", nb_iter=20, thin=1):
    """Inversion courte lancée depuis l'onglet Inversion, puis affichée dans l'onglet Résultats."""
    view.cleanup_tab.apply()
    tab = view.inversion_tab
    view.tabs.setCurrentIndex(view.INVERSION)
    tab.advanced.setChecked(True)
    for field, value in ((tab.nb_cells, 20), (tab.nb_chains, 7), (tab.nb_iter, nb_iter), (tab.burn_in_max, 20), (tab.n_field_samples, 14),
                         (tab.thin, thin)):
        field.set_value(value)
    tab.store_fields.setCurrentIndex(tab.store_fields.findData(store_fields))
    assert tab.run_button.isEnabled(), tab.issues.text()
    tab.run()
    wait_task(qtbot, window)
    tab.show_results.emit(tab._last_id)
    return view.results_tab


def predictive_collections(results):
    return [[c for c in ax.collections if c.get_gid() == "predictive"] for ax in results.sensors.figure.axes]


def test_predictive_band_and_its_coverage_come_from_questions(qtbot, window):
    from molonaviz2.questions import (
        coverage,
        observed_at,
        predictive_band,
        sensor_samples,
    )
    from molonaviz2.ui.predictive import PREDICTIVE_SEED

    view = open_point(qtbot, window)
    results = run_small_inversion(qtbot, window, view)
    assert results.inversion.sensor_temperature is not None and results.predictive_note.isHidden()
    sigma, _ = results.sigma_t()
    samples = sensor_samples(results.inversion).data
    band = predictive_band(samples["temperature"].value, sigma, 0.90, seed=PREDICTIVE_SEED).data["temperature"]
    low, high = band.interval
    collections = predictive_collections(results)
    assert all(len(found) == 1 for found in collections)
    n = len(samples["time"].value)
    for i, (found,) in enumerate(collections):
        vertices = found.get_paths()[0].vertices[:, 1]
        np.testing.assert_array_equal(vertices[1:n + 1], low[i])  # sommets bas, dans l'ordre des dates
        np.testing.assert_array_equal(vertices[n + 2:2 * n + 2][::-1], high[i])  # sommets hauts, en sens inverse

    measured = observed_at(results.forward.times, results.forward.observed, samples["time"].value).data["temperature"].value
    expected = coverage(measured, band).data
    label = f"Predictive 90 % — {expected['coverage'].value * 100:.0f} % of measurements (n = {expected['points'].value})"
    legend = [text.get_text() for text in results.sensors.figure.axes[0].get_legend().get_texts()]
    assert label in legend and "Credible interval" in legend
    assert results.predictive.coverage == expected["coverage"].value and results.predictive.points == expected["points"].value

    results.measurement_level.setCurrentIndex(2)  # 95 % : la bande prédictive suit le niveau partagé
    wider = predictive_band(samples["temperature"].value, sigma, 0.95, seed=PREDICTIVE_SEED).data["temperature"].interval[0]
    vertices = predictive_collections(results)[0][0].get_paths()[0].vertices[:, 1]
    np.testing.assert_array_equal(vertices[1:n + 1], wider[0])
    assert any(text.get_text().startswith("Predictive 95 %") for text in results.sensors.figure.axes[0].get_legend().get_texts())


def test_predictive_band_is_explained_when_no_field_was_stored(qtbot, window):
    view = open_point(qtbot, window)
    results = run_small_inversion(qtbot, window, view, store_fields="none")
    assert not results.predictive_note.isHidden()
    assert "No temperature fields were stored" in results.predictive_note.text()
    assert all(found == [] for found in predictive_collections(results))
    legend = [text.get_text() for text in results.sensors.figure.axes[0].get_legend().get_texts()]
    assert "Credible interval" in legend and any("σT (90 %)" in label for label in legend)


def chain_lines(results, row=0):
    """{chaîne: (abscisses, ordonnées)} des traces du graphe `row` de la vue Chaînes."""
    ax = results.traces.figure.axes[row]
    return {int(line.get_gid().removeprefix("chain-")): line.get_data() for line in ax.lines if (line.get_gid() or "").startswith("chain-")}


def test_chains_view_selects_chains_thinning_and_iterations(qtbot, window):
    from molonaviz2.questions import chain_data

    view = open_point(qtbot, window)
    results = run_small_inversion(qtbot, window, view, nb_iter=21, thin=1)
    results.tabs.setCurrentIndex(results.parameters_index)
    results.parameter_views.setCurrentIndex(results.chains_index)
    assert len(results.traces.figure.axes) == 5  # 4 paramètres et l'énergie
    from molonaviz2.ui.dialogs.notice import key_for

    assert {key_for(w) for w in (results.chain_list, results.display_thin, results.iteration_from, results.iteration_to)} == {"results.chains"}
    assert key_for(results.sensors.canvas) == "results.predictive_band" and key_for(results.predictive_note) == "results.predictive_band"
    assert results.chain_list.count() == 7 and results.checked_chains() == list(range(7))
    full = chain_data(results.inversion).data
    for i in range(7):
        text = results.chain_list.item(i).text()
        assert text.startswith(f"Chain {i} — acceptance {full['acceptance'].value[i] * 100:.0f} %")
        assert f"{int(full['resets'].value[i])} reset(s)" in text
    lines = chain_lines(results)
    assert sorted(lines) == list(range(7)) and all(len(x) == 21 for x, _ in lines.values())
    np.testing.assert_array_equal(lines[3][1], full["samples.neg_log10_k"].value[:, 3, 0])
    np.testing.assert_array_equal(chain_lines(results, row=4)[3][1], full["energy"].value[:, 3])
    assert results.chain_count.text().startswith("147 samples shown")  # 7 × 21

    results.chain_list.item(2).setCheckState(QtCore.Qt.CheckState.Unchecked)  # décocher une chaîne retire sa trace
    for row in range(5):
        assert 2 not in chain_lines(results, row) and len(chain_lines(results, row)) == 6
    assert results.chain_count.text().startswith("126 samples shown")

    results.display_thin.setValue(2)  # une itération gardée sur 2 : ⌊21 / 2⌋ = 10 points, itérations 2, 4…, 20
    x, _ = chain_lines(results)[0]
    np.testing.assert_array_equal(x, np.arange(2, 21, 2))
    assert results.chain_count.text().startswith("60 samples shown")  # 6 × 10

    results.iteration_from.setValue(5)
    results.iteration_to.setValue(12)
    x, y = chain_lines(results)[0]
    np.testing.assert_array_equal(x, [6, 8, 10, 12])
    np.testing.assert_array_equal(y, results.inversion.samples[[5, 7, 9, 11], 0, 0, 0])
