"""Fenêtre d'un twin : config figée, identifiant, état dans le registre, exécution, résultats."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from molonaviz2.facade import (
    Issue,
    TwinConfigError,
    TwinStatus,
    config_path,
    load_results,
    run_twin,
    twin_status,
)

from ..i18n import t
from ..workers.tasks import Task, TaskManager
from .dialogs.notice import attach
from .errors import error_text
from .plots.maps import (
    FigurePanel,
    _date_axis,
    depth_time_map,
    predictive_band,
    robust_ylim,
    sensor_comparison,
)
from .plots.style import RIVER_COLOR, THERMOMETER_COLORS
from .predictive import try_predictive_at_sensors
from .units import (
    level_label,
    measurement_interval,
    measurement_level,
    temperature_unit,
)
from .widgets import format_number, heading, hint

INFILTRATION_COLOR = "#0072b2"
EXFILTRATION_COLOR = "#d55e00"


def state_text(status: TwinStatus) -> str:
    if status.state == "new":
        return t("Valid configuration, not run yet.")
    if status.state == "computed":
        return t("Computed on {date}.").format(date=status.source.created_at.replace("T", " "))
    if status.state == "duplicate":
        return t("Duplicate of {name}: same configuration, same data and same code. The existing results are shown.").format(name=status.source.name)
    return t("The configuration was modified after it was run (id {previous}). A configuration is frozen: create a new twin to change a parameter.").format(
        previous=status.previous_id)


class TwinWindow(QtWidgets.QMainWindow):
    def __init__(self, path: str | Path, tasks: TaskManager, parent=None):
        super().__init__(parent)
        self.path = config_path(path)
        self.tasks = tasks
        self.status: TwinStatus | None = None
        self.results = None
        self._task: Task | None = None
        self.resize(1280, 820)

        self.title = QtWidgets.QLabel()
        self.title.setProperty("role", "heading")
        self.identity = QtWidgets.QLabel()
        self.identity.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.state = QtWidgets.QLabel()
        self.state.setWordWrap(True)
        self.result_warnings = hint("")  # avertissements de la relecture (résultats 1.x…)
        self.result_warnings.hide()
        self.run_button = QtWidgets.QPushButton(t("Run  (F5)"))
        self.run_button.setDefault(True)
        self.run_button.clicked.connect(self.run)
        self.cancel_button = QtWidgets.QPushButton(t("Cancel  (Esc)"))
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.tasks.cancel)
        self.open_results = QtWidgets.QPushButton(t("Show results folder"))
        self.open_results.clicked.connect(lambda: self._reveal(self.status.results_dir))
        reload_button = QtWidgets.QPushButton(t("Reload"))
        reload_button.setToolTip(t("Read twin.yaml again (after editing it in a text editor)."))
        reload_button.clicked.connect(self.refresh)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.progress)
        buttons.addStretch()
        buttons.addWidget(reload_button)
        buttons.addWidget(self.open_results)

        self.config_text = QtWidgets.QPlainTextEdit()
        self.config_text.setReadOnly(True)
        self.config_text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.issues = QtWidgets.QLabel()
        self.issues.setProperty("role", "error")
        self.issues.setWordWrap(True)
        self.issues.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(heading("twin.yaml"))
        left_layout.addWidget(hint(t("Frozen configuration: edit it only before the first run. To change a parameter, create a new twin.")))
        left_layout.addWidget(self.config_text, 1)
        left_layout.addWidget(self.issues)

        self.tabs = QtWidgets.QTabWidget()
        self.temperature = FigurePanel(height=4)
        self.flux = FigurePanel(height=4)
        self.residuals = FigurePanel(height=4.5)
        residuals_page = QtWidgets.QWidget()
        residuals_layout = QtWidgets.QVBoxLayout(residuals_page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Measurement interval")))
        level = measurement_level.combo()
        attach(level, "results.measurement_level")
        row.addWidget(level)
        row.addStretch()
        residuals_layout.addLayout(row)
        residuals_layout.addWidget(self.residuals, 1)
        self.sensors = FigurePanel(height=4.5)
        self.predictive = None  # PredictiveBand dessinée (twin mcmc avec champs stockés)
        self.predictive_note = hint("")  # bande prédictive indisponible : résultat 1.x, ou mcmc sans champs
        self.predictive_note.hide()
        attach(self.predictive_note, "results.predictive_band")
        self.sensors.setProperty("notice_key", "results.predictive_band")  # Maj+F1 sur le graphe, sans infobulle qui le masque
        sensors_page = QtWidgets.QWidget()
        sensors_layout = QtWidgets.QVBoxLayout(sensors_page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Measurement interval")))
        sensors_level = measurement_level.combo()
        attach(sensors_level, "results.measurement_level")
        row.addWidget(sensors_level)
        row.addStretch()
        sensors_layout.addLayout(row)
        sensors_layout.addWidget(self.predictive_note)
        sensors_layout.addWidget(self.sensors, 1)
        self.metadata = QtWidgets.QPlainTextEdit()
        self.metadata.setReadOnly(True)
        self.metadata.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        self.flux_summary = hint("")
        flux_page = QtWidgets.QWidget()
        flux_layout = QtWidgets.QVBoxLayout(flux_page)
        flux_layout.addWidget(self.flux_summary)
        flux_layout.addWidget(self.flux, 1)
        self.tabs.addTab(self.temperature, t("Temperature θ(z, t)"))
        self.tabs.addTab(flux_page, t("Water flux q(t)"))
        self.sensors_index = self.tabs.addTab(sensors_page, t("Sensors t₁–t₃"))
        self.tabs.addTab(residuals_page, t("Residuals t₁–t₃"))
        self.tabs.addTab(self.metadata, t("Metadata"))
        self.no_results = hint(t("No results yet: run the twin."))
        self.no_results.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.tabs, 1)
        right_layout.addWidget(self.no_results, 1)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([420, 860])
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addWidget(self.title)
        layout.addWidget(self.identity)
        layout.addWidget(self.state)
        layout.addWidget(self.result_warnings)
        layout.addLayout(buttons)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        QtGui.QShortcut(QtGui.QKeySequence("F5"), self, self.run)
        QtGui.QShortcut(QtGui.QKeySequence("Escape"), self, self.tasks.cancel)
        tasks.task_started.connect(lambda _: self._update_buttons())
        tasks.task_finished.connect(lambda _: self._update_buttons())
        temperature_unit.changed.connect(self._draw)
        measurement_level.changed.connect(self._draw)
        self.refresh()

    # État -----------------------------------------------------------------------------
    def refresh(self) -> None:
        self.config_text.setPlainText(self.path.read_text(encoding="utf-8") if self.path.is_file() else "")
        self.status, self.results = None, None
        self.issues.setText("")
        try:
            self.status = twin_status(self.path)
        except TwinConfigError as exc:
            folder = self.path.parent.name
            self.setWindowTitle(f"{folder} — Molonaviz")
            self.title.setText(folder)
            self.identity.setText("")
            self.state.setText(t("Invalid configuration: fix twin.yaml, then click Reload."))
            self.issues.setText("\n".join(f"• {error_text(issue)}" for issue in exc.issues))
        else:
            config = self.status.config
            self.setWindowTitle(f"{config.name} — Molonaviz")
            mode = t("direct mode") if config.mode == "direct" else t("mcmc mode (experimental)")
            point = f" · {config.point}" if config.point else ""
            self.title.setText(f"{config.name}{point} · {mode}")
            self.identity.setText(f"twin_id <b>{self.status.twin_id}</b>" + (f" · {config.description}" if config.description else ""))
            self.state.setText(state_text(self.status))
            if self.status.has_results:
                self.results = load_results(self.status.results_dir)
        warnings = self.results.warnings if self.results is not None else []
        self.result_warnings.setText("\n".join(f"⚠ {error_text(warning)}" for warning in warnings))
        self.result_warnings.setVisible(bool(warnings))
        self.tabs.setVisible(self.results is not None)
        self.no_results.setVisible(self.results is None)
        self._update_buttons()
        self._draw()

    def _update_buttons(self) -> None:
        running = self._task is not None
        can_run = self.status is not None and self.status.state == "new" and not self.tasks.busy
        self.run_button.setEnabled(can_run)
        if self.status is None:
            reason = t("Fix the configuration first.")
        elif self.status.state != "new":
            reason = t("Already computed: results are shown.") if self.status.has_results else t("A frozen twin cannot be run again with a modified configuration.")
        elif self.tasks.busy and not running:
            reason = t("A computation is already running.")
        else:
            reason = ""
        self.run_button.setToolTip(reason)
        self.cancel_button.setEnabled(running)
        self.open_results.setEnabled(self.results is not None)

    # Exécution ------------------------------------------------------------------------
    def run(self) -> None:
        if not self.run_button.isEnabled():
            return
        path = self.path
        task = Task(t("Twin — {name}").format(name=self.status.config.name),
                    lambda ctx: run_twin(path, on_progress=ctx.report, should_cancel=ctx.cancelled))
        task.signals.progress.connect(self._progress)
        task.signals.succeeded.connect(lambda run: self._finished())
        task.signals.cancelled.connect(lambda: self._stopped(t("Run cancelled. Nothing was saved.")))
        task.signals.failed.connect(lambda _: self._stopped(t("The run stopped because of an error.")))
        self._task = task
        self.progress.setRange(0, 0)
        self.progress.show()
        self.state.setText(t("Running…"))
        self.tasks.start(task)
        self._update_buttons()

    def _progress(self, progress) -> None:
        if progress.phase in ("burn_in", "sampling") and progress.total:
            self.progress.setRange(0, progress.total)
            self.progress.setValue(progress.iteration)
            phase = t("Burn-in") if progress.phase == "burn_in" else t("Sampling")
            self.state.setText(f"{phase} · {progress.iteration} / {progress.total}")
        else:
            self.progress.setRange(0, 0)

    def _finished(self) -> None:
        self._task = None
        self.progress.hide()
        self.refresh()

    def _stopped(self, message: str) -> None:
        self._task = None
        self.progress.hide()
        self.state.setText(message)
        self._update_buttons()

    def _reveal(self, folder: Path) -> None:
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    # Graphiques -----------------------------------------------------------------------
    def _draw(self) -> None:
        if self.results is None:
            return
        results = self.results
        unit = temperature_unit.unit
        meta = results.metadata

        figure = self.temperature.figure
        figure.clear()
        ax = figure.add_subplot()
        depth_time_map(ax, results.times, results.depths, temperature_unit.convert(results.temperature),
                       t("Temperature ({unit})").format(unit=unit), sensor_depths=results.sensor_depths)
        ax.set_title(t("Simulated temperature θ(z, t) — dashed lines: sensors t₁–t₃"), fontsize=10, loc="left")
        self.temperature.draw()

        flux = results.flux
        q = flux["q_riverbed_m_per_s"].to_numpy(float)
        dates = pd.to_datetime(flux["date"]).to_numpy()
        figure = self.flux.figure
        figure.clear()
        ax = figure.add_subplot()
        low_col = next((c for c in flux.columns if c.startswith("q_riverbed_q")), None)
        quantile_cols = [c for c in flux.columns if c.startswith("q_riverbed_q")]
        if len(quantile_cols) >= 2:
            band = flux[quantile_cols].dropna()
            ax.fill_between(pd.to_datetime(flux.loc[band.index, "date"]).to_numpy(), band[quantile_cols[0]], band[quantile_cols[-1]],
                            color="#444444", alpha=0.2, lw=0, label=t("Credible interval"))
        ax.fill_between(dates, 0, q, where=q >= 0, color=INFILTRATION_COLOR, alpha=0.35, lw=0, label=t("Infiltration (river → aquifer)"))
        ax.fill_between(dates, 0, q, where=q < 0, color=EXFILTRATION_COLOR, alpha=0.35, lw=0, label=t("Exfiltration (aquifer → river)"))
        ax.plot(dates, q, color="#222222", lw=0.8)
        ax.axhline(0, color="#888888", lw=0.6)
        ax.set_ylabel(t("q at the river bed (m/s)"))
        _date_axis(ax)
        robust_ylim(ax, q)
        ax.legend(fontsize=8, loc="upper right")
        self.flux.draw()
        infiltration = float(np.mean(q >= 0)) if len(q) else 0.0
        self.flux_summary.setText(
            t("Infiltration {infiltration:.0%} of the time, exfiltration {exfiltration:.0%}. Mean flux {mean:.2e} m/s ({per_day:.3f} m/day).").format(
                infiltration=infiltration, exfiltration=1 - infiltration, mean=float(np.mean(q)), per_day=float(np.mean(q)) * 86400)
            + ("" if low_col is None else " " + t("Grey band: credible interval (mcmc)."))
        )

        res_dates = pd.to_datetime(results.residuals["date"]).to_numpy()
        rmse = meta.get("rmse_celsius", {})
        sigma = meta.get("config", {}).get("sensors", {}).get("sigma_temperature")
        if sigma is None and self.status is not None:
            sigma = self.status.config.sigma_temperature  # mode direct : σT n'est pas dans la config canonique
        self._draw_sensors(res_dates, sigma, rmse)

        figure = self.residuals.figure
        figure.clear()
        axes = figure.subplots(3, 1, sharex=True)
        for i, ax in enumerate(axes):
            values = temperature_unit.convert_difference(results.residuals[f"t{i + 1}_residual"].to_numpy(float))
            ax.plot(res_dates, values, color=THERMOMETER_COLORS[i], lw=0.9)
            ax.axhline(0, color="#888888", lw=0.6)
            if sigma:
                half = measurement_level.z * float(temperature_unit.convert_difference(sigma))
                label = t("± {z} σT ({level})").format(z=format_number(round(measurement_level.z, 3)), level=level_label(measurement_level.level))
                for sign in (-1, 1):
                    ax.axhline(sign * half, color=RIVER_COLOR, lw=0.6, ls="--", label=label if sign > 0 else None)
                if i == 0:
                    ax.legend(fontsize=8, loc="upper right")
            value = temperature_unit.convert_difference(rmse.get(f"t{i + 1}", float("nan")))
            ax.set_title(t("t{n}: simulated − measured — RMSE {rmse:.3f}").format(n=i + 1, rmse=float(value)), fontsize=9, loc="left")
            ax.set_ylabel(unit if unit != "°C" else "°C")
            _date_axis(ax)
        self.residuals.draw()
        self.metadata.setPlainText(json.dumps(meta, indent=2, ensure_ascii=False))

    def _draw_sensors(self, dates, sigma, rmse: dict) -> None:
        """Mesures et simulation aux capteurs, intervalle de mesure et, pour un twin mcmc dont les champs
        existent, bande prédictive et couverture calculées par questions."""
        results = self.results
        observed = np.stack([results.residuals[f"t{i + 1}_measured"].to_numpy(float) for i in range(3)], axis=1)
        simulated = np.stack([results.residuals[f"t{i + 1}_simulated"].to_numpy(float) for i in range(3)], axis=1)
        level = measurement_level.level
        figure = self.sensors.figure
        figure.clear()
        axes = list(figure.subplots(3, 1, sharex=True))
        measured_band = None
        if sigma:
            low, high = measurement_interval(observed, sigma, level)
            measured_band = (temperature_unit.convert(low), temperature_unit.convert(high))
        sensor_comparison(axes, dates, temperature_unit.convert(observed), temperature_unit.convert(simulated), unit=temperature_unit.unit,
                          rmse=temperature_unit.convert_difference(np.array([rmse.get(f"t{i + 1}", np.nan) for i in range(3)], float)),
                          measured_band=measured_band,
                          measured_label=t("Measured ± {z} σT ({level})").format(z=format_number(round(measurement_level.z, 3)),
                                                                                 level=level_label(level)))
        self.predictive, note = None, None
        if not results.results_format.startswith("2."):
            note = Issue(None, "W_FIELD_UNAVAILABLE")
        elif self.status is not None and self.status.config.mode == "mcmc" and sigma:
            self.predictive, note = try_predictive_at_sensors(results, dates, observed, float(sigma), level)
        if self.predictive is not None:
            predictive_band(axes, self.predictive.times, temperature_unit.convert(self.predictive.low),
                            temperature_unit.convert(self.predictive.high), self.predictive.label())
        self.predictive_note.setText("" if note is None else t("Predictive band unavailable: {reason}").format(reason=error_text(note)))
        self.predictive_note.setVisible(note is not None)
        self.sensors.draw()

    def closeEvent(self, event) -> None:  # API Qt
        if self._task is not None:
            self.tasks.cancel()
        super().closeEvent(event)
