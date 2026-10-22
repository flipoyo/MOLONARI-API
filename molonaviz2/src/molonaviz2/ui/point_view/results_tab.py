"""Onglet « Résultats » : températures, profils, flux, paramètres a posteriori, exports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from PySide6 import QtCore, QtGui, QtWidgets

from molonaviz2 import facade as api
from molonaviz2.facade import (
    ComputationInfo,
    ForwardConfig,
    ForwardResult,
    InversionConfig,
    InversionResult,
    chain_data,
)

from ...i18n import t
from ..dialogs.notice import attach
from ..errors import error_text
from ..plots import posterior
from ..plots.maps import (
    FigurePanel,
    depth_time_map,
    predictive_band,
    profiles,
    sensor_comparison,
    time_series_with_band,
)
from ..plots.style import CHAIN_COLORMAP
from ..predictive import try_predictive_at_sensors
from ..units import (
    level_label,
    measurement_interval,
    measurement_level,
    temperature_unit,
)
from ..widgets import DataFrameModel, ask, format_number, hint, table_view
from .context import PointContext


def _duration(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 60 else f"{int(seconds // 60)} min {int(seconds % 60):02d} s"


class ResultsTab(QtWidgets.QWidget):
    load_forward_config = QtCore.Signal(object)
    load_inversion_config = QtCore.Signal(object)
    create_twin_requested = QtCore.Signal(object)  # ComputationInfo

    def __init__(self, context: PointContext, parent=None):
        super().__init__(parent)
        self.context = context
        self._infos: list[ComputationInfo] = []
        self.forward: ForwardResult | None = None
        self.inversion: InversionResult | None = None
        self.predictive = None  # PredictiveBand dessinée aux capteurs (inversion avec champs stockés)
        self.info: ComputationInfo | None = None

        self.selector = QtWidgets.QComboBox()
        self.selector.setMinimumWidth(320)
        self.selector.currentIndexChanged.connect(self._select)
        self.reload_button = QtWidgets.QPushButton(t("Reuse this configuration"))
        self.reload_button.setToolTip(t("Loads these settings into the Direct model or Inversion tab, to run the same computation again."))
        self.reload_button.clicked.connect(self._reuse)
        export = QtWidgets.QToolButton()
        export.setText(t("Export"))
        export.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QtWidgets.QMenu(export)
        menu.addAction(t("Data as CSV files…"), self.export_csv)
        self.netcdf_action = menu.addAction(t("Posterior samples (NetCDF, ArviZ)…"), self.export_netcdf)
        export.setMenu(menu)
        self.twin_button = QtWidgets.QPushButton(t("Create a twin…"))
        self.twin_button.setToolTip(t("Freeze this computation as a twin: twin.yaml + measurements, in the twins folder."))
        self.twin_button.clicked.connect(lambda: self.info is not None and self.create_twin_requested.emit(self.info))
        self.delete_button = QtWidgets.QPushButton(t("Delete"))
        self.delete_button.clicked.connect(self._delete)
        self.provenance = hint("")
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel(t("Computation")))
        top.addWidget(self.selector, 1)
        top.addWidget(self.reload_button)
        top.addWidget(export)
        top.addWidget(self.twin_button)
        top.addWidget(self.delete_button)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)

        # Températures
        self.temperature_map = FigurePanel(height=3.2)
        self.sensors = FigurePanel(height=4.5)
        self.measurement_level = measurement_level.combo()
        attach(self.measurement_level, "results.measurement_level")
        self.sigma_source = hint("")
        self.predictive_note = hint("")  # bande prédictive indisponible (pas de champs stockés)
        self.predictive_note.hide()
        attach(self.predictive_note, "results.predictive_band")
        self.sensors.setProperty("notice_key", "results.predictive_band")  # Maj+F1 sur le graphe, sans infobulle qui le masque
        sensors_page = QtWidgets.QWidget()
        sensors_layout = QtWidgets.QVBoxLayout(sensors_page)
        sensors_layout.setContentsMargins(0, 0, 0, 0)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Measurement interval")))
        row.addWidget(self.measurement_level)
        row.addWidget(self.sigma_source, 1)
        sensors_layout.addLayout(row)
        sensors_layout.addWidget(self.predictive_note)
        sensors_layout.addWidget(self.sensors, 1)
        temps = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        temps.addWidget(self.temperature_map)
        temps.addWidget(sensors_page)
        self.tabs.addTab(temps, t("Temperatures"))

        # Profils
        self.profile_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.profile_slider.valueChanged.connect(self._draw_profile)
        self.profile_date = QtWidgets.QLabel()
        self.profile = FigurePanel(height=5)
        profile_page = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profile_page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Date")))
        row.addWidget(self.profile_slider, 1)
        row.addWidget(self.profile_date)
        profile_layout.addLayout(row)
        profile_layout.addWidget(self.profile, 1)
        self.tabs.addTab(profile_page, t("Profiles"))

        # Flux
        self.depth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.depth_slider.valueChanged.connect(self._draw_water_flux)
        self.depth_label = QtWidgets.QLabel()
        self.water_flux = FigurePanel(height=3)
        self.heat_kind = QtWidgets.QComboBox()
        self.heat_kind.addItem(t("Total heat flux"), "total")
        self.heat_kind.addItem(t("Advective heat flux"), "advective")
        self.heat_kind.addItem(t("Conductive heat flux"), "conductive")
        self.heat_kind.currentIndexChanged.connect(self._draw_heat_map)
        self.heat_map = FigurePanel(height=3.2)
        flux_page = QtWidgets.QWidget()
        flux_layout = QtWidgets.QVBoxLayout(flux_page)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Depth")))
        row.addWidget(self.depth_slider, 1)
        row.addWidget(self.depth_label)
        flux_layout.addLayout(row)
        flux_layout.addWidget(self.water_flux, 1)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Map")))
        row.addWidget(self.heat_kind)
        row.addStretch()
        flux_layout.addLayout(row)
        flux_layout.addWidget(self.heat_map, 1)
        self.tabs.addTab(flux_page, t("Fluxes"))

        # Paramètres (inversion)
        self.summary_model = DataFrameModel()
        self.summary_view = table_view(self.summary_model)
        self.summary_view.setMaximumHeight(220)
        self.layer_choice = QtWidgets.QComboBox()
        self.layer_choice.currentIndexChanged.connect(self._draw_parameters)
        self.bins = QtWidgets.QSpinBox()
        self.bins.setRange(5, 200)
        self.bins.setValue(40)
        self.bins.valueChanged.connect(self._draw_parameters)
        self.histograms = FigurePanel(height=3.5)
        # Chaînes : tout passe par questions.chain_data
        self.chain_list = QtWidgets.QListWidget()
        self.chain_list.setWordWrap(True)  # « Chaîne n — acceptation … · … réinitialisation(s) » en entier
        self.chain_list.itemChanged.connect(lambda _: self._draw_chains())
        self.display_thin = QtWidgets.QSpinBox()
        self.display_thin.setMinimum(1)
        self.display_thin.valueChanged.connect(self._draw_chains)
        self.iteration_from = QtWidgets.QSpinBox()
        self.iteration_to = QtWidgets.QSpinBox()
        for box in (self.iteration_from, self.iteration_to):
            box.valueChanged.connect(self._draw_chains)
        self.chain_count = hint("")
        for widget in (self.chain_list, self.display_thin, self.iteration_from, self.iteration_to):
            attach(widget, "results.chains")
        self.traces = FigurePanel(height=5)
        parameters_page = QtWidgets.QWidget()
        parameters_layout = QtWidgets.QVBoxLayout(parameters_page)
        parameters_layout.addWidget(hint(t("Best: lowest-energy parameters · HDI 90 %: highest density interval · ESS: effective sample size · R-hat: convergence (≤ 1.01 is good)")))
        parameters_layout.addWidget(self.summary_view)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Layer")))
        row.addWidget(self.layer_choice)
        row.addSpacing(20)
        row.addWidget(QtWidgets.QLabel(t("Histogram bins")))
        row.addWidget(self.bins)
        row.addStretch()
        parameters_layout.addLayout(row)
        chain_controls = QtWidgets.QWidget()
        controls = QtWidgets.QFormLayout(chain_controls)
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addRow(self.chain_list)
        controls.addRow(t("Display thinning"), self.display_thin)
        controls.addRow(t("From iteration"), self.iteration_from)
        controls.addRow(t("To iteration"), self.iteration_to)
        controls.addRow(self.chain_count)
        chain_controls.setMaximumWidth(420)
        chains_page = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        chains_page.addWidget(chain_controls)
        chains_page.addWidget(self.traces)
        chains_page.setStretchFactor(1, 1)
        chains_page.setSizes([380, 900])
        self.parameter_views = QtWidgets.QTabWidget()
        self.parameter_views.addTab(self.histograms, t("Histograms"))
        self.chains_index = self.parameter_views.addTab(chains_page, t("Chains"))
        parameters_layout.addWidget(self.parameter_views, 1)
        self.parameters_index = self.tabs.addTab(parameters_page, t("Parameters"))

        self.empty = hint(t("No computation yet for this point. Run the direct model or an inversion first."))
        self.empty.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.provenance)
        layout.addWidget(self.tabs, 1)
        layout.addWidget(self.empty, 1)

        temperature_unit.changed.connect(self._draw_all)
        measurement_level.changed.connect(self._draw_temperatures)
        context.changed.connect(self.refresh)
        self.refresh()

    # Sélection ------------------------------------------------------------------------
    def refresh(self, select_id: int | None = None) -> None:
        current = self.info.id if self.info else None
        self._infos = self.context.project.computations(self.context.study, self.context.name)
        self.selector.blockSignals(True)
        self.selector.clear()
        for info in self._infos:
            kind = t("Inversion") if info.kind == "inversion" else t("Direct model")
            self.selector.addItem(f"{kind} — {info.created_at:%Y-%m-%d %H:%M} — {_duration(info.duration_s)}", info.id)
        self.selector.blockSignals(False)
        has = bool(self._infos)
        self.tabs.setVisible(has)
        self.empty.setVisible(not has)
        for widget in (self.selector, self.reload_button, self.twin_button, self.delete_button):
            widget.setEnabled(has)
        if not has:
            self.info = self.forward = self.inversion = None
            self.provenance.setText("")
            return
        target = select_id if select_id is not None else current
        index = self.selector.findData(target) if target is not None else 0
        self.selector.setCurrentIndex(max(index, 0))
        self._select()

    def show_computation(self, computation_id: int) -> None:
        self.refresh(select_id=computation_id)

    def _select(self) -> None:
        index = self.selector.currentIndex()
        if index < 0 or index >= len(self._infos):
            return
        self.info = self._infos[index]
        arrays = self.context.project.load_result(self.info.id)
        if self.info.kind == "inversion":
            self.inversion = InversionResult.from_arrays(arrays)
            self.forward = self.inversion.best
        else:
            self.inversion = None
            self.forward = ForwardResult.from_arrays(arrays)
        config = self.info.config
        details = [
            t("Engine: {engine}").format(engine=self.info.engine),
            t("Device: {device}").format(device=self.info.device),
            t("Duration: {time}").format(time=_duration(self.info.duration_s)),
            t("{n} layer(s), {cells} cells").format(n=len(config["layers"]), cells=config["nb_cells"]),
        ]
        if self.inversion is not None:
            layout = api.sampling_layout(InversionConfig.from_dict(config), len(self.forward.times))
            details += [
                t("seed {seed}").format(seed=config["seed"]),
                t("{chains} chains").format(chains=config["nb_chains"]),
                t("burn-in {n} iterations per chain ({state})").format(
                    n=self.inversion.burn_in_iterations,
                    state=t("converged") if self.inversion.burn_in_converged else t("not converged"),
                ),
                t("{n} iterations kept per chain").format(n=self.inversion.samples.shape[0]),
                t("{k} fields per chain, one every {m} iterations").format(k=layout.fields_per_chain, m=layout.interval),
            ]
        self.provenance.setText(" · ".join(details))
        self.tabs.setTabEnabled(self.parameters_index, self.inversion is not None)
        self.tabs.setTabToolTip(self.parameters_index, "" if self.inversion is not None else t("Available after an inversion."))
        self.netcdf_action.setEnabled(self.inversion is not None)
        self.profile_slider.setRange(0, len(self.forward.times) - 1)
        self.depth_slider.setRange(0, len(self.forward.depths) - 1)
        self.depth_slider.setValue(int(self.forward.sensor_index[0]))
        if self.inversion is not None:
            self.layer_choice.blockSignals(True)
            self.layer_choice.clear()
            self.layer_choice.addItems([layer["name"] for layer in config["layers"]])
            self.layer_choice.blockSignals(False)
            table = api.summary_table(self.inversion, self._layer_names())
            self.summary_model._headers = {
                "parameter": t("Parameter"), "best": t("Best"), "mean": t("Mean"), "sd": t("Std. dev."),
                "hdi90_lb": t("HDI 90 % low"), "hdi90_ub": t("HDI 90 % high"), "ess_bulk": "ESS", "r_hat": "R-hat",
            }
            self.summary_model.set_frame(table.reset_index(names="parameter"))
            self.summary_view.resizeColumnsToContents()
            self._fill_chain_controls()
        self._draw_all()

    def _layer_names(self) -> list[str]:
        return [layer["name"] for layer in self.info.config["layers"]]

    # Dessin ---------------------------------------------------------------------------
    def _draw_all(self) -> None:
        if self.forward is None:
            return
        self._draw_temperatures()
        self._draw_profile()
        self._draw_water_flux()
        self._draw_heat_map()
        if self.inversion is not None:
            self._draw_parameters()

    def _band_at_sensors(self):
        """Bande [quantile min, quantile max] aux 3 capteurs, sur les dates sous-échantillonnées."""
        if self.inversion is None or not self.inversion.quantiles:
            return None, None
        quantiles = sorted(self.inversion.quantiles, key=lambda q: q.quantile)
        depths = self.inversion.field_depths
        rows = [int(np.argmin(np.abs(depths - d))) for d in self.forward.sensor_depths]
        low = quantiles[0].temperature[rows].T
        high = quantiles[-1].temperature[rows].T
        return self.inversion.field_times, (temperature_unit.convert(low), temperature_unit.convert(high))

    def sigma_t(self) -> tuple[float, str]:
        """Incertitude des mesures de température (°C) et son origine, pour l'intervalle de mesure."""
        if self.inversion is not None and self.info is not None and "sigma_t" in self.info.config:
            return float(self.info.config["sigma_t"]), t("σT of the inversion")
        thermometer = self.context.sensors().thermometer
        if thermometer is not None and thermometer.sigma > 0:
            return float(thermometer.sigma), t("uncertainty of thermometer {name}").format(name=thermometer.name)
        return 0.1, t("default value: no uncertainty is given for the thermometer")

    def _draw_temperatures(self) -> None:
        if self.forward is None:
            return
        result = self.forward
        unit = temperature_unit.unit
        figure = self.temperature_map.figure
        figure.clear()
        ax = figure.add_subplot()
        depth_time_map(ax, result.times, result.depths, temperature_unit.convert(result.temperature), t("Temperature ({unit})").format(unit=unit),
                       sensor_depths=result.sensor_depths)
        ax.set_title(t("Simulated temperature (best fit)") if self.inversion is not None else t("Simulated temperature"), fontsize=10, loc="left")
        self.temperature_map.draw()

        figure = self.sensors.figure
        figure.clear()
        axes = figure.subplots(3, 1, sharex=True)
        band_times, band = self._band_at_sensors()
        sigma, source = self.sigma_t()
        level = measurement_level.level
        low, high = measurement_interval(result.observed, sigma, level)
        label = t("Measured ± {z} σT ({level})").format(z=format_number(round(measurement_level.z, 3)), level=level_label(level))
        self.sigma_source.setText(t("σT = {sigma} {unit} ({source})").format(
            sigma=format_number(float(temperature_unit.convert_difference(sigma))), unit=unit, source=source))
        sensor_comparison(list(axes), result.times, temperature_unit.convert(result.observed),
                          temperature_unit.convert(result.simulated_at_sensors()), unit=unit,
                          rmse=temperature_unit.convert_difference(result.rmse),
                          measured_band=(temperature_unit.convert(low), temperature_unit.convert(high)), measured_label=label)
        if band is not None:
            for i, ax in enumerate(axes):
                ax.fill_between(pd.to_datetime(band_times).to_numpy(), band[0][:, i], band[1][:, i], color="#444444", alpha=0.18, lw=0,
                                label=t("Credible interval") if i == 0 else None)
            axes[0].legend(loc="upper right", fontsize=8)
        self.predictive, error = (None, None) if self.inversion is None else try_predictive_at_sensors(
            self.inversion, result.times, result.observed, sigma, level)
        if self.predictive is not None:
            predictive_band(list(axes), self.predictive.times, temperature_unit.convert(self.predictive.low),
                            temperature_unit.convert(self.predictive.high), self.predictive.label())
        self.predictive_note.setText("" if error is None else t("Predictive band unavailable: {reason}").format(reason=error_text(error)))
        self.predictive_note.setVisible(error is not None)
        self.sensors.draw()

    def _draw_profile(self) -> None:
        if self.forward is None:
            return
        result = self.forward
        j = self.profile_slider.value()
        date = pd.Timestamp(result.times[j])
        self.profile_date.setText(date.strftime("%Y-%m-%d %H:%M"))
        figure = self.profile.figure
        figure.clear()
        ax = figure.add_subplot()
        unit = temperature_unit.unit
        curves = [(t("Simulated"), temperature_unit.convert(result.temperature[:, j]))]
        if self.inversion is not None and self.inversion.quantiles:
            times = pd.to_datetime(self.inversion.field_times)
            k = int(np.argmin(np.abs((times - date).total_seconds())))
            quantiles = sorted(self.inversion.quantiles, key=lambda q: q.quantile)
            ax.fill_betweenx(self.inversion.field_depths, temperature_unit.convert(quantiles[0].temperature[:, k]),
                             temperature_unit.convert(quantiles[-1].temperature[:, k]), color="#444444", alpha=0.18, lw=0,
                             label=t("Credible interval"))
        observed = (result.sensor_depths, [temperature_unit.convert(result.observed[j])])
        profiles(ax, result.depths, curves, unit, observed=observed)
        ax.set_title(t("Temperature profile — dots: measurements"), fontsize=10, loc="left")
        self.profile.draw()

    def _draw_water_flux(self) -> None:
        if self.forward is None:
            return
        result = self.forward
        i = self.depth_slider.value()
        depth = result.depths[i]
        self.depth_label.setText(f"{format_number(depth)} m")
        band = None
        if self.inversion is not None and self.inversion.quantiles:
            quantiles = sorted(self.inversion.quantiles, key=lambda q: q.quantile)
            row = int(np.argmin(np.abs(self.inversion.field_depths - depth)))
            band_times = self.inversion.field_times
            band = (quantiles[0].water_flux[row], quantiles[-1].water_flux[row])
        figure = self.water_flux.figure
        figure.clear()
        ax = figure.add_subplot()
        if band is not None and len(band_times) != len(result.times):
            ax.fill_between(pd.to_datetime(band_times).to_numpy(), band[0], band[1], color="#444444", alpha=0.2, lw=0, label=t("Credible interval"))
            band = None
        time_series_with_band(ax, result.times, result.water_flux[i], band, label=t("Specific discharge"), y_label=t("Water flux (m/s), positive downwards"))
        self.water_flux.draw()

    def _draw_heat_map(self) -> None:
        if self.forward is None:
            return
        result = self.forward
        kind = self.heat_kind.currentData()
        values = {
            "total": result.total_heat_flux,
            "advective": result.advective_flux,
            "conductive": result.conductive_flux_down,
        }[kind]
        figure = self.heat_map.figure
        figure.clear()
        ax = figure.add_subplot()
        if not np.any(np.isfinite(values)) or np.nanmax(np.abs(values)) == 0:
            ax.axis("off")
            ax.text(0.5, 0.5, t("This flux is zero for this computation."), ha="center", va="center")
        else:
            depth_time_map(ax, result.times, result.depths, values, t("Heat flux (W/m²), positive downwards"), signed=True, sensor_depths=result.sensor_depths)
            ax.set_title(self.heat_kind.currentText(), fontsize=10, loc="left")
        self.heat_map.draw()

    def _draw_parameters(self) -> None:
        if self.inversion is None:
            return
        layer = max(self.layer_choice.currentIndex(), 0)
        posterior.histograms(self.histograms.figure, self.inversion, layer, self.bins.value())
        self.histograms.draw()
        self._draw_chains()

    def _fill_chain_controls(self) -> None:
        """Liste des chaînes (acceptation, réinitialisations), thin d'affichage et plage d'itérations du calcul choisi."""
        everything = chain_data(self.inversion).data
        iterations = everything["iteration"].value
        widgets = (self.chain_list, self.display_thin, self.iteration_from, self.iteration_to)
        for widget in widgets:
            widget.blockSignals(True)
        self.chain_list.clear()
        for chain, acceptance, resets in zip(everything["chain"].value, everything["acceptance"].value, everything["resets"].value):
            item = QtWidgets.QListWidgetItem(t("Chain {n} — acceptance {acceptance} · {resets} reset(s)").format(
                n=int(chain), acceptance=f"{float(acceptance) * 100:.0f} %", resets=int(resets)))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, int(chain))
            item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(QtCore.Qt.CheckState.Checked)
            color = QtGui.QPixmap(12, 12)
            color.fill(QtGui.QColor(to_hex(CHAIN_COLORMAP(int(chain) % 10))))
            item.setIcon(QtGui.QIcon(color))
            self.chain_list.addItem(item)
        self.display_thin.setRange(1, max(len(iterations), 1))
        self.display_thin.setValue(1)
        first, last = (int(iterations[0]), int(iterations[-1])) if len(iterations) else (0, 0)
        step = int(iterations[1] - iterations[0]) if len(iterations) > 1 else 1
        for box, value in ((self.iteration_from, first), (self.iteration_to, last)):
            box.setRange(first, last)
            box.setSingleStep(step)
            box.setValue(value)
        for widget in widgets:
            widget.blockSignals(False)

    def checked_chains(self) -> list[int]:
        items = (self.chain_list.item(i) for i in range(self.chain_list.count()))
        return [item.data(QtCore.Qt.ItemDataRole.UserRole) for item in items if item.checkState() == QtCore.Qt.CheckState.Checked]

    def _draw_chains(self) -> None:
        if self.inversion is None:
            return
        answer = chain_data(self.inversion, chains=self.checked_chains(), thin=self.display_thin.value(),
                            iterations=(self.iteration_from.value(), self.iteration_to.value()))
        posterior.chain_traces(self.traces.figure, answer, max(self.layer_choice.currentIndex(), 0))
        self.traces.draw()
        self.chain_count.setText(t("{count} samples shown ({chains} chains × {iterations} iterations)").format(
            count=answer.data["count"].value, chains=len(answer.data["chain"].value), iterations=len(answer.data["iteration"].value)))

    # Actions --------------------------------------------------------------------------
    def _reuse(self) -> None:
        if self.info is None:
            return
        if self.info.kind == "inversion":
            self.load_inversion_config.emit(InversionConfig.from_dict(self.info.config))
        else:
            self.load_forward_config.emit(ForwardConfig.from_dict(self.info.config))

    def _delete(self) -> None:
        if self.info is None:
            return
        if ask(self, t("Delete this computation"), t("The results of this computation will be deleted. This cannot be undone."), t("Delete")):
            self.context.project.delete_computation(self.info.id)
            self.info = None
            self.context.notify()

    def export_csv(self, folder: str | None = None) -> None:
        if self.forward is None:
            return
        if folder is None:
            folder = QtWidgets.QFileDialog.getExistingDirectory(self, t("Folder for the CSV files"))
            if not folder:
                return
        write_csv_exports(Path(folder), self.forward, self.inversion, self._layer_names())

    def export_netcdf(self, path: str | None = None) -> None:
        if self.inversion is None:
            return
        if path is None:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, t("Export posterior samples"), f"{self.context.name}_posterior.nc", "NetCDF (*.nc)")
            if not path:
                return
        api.export_netcdf(self.inversion, self._layer_names(), path)


def write_csv_exports(folder: Path, forward: ForwardResult, inversion: InversionResult | None, layer_names: list[str]) -> list[Path]:
    """Écrit les résultats en CSV (°C, m/s, W/m²). Les cartes : une ligne par profondeur, une colonne par date.

    Flux d'eau et de chaleur positifs vers le bas."""
    folder.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(forward.times).strftime("%Y-%m-%d %H:%M:%S")
    depths = pd.Index(np.round(forward.depths, 6), name="depth_m")
    written = []
    for name, values in (
        ("temperature_C", forward.temperature),
        ("water_flux_m_per_s", forward.water_flux),
        ("advective_heat_flux_W_m2", forward.advective_flux),
        ("conductive_heat_flux_down_W_m2", forward.conductive_flux_down),
    ):
        path = folder / f"{name}.csv"
        pd.DataFrame(values, index=depths, columns=times).to_csv(path)
        written.append(path)
    sensors = pd.DataFrame({"date": times})
    for i in range(3):
        sensors[f"measured_{i + 1}_C"] = forward.observed[:, i]
        sensors[f"simulated_{i + 1}_C"] = forward.simulated_at_sensors()[:, i]
    path = folder / "sensors.csv"
    sensors.to_csv(path, index=False)
    written.append(path)
    if inversion is not None:
        table = api.summary_table(inversion, layer_names)
        path = folder / "posterior_summary.csv"
        table.to_csv(path)
        written.append(path)
        flat = inversion.samples.reshape(-1, inversion.samples.shape[2], inversion.samples.shape[3])
        columns = {f"{p}[{layer}]": flat[:, i, k] for i, layer in enumerate(layer_names) for k, p in enumerate(posterior.INVERTED_PARAMETERS)}
        path = folder / "posterior_samples.csv"
        pd.DataFrame(columns).to_csv(path, index=False)
        written.append(path)
    return written
