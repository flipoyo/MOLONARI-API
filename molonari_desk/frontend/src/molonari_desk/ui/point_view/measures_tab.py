"""Onglet « Mesures » : graphes et tableau des mesures brutes ou nettoyées."""

from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtWidgets

from ...i18n import t
from ..plots.series import TimeSeriesPlot, link_time_axes
from ..plots.style import PRESSURE_COLOR, RIVER_COLOR, THERMOMETER_COLORS
from ..units import temperature_unit
from ..widgets import DataFrameModel, hint, table_view
from .context import PointContext

TEMPERATURE_COLUMNS = ["temp_bed", "t1", "t2", "t3", "t4"]


def column_labels() -> dict[str, str]:
    unit = temperature_unit.unit
    return {
        "date": t("Date"),
        "temp_bed": t("River / sensor ({unit})").format(unit=unit),
        "t1": t("Sensor 1 ({unit})").format(unit=unit),
        "t2": t("Sensor 2 ({unit})").format(unit=unit),
        "t3": t("Sensor 3 ({unit})").format(unit=unit),
        "t4": t("Sensor 4 ({unit})").format(unit=unit),
        "voltage": t("Voltage (V)"),
        "pressure": t("Head difference ΔH (m)"),
    }


class MeasuresTab(QtWidgets.QWidget):
    def __init__(self, context: PointContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.raw_button = QtWidgets.QRadioButton(t("Raw measurements"))
        self.cleaned_button = QtWidgets.QRadioButton(t("Cleaned measurements"))
        self.raw_button.setChecked(True)
        self.summary = hint("")
        top = QtWidgets.QHBoxLayout()
        top.addWidget(self.raw_button)
        top.addWidget(self.cleaned_button)
        top.addSpacing(20)
        top.addWidget(self.summary, 1)

        self.temperature_plot = TimeSeriesPlot()
        self.pressure_plot = TimeSeriesPlot()
        link_time_axes([self.temperature_plot, self.pressure_plot])
        self.model = DataFrameModel()
        self.table = table_view(self.model)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        splitter.addWidget(self.temperature_plot)
        splitter.addWidget(self.pressure_plot)
        splitter.addWidget(self.table)
        splitter.setSizes([300, 200, 220])

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        self.raw_button.toggled.connect(self.refresh)
        context.changed.connect(self.refresh)
        temperature_unit.changed.connect(self.refresh)
        self.refresh()

    def _frame(self) -> pd.DataFrame:
        project, study, name = self.context.project, self.context.study, self.context.name
        if self.raw_button.isChecked():
            temperature = project.raw_temperature(study, name)
            pressure = project.raw_pressure(study, name)
            return temperature.merge(pressure, on="date", how="outer").sort_values("date").reset_index(drop=True)
        return project.cleaned(study, name)

    def refresh(self) -> None:
        has_cleaned = self.context.stage >= 1
        self.cleaned_button.setEnabled(has_cleaned)
        self.cleaned_button.setToolTip("" if has_cleaned else t("Clean the measurements first (Cleaning tab)."))
        if not has_cleaned and self.cleaned_button.isChecked():
            self.raw_button.setChecked(True)
            return
        frame = self._frame()
        raw = self.raw_button.isChecked()
        unit = temperature_unit.unit
        labels = column_labels()
        self.temperature_plot.reset()
        self.temperature_plot.set_y_label(t("Temperature ({unit})").format(unit=unit))
        for column, color in zip(TEMPERATURE_COLUMNS, [RIVER_COLOR, *THERMOMETER_COLORS]):
            data = frame[["date", column]].dropna()
            self.temperature_plot.plot_series(data["date"], temperature_unit.convert(data[column]), labels[column], color)
        self.pressure_plot.reset()
        pressure_column = "voltage" if raw else "pressure"
        self.pressure_plot.set_y_label(labels[pressure_column])
        data = frame[["date", pressure_column]].dropna()
        self.pressure_plot.plot_series(data["date"], data[pressure_column], labels[pressure_column], PRESSURE_COLOR)

        shown = frame.copy()
        for column in TEMPERATURE_COLUMNS:
            if column in shown:
                shown[column] = temperature_unit.convert(shown[column])
        self.model._headers = labels
        self.model.set_frame(shown)
        if len(frame):
            self.summary.setText(
                t("{n} dates · {start} → {end}").format(
                    n=len(frame), start=frame["date"].iloc[0].strftime("%Y-%m-%d %H:%M"), end=frame["date"].iloc[-1].strftime("%Y-%m-%d %H:%M")
                )
            )
        else:
            self.summary.setText(t("No measurements."))
