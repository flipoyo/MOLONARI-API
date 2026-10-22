"""Onglet « Nettoyage » : période, valeurs aberrantes, exclusions manuelles, aperçu en direct."""

from __future__ import annotations

import pandas as pd
from PySide6 import QtCore, QtWidgets

from molonaviz2.facade import (
    VARIABLES,
    CleanupResult,
    CleanupSettings,
    PointStage,
    UserError,
    clean,
    read_cleaned_csv,
    write_cleaned_csv,
)

from ...i18n import N_, t
from ..dialogs.notice import attach
from ..errors import error_text, report_exception
from ..plots.series import RegionSelector, TimeSeriesPlot
from ..plots.style import (
    PRESSURE_COLOR,
    REJECTED_COLOR,
    RIVER_COLOR,
    THERMOMETER_COLORS,
)
from ..units import temperature_unit
from ..widgets import ask, from_qdatetime, heading, hint, to_qdatetime
from .context import PointContext

METHOD_LABELS = {"none": N_("None"), "iqr": N_("IQR (1.5 × interquartile range)"), "zscore": N_("Z-score (|z| > 3)")}
REASON_LABELS = {
    "outside_window": N_("outside the period"),
    "manual": N_("excluded by hand"),
    "missing": N_("missing value"),
    "iqr": N_("IQR outlier"),
    "zscore": N_("Z-score outlier"),
}
VARIABLE_COLORS = dict(zip(VARIABLES, [RIVER_COLOR, *THERMOMETER_COLORS, PRESSURE_COLOR]))


def variable_label(variable: str) -> str:
    return {
        "temp_bed": t("River / pressure sensor temperature"),
        "t1": t("Sensor 1"),
        "t2": t("Sensor 2"),
        "t3": t("Sensor 3"),
        "t4": t("Sensor 4 (aquifer)"),
        "pressure": t("Head difference ΔH"),
    }[variable]


class CleanupTab(QtWidgets.QWidget):
    def __init__(self, context: PointContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.result: CleanupResult | None = None
        self._raw_t = self._raw_p = None
        self._excluded: list[pd.Timestamp] = []
        self._timer = QtCore.QTimer(self, singleShot=True, interval=150)
        self._timer.timeout.connect(self.preview)

        # Réglages
        self.start = QtWidgets.QDateTimeEdit()
        self.end = QtWidgets.QDateTimeEdit()
        for edit in (self.start, self.end):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
            edit.dateTimeChanged.connect(self._schedule)
        whole = QtWidgets.QPushButton(t("Whole period"))
        whole.clicked.connect(self._whole_period)
        for widget in (self.start, self.end, whole):
            attach(widget, "cleaning.period")
        period = QtWidgets.QFormLayout()
        period.addRow(t("Start"), self.start)
        period.addRow(t("End"), self.end)
        period.addRow("", whole)

        self.methods: dict[str, QtWidgets.QComboBox] = {}
        outliers = QtWidgets.QFormLayout()
        for variable in VARIABLES:
            combo = QtWidgets.QComboBox()
            for key, label in METHOD_LABELS.items():
                combo.addItem(t(label), key)
            combo.currentIndexChanged.connect(self._schedule)
            attach(combo, "cleaning.outliers")
            self.methods[variable] = combo
            outliers.addRow(variable_label(variable), combo)

        self.select_button = QtWidgets.QPushButton(t("Select a period on the graph"))
        self.select_button.setCheckable(True)
        self.select_button.toggled.connect(self._toggle_selection)
        self.exclude_button = QtWidgets.QPushButton(t("Exclude the selected period"))
        self.exclude_button.setEnabled(False)
        self.exclude_button.clicked.connect(self._exclude_selection)
        for widget in (self.select_button, self.exclude_button):
            attach(widget, "cleaning.exclusions")
        self.clear_exclusions = QtWidgets.QPushButton()
        self.clear_exclusions.clicked.connect(self._clear_exclusions)
        manual = QtWidgets.QVBoxLayout()
        manual.addWidget(hint(t("Drag the band on the graph over the dates to remove, then exclude them.")))
        manual.addWidget(self.select_button)
        manual.addWidget(self.exclude_button)
        manual.addWidget(self.clear_exclusions)

        self.summary = QtWidgets.QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.apply_button = QtWidgets.QPushButton(t("Apply cleaning"))
        self.apply_button.setDefault(True)
        self.apply_button.clicked.connect(self.apply)
        import_button = QtWidgets.QPushButton(t("Import cleaned measurements…"))
        import_button.clicked.connect(self._import_cleaned)
        self.export_button = QtWidgets.QPushButton(t("Export cleaned measurements…"))
        self.export_button.clicked.connect(self._export_cleaned)

        settings = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(settings)
        side.addWidget(heading(t("Period")))
        side.addLayout(period)
        side.addWidget(heading(t("Outliers")))
        side.addWidget(hint(t("Computed on the selected period. A date is removed as soon as one variable is an outlier.")))
        side.addLayout(outliers)
        side.addWidget(heading(t("Manual exclusion")))
        side.addLayout(manual)
        side.addWidget(heading(t("Result")))
        side.addWidget(self.summary)
        side.addWidget(self.apply_button)
        side.addWidget(import_button)
        side.addWidget(self.export_button)
        side.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(settings)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(460)

        # Graphe
        self.variable = QtWidgets.QComboBox()
        for variable in VARIABLES:
            self.variable.addItem(variable_label(variable), variable)
        self.variable.currentIndexChanged.connect(self._draw)
        self.plot = TimeSeriesPlot()
        self.selector = RegionSelector(self.plot)
        graph = QtWidgets.QVBoxLayout()
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel(t("Variable shown")))
        row.addWidget(self.variable, 1)
        graph.addLayout(row)
        graph.addWidget(self.plot, 1)
        graph.addWidget(hint(t("Kept measurements as a line; removed measurements as red dots. Nothing is saved until you apply.")))

        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(graph, 1)
        temperature_unit.changed.connect(self._draw)
        self.load()

    # Chargement -------------------------------------------------------------------------
    def load(self) -> None:
        project, study, name = self.context.project, self.context.study, self.context.name
        self._raw_t = project.raw_temperature(study, name)
        self._raw_p = project.raw_pressure(study, name)
        settings = CleanupSettings.from_dict(project.cleanup_settings(study, name))
        dates = pd.concat([self._raw_t["date"], self._raw_p["date"]])
        for edit in (self.start, self.end):
            edit.blockSignals(True)
        if len(dates):
            self.start.setDateTime(to_qdatetime(settings.start or dates.min()))
            self.end.setDateTime(to_qdatetime(settings.end or dates.max()))
        for edit in (self.start, self.end):
            edit.blockSignals(False)
        for variable, combo in self.methods.items():
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(settings.methods.get(variable, "none"))))
            combo.blockSignals(False)
        self._excluded = [pd.Timestamp(d) for d in settings.excluded]
        self.preview()

    def _settings(self) -> CleanupSettings:
        return CleanupSettings(
            start=from_qdatetime(self.start.dateTime()),
            end=from_qdatetime(self.end.dateTime()),
            methods={v: combo.currentData() for v, combo in self.methods.items()},
            excluded=[d.to_pydatetime() for d in self._excluded],
        )

    def _schedule(self) -> None:
        self._timer.start()

    def _whole_period(self) -> None:
        dates = pd.concat([self._raw_t["date"], self._raw_p["date"]])
        if len(dates):
            self.start.setDateTime(to_qdatetime(dates.min()))
            self.end.setDateTime(to_qdatetime(dates.max()))

    # Aperçu -----------------------------------------------------------------------------
    def preview(self) -> None:
        self.clear_exclusions.setText(t("Restore manually excluded dates ({n})").format(n=len(self._excluded)))
        self.clear_exclusions.setEnabled(bool(self._excluded))
        try:
            self.result = clean(self._raw_t, self._raw_p, self.context.sensors().pressure, self._settings())
        except UserError as exc:
            self.result = None
            self.summary.setText(f"<span style='color:#c0392b'>{error_text(exc)}</span>")
            self.apply_button.setEnabled(False)
            self.plot.reset()
            return
        kept = int(self.result.kept.sum())
        total = len(self.result.measures)
        lines = [t("<b>{kept}</b> dates kept out of {total}").format(kept=kept, total=total)]
        for reason, count in sorted(self.result.counts().items()):
            kind, _, variable = reason.partition(":")
            label = t(REASON_LABELS.get(kind, kind))
            if variable:
                label += f" — {variable_label(variable)}"
            lines.append(f"{count} × {label}")
        if self.result.unmatched_temperature or self.result.unmatched_pressure:
            lines.append(
                t("Dates without a match in the other file: {t} (temperature), {p} (pressure).").format(
                    t=self.result.unmatched_temperature, p=self.result.unmatched_pressure
                )
            )
        if kept < 3:
            lines.append("<span style='color:#c0392b'>" + t("At least 3 dates are needed to run a model.") + "</span>")
        self.summary.setText("<br>".join(lines))
        self.apply_button.setEnabled(kept >= 3)
        self._draw()

    def _draw(self) -> None:
        self.plot.reset()
        if self.result is None:
            return
        variable = self.variable.currentData()
        measures = self.result.measures
        kept = self.result.kept
        is_temperature = variable != "pressure"
        convert = temperature_unit.convert if is_temperature else (lambda v: v)
        self.plot.set_y_label(
            t("Temperature ({unit})").format(unit=temperature_unit.unit) if is_temperature else t("Head difference ΔH (m)")
        )
        kept_data = measures.loc[kept, ["date", variable]]
        removed = measures.loc[~kept, ["date", variable]].dropna()
        self.plot.plot_series(kept_data["date"], convert(kept_data[variable]), t("Kept"), VARIABLE_COLORS[variable])
        if len(removed):
            self.plot.plot_points(removed["date"], convert(removed[variable]), t("Removed"), REJECTED_COLOR, size=5)
        if self.select_button.isChecked():
            self.plot.addItem(self.selector.region, ignoreBounds=True)

    # Exclusions manuelles ---------------------------------------------------------------
    def _toggle_selection(self, active: bool) -> None:
        if active:
            self.selector.show_default()
        else:
            self.selector.hide()
        self.exclude_button.setEnabled(active)

    def _exclude_selection(self) -> None:
        bounds = self.selector.bounds()
        if bounds is None or self.result is None:
            return
        dates = self.result.measures["date"]
        chosen = dates[(dates >= bounds[0]) & (dates <= bounds[1])]
        self._excluded = sorted(set(self._excluded) | set(pd.to_datetime(chosen)))
        self.select_button.setChecked(False)
        self.preview()

    def _clear_exclusions(self) -> None:
        self._excluded = []
        self.preview()

    # Enregistrement ---------------------------------------------------------------------
    def apply(self) -> None:
        if self.result is None:
            return
        project, study, name = self.context.project, self.context.study, self.context.name
        if self.context.stage >= PointStage.DIRECT_MODEL and not ask(
            self,
            t("Replace the cleaned measurements"),
            t("The computations of this point were made with the current cleaned measurements. Applying a new cleaning deletes them."),
            t("Apply and delete computations"),
        ):
            return
        project.set_cleaned(study, name, self.result.cleaned, self._settings().to_dict())
        self.context.notify()

    def _import_cleaned(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("Import cleaned measurements"), "", "CSV (*.csv)")
        if not path:
            return
        try:
            cleaned = read_cleaned_csv(path)
        except (UserError, ValueError, OSError) as exc:
            report_exception(exc if isinstance(exc, UserError) else UserError("E_SYSTEM_ERROR", detail=str(exc)))
            return
        if self.context.stage >= PointStage.CLEANED and not ask(
            self, t("Replace the cleaned measurements"), t("This replaces the current cleaned measurements and deletes the computations of this point."), t("Replace")
        ):
            return
        self.context.project.set_cleaned(self.context.study, self.context.name, cleaned, {"source": "file", "path": path})
        self.context.notify()

    def _export_cleaned(self) -> None:
        cleaned = self.context.project.cleaned(self.context.study, self.context.name)
        if cleaned.empty:
            report_exception(UserError("E_NO_CLEANED_MEASURES"))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, t("Export cleaned measurements"), f"{self.context.name}_cleaned.csv", "CSV (*.csv)")
        if path:
            write_cleaned_csv(cleaned, path)
