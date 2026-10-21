"""Import d'un point : fichiers repérés automatiquement, champs pré-remplis, aperçu des mesures.

Rien n'est écrit dans le projet avant de cliquer sur « Importer ». Chaque fichier de
mesures est lu dès qu'il est choisi : l'aperçu indique les lignes gardées, le format
de date détecté, la période couverte, les conversions d'unités et les lignes écartées
(avec leur numéro).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from ...domain.models import Lab, SamplingPoint
from ...errors import UserError
from ...i18n import t
from ...io.lab_import import read_text
from ...io.point_import import ParsedMeasures, find_point_files, read_point_info, read_pressure_file, read_temperature_file
from ..errors import error_text
from ..widgets import NumberField, hint, to_qdatetime

MAX_DROPPED_SHOWN = 6


class _FileRow(QtWidgets.QWidget):
    changed = QtCore.Signal()

    def __init__(self, file_filter: str, parent=None):
        super().__init__(parent)
        self._filter = file_filter
        self.edit = QtWidgets.QLineEdit()
        self.edit.setReadOnly(True)
        self.edit.setPlaceholderText(t("No file"))
        browse = QtWidgets.QPushButton(t("Choose…"))
        browse.clicked.connect(self._browse)
        clear = QtWidgets.QToolButton()
        clear.setText("×")
        clear.setToolTip(t("Remove this file"))
        clear.clicked.connect(lambda: self.set_path(None))
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(browse)
        layout.addWidget(clear)
        self.path: Path | None = None

    def _browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, t("Choose a file"), str(self.path.parent) if self.path else "", self._filter)
        if path:
            self.set_path(Path(path))

    def set_path(self, path: Path | None) -> None:
        self.path = path
        self.edit.setText(path.name if path else "")
        self.edit.setToolTip(str(path) if path else "")
        self.changed.emit()


class PointImportDialog(QtWidgets.QDialog):
    def __init__(self, study: str, lab: Lab, existing_points: list[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle(t("Import a sampling point"))
        self.resize(900, 680)
        self._study = study
        self._lab = lab
        self._existing = set(existing_points)
        self._pressure: ParsedMeasures | None = None
        self._temperature: ParsedMeasures | None = None

        # Fichiers
        folder_button = QtWidgets.QPushButton(t("Fill from a point folder…"))
        folder_button.clicked.connect(self._choose_folder)
        self.info_file = _FileRow(t("Info file (*.csv *.txt)"))
        self.pressure_file = _FileRow(t("Measurements (*.csv *.txt)"))
        self.temperature_file = _FileRow(t("Measurements (*.csv *.txt)"))
        self.notice_file = _FileRow(t("Text (*.txt)"))
        self.scheme_file = _FileRow(t("Images (*.png *.jpg *.jpeg)"))
        files = QtWidgets.QFormLayout()
        files.addRow(t("Info file"), self.info_file)
        files.addRow(t("Pressure measurements"), self.pressure_file)
        files.addRow(t("Temperature measurements"), self.temperature_file)
        files.addRow(t("Notice"), self.notice_file)
        files.addRow(t("Installation scheme"), self.scheme_file)
        files_box = QtWidgets.QGroupBox(t("Files"))
        files_layout = QtWidgets.QVBoxLayout(files_box)
        files_layout.addWidget(folder_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        files_layout.addLayout(files)

        # Point
        self.name = QtWidgets.QLineEdit()
        self.pressure_sensor = QtWidgets.QComboBox()
        self.pressure_sensor.addItems([p.name for p in lab.pressure_sensors])
        self.shaft = QtWidgets.QComboBox()
        self.shaft.addItems([s.name for s in lab.shafts])
        self.setup_date = QtWidgets.QDateTimeEdit()
        self.measure_date = QtWidgets.QDateTimeEdit()
        for edit in (self.setup_date, self.measure_date):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
            edit.setDateTime(QtCore.QDateTime.currentDateTime())
        self.river_bed = NumberField(0.0, "m", minimum=0.0)
        self.offset = NumberField(0.0, "m")
        point_box = QtWidgets.QGroupBox(t("Point"))
        point_form = QtWidgets.QFormLayout(point_box)
        point_form.addRow(t("Name"), self.name)
        point_form.addRow(t("Pressure sensor"), self.pressure_sensor)
        point_form.addRow(t("Shaft"), self.shaft)
        point_form.addRow(t("Installation date"), self.setup_date)
        point_form.addRow(t("Measurement date"), self.measure_date)
        point_form.addRow(t("River bed depth"), self.river_bed)
        point_form.addRow(t("Sensor offset (Δh)"), self.offset)
        self.info_warnings = hint("")
        point_form.addRow("", self.info_warnings)

        # Aperçu
        self.pressure_preview = QtWidgets.QLabel()
        self.temperature_preview = QtWidgets.QLabel()
        for label in (self.pressure_preview, self.temperature_preview):
            label.setWordWrap(True)
            label.setTextFormat(QtCore.Qt.TextFormat.RichText)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        preview_box = QtWidgets.QGroupBox(t("Measurement preview"))
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.addWidget(self.pressure_preview)
        preview_layout.addWidget(self.temperature_preview)
        preview_layout.addStretch()

        self.problems = QtWidgets.QLabel()
        self.problems.setProperty("role", "error")
        self.problems.setWordWrap(True)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.import_button = buttons.addButton(t("Import"), QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(files_box)
        left.addWidget(point_box)
        left.addStretch()
        columns = QtWidgets.QHBoxLayout()
        columns.addLayout(left, 3)
        columns.addWidget(preview_box, 2)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(columns, 1)
        layout.addWidget(hint(t("Nothing is written to the project until you click Import.")))
        layout.addWidget(self.problems)
        layout.addWidget(buttons)

        self.info_file.changed.connect(self._read_info)
        self.pressure_file.changed.connect(self._read_pressure)
        self.temperature_file.changed.connect(self._read_temperature)
        for signal in (self.name.textChanged, self.pressure_sensor.currentIndexChanged, self.shaft.currentIndexChanged,
                       self.river_bed.valueChanged, self.offset.valueChanged):
            signal.connect(self._validate)
        self._read_pressure()
        self._read_temperature()

    # Remplissage ------------------------------------------------------------------------
    def _choose_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, t("Point folder"))
        if folder:
            self.fill_from_folder(Path(folder))

    def fill_from_folder(self, folder: Path) -> None:
        files = find_point_files(folder)
        self.info_file.set_path(files.info)
        self.pressure_file.set_path(files.pressure)
        self.temperature_file.set_path(files.temperature)
        self.notice_file.set_path(files.notice)
        self.scheme_file.set_path(files.scheme)
        if not self.name.text():
            self.name.setText(folder.name)

    def _read_info(self) -> None:
        path = self.info_file.path
        self.info_warnings.setText("")
        if path is None:
            return
        try:
            info = read_point_info(path)
        except (UserError, OSError) as exc:
            self.info_warnings.setText(error_text(exc))
            return
        notes = [error_text(warning) for warning in info.warnings]
        if info.name:
            self.name.setText(info.name)
        for combo, value, label in ((self.pressure_sensor, info.pressure_sensor, t("pressure sensor")), (self.shaft, info.shaft, t("shaft"))):
            if value:
                index = combo.findText(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
                else:
                    notes.append(t("The info file names {kind} \"{name}\", which is not in laboratory \"{lab}\".").format(kind=label, name=value, lab=self._lab.name))
            else:
                notes.append(t("The info file does not name a {kind}: choose it below.").format(kind=label))
        for edit, value in ((self.setup_date, info.setup_date), (self.measure_date, info.measure_date)):
            if value:
                edit.setDateTime(to_qdatetime(value))
        if info.river_bed is not None:
            self.river_bed.set_value(info.river_bed)
        else:
            notes.append(t("River bed depth missing from the info file: enter it below."))
        if info.offset is not None:
            self.offset.set_value(info.offset)
        self.info_warnings.setText("\n".join(notes))
        self._validate()

    def _describe(self, parsed: ParsedMeasures | None, error: str, title: str) -> str:
        if error:
            return f"<b>{title}</b><br><span style='color:#c0392b'>{error}</span>"
        if parsed is None:
            return f"<b>{title}</b><br>{t('No file selected.')}"
        frame = parsed.frame
        lines = [
            f"<b>{title}</b>",
            t("{kept} measurements kept out of {total} lines").format(kept=parsed.kept_rows, total=parsed.total_rows),
            t("Period: {start} → {end}").format(start=frame['date'].iloc[0].strftime('%Y-%m-%d %H:%M'), end=frame['date'].iloc[-1].strftime('%Y-%m-%d %H:%M')),
            t("Date format: {fmt}").format(fmt=parsed.date_format.replace("%", "")),
            t("Columns used: {cols}").format(cols=" · ".join(c[:40] for c in parsed.columns_used)),
        ]
        if " + " in parsed.date_format:
            lines.append("<span style='color:#b9770e'>" + t("Several date formats in the same file: check the period above.") + "</span>")
        if parsed.converted_units:
            lines.append(t("Converted: {units}").format(units=", ".join(parsed.converted_units)))
        if parsed.dropped:
            shown = ", ".join(
                (t("line {n}").format(n=line) if line > 0 else "") + f" ({error_text(reason)})" for line, reason in parsed.dropped[:MAX_DROPPED_SHOWN]
            )
            more = len(parsed.dropped) - MAX_DROPPED_SHOWN
            lines.append(t("Lines skipped: {list}").format(list=shown) + (t(" and {n} more").format(n=more) if more > 0 else ""))
        return "<br>".join(lines)

    def _read_measures(self, row: _FileRow, reader, title: str, target: QtWidgets.QLabel) -> ParsedMeasures | None:
        parsed, error = None, ""
        if row.path is not None:
            try:
                parsed = reader(row.path)
            except UserError as exc:
                error = error_text(exc)
            except OSError as exc:
                error = str(exc)
        target.setText(self._describe(parsed, error, title))
        return parsed

    def _read_pressure(self) -> None:
        self._pressure = self._read_measures(self.pressure_file, read_pressure_file, t("Pressure"), self.pressure_preview)
        self._validate()

    def _read_temperature(self) -> None:
        self._temperature = self._read_measures(self.temperature_file, read_temperature_file, t("Temperatures"), self.temperature_preview)
        self._validate()

    # Validation -------------------------------------------------------------------------
    def problems_list(self) -> list[str]:
        problems = []
        name = self.name.text().strip()
        if not name:
            problems.append(t("The point needs a name."))
        elif name in self._existing:
            problems.append(t("A point named \"{name}\" already exists in this study.").format(name=name))
        if self.pressure_sensor.currentIndex() < 0:
            problems.append(t("This laboratory has no pressure sensor."))
        if self.shaft.currentIndex() < 0:
            problems.append(t("This laboratory has no shaft."))
        if self._pressure is None:
            problems.append(t("Valid pressure measurements are required."))
        if self._temperature is None:
            problems.append(t("Valid temperature measurements are required."))
        if not self.river_bed.is_valid() or not self.offset.is_valid():
            problems.append(t("Check the river bed depth and the offset."))
        return problems

    def _validate(self) -> None:
        problems = self.problems_list()
        self.problems.setText(" ".join(problems))
        self.import_button.setEnabled(not problems)

    def point_data(self) -> tuple[SamplingPoint, object, object]:
        notice = read_text(self.notice_file.path) if self.notice_file.path else ""
        scheme = self.scheme_file.path.read_bytes() if self.scheme_file.path else None
        point = SamplingPoint(
            name=self.name.text().strip(),
            study=self._study,
            pressure_sensor=self.pressure_sensor.currentText(),
            shaft=self.shaft.currentText(),
            setup_date=_py(self.setup_date),
            measure_date=_py(self.measure_date),
            river_bed=float(self.river_bed.value()),
            offset=float(self.offset.value()),
            notice=notice,
            scheme=scheme,
        )
        return point, self._temperature.frame, self._pressure.frame


def _py(edit: QtWidgets.QDateTimeEdit) -> datetime:
    return edit.dateTime().toPython()
