"""Onglet « Infos » : installation, capteurs, notice et schéma."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

from ...errors import UserError
from ...i18n import t
from ..errors import error_text
from ..widgets import format_number, heading
from .context import PointContext


class InfoTab(QtWidgets.QWidget):
    def __init__(self, context: PointContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.details = QtWidgets.QLabel()
        self.details.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.details.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        self.notice = QtWidgets.QPlainTextEdit()
        self.notice.setReadOnly(True)
        self.scheme = QtWidgets.QLabel()
        self.scheme.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.scheme.setMinimumSize(240, 200)
        self._pixmap: QtGui.QPixmap | None = None

        left = QtWidgets.QVBoxLayout()
        left.addWidget(heading(t("Installation")))
        left.addWidget(self.details)
        left.addWidget(heading(t("Notice")))
        left.addWidget(self.notice, 1)
        right = QtWidgets.QVBoxLayout()
        right.addWidget(heading(t("Installation scheme")))
        right.addWidget(self.scheme, 1)
        layout = QtWidgets.QHBoxLayout(self)
        layout.addLayout(left, 1)
        layout.addLayout(right, 1)
        context.changed.connect(self.refresh)
        self.refresh()

    def refresh(self) -> None:
        point = self.context.point
        rows = [
            (t("Study"), point.study),
            (t("Laboratory"), self.context.lab.name),
            (t("Installation date"), point.setup_date.strftime("%Y-%m-%d %H:%M") if point.setup_date else "—"),
            (t("Measurement date"), point.measure_date.strftime("%Y-%m-%d %H:%M") if point.measure_date else "—"),
            (t("River bed depth"), f"{format_number(point.river_bed)} m"),
            (t("Sensor offset (Δh)"), f"{format_number(point.offset)} m"),
        ]
        try:
            sensors = self.context.sensors()
            rows += [
                (t("Pressure sensor"), f"{sensors.pressure.name} · dU/dH = {format_number(sensors.pressure.du_dh)} V/m · "
                 f"dU/dT = {format_number(sensors.pressure.du_dt)} V/°C · intercept = {format_number(sensors.pressure.intercept)} V"),
                (t("Shaft"), f"{sensors.shaft.name} · " + t("depths {d} m").format(d=", ".join(format_number(d) for d in sensors.shaft.depths))),
                (t("Thermometers"), f"{sensors.thermometer.name} · σ = {format_number(sensors.thermometer.sigma)} °C" if sensors.thermometer else "—"),
            ]
        except UserError as exc:
            rows.append((t("Sensors"), error_text(exc)))
        self.details.setText("<table cellspacing='4'>" + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>")
        self.notice.setPlainText(point.notice or t("No notice for this point."))
        self._pixmap = None
        if point.scheme:
            pixmap = QtGui.QPixmap()
            if pixmap.loadFromData(point.scheme):
                self._pixmap = pixmap
        if self._pixmap is None:
            self.scheme.setText(t("No installation scheme for this point."))
        self._scale()

    def _scale(self) -> None:
        if self._pixmap is not None:
            self.scheme.setPixmap(self._pixmap.scaled(self.scheme.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                                      QtCore.Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event) -> None:  # API Qt
        super().resizeEvent(event)
        self._scale()

