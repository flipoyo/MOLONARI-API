"""Éléments d'interface réutilisables."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtGui, QtWidgets

from ..i18n import t
from .theme import set_invalid

# Index invalide : la racine d'un modèle Qt (parent par défaut de rowCount et columnCount).
NO_PARENT = QtCore.QModelIndex()


def heading(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setProperty("role", "heading")
    return label


def hint(text: str) -> QtWidgets.QLabel:
    label = QtWidgets.QLabel(text)
    label.setProperty("role", "hint")
    label.setWordWrap(True)
    return label


def format_number(value: float) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.4g}"
    return f"{value:.6g}"


class NumberField(QtWidgets.QWidget):
    """Champ numérique avec unité, validation immédiate et message d'erreur sous le champ.

    Accepte la virgule décimale et la notation scientifique (1e6).
    """

    valueChanged = QtCore.Signal()

    def __init__(self, value: float = 0.0, unit: str = "", integer: bool = False, minimum: float | None = None,
                 maximum: float | None = None, parent=None):
        super().__init__(parent)
        self._integer = integer
        self._minimum = minimum
        self._maximum = maximum
        self._external_error = ""
        self.edit = QtWidgets.QLineEdit()
        self.edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self.edit.setMinimumWidth(80)
        self.unit = QtWidgets.QLabel(unit)
        self.unit.setProperty("role", "hint")
        self.error = QtWidgets.QLabel()
        self.error.setProperty("role", "error")
        self.error.setWordWrap(True)
        self.error.hide()
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        if unit:
            row.addWidget(self.unit)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addLayout(row)
        layout.addWidget(self.error)
        self.set_value(value)
        self.edit.textChanged.connect(self._on_text)

    def _parse(self) -> float | None:
        text = self.edit.text().strip().replace(",", ".").replace(" ", "")
        try:
            value = float(text)
        except ValueError:
            return None
        if self._integer and value != int(value):
            return None
        return value

    def _local_error(self) -> str:
        value = self._parse()
        if value is None:
            return t("Enter a whole number.") if self._integer else t("Enter a number.")
        if self._minimum is not None and value < self._minimum:
            return t("Minimum: {v}").format(v=format_number(self._minimum))
        if self._maximum is not None and value > self._maximum:
            return t("Maximum: {v}").format(v=format_number(self._maximum))
        return ""

    def _on_text(self) -> None:
        self._external_error = ""
        self._refresh()
        self.valueChanged.emit()

    def _refresh(self) -> None:
        message = self._local_error() or self._external_error
        set_invalid(self.edit, bool(message))
        self.error.setText(message)
        self.error.setVisible(bool(message))

    def set_error(self, message: str) -> None:
        """Erreur venant d'une validation d'ensemble (ex. « min < max »)."""
        self._external_error = message
        self._refresh()

    def value(self) -> float | None:
        if self._local_error():
            return None
        value = self._parse()
        return int(value) if self._integer else value

    def is_valid(self) -> bool:
        return not self._local_error() and not self._external_error

    def set_value(self, value: float) -> None:
        self.edit.setText(str(int(value)) if self._integer else format_number(float(value)))


class DataFrameModel(QtCore.QAbstractTableModel):
    """Affiche un DataFrame en lecture seule (dates lisibles, nombres arrondis)."""

    def __init__(self, frame: pd.DataFrame | None = None, headers: dict[str, str] | None = None, parent=None):
        super().__init__(parent)
        self._frame = pd.DataFrame()
        self._headers = headers or {}
        self._marked: np.ndarray | None = None
        if frame is not None:
            self.set_frame(frame)

    def set_frame(self, frame: pd.DataFrame, marked: np.ndarray | None = None) -> None:
        self.beginResetModel()
        self._frame = frame.reset_index(drop=True)
        self._marked = marked
        self.endResetModel()

    def rowCount(self, parent=NO_PARENT) -> int:  # API Qt
        return 0 if parent.isValid() else len(self._frame)

    def columnCount(self, parent=NO_PARENT) -> int:  # API Qt
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        value = self._frame.iat[index.row(), index.column()]
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if isinstance(value, pd.Timestamp | np.datetime64):
                return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            if isinstance(value, float | np.floating):
                return format_number(float(value))
            return str(value)
        if role == QtCore.Qt.ItemDataRole.TextAlignmentRole and not isinstance(value, str):
            return int(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        if role == QtCore.Qt.ItemDataRole.ForegroundRole and self._marked is not None and self._marked[index.row()]:
            return QtGui.QColor("#c0392b")
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):  # API Qt
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            name = str(self._frame.columns[section])
            return self._headers.get(name, name)
        return str(section + 1)


def table_view(model: QtCore.QAbstractItemModel) -> QtWidgets.QTableView:
    view = QtWidgets.QTableView()
    view.setModel(model)
    view.setAlternatingRowColors(True)
    view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
    view.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
    view.horizontalHeader().setStretchLastSection(True)
    view.verticalHeader().setDefaultSectionSize(22)
    return view


class TypedConfirmation(QtWidgets.QDialog):
    """Confirmation d'une action destructive : l'utilisateur retape le nom de l'élément."""

    def __init__(self, title: str, message: str, name: str, action: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self._name = name
        layout = QtWidgets.QVBoxLayout(self)
        text = QtWidgets.QLabel(message)
        text.setWordWrap(True)
        layout.addWidget(text)
        layout.addWidget(QtWidgets.QLabel(t("Type \"{name}\" to confirm:").format(name=name)))
        self.edit = QtWidgets.QLineEdit()
        layout.addWidget(self.edit)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.confirm = buttons.addButton(action, QtWidgets.QDialogButtonBox.ButtonRole.DestructiveRole)
        self.confirm.setEnabled(False)
        self.confirm.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.edit.textChanged.connect(lambda text: self.confirm.setEnabled(text.strip() == name))
        self.setMinimumWidth(420)


def confirm_destructive(parent, title: str, message: str, name: str, action: str) -> bool:
    return TypedConfirmation(title, message, name, action, parent).exec() == QtWidgets.QDialog.DialogCode.Accepted


def message(parent, title: str, text: str, details: str = "", icon=QtWidgets.QMessageBox.Icon.Information) -> None:
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(icon)
    box.setWindowTitle(title)
    box.setText(text)
    if details:
        box.setDetailedText(details)
    box.exec()


def ask(parent, title: str, text: str, action: str) -> bool:
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Icon.Question)
    box.setWindowTitle(title)
    box.setText(text)
    ok = box.addButton(action, QtWidgets.QMessageBox.ButtonRole.AcceptRole)
    box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
    box.exec()
    return box.clickedButton() is ok


def to_qdatetime(value) -> QtCore.QDateTime:
    ts = pd.Timestamp(value)
    return QtCore.QDateTime(QtCore.QDate(ts.year, ts.month, ts.day), QtCore.QTime(ts.hour, ts.minute, ts.second))


def from_qdatetime(value: QtCore.QDateTime):
    return value.toPython()
