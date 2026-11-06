"""Tableau d'édition des couches (modèle direct : valeurs ; inversion : bornes a priori).

Chaque cellule est validée à la saisie : une valeur invalide est surlignée et son
infobulle explique pourquoi. Le bas de la dernière couche est fixé à la profondeur
de la colonne (non modifiable).
"""

from __future__ import annotations

from molonari_twin.facade import (
    DEFAULT_PRIORS,
    INVERTED_PARAMETERS,
    PARAMETER_LABELS,
    PARAMETER_UNITS,
    PHYSICAL_BOUNDS,
    LayerSpec,
    PriorRange,
)
from PySide6 import QtCore, QtGui, QtWidgets

from ...i18n import t
from ...notice import BY_KEY
from ..widgets import format_number

INVALID_BACKGROUND = QtGui.QColor(192, 57, 43, 60)


def _parse(text: str) -> float | None:
    try:
        return float(text.strip().replace(",", ".").replace(" ", ""))
    except ValueError:
        return None


class LayersTable(QtWidgets.QTableWidget):
    edited = QtCore.Signal()

    def __init__(self, mode: str, column_depth: float, parent=None):
        super().__init__(parent)
        assert mode in ("values", "priors")
        self.mode = mode
        self.column_depth = column_depth
        self._errors: dict[tuple[int, int], str] = {}
        headers = [t("Name"), t("Bottom (m)")]
        for name in INVERTED_PARAMETERS:
            label = f"{PARAMETER_LABELS[name]}\n{t(PARAMETER_UNITS[name])}"
            headers += [label] if mode == "values" else [f"{PARAMETER_LABELS[name]} min\n{t(PARAMETER_UNITS[name])}", "max"]
        self.setColumnCount(len(headers))
        self.setHorizontalHeaderLabels(headers)
        for column in range(len(headers)):
            self.horizontalHeaderItem(column).setToolTip(BY_KEY[self._column_key(column)].text("summary") + "\n" + t("Shift+F1: parameter guide"))
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.itemChanged.connect(self._on_change)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)

    def _column_key(self, column: int) -> str:
        """Fiche de la notice pour une colonne du tableau."""
        if column == 0:
            return "layers.name"
        if column == 1:
            return "layers.z_low"
        index = column - 2 if self.mode == "values" else (column - 2) // 2
        return f"layers.{INVERTED_PARAMETERS[index]}"

    def notice_key(self) -> str:
        """Fiche de la colonne courante (Maj+F1)."""
        return self._column_key(max(self.currentColumn(), 0))

    def _fit_height(self) -> None:
        """Hauteur ajustée au nombre de couches (jusqu'à 6 lignes visibles), sans espace vide."""
        rows = min(max(self.rowCount(), 1), 6)
        height = self.horizontalHeader().sizeHint().height() + rows * self.verticalHeader().defaultSectionSize() + 2 * self.frameWidth()
        if self.horizontalScrollBar().isVisible() or self.horizontalHeader().length() > self.viewport().width():
            height += self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(height + 2)

    # Contenu --------------------------------------------------------------------------
    def set_layers(self, layers: list[LayerSpec], priors: list[dict[str, PriorRange]] | None = None) -> None:
        self.blockSignals(True)
        self.setRowCount(0)
        for i, layer in enumerate(layers):
            self._append_row(layer, priors[i] if priors else None)
        self.blockSignals(False)
        self._fix_last_row()
        self.validate()
        self._fit_height()

    def _append_row(self, layer: LayerSpec, priors: dict[str, PriorRange] | None) -> None:
        row = self.rowCount()
        self.insertRow(row)
        values = [layer.name, format_number(layer.z_low)]
        for name in INVERTED_PARAMETERS:
            if self.mode == "values":
                values.append(format_number(getattr(layer, name)))
            else:
                prior = (priors or {}).get(name, DEFAULT_PRIORS[name])
                values += [format_number(prior.low), format_number(prior.high)]
        for column, value in enumerate(values):
            self.setItem(row, column, QtWidgets.QTableWidgetItem(value))

    def _fix_last_row(self) -> None:
        self.blockSignals(True)
        for row in range(self.rowCount()):
            item = self.item(row, 1)
            last = row == self.rowCount() - 1
            flags = item.flags()
            if last:
                item.setText(format_number(self.column_depth))
                item.setFlags(flags & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(t("The last layer ends at the column depth (deepest thermometer + offset)."))
                item.setForeground(self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
            else:
                item.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEditable)
                item.setForeground(self.palette().color(QtGui.QPalette.ColorRole.Text))
        self.blockSignals(False)

    def add_layer(self) -> None:
        layers = self.layers(strict=False)
        previous = layers[-2].z_low if len(layers) > 1 else 0.0
        new_bottom = round((previous + self.column_depth) / 2, 4)
        template = layers[-1] if layers else LayerSpec("", self.column_depth)
        new = LayerSpec(t("Layer {n}").format(n=len(layers) + 1), new_bottom, template.neg_log10_k, template.porosity,
                        template.lambda_s, template.rhos_cs)
        priors = self.priors(strict=False) if self.mode == "priors" else None
        layers.insert(len(layers) - 1 if layers else 0, new)
        if priors is not None:
            priors.insert(len(priors) - 1 if priors else 0, {k: PriorRange(v.low, v.high) for k, v in (priors[-1] if priors else DEFAULT_PRIORS).items()})
        self.set_layers(layers, priors)
        self.edited.emit()

    def remove_selected(self) -> None:
        rows = sorted({index.row() for index in self.selectedIndexes()}, reverse=True)
        if not rows or self.rowCount() - len(rows) < 1:
            return
        layers = self.layers(strict=False)
        priors = self.priors(strict=False) if self.mode == "priors" else None
        for row in rows:
            del layers[row]
            if priors is not None:
                del priors[row]
        self.set_layers(layers, priors)
        self.edited.emit()

    def _cell(self, row: int, column: int) -> float | None:
        item = self.item(row, column)
        return _parse(item.text()) if item else None

    def layers(self, strict: bool = True) -> list[LayerSpec]:
        """Couches saisies. strict=False remplace les valeurs illisibles par les valeurs par défaut."""
        result = []
        defaults = LayerSpec("", 0.0)
        for row in range(self.rowCount()):
            name = self.item(row, 0).text().strip() or t("Layer {n}").format(n=row + 1)
            z_low = self.column_depth if row == self.rowCount() - 1 else self._cell(row, 1)
            values = {}
            if self.mode == "values":
                for k, param in enumerate(INVERTED_PARAMETERS):
                    values[param] = self._cell(row, 2 + k)
            if strict and (z_low is None or None in values.values()):
                raise ValueError("invalid layer table")
            values = {k: (v if v is not None else getattr(defaults, k)) for k, v in values.items()}
            result.append(LayerSpec(name, z_low if z_low is not None else 0.0, **values))
        return result

    def priors(self, strict: bool = True) -> list[dict[str, PriorRange]]:
        result = []
        for row in range(self.rowCount()):
            priors = {}
            for k, param in enumerate(INVERTED_PARAMETERS):
                low, high = self._cell(row, 2 + 2 * k), self._cell(row, 3 + 2 * k)
                if strict and (low is None or high is None):
                    raise ValueError("invalid prior table")
                default = DEFAULT_PRIORS[param]
                priors[param] = PriorRange(low if low is not None else default.low, high if high is not None else default.high)
            result.append(priors)
        return result

    # Validation -----------------------------------------------------------------------
    def _on_change(self, _item) -> None:
        self.validate()
        self.edited.emit()

    def validate(self) -> list[str]:
        errors: dict[tuple[int, int], str] = {}
        previous = 0.0
        for row in range(self.rowCount()):
            if not self.item(row, 0).text().strip():
                errors[(row, 0)] = t("The layer needs a name.")
            z_low = self.column_depth if row == self.rowCount() - 1 else self._cell(row, 1)
            if z_low is None:
                errors[(row, 1)] = t("Enter a number.")
            elif z_low <= previous or z_low > self.column_depth:
                errors[(row, 1)] = t("Bottoms must increase from top to bottom and stay within the column ({depth} m).").format(
                    depth=format_number(self.column_depth)
                )
            else:
                previous = z_low
            for k, param in enumerate(INVERTED_PARAMETERS):
                low_bound, high_bound = PHYSICAL_BOUNDS[param]
                columns = [2 + k] if self.mode == "values" else [2 + 2 * k, 3 + 2 * k]
                values = [self._cell(row, c) for c in columns]
                for column, value in zip(columns, values):
                    if value is None:
                        errors[(row, column)] = t("Enter a number.")
                    elif not low_bound <= value <= high_bound:
                        errors[(row, column)] = t("Value outside the physical range [{low:g}, {high:g}].").format(low=low_bound, high=high_bound)
                if self.mode == "priors" and None not in values and values[0] >= values[1]:
                    errors[(row, columns[1])] = t("Minimum must be lower than maximum.")
        self._errors = errors
        self.blockSignals(True)
        for row in range(self.rowCount()):
            for column in range(self.columnCount()):
                item = self.item(row, column)
                if item is None:
                    continue
                message = errors.get((row, column), "")
                item.setBackground(INVALID_BACKGROUND if message else QtGui.QBrush())
                if not (row == self.rowCount() - 1 and column == 1):
                    item.setToolTip(message)
        self.blockSignals(False)
        return sorted(set(errors.values()))

    def resizeEvent(self, event) -> None:  # API Qt
        super().resizeEvent(event)
        self._fit_height()

    def is_valid(self) -> bool:
        return not self._errors
