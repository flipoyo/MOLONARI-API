"""Unité d'affichage des températures (°C, K, °F). Les données restent en °C."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore


class TemperatureUnit(QtCore.QObject):
    changed = QtCore.Signal(str)

    def __init__(self, unit: str = "°C"):
        super().__init__()
        self.unit = unit

    def set(self, unit: str) -> None:
        if unit != self.unit:
            self.unit = unit
            self.changed.emit(unit)

    def convert(self, celsius):
        values = np.asarray(celsius, dtype=float)
        if self.unit == "K":
            return values + 273.15
        if self.unit == "°F":
            return values * 9.0 / 5.0 + 32.0
        return values

    def convert_difference(self, celsius):
        """Écarts (ex. RMSE) : le kelvin vaut le degré Celsius, le °F en vaut 9/5."""
        values = np.asarray(celsius, dtype=float)
        return values * 9.0 / 5.0 if self.unit == "°F" else values


temperature_unit = TemperatureUnit()
