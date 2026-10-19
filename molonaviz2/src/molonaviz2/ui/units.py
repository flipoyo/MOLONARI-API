"""Unité d'affichage des températures (°C, K, °F) et niveau de l'intervalle de mesure. Les données restent en °C."""

from __future__ import annotations

import numpy as np
from PySide6 import QtCore, QtWidgets
from scipy.stats import norm


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


# Intervalle de mesure ---------------------------------------------------------------------
MEASUREMENT_LEVELS = (0.6827, 0.90, 0.95)  # 1σ, 90 %, 95 %


def z_score(level: float) -> float:
    """Demi-largeur, en σ, de l'intervalle centré qui contient `level` d'une loi normale (90 % → 1,645)."""
    return float(norm.ppf(0.5 + level / 2))


def level_label(level: float) -> str:
    return "1σ (68 %)" if abs(level - 0.6827) < 1e-4 else f"{level * 100:.0f} %"


def measurement_interval(observed_celsius, sigma_celsius: float, level: float):
    """Bornes (°C) de l'intervalle mesure ± z·σT, z venant du niveau."""
    observed = np.asarray(observed_celsius, dtype=float)
    half = z_score(level) * float(sigma_celsius)
    return observed - half, observed + half


class MeasurementLevel(QtCore.QObject):
    """Niveau de l'intervalle mesure ± z·σT, partagé par tous les graphes qui l'affichent."""

    changed = QtCore.Signal(float)

    def __init__(self, level: float = 0.90):
        super().__init__()
        self.level = level

    @property
    def z(self) -> float:
        return z_score(self.level)

    def set(self, level: float) -> None:
        if level != self.level:
            self.level = level
            self.changed.emit(level)

    def combo(self, parent=None) -> QtWidgets.QComboBox:
        """Liste déroulante liée au niveau, dans les deux sens."""
        return _LevelCombo(self, parent)


class _LevelCombo(QtWidgets.QComboBox):
    def __init__(self, shared: MeasurementLevel, parent=None):
        super().__init__(parent)
        for level in MEASUREMENT_LEVELS:
            self.addItem(level_label(level), level)
        self.setCurrentIndex(MEASUREMENT_LEVELS.index(shared.level))
        self.currentIndexChanged.connect(lambda i: shared.set(self.itemData(i)))
        shared.changed.connect(self._follow)  # méthode d'un QObject : déconnectée à la destruction du widget

    def _follow(self, level: float) -> None:
        self.setCurrentIndex(MEASUREMENT_LEVELS.index(level))


measurement_level = MeasurementLevel()
