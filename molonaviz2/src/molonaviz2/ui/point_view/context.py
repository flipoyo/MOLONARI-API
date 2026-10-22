"""Contexte d'un point ouvert, partagé par ses onglets."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtCore

from molonaviz2.facade import (
    ColumnInputs,
    Lab,
    PointStage,
    PressureSensor,
    Project,
    SamplingPoint,
    Shaft,
    Thermometer,
    UserError,
)

from ...i18n import N_

STAGE_LABELS = {
    PointStage.RAW: N_("Raw measurements"),
    PointStage.CLEANED: N_("Cleaned"),
    PointStage.DIRECT_MODEL: N_("Direct model"),
    PointStage.INVERSION: N_("Inversion"),
}


@dataclass
class Sensors:
    pressure: PressureSensor
    shaft: Shaft
    thermometer: Thermometer | None


class PointContext(QtCore.QObject):
    """Accès au point et à ses capteurs ; `changed` est émis après toute modification du point."""

    changed = QtCore.Signal()

    def __init__(self, project: Project, study: str, name: str, parent=None):
        super().__init__(parent)
        self.project = project
        self.study = study
        self.name = name
        self.reload()

    def reload(self) -> None:
        self.point: SamplingPoint = self.project.get_point(self.study, self.name)
        self.lab: Lab = self.project.study_lab(self.study)
        self.stage: PointStage = self.project.stage(self.study, self.name)

    def notify(self) -> None:
        self.reload()
        self.changed.emit()

    def sensors(self) -> Sensors:
        pressure = next((p for p in self.lab.pressure_sensors if p.name == self.point.pressure_sensor), None)
        shaft = next((s for s in self.lab.shafts if s.name == self.point.shaft), None)
        if pressure is None or shaft is None:
            raise UserError("E_POINT_SENSORS_MISSING", lab=self.lab.name)
        thermometer = next((th for th in self.lab.thermometers if th.name == shaft.thermometer), None)
        return Sensors(pressure, shaft, thermometer)

    @property
    def column_depth(self) -> float:
        """Profondeur de la colonne modélisée : thermomètre le plus profond + offset (m)."""
        return float(self.sensors().shaft.depths[-1] + self.point.offset)

    def default_sigma_t(self) -> float:
        thermometer = self.sensors().thermometer
        return float(thermometer.sigma) if thermometer and thermometer.sigma > 0 else 0.1

    def column_inputs(self) -> ColumnInputs:
        """Entrées du moteur, construites depuis les mesures nettoyées (lecture dans le thread de l'interface)."""
        sensors = self.sensors()
        cleaned = self.project.cleaned(self.study, self.name)
        if cleaned.empty:
            raise UserError("E_POINT_NOT_CLEANED")
        return ColumnInputs.from_cleaned(cleaned, sensors.shaft.depths, self.point.offset, sensors.pressure.sigma, self.default_sigma_t())

    @property
    def title(self) -> str:
        return f"{self.name} · {self.study}"
