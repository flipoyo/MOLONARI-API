"""Objets du domaine : laboratoire, capteurs, études, points de prélèvement.

Conventions d'unités dans tout le code :
- températures en °C dans les données et le stockage (le moteur convertit en K) ;
- profondeurs en m, positives vers le bas ;
- dates naïves (heure locale du logger), sans fuseau.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum


@dataclass
class Thermometer:
    name: str
    manufacturer: str
    reference: str
    sigma: float  # incertitude de mesure (°C)


@dataclass
class PressureSensor:
    name: str
    datalogger: str
    calibration_date: datetime | None
    intercept: float  # V
    du_dh: float  # V/m
    du_dt: float  # V/°C
    sigma: float  # incertitude de mesure (m)
    thermometer: str

    def differential_head(self, voltage, temperature):
        """Charge différentielle (m) à partir de la tension (V) et de la température du capteur (°C)."""
        return (voltage - temperature * self.du_dt - self.intercept) / self.du_dh


@dataclass
class Shaft:
    name: str
    datalogger: str
    thermometer: str
    depths: tuple[float, float, float, float]  # profondeur des 4 thermomètres (m)


@dataclass
class Lab:
    name: str
    thermometers: list[Thermometer] = field(default_factory=list)
    pressure_sensors: list[PressureSensor] = field(default_factory=list)
    shafts: list[Shaft] = field(default_factory=list)


@dataclass
class Study:
    name: str
    lab: str


@dataclass
class SamplingPoint:
    name: str
    study: str
    pressure_sensor: str
    shaft: str
    setup_date: datetime | None = None
    measure_date: datetime | None = None
    river_bed: float = 0.0  # m
    offset: float = 0.0  # décalage des capteurs par rapport au lit (Delta_h dans les fichiers info), m
    notice: str = ""
    scheme: bytes | None = None  # image PNG/JPEG du schéma d'installation


class PointStage(IntEnum):
    """Avancement d'un point dans le flux de travail. L'ordre compte."""

    RAW = 0
    CLEANED = 1
    DIRECT_MODEL = 2
    INVERSION = 3


# Colonnes normalisées des tableaux de mesures (pandas).
RAW_TEMPERATURE_COLUMNS = ["date", "t1", "t2", "t3", "t4"]
RAW_PRESSURE_COLUMNS = ["date", "voltage", "temp_bed"]
CLEANED_COLUMNS = ["date", "temp_bed", "t1", "t2", "t3", "t4", "pressure"]
