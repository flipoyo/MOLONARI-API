"""Pipeline du brut au résultat : des fichiers d'un point et d'un laboratoire aux entrées d'une colonne.

    laboratoire  io.lab_import.import_lab_directory       capteurs, tiges, incertitudes σ
    point        find_point_files, read_*_file            fichier info, mesures de pression et de température
    nettoyage    processing.cleanup.clean(…, CleanupSettings)
    colonne      engine.column.ColumnInputs.from_cleaned

Ce module enchaîne les lecteurs et le nettoyage qui existent déjà (aucun calcul n'est dupliqué) et décide de ce
qui n'est pas pris en charge. Il le refuse par une erreur structurée (`UserError` avec un code `E_…`, voir
errors.py), pour ne jamais lire de travers : un format inconnu ne doit pas donner un twin faux sans le dire.
Sont refusés :
- les mesures qui ne sont pas des exports CSV du logiciel du logger (tableur, fichier `.hobo`…) ;
- une unité de pression qui n'est pas la tension du capteur (E_MEASURES_UNIT_UNSUPPORTED, par le lecteur) ;
- un fichier dont les dates ne sont pas chronologiques (E_MEASURES_NOT_CHRONOLOGICAL, par le lecteur) ;
- un point sans fichier de pression ou de température, sans fichier info, ou dont le capteur, la tige ou le
  thermomètre n'est pas dans le laboratoire ;
- un fuseau en paramètre qui n'est pas un décalage fixe (« Europe/Paris »), qui contredit celui des fichiers
  (E_PIPELINE_TIMEZONE_CONFLICT) ou que les fichiers ne déclarent pas (E_PIPELINE_TIMEZONE_UNDECLARED) ;
- des fichiers de mesures dont un seul déclare son fuseau (E_PIPELINE_TIMEZONE_AMBIGUOUS), ou dont une date
  contredit le fuseau déclaré (E_MEASURES_TIMEZONE_CONFLICT, par le lecteur).

Fuseau des dates (P2.1f). La convention est celle des données brutes : l'en-tête de la colonne des dates de chaque
export déclare un décalage fixe à UTC (« Date Heure, GMT+01:00 »), sans heure d'été ni nom de zone, et une date peut
porter le sien. Normalisation : les dates du point restent des dates sans fuseau, écrites dans ce repère (celui de
la température) ; `utc_offset` donne le décalage, et UTC = date - `utc_offset`. Si la pression est déclarée dans un
autre décalage que la température, ses dates sont décalées vers celui de la température (exact, avec un
avertissement), pour que l'association par date reste juste. Le paramètre `timezone` ne choisit rien : c'est une
vérification (« +01:00 », « UTC », « GMT+1 ») contre ce que déclarent les fichiers. Rien n'est supposé quand rien
n'est déclaré : les dates restent telles qu'écrites, `utc_offset` est None et un avertissement le dit.

Les incertitudes σ des instruments viennent du laboratoire : `Sigma_Meas_P` du capteur de pression, `Sigma_Meas_T`
du thermomètre de la tige. Elles peuvent être remplacées explicitement.

Rien ici ne charge JAX : les questions BN-03 et BN-04 s'appuient sur ce module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import pandas as pd

from ..domain.models import Lab, PressureSensor, Shaft, Thermometer
from ..engine.column import ColumnInputs
from ..errors import Issue, UserError
from ..io.dates import format_offset, parse_offset_parameter
from ..io.lab_import import ImportIssue, import_lab_directory
from ..io.point_import import (
    ParsedMeasures,
    PointFiles,
    PointInfo,
    find_point_files,
    read_point_info,
    read_pressure_file,
    read_temperature_file,
)
from ..processing.cleanup import CleanupResult, CleanupSettings, clean

# Fichiers de mesures que l'on reconnaît sans pouvoir les lire : exports de tableur ou fichiers propres au logger.
UNSUPPORTED_SUFFIXES = (".xlsx", ".xls", ".ods", ".hobo", ".hproj", ".dtf", ".json")


@dataclass
class RawPoint:
    """Un point tel que ses fichiers le donnent : info et mesures brutes (None si le fichier manque et qu'on l'admet)."""

    folder: Path
    files: PointFiles
    info: PointInfo | None
    temperature: ParsedMeasures | None
    pressure: ParsedMeasures | None
    warnings: list[Issue] = field(default_factory=list)
    utc_offset: timedelta | None = None  # décalage à UTC des dates (UTC = date - utc_offset), None si non déclaré

    @property
    def name(self) -> str:
        return (self.info.name if self.info and self.info.name else None) or self.folder.name


@dataclass(frozen=True)
class Instruments:
    """Instruments d'un point et leurs incertitudes : σ de la pression (m) et de la température (°C)."""

    pressure_sensor: PressureSensor
    shaft: Shaft
    thermometer: Thermometer | None
    sigma_pressure: float
    sigma_temperature: float


@dataclass
class LoadedPoint:
    raw: RawPoint
    lab: Lab
    lab_issues: list[ImportIssue]
    instruments: Instruments

    @property
    def utc_offset(self) -> timedelta | None:
        return self.raw.utc_offset

    @property
    def offset(self) -> float:
        return float(self.raw.info.offset or 0.0) if self.raw.info else 0.0


@dataclass
class PipelineResult:
    point: LoadedPoint
    cleanup: CleanupResult
    inputs: ColumnInputs

    @property
    def cleaned(self) -> pd.DataFrame:
        return self.cleanup.cleaned

    @property
    def utc_offset(self) -> timedelta | None:
        return self.point.raw.utc_offset


def check_timezone(timezone: str | None, declared: timedelta | None) -> None:
    """Vérifie le fuseau demandé contre celui des fichiers (la convention est celle des données brutes)."""
    if timezone is None:
        return
    requested = parse_offset_parameter(timezone)
    if declared is None:
        raise UserError("E_PIPELINE_TIMEZONE_UNDECLARED", requested=format_offset(requested))
    if requested != declared:
        raise UserError("E_PIPELINE_TIMEZONE_CONFLICT", requested=format_offset(requested), declared=format_offset(declared))


def align_time_frames(files: PointFiles, temperature: ParsedMeasures | None, pressure: ParsedMeasures | None) -> tuple[timedelta | None, list[Issue]]:
    """Met les dates de pression dans le repère de la température ; retourne le décalage commun et les avertissements.

    Les dates sont sans fuseau, écrites au décalage déclaré par chaque fichier. Deux déclarations différentes se
    ramènent exactement l'une à l'autre ; une déclaration d'un côté seulement ne le permet pas (refusé)."""
    parsed = [m for m in (temperature, pressure) if m is not None]
    warnings = [w for m in parsed for w in m.warnings]
    if temperature is None or pressure is None:
        return (parsed[0].utc_offset if parsed else None), warnings
    reference, other = temperature.utc_offset, pressure.utc_offset
    if (reference is None) != (other is None):
        t_name, p_name = files.temperature.name, files.pressure.name
        declared_file, other_file = (t_name, p_name) if reference is not None else (p_name, t_name)
        raise UserError(
            "E_PIPELINE_TIMEZONE_AMBIGUOUS",
            declared_file=declared_file, declared=format_offset(reference if reference is not None else other), other_file=other_file,
        )
    if reference is not None and other != reference:
        pressure.frame["date"] = pressure.frame["date"] + (reference - other)
        warnings.append(
            Issue(None, "W_PIPELINE_TIMEZONE_SHIFTED", file=files.pressure.name, declared=format_offset(other), reference_file=files.temperature.name,
                  reference=format_offset(reference), shift=format_offset(reference - other))
        )
    return reference, warnings


# Lecture d'un point -------------------------------------------------------------------------------------------


def _unsupported_files(folder: Path) -> list[str]:
    return sorted(path.name for path in folder.iterdir() if path.suffix.lower() in UNSUPPORTED_SUFFIXES)


def _missing(folder: Path, pressure: bool) -> UserError:
    """Fichier de mesures absent : s'il y a des fichiers d'un format connu mais non pris en charge, c'est le format."""
    unsupported = _unsupported_files(folder)
    if unsupported:
        return UserError("E_PIPELINE_FORMAT_UNSUPPORTED", folder=folder, files=", ".join(unsupported))
    if pressure:
        return UserError("E_PIPELINE_NO_PRESSURE_FILE", folder=folder)
    return UserError("E_PIPELINE_NO_TEMPERATURE_FILE", folder=folder)


def _read(reader, path: Path) -> ParsedMeasures:
    try:
        return reader(path)
    except OSError as exc:
        raise UserError("E_SYSTEM_ERROR", detail=str(exc)) from None


def read_raw_point(folder: Path, require_measures: bool = True) -> RawPoint:
    """Lit les fichiers d'un dossier de point : info (s'il y en a un) et mesures de pression et de température.

    Un fichier de mesures absent est une erreur (E_PIPELINE_NO_TEMPERATURE_FILE, E_PIPELINE_NO_PRESSURE_FILE ou
    E_PIPELINE_FORMAT_UNSUPPORTED). Avec `require_measures=False`, c'est un avertissement tant qu'il reste au moins
    un fichier de mesures à lire. Un fichier illisible (dates, unités…) est toujours une erreur."""
    folder = Path(folder)
    if not folder.is_dir():
        raise UserError("E_FOLDER_NOT_FOUND", path=folder)
    files = find_point_files(folder)
    warnings: list[Issue] = []
    missing = []
    if files.temperature is None:
        missing.append(_missing(folder, pressure=False))
    if files.pressure is None:
        missing.append(_missing(folder, pressure=True))
    if missing and (require_measures or len(missing) == 2):
        raise missing[0]
    warnings += [Issue.from_error(None, error) for error in missing]
    try:
        info = read_point_info(files.info) if files.info else None
    except OSError as exc:
        raise UserError("E_SYSTEM_ERROR", detail=str(exc)) from None
    if info is not None:
        warnings += info.warnings
    temperature = _read(read_temperature_file, files.temperature) if files.temperature else None
    pressure = _read(read_pressure_file, files.pressure) if files.pressure else None
    utc_offset, time_warnings = align_time_frames(files, temperature, pressure)
    return RawPoint(folder, files, info, temperature, pressure, warnings + time_warnings, utc_offset)


# Instruments --------------------------------------------------------------------------------------------------


def _positive(sigma: float, name: str) -> float:
    if not sigma > 0:
        raise UserError("E_PIPELINE_SIGMA_POSITIVE", name=name, sigma=sigma)
    return float(sigma)


def resolve_instruments(
    raw: RawPoint, lab: Lab, sigma_temperature: float | None = None, sigma_pressure: float | None = None
) -> Instruments:
    """Capteur de pression, tige et thermomètre du point, cherchés dans le laboratoire d'après le fichier info.

    σ de la pression : `Sigma_Meas_P` du capteur. σ de la température : `Sigma_Meas_T` du thermomètre de la tige.
    Une valeur donnée en paramètre remplace celle du laboratoire (et dispense de trouver le thermomètre)."""
    info = raw.info
    if info is None:
        raise UserError("E_PIPELINE_NO_INFO", folder=raw.folder)
    if not info.pressure_sensor:
        raise UserError("E_PIPELINE_NO_PRESSURE_SENSOR", point=raw.name)
    if not info.shaft:
        raise UserError("E_PIPELINE_NO_SHAFT", point=raw.name)
    sensor = next((p for p in lab.pressure_sensors if p.name == info.pressure_sensor), None)
    if sensor is None:
        raise UserError("E_POINT_PRESSURE_SENSOR_UNKNOWN", name=info.pressure_sensor, lab=lab.name)
    shaft = next((s for s in lab.shafts if s.name == info.shaft), None)
    if shaft is None:
        raise UserError("E_POINT_SHAFT_UNKNOWN", name=info.shaft, lab=lab.name)
    thermometer = next((t for t in lab.thermometers if t.name == shaft.thermometer), None)
    if thermometer is None and sigma_temperature is None:
        raise UserError("E_PIPELINE_THERMOMETER_UNKNOWN", name=shaft.thermometer, shaft=shaft.name, lab=lab.name)
    sigma_pressure = _positive(sensor.sigma if sigma_pressure is None else sigma_pressure, sensor.name if sigma_pressure is None else "sigma_pressure")
    sigma_temperature = _positive(
        thermometer.sigma if sigma_temperature is None else sigma_temperature, thermometer.name if sigma_temperature is None else "sigma_temperature"
    )
    return Instruments(sensor, shaft, thermometer, sigma_pressure, sigma_temperature)


# Du brut au résultat ------------------------------------------------------------------------------------------


def load_point(
    point_folder: Path,
    lab_folder: Path,
    timezone: str | None = None,
    sigma_temperature: float | None = None,
    sigma_pressure: float | None = None,
) -> LoadedPoint:
    """Étapes 1 à 3 : le laboratoire, les fichiers du point, puis ses instruments. Rien n'est encore nettoyé.

    `timezone` : vérification du fuseau déclaré par les fichiers (voir l'en-tête du module)."""
    if timezone is not None:
        parse_offset_parameter(timezone)  # un nom de zone est refusé avant de lire quoi que ce soit
    imported = import_lab_directory(lab_folder)
    raw = read_raw_point(point_folder)
    check_timezone(timezone, raw.utc_offset)
    instruments = resolve_instruments(raw, imported.lab, sigma_temperature, sigma_pressure)
    return LoadedPoint(raw, imported.lab, imported.issues, instruments)


def clean_point(point: LoadedPoint, settings: CleanupSettings | None = None) -> CleanupResult:
    """Étape 4 : association par date, charge différentielle, fenêtre, valeurs aberrantes, dates écartées."""
    raw = point.raw
    return clean(raw.temperature.frame, raw.pressure.frame, point.instruments.pressure_sensor, settings or CleanupSettings())


def column_inputs(point: LoadedPoint, cleanup: CleanupResult) -> ColumnInputs:
    """Étape 5 : les entrées d'une colonne, avec la profondeur des capteurs de la tige et les σ des instruments."""
    instruments = point.instruments
    return ColumnInputs.from_cleaned(
        cleanup.cleaned, instruments.shaft.depths, point.offset, instruments.sigma_pressure, instruments.sigma_temperature
    )


def run_pipeline(
    point_folder: Path,
    lab_folder: Path,
    settings: CleanupSettings | None = None,
    timezone: str | None = None,
    sigma_temperature: float | None = None,
    sigma_pressure: float | None = None,
) -> PipelineResult:
    """Du brut au résultat : lit le laboratoire et le point, nettoie, et prépare les entrées d'une colonne.

    `settings` : fenêtre de dates, méthodes de valeurs aberrantes et dates écartées (aucun nettoyage par défaut).
    `timezone` (« +01:00 », « UTC ») est vérifié contre le fuseau que déclarent les fichiers, il ne le remplace pas."""
    point = load_point(point_folder, lab_folder, timezone, sigma_temperature, sigma_pressure)
    cleanup = clean_point(point, settings)
    return PipelineResult(point, cleanup, column_inputs(point, cleanup))
