"""Import d'un laboratoire depuis un dossier de fichiers CSV « clé,valeur ».

Structure attendue (identique à Molonaviz, utils/general.extractDetectorsDF) :
    <dossier>/temperature_sensors/*.csv   4 lignes : Cons_Name, Cons_Ref, T_Sensor_Name, Sigma_Meas_T
    <dossier>/pressure_sensors/*.csv      8 lignes : nom, Datalogger, Calibration_Date, Intercept,
                                          dU/dH, dU/dT, Sigma_Meas_P, Thermometer name
    <dossier>/shafts/*.csv                4 lignes : Shaft_Name, Datalogger, T_Sensor_Name, Sensors_Depth
Les valeurs sont lues par position, comme Molonaviz ; la virgule décimale est acceptée.
Un fichier invalide est écarté et signalé, les autres sont importés.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field
from pathlib import Path

from ..domain.models import Lab, PressureSensor, Shaft, Thermometer
from ..errors import UserError
from ..i18n import t
from .dates import parse_date


@dataclass
class ImportIssue:
    file: str
    message: str
    line: int | None = None

    def __str__(self) -> str:
        where = f"{self.file}, " + t("line {n}").format(n=self.line) if self.line else self.file
        return f"{where} : {self.message}"


@dataclass
class LabImport:
    lab: Lab
    issues: list[ImportIssue] = field(default_factory=list)


def read_text(path: Path) -> str:
    """Lit un fichier texte de terrain (UTF-8 avec ou sans BOM, sinon Windows-1252)."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")


def read_key_values(path: Path) -> list[tuple[str, str]]:
    rows = []
    for row in csv.reader(read_text(path).splitlines()):
        if not row or not any(cell.strip() for cell in row):
            continue
        key = row[0].strip()
        value = ",".join(row[1:]).strip() if len(row) > 2 else (row[1].strip() if len(row) > 1 else "")
        rows.append((key, value))
    return rows


class _InvalidFile(Exception):
    def __init__(self, message: str, line: int | None = None):
        super().__init__(message)
        self.line = line


def _number(rows, index: int) -> float:
    key, value = rows[index]
    try:
        return float(value.replace(",", "."))
    except ValueError:
        raise _InvalidFile(t("\"{key}\" is not a number: \"{value}\".").format(key=key, value=value), index + 1) from None


def _text(rows, index: int) -> str:
    key, value = rows[index]
    if not value:
        raise _InvalidFile(t("\"{key}\" is empty.").format(key=key), index + 1)
    return value


def _expect_rows(rows, count: int) -> None:
    if len(rows) != count:
        raise _InvalidFile(t("Expected {expected} lines, found {found}.").format(expected=count, found=len(rows)))


def parse_thermometer(path: Path) -> Thermometer:
    rows = read_key_values(path)
    _expect_rows(rows, 4)
    return Thermometer(name=_text(rows, 2), manufacturer=_text(rows, 0), reference=_text(rows, 1), sigma=_number(rows, 3))


def parse_pressure_sensor(path: Path) -> PressureSensor:
    rows = read_key_values(path)
    _expect_rows(rows, 8)
    try:
        calibration = parse_date(rows[2][1])
    except UserError as exc:
        raise _InvalidFile(str(exc), 3) from None
    return PressureSensor(
        name=_text(rows, 0),
        datalogger=rows[1][1],
        calibration_date=calibration,
        intercept=_number(rows, 3),
        du_dh=_number(rows, 4),
        du_dt=_number(rows, 5),
        sigma=_number(rows, 6),
        thermometer=_text(rows, 7),
    )


def parse_shaft(path: Path) -> Shaft:
    rows = read_key_values(path)
    _expect_rows(rows, 4)
    text = _text(rows, 3).replace(";", ",")
    try:
        depths = tuple(float(d) for d in ast.literal_eval(text))
    except (ValueError, SyntaxError, TypeError):
        raise _InvalidFile(t("Sensor depths must be a list of 4 numbers, e.g. [0.1, 0.2, 0.3, 0.4]."), 4) from None
    if len(depths) != 4 or any(b <= a for a, b in zip(depths, depths[1:])) or depths[0] <= 0:
        raise _InvalidFile(t("Sensor depths must be 4 strictly increasing positive values (m)."), 4)
    return Shaft(name=_text(rows, 0), datalogger=rows[1][1], thermometer=_text(rows, 2), depths=depths)  # type: ignore[arg-type]


SUBFOLDERS = {
    "temperature_sensors": ("thermometers", parse_thermometer),
    "pressure_sensors": ("pressure_sensors", parse_pressure_sensor),
    "shafts": ("shafts", parse_shaft),
}


def import_lab_directory(directory: Path, name: str | None = None) -> LabImport:
    directory = Path(directory)
    if not directory.is_dir():
        raise UserError(t("Folder not found: {path}").format(path=directory))
    if not any((directory / sub).is_dir() for sub in SUBFOLDERS):
        raise UserError(
            t("This folder does not look like a laboratory: it should contain temperature_sensors, pressure_sensors and/or shafts.")
        )
    lab = Lab(name=(name or directory.name).strip())
    result = LabImport(lab)
    for sub, (attribute, parser) in SUBFOLDERS.items():
        names: set[str] = set()
        for path in sorted((directory / sub).glob("*.csv")):
            relative = f"{sub}/{path.name}"
            try:
                item = parser(path)
            except _InvalidFile as exc:
                result.issues.append(ImportIssue(relative, str(exc), exc.line))
                continue
            except OSError as exc:
                result.issues.append(ImportIssue(relative, str(exc)))
                continue
            if item.name in names:
                result.issues.append(ImportIssue(relative, t("Duplicate name \"{name}\": file ignored.").format(name=item.name)))
                continue
            names.add(item.name)
            getattr(lab, attribute).append(item)
    known = {th.name for th in lab.thermometers}
    for sensor in [*lab.pressure_sensors, *lab.shafts]:
        if known and sensor.thermometer not in known:
            result.issues.append(
                ImportIssue(sensor.name, t("Refers to thermometer \"{name}\", which is not in this laboratory.").format(name=sensor.thermometer))
            )
    return result
