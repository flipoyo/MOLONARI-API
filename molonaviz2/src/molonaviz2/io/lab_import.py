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
from ..errors import Issue, UserError
from ..i18n import N_, render
from .dates import parse_date


@dataclass
class ImportIssue:
    """Fichier écarté à l'import : `message` est un `Issue` (ou le texte d'une erreur système)."""

    file: str
    message: str
    line: int | None = None

    def __str__(self) -> str:
        return self.localized(lambda text: text)

    def localized(self, tr) -> str:
        """Texte traduit par `tr` (voir `i18n.render`)."""
        where = f"{self.file}, " + tr(N_("line {n}")).format(n=self.line) if self.line else self.file
        return f"{where} : {render(self.message, tr)}"


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
    def __init__(self, issue: Issue, line: int | None = None):
        super().__init__(issue)
        self.issue = issue
        self.line = line


def _number(rows, index: int) -> float:
    key, value = rows[index]
    try:
        return float(value.replace(",", "."))
    except ValueError:
        raise _InvalidFile(Issue(None, "E_LAB_FILE_NOT_A_NUMBER", key=key, value=value), index + 1) from None


def _text(rows, index: int) -> str:
    key, value = rows[index]
    if not value:
        raise _InvalidFile(Issue(None, "E_LAB_FILE_EMPTY_VALUE", key=key), index + 1)
    return value


def _expect_rows(rows, count: int) -> None:
    if len(rows) != count:
        raise _InvalidFile(Issue(None, "E_LAB_FILE_LINE_COUNT", expected=count, found=len(rows)))


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
        raise _InvalidFile(Issue.from_error(None, exc), 3) from None
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
        raise _InvalidFile(Issue(None, "E_SHAFT_DEPTHS_FORMAT"), 4) from None
    if len(depths) != 4 or any(b <= a for a, b in zip(depths, depths[1:])) or depths[0] <= 0:
        raise _InvalidFile(Issue(None, "E_SHAFT_DEPTHS_ORDER"), 4)
    return Shaft(name=_text(rows, 0), datalogger=rows[1][1], thermometer=_text(rows, 2), depths=depths)  # type: ignore[arg-type]


SUBFOLDERS = {
    "temperature_sensors": ("thermometers", parse_thermometer),
    "pressure_sensors": ("pressure_sensors", parse_pressure_sensor),
    "shafts": ("shafts", parse_shaft),
}


def import_lab_directory(directory: Path, name: str | None = None) -> LabImport:
    directory = Path(directory)
    if not directory.is_dir():
        raise UserError("E_FOLDER_NOT_FOUND", path=directory)
    if not any((directory / sub).is_dir() for sub in SUBFOLDERS):
        raise UserError(
            "E_LAB_FOLDER_INVALID"
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
                result.issues.append(ImportIssue(relative, exc.issue, exc.line))
                continue
            except OSError as exc:
                result.issues.append(ImportIssue(relative, str(exc)))
                continue
            if item.name in names:
                result.issues.append(ImportIssue(relative, Issue(None, "W_LAB_DUPLICATE_NAME", name=item.name)))
                continue
            names.add(item.name)
            getattr(lab, attribute).append(item)
    known = {th.name for th in lab.thermometers}
    for sensor in [*lab.pressure_sensors, *lab.shafts]:
        if known and sensor.thermometer not in known:
            result.issues.append(
                ImportIssue(sensor.name, Issue(None, "W_LAB_UNKNOWN_THERMOMETER", name=sensor.thermometer))
            )
    return result
