"""Import d'un point de prélèvement : fichier info, mesures de pression et de température.

Les fichiers de mesures viennent de loggers HOBO (export HOBOware) ou ont déjà été
mis au format propre de Molonaviz. Le lecteur gère les deux :
- ligne de titre éventuelle (« Titre de tracé », « Plot Title ») ;
- colonne d'index « # » éventuelle ;
- colonnes repérées d'après l'en-tête (tension, températures) et unités converties
  en °C (°F et K acceptés) ; à défaut d'en-tête, lecture par position ;
- lignes d'événements du logger (valeurs vides) écartées et comptées ;
- séparateur « , », « ; » ou tabulation ; virgule décimale si le séparateur n'est pas « , ».

Fuseau des dates (P2.1f). La convention est celle des données brutes : l'en-tête de la colonne des dates déclare un
décalage fixe à UTC (« Date Heure, GMT+01:00 »), et une date peut porter le sien (« 2016-06-27T12:00:00+01:00 »).
Les dates restent écrites telles quelles, sans fuseau (`ParsedMeasures.utc_offset` donne le décalage). Rien n'est
supposé : sans déclaration, le décalage est None et un avertissement le dit.

CONVENTION PROVISOIRE (TODO(fuseau), à régler les années suivantes) : un en-tête qui déclare « GMT+01:00 » (tous les
exports actuels, en été comme en hiver) est lu comme UTC+2 toute l'année, `PROVISIONAL_UTC_OFFSET`, avec
l'avertissement W_MEASURES_TIMEZONE_OVERRIDDEN. L'heure réelle des loggers reste à confirmer auprès des opérateurs
terrain, puis la gestion CET/CEST à traiter proprement. Un en-tête « GMT+02:00 » est gardé tel quel, et tout autre
décalage d'en-tête aussi. Des dates qui portent elles-mêmes leur décalage (exact) ne sont jamais remplacées. Une date dont le décalage contredit
celui du fichier, ou des décalages mêlés sans déclaration, sont refusés (E_MEASURES_TIMEZONE_CONFLICT,
E_MEASURES_TIMEZONE_MIXED, E_MEASURES_TIMEZONE_PARTIAL).

Un fichier dont l'en-tête annonce une unité de pression (kPa, psi, bar…) à la place de la tension du capteur
est refusé (E_MEASURES_UNIT_UNSUPPORTED) : le lire par position donnerait des charges fausses sans le dire.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.models import RAW_PRESSURE_COLUMNS, RAW_TEMPERATURE_COLUMNS
from ..errors import Issue, UserError
from .dates import (
    HEADER_OFFSET_OVERRIDDEN,
    PROVISIONAL_UTC_OFFSET,
    format_offset,
    parse_date,
    parse_dates,
    parse_header_offset,
    split_cell_offset,
)
from .lab_import import read_key_values, read_text

# Clés des fichiers info (insensibles à la casse), par champ.
INFO_KEYS = {
    "name": ("point_name", "name"),
    "pressure_sensor": ("p_sensor_name", "psensor", "pressure_sensor"),
    "shaft": ("shaft_name", "shaft"),
    "setup_date": ("implantation_date", "implatation date", "setup_date"),
    "measure_date": ("meas_date", "measure_date"),
    "river_bed": ("river_bed", "river_deb", "riverbed"),
    "offset": ("delta_h", "offset"),
}


@dataclass
class PointInfo:
    name: str | None = None
    pressure_sensor: str | None = None
    shaft: str | None = None
    setup_date: datetime | None = None
    measure_date: datetime | None = None
    river_bed: float | None = None
    offset: float | None = None
    warnings: list[Issue] = field(default_factory=list)


def read_point_info(path: Path) -> PointInfo:
    info = PointInfo()
    rows = read_key_values(path)
    by_key = {key.strip().lower(): value for key, value in rows}
    for attribute, aliases in INFO_KEYS.items():
        value = next((by_key[a] for a in aliases if a in by_key), None)
        if value is None or not value.strip():
            continue
        try:
            if attribute in ("setup_date", "measure_date"):
                setattr(info, attribute, parse_date(value))
            elif attribute in ("river_bed", "offset"):
                setattr(info, attribute, float(value.replace(",", ".")))
            else:
                setattr(info, attribute, value.strip())
        except (UserError, ValueError):
            info.warnings.append(Issue(None, "W_POINT_INFO_UNREADABLE", value=value, field=attribute))
    # Fichiers sans clés reconnues : les 3 premières lignes sont nom, capteur de pression, tige (Molonaviz).
    if info.name is None and len(rows) >= 3:
        info.name, info.pressure_sensor, info.shaft = (rows[i][1] or None for i in range(3))
    return info


@dataclass
class ParsedMeasures:
    frame: pd.DataFrame
    date_format: str
    total_rows: int
    dropped: list[tuple[int, Issue]]  # (numéro de ligne dans le fichier, raison)
    columns_used: list[str]
    converted_units: list[str] = field(default_factory=list)  # ex. « °F → °C »
    utc_offset: timedelta | None = None  # décalage à UTC déclaré par le fichier (dates écrites à ce décalage), None si aucun
    warnings: list[Issue] = field(default_factory=list)

    @property
    def kept_rows(self) -> int:
        return len(self.frame)


_VOLTAGE = re.compile(r"tension|volt|\bv\b|, v", re.IGNORECASE)
_TEMPERATURE = re.compile(r"temp", re.IGNORECASE)
_PRESSURE_UNIT = re.compile(r"\b(?:[kmh]?pa|mbar|bar|psi|mmhg|inhg|cmh2o|mh2o)\b", re.IGNORECASE)


def _unit(header: str) -> str:
    if "°F" in header or "°f" in header:
        return "F"
    if re.search(r"\bK\b", header) and "°C" not in header:
        return "K"
    return "C"


def _to_celsius(values: np.ndarray, unit: str) -> np.ndarray:
    if unit == "F":
        return (values - 32.0) * 5.0 / 9.0
    if unit == "K":
        return values - 273.15
    return values


def _read_rows(path: Path) -> tuple[list[list[str]], str]:
    text = read_text(path)
    lines = text.splitlines()
    sample = "\n".join(lines[:10])
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        delimiter = ","
    return list(csv.reader(lines, delimiter=delimiter)), delimiter


def _looks_like_date(cell: str) -> bool:
    return bool(re.match(r"^\s*\d{1,4}[/:\-]\d{1,2}[/:\-]\d{1,4}", cell or ""))


def _read_measures(path: Path, kind: str) -> ParsedMeasures:
    path = Path(path)
    rows, delimiter = _read_rows(path)
    if not rows:
        raise UserError("E_MEASURES_FILE_EMPTY", file=path.name)

    # Repérer l'en-tête : première ligne (parmi les 5 premières) suivie d'une ligne de données.
    header_index = None
    for i, row in enumerate(rows[:5]):
        cells = [c.strip() for c in row]
        first_data = next((j for j, c in enumerate(cells) if _looks_like_date(c)), None)
        if first_data is not None:
            header_index = i - 1 if i > 0 else None
            break
    else:
        raise UserError("E_MEASURES_NO_DATE", file=path.name)
    header = [c.strip() for c in rows[header_index]] if header_index is not None else []
    data_start = (header_index + 1) if header_index is not None else 0

    offset = 1 if header and header[0] in ("#", "N°", "No") else 0
    if not header and rows[data_start] and not _looks_like_date(rows[data_start][0]):
        offset = 1  # colonne d'index sans en-tête
    date_col = offset
    value_headers = header[offset + 1 :] if header else []

    if kind == "pressure":
        wanted = 2
        voltage = next((i for i, h in enumerate(value_headers) if _VOLTAGE.search(h)), None)
        temps = [i for i, h in enumerate(value_headers) if _TEMPERATURE.search(h)]
        if voltage is None:
            unit = next((h for h in value_headers if _PRESSURE_UNIT.search(h)), None)
            if unit is not None:
                raise UserError("E_MEASURES_UNIT_UNSUPPORTED", file=path.name, column=unit)
        if voltage is not None and temps:
            picks = [voltage, next((i for i in temps if i > voltage), temps[0])]
        else:
            picks = [0, 1]
    else:
        wanted = 4
        temps = [i for i, h in enumerate(value_headers) if _TEMPERATURE.search(h)]
        picks = temps[:4] if len(temps) >= 4 else [0, 1, 2, 3]
    columns = [date_col + 1 + p for p in picks]
    units = ["C"] * wanted
    if value_headers:
        units = [_unit(value_headers[p]) if p < len(value_headers) else "C" for p in picks]
        if kind == "pressure":
            units[0] = "V"

    dates, values, lines, dropped = [], [], [], []
    decimal_comma = delimiter != ","
    for line_number, row in enumerate(rows[data_start:], start=data_start + 1):
        if not row or not any(c.strip() for c in row):
            continue
        if len(row) <= max(columns):
            dropped.append((line_number, Issue(None, "W_ROW_MISSING_COLUMNS")))
            continue
        date = row[date_col].strip()
        cells = [row[c].strip() for c in columns]
        if not date:
            dropped.append((line_number, Issue(None, "W_ROW_NO_DATE")))
            continue
        if any(not c for c in cells):
            dropped.append((line_number, Issue(None, "W_ROW_EMPTY_VALUES")))
            continue
        try:
            numbers = [float(c.replace(",", ".") if decimal_comma else c) for c in cells]
        except ValueError:
            bad = next(c for c in cells if not _is_number(c, decimal_comma))
            dropped.append((line_number, Issue(None, "W_ROW_NOT_A_NUMBER", value=bad)))
            continue
        dates.append(date)
        values.append(numbers)
        lines.append(line_number)

    if len(dates) < 2:
        raise UserError("E_MEASURES_TOO_FEW", file=path.name)
    header_cell = header[date_col] if header and date_col < len(header) else None
    dates, utc_offset, overridden = _declared_offset(path.name, header_cell, dates, lines)
    parsed_dates, date_format, chronological = parse_dates(dates)
    if not chronological:
        backwards = int(np.flatnonzero(np.diff(parsed_dates.to_numpy().astype("datetime64[s]").astype(np.int64)) < 0)[0]) + 1
        raise UserError(
            "E_MEASURES_NOT_CHRONOLOGICAL", file=path.name, line=lines[backwards], value=dates[backwards], previous_line=lines[backwards - 1], previous=dates[backwards - 1]
        )
    array = np.asarray(values, np.float64)
    converted = []
    for j, unit in enumerate(units):
        if unit in ("F", "K"):
            array[:, j] = _to_celsius(array[:, j], unit)
            converted.append(f"°{unit} → °C" if unit == "F" else "K → °C")
    names = RAW_PRESSURE_COLUMNS if kind == "pressure" else RAW_TEMPERATURE_COLUMNS
    frame = pd.DataFrame(array, columns=names[1:])
    frame.insert(0, "date", parsed_dates.to_numpy())
    duplicated = frame["date"].duplicated()
    if duplicated.any():
        for idx in np.flatnonzero(duplicated.to_numpy()):
            dropped.append((-1, Issue(None, "W_ROW_DUPLICATE_DATE", date=frame["date"].iloc[idx])))
        frame = frame[~duplicated]
    frame = frame.reset_index(drop=True)
    used = [header[c] if header and c < len(header) else f"#{c + 1}" for c in [date_col, *columns]]
    warnings = [] if utc_offset is not None else [Issue(None, "W_MEASURES_TIMEZONE_UNDECLARED", file=path.name)]
    if overridden is not None:
        warnings.append(
            Issue(None, "W_MEASURES_TIMEZONE_OVERRIDDEN", file=path.name, declared=format_offset(overridden), applied=format_offset(utc_offset))
        )
    return ParsedMeasures(frame, date_format, len(rows) - data_start, dropped, used, sorted(set(converted)), utc_offset, warnings)


def _declared_offset(
    file: str, header: str | None, dates: list[str], lines: list[int]
) -> tuple[list[str], timedelta | None, timedelta | None]:
    """Décalage à UTC déclaré par le fichier, et les dates débarrassées de leur décalage explicite.

    Source : l'en-tête de la colonne des dates, ou à défaut le décalage porté par les dates elles-mêmes. Un décalage
    explicite qui contredit l'en-tête est un conflit ; des décalages explicites différents, ou portés par une partie
    seulement des dates, sans en-tête pour trancher, sont ambigus. Sans aucune déclaration : (dates, None, None).

    Le troisième élément est le décalage d'en-tête remplacé par la convention provisoire (None si aucun) : un en-tête
    +01:00 sans décalage explicite dans les dates vaut PROVISIONAL_UTC_OFFSET."""
    declared = parse_header_offset(header)
    texts, offsets = [], []
    for value in dates:
        text, offset = split_cell_offset(value)
        texts.append(text)
        offsets.append(offset)
    carried = {offset for offset in offsets if offset is not None}
    if declared is not None:
        for line, value, offset in zip(lines, dates, offsets):
            if offset is not None and offset != declared:
                raise UserError(
                    "E_MEASURES_TIMEZONE_CONFLICT",
                    file=file, line=line, value=value, offset=format_offset(offset), declared=format_offset(declared),
                )
        if declared == HEADER_OFFSET_OVERRIDDEN and not carried:
            return texts, PROVISIONAL_UTC_OFFSET, declared
        return texts, declared, None
    if not carried:
        return texts, None, None
    if len(carried) > 1:
        shown = ", ".join(format_offset(o) for o in sorted(carried))
        raise UserError("E_MEASURES_TIMEZONE_MIXED", file=file, offsets=shown)
    if any(offset is None for offset in offsets):
        raise UserError("E_MEASURES_TIMEZONE_PARTIAL", file=file)
    return texts, next(iter(carried)), None


def _is_number(cell: str, decimal_comma: bool) -> bool:
    try:
        float(cell.replace(",", ".") if decimal_comma else cell)
        return True
    except ValueError:
        return False


def read_pressure_file(path: Path) -> ParsedMeasures:
    """Mesures du capteur de pression : date, tension (V), température du capteur (°C)."""
    return _read_measures(path, "pressure")


def read_temperature_file(path: Path) -> ParsedMeasures:
    """Mesures de la tige : date et 4 températures (°C), de la moins à la plus profonde."""
    return _read_measures(path, "temperature")


@dataclass
class PointFiles:
    info: Path | None = None
    pressure: Path | None = None
    temperature: Path | None = None
    notice: Path | None = None
    scheme: Path | None = None


def find_point_files(directory: Path) -> PointFiles:
    """Repère les fichiers d'un dossier de point. Les mesures sont reconnues à leur contenu."""
    directory = Path(directory)
    files = PointFiles()
    for path in sorted(directory.iterdir()):
        lower = path.name.lower()
        if path.suffix.lower() in (".png", ".jpg", ".jpeg") and files.scheme is None:
            files.scheme = path
        elif "info" in lower and path.suffix.lower() in (".csv", ".txt") and files.info is None:
            files.info = path
        elif path.suffix.lower() == ".txt" and files.notice is None and "geometr" not in lower:
            files.notice = path
        elif path.suffix.lower() == ".csv":
            kind = _guess_kind(path)
            if kind == "pressure" and files.pressure is None:
                files.pressure = path
            elif kind == "temperature" and files.temperature is None:
                files.temperature = path
    return files


def _guess_kind(path: Path) -> str | None:
    try:
        head = read_text(path)[:2000]
    except OSError:
        return None
    header_lines = head.splitlines()[:3]
    if any(_VOLTAGE.search(line) for line in header_lines):
        return "pressure"
    if any(len(_TEMPERATURE.findall(line)) >= 4 for line in header_lines):
        return "temperature"
    stem = path.stem.lower()
    if re.search(r"(^p|_p_)", stem):
        return "pressure"
    if re.search(r"(^t|_t_)", stem):
        return "temperature"
    return None
