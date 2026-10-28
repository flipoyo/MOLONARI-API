"""Lecture des dates des fichiers de terrain.

Portage de `convertDates` (Molonaviz, utils/general.py) : on essaie une liste de
formats rencontrés sur le terrain et on retient le premier qui lit toutes les
dates ET donne une série croissante (ce qui départage jour/mois inversés).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from datetime import datetime, timedelta

import pandas as pd

from ..errors import UserError


def _formats() -> list[str]:
    dates = []
    for sep in ("/", "-", ":"):
        for order in ("%m{s}%d{s}%y", "%m{s}%d{s}%Y", "%d{s}%m{s}%y", "%d{s}%m{s}%Y", "%y{s}%m{s}%d", "%Y{s}%m{s}%d"):
            dates.append(order.format(s=sep))
    times = ("%H:%M:%S", "%I:%M:%S %p", "%H:%M", "%I:%M %p")
    # Années à 4 chiffres d'abord pour l'ordre année-mois-jour : « 2016/06/27 » ne doit pas être lu %y.
    ordered = sorted(dates, key=lambda f: (not f.startswith("%Y"), not f.endswith("%Y")))
    return [f"{d} {tm}" for d in ordered for tm in times] + ["%Y-%m-%dT%H:%M:%S"]


DATE_FORMATS = _formats()
_SPACES = re.compile(r"\s+")


def normalize(value: str) -> str:
    return _SPACES.sub(" ", str(value).strip())


def _shape(value: str) -> str:
    """Forme d'une date (chiffres remplacés), pour regrouper les valeurs écrites de la même façon."""
    return re.sub(r"\d+", "9", value).upper()


def _candidates(values: pd.Series) -> list[tuple[str, pd.Series]]:
    found = []
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        if not parsed.isna().any():
            found.append((fmt, parsed))
    return found


MAX_COMBINATIONS = 256


def parse_dates(values: Sequence[str]) -> tuple[pd.Series, str, bool]:
    """Convertit une série de dates texte. Retourne (dates, format détecté, chronologique).

    Un fichier peut mélanger plusieurs écritures (par exemple après une retouche dans
    un tableur) : les valeurs sont regroupées par forme, un format est choisi par
    groupe, et on retient la combinaison qui donne une série chronologique. Le
    format retourné liste alors les formats séparés par « + ».

    Si aucune lecture n'est chronologique, la première lecture possible est
    retournée avec chronologique = False : c'est à l'appelant de décider (les
    fichiers de terrain non chronologiques ont en général été altérés par un tableur).

    Lève UserError si une valeur ne correspond à aucun format connu.
    """
    series = pd.Series([normalize(v) for v in values], dtype="object")
    if series.empty:
        raise UserError("E_DATES_EMPTY")
    shapes = series.map(_shape)
    groups = []
    for shape in dict.fromkeys(shapes):  # ordre d'apparition
        index = shapes.index[shapes == shape]
        candidates = _candidates(series[index])
        if not candidates:
            sample = series[index[0]]
            raise UserError("E_DATE_FORMAT_UNKNOWN", value=sample)
        groups.append((index, candidates))

    first = None
    for combination in itertools.islice(itertools.product(*(c for _, c in groups)), MAX_COMBINATIONS):
        result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
        for (index, _), (_, parsed) in zip(groups, combination):
            result[index] = parsed.to_numpy()
        label = " + ".join(dict.fromkeys(fmt for fmt, _ in combination))
        if result.is_monotonic_increasing:
            return result, label, True
        first = first or (result, label, False)
    return first


def parse_date(value: str | None) -> datetime | None:
    """Convertit une date isolée (fichiers info, étalonnage). Vide → None."""
    if value is None or not normalize(value):
        return None
    text = normalize(value)
    for fmt in DATE_FORMATS + ["%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d", "%d_%m_%y", "%y_%m_%d"]:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise UserError("E_DATE_FORMAT_UNKNOWN", value=text)


# Fuseau déclaré par les fichiers (P2.1f) ---------------------------------------------------------------------
#
# Convention : celle des données brutes. Les exports HOBOware déclarent dans l'en-tête de la colonne des dates un
# décalage FIXE par rapport à UTC (« Date Heure, GMT+01:00 »), sans heure d'été ni nom de zone. Une date peut
# aussi porter son propre décalage (« 2016-06-27T12:00:00+01:00 »). Rien n'est supposé quand rien n'est déclaré.

_HEADER_OFFSET = re.compile(r"\b(?:GMT|UTC)\s*(?:(?P<sign>[+\-\u2212])\s*(?P<h>\d{1,2})(?::?(?P<m>\d{2}))?)?(?![\w:])", re.IGNORECASE)
# Décalage en fin de date, seulement après une heure (« 12:00:00+01:00 », « 12:00 PM Z ») : « 27-06-2016 » n'en a pas.
_CELL_OFFSET = re.compile(r"(?<=\d:\d\d)(?P<time>(?::\d\d)?(?:\s*[AaPp][Mm])?)\s*(?P<z>Z|(?P<sign>[+\-])(?P<h>\d{2})(?::?(?P<m>\d{2}))?)$")


def _offset(sign: str | None, hours: str | None, minutes: str | None) -> timedelta:
    delta = timedelta(hours=int(hours or 0), minutes=int(minutes or 0))
    return -delta if sign in ("-", "\u2212") else delta


def parse_header_offset(header: str | None) -> timedelta | None:
    """Décalage à UTC déclaré par l'en-tête d'une colonne de dates (« Date Heure, GMT+01:00 »), None s'il n'y en a
    pas. « GMT » ou « UTC » seul vaut UTC. Un décalage hors de ±14 h est refusé (E_DATE_OFFSET_INVALID)."""
    if not header:
        return None
    match = _HEADER_OFFSET.search(header)
    if match is None:
        return None
    delta = _offset(match["sign"], match["h"], match["m"])
    if abs(delta) > timedelta(hours=14) or (match["m"] and int(match["m"]) > 59):
        raise UserError("E_DATE_OFFSET_INVALID", value=match.group(0).strip())
    return delta


def split_cell_offset(value: str) -> tuple[str, timedelta | None]:
    """Sépare une date de son décalage explicite éventuel (« 2016-06-27T12:00:00+01:00 » → date, +1 h)."""
    text = normalize(value)
    match = _CELL_OFFSET.search(text)
    if match is None:
        return text, None
    z = match["z"]
    delta = timedelta() if z.upper() == "Z" else _offset(match["sign"], match["h"], match["m"])
    if abs(delta) > timedelta(hours=14) or (match["m"] and int(match["m"]) > 59):
        raise UserError("E_DATE_OFFSET_INVALID", value=z)
    return text[: match.start("z")].rstrip(), delta


def parse_offset_parameter(value: str) -> timedelta:
    """Décalage donné en paramètre : « UTC », « Z », « +01:00 », « GMT+1 », « UTC-05:30 ». Un nom de zone
    (« Europe/Paris ») n'est pas un décalage fixe : refusé (E_PIPELINE_TIMEZONE_INVALID)."""
    text = str(value).strip()
    if not text:
        raise UserError("E_PIPELINE_TIMEZONE_INVALID", value=text)
    candidate = text if re.match(r"(?i)^(gmt|utc|z)\b", text) else f"UTC{text}"
    if text.upper() == "Z":
        return timedelta()
    match = re.fullmatch(r"(?i)(?:gmt|utc)\s*(?:(?P<sign>[+\-\u2212])\s*(?P<h>\d{1,2})(?::?(?P<m>\d{2}))?)?", candidate)
    if match is None:
        raise UserError("E_PIPELINE_TIMEZONE_INVALID", value=text)
    delta = _offset(match["sign"], match["h"], match["m"])
    if abs(delta) > timedelta(hours=14) or (match["m"] and int(match["m"]) > 59):
        raise UserError("E_PIPELINE_TIMEZONE_INVALID", value=text)
    return delta


def format_offset(delta: timedelta | None) -> str:
    """« +01:00 » (« UTC » pour zéro, « non déclaré » pour None, repris par les messages)."""
    if delta is None:
        return "none"
    minutes = int(delta.total_seconds() // 60)
    sign = "-" if minutes < 0 else "+"
    return f"{sign}{abs(minutes) // 60:02d}:{abs(minutes) % 60:02d}"
