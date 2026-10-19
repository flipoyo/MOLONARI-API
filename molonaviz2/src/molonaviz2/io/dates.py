"""Lecture des dates des fichiers de terrain.

Portage de `convertDates` (Molonaviz, utils/general.py) : on essaie une liste de
formats rencontrés sur le terrain et on retient le premier qui lit toutes les
dates ET donne une série croissante (ce qui départage jour/mois inversés).
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence
from datetime import datetime

import pandas as pd

from ..errors import UserError
from ..i18n import t


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
        raise UserError(t("No dates to read."))
    shapes = series.map(_shape)
    groups = []
    for shape in dict.fromkeys(shapes):  # ordre d'apparition
        index = shapes.index[shapes == shape]
        candidates = _candidates(series[index])
        if not candidates:
            sample = series[index[0]]
            raise UserError(t("Unrecognised date format (first value: \"{value}\").").format(value=sample))
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
    raise UserError(t("Unrecognised date format (first value: \"{value}\").").format(value=text))
