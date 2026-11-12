"""Détecteur de fuite de valeurs, commun aux tests d'export (T10) et de rapport public (F1).

Il cherche une valeur dans un dossier : un **jeton numérique isolé** de même valeur dans les textes (JSON, SVG, README,
CSV) et `np.any(array == valeur)` dans les tableaux numériques des `.npz`. Le jeton isolé ne voit pas `142.42` comme
`4242`, et trouve `731942.0` ou `7.31942E+05`.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

NUMBER = re.compile(r"(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])")
SEED_KEY = re.compile(r'"seed"|\bseed\s*[:=]')


def text_forms(content: str, value: float, key: re.Pattern | None = None) -> list[str]:
    """Les formes de `value` dans un texte : un nombre isolé de même valeur (731942, 731942.0, 7.31942e5...), et, avec `key`,
    la clé que ce motif reconnaît."""
    forms = [f"nombre {token}" for token in NUMBER.findall(content) if float(token) == float(value)]
    if key is not None and key.search(content):
        forms.append("clé")
    return forms


def value_occurrences(folder: Path, value: float, key: re.Pattern | None = None) -> list[str]:
    """Une description « chemin relatif: forme » par occurrence de `value` dans un dossier ; vide s'il n'y en a pas.
    `key` : motif d'une clé à signaler aussi (la graine ajoute la clé « seed »)."""
    found = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        name = path.relative_to(folder).as_posix()
        if path.suffix != ".npz":
            found += [f"{name}: {form}" for form in text_forms(path.read_text(encoding="utf-8"), value, key)]
            continue
        with np.load(path, allow_pickle=False) as archive:
            for array_key in archive.files:
                if (key is not None and "seed" in array_key) or (float(value).is_integer() and str(int(value)) in array_key):
                    found.append(f"{name}: tableau nommé {array_key}")
                array = archive[array_key]
                if array.dtype.kind in "iuf" and np.any(array == value):
                    found.append(f"{name}[{array_key}]: valeur numérique")
                elif array.dtype.kind in "US":
                    found += [f"{name}[{array_key}]: {form}" for item in array.ravel() for form in text_forms(str(item), value, key)]
    return found


def seed_occurrences(folder: Path, seed: int) -> list[str]:
    """Les occurrences de la graine (ou de la clé « seed ») dans un dossier ; vide s'il n'y en a pas."""
    return value_occurrences(folder, seed, SEED_KEY)
