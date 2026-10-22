"""Réponses typées des questions : des valeurs avec leur unité, et, s'il y a lieu, leur intervalle.

Ce sont des données, sans méthode : l'interface les affiche, le futur backend les sérialisera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..errors import Issue


@dataclass(frozen=True)
class Quantity:
    value: Any  # nombre ou tableau numpy
    unit: str  # "1" pour une grandeur sans dimension
    interval: tuple[Any, Any] | None = None  # (bas, haut), de la forme de value


@dataclass(frozen=True)
class Answer:
    data: dict[str, Quantity]
    provenance: dict  # {"question": nom, "parameters": {paramètres scalaires}}
    warnings: list[Issue] = field(default_factory=list)
