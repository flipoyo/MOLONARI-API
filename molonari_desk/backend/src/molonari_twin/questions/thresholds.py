"""Seuils du verdict de fiabilité (HN-41). **Provisoires : ils n'ont pas été validés par un hydrogéologue** (choix C9)
et doivent l'être avant l'API 1.0.0. Les réponses disent `thresholds_validated: false` tant que c'est le cas.

Chaque critère a un seuil « vert » et un seuil « orange » ; au-delà, il est rouge. Les seuils se surchargent, dans cet
ordre de priorité croissante : les valeurs ci-dessous, un fichier YAML dont le chemin est dans la variable
d'environnement MOLONARI_THRESHOLDS, puis le paramètre `thresholds` de la question (un objet partiel).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path

from ..errors import UserError

ENVIRONMENT_VARIABLE = "MOLONARI_THRESHOLDS"
THRESHOLDS_VALIDATED = False  # passer à True quand un hydrogéologue aura validé les valeurs par défaut


@dataclass(frozen=True)
class ReliabilityThresholds:
    rhat_green: float = 1.05  # R̂ maximal (rank-normalized) : vert jusqu'ici
    rhat_orange: float = 1.2
    ess_green: float = 400.0  # ESS minimal (bulk et tail) : vert à partir d'ici
    ess_orange: float = 100.0
    fit_green: float = 1.5  # RMSE/σT maximal sur les capteurs
    fit_orange: float = 3.0
    coverage_green: float = 0.05  # écart entre la couverture prédictive et son niveau
    coverage_orange: float = 0.15
    identifiability_green: float = 0.5  # écart-type a posteriori / a priori maximal
    identifiability_orange: float = 0.8
    gaps_green: float = 0.05  # part des mesures attendues qui manquent
    gaps_orange: float = 0.2

    def as_dict(self) -> dict[str, float]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


# Sens de chaque couple (vert, orange) : le vert est plus petit que l'orange, sauf pour l'ESS.
HIGHER_IS_BETTER = {"ess"}


def _check(thresholds: ReliabilityThresholds) -> None:
    values = thresholds.as_dict()
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0:
            raise UserError("E_THRESHOLDS_INVALID", detail=f"{name} must be a positive number (got {value!r})")
    for name in values:
        stem, _, grade = name.rpartition("_")
        if grade != "green":
            continue
        green, orange = values[name], values[f"{stem}_orange"]
        if (green < orange) if stem in HIGHER_IS_BETTER else (green > orange):
            order = "at least" if stem in HIGHER_IS_BETTER else "at most"
            raise UserError("E_THRESHOLDS_INVALID", detail=f"{name} must be {order} {stem}_orange ({green:g} against {orange:g})")


def _apply(base: ReliabilityThresholds, override, source: str) -> ReliabilityThresholds:
    known = {field.name for field in fields(base)}
    if not isinstance(override, dict):
        raise UserError("E_THRESHOLDS_INVALID", detail=f"{source}: expected an object of thresholds")
    unknown = sorted(set(override) - known)
    if unknown:
        raise UserError("E_THRESHOLDS_INVALID", detail=f"{source}: unknown threshold {unknown[0]!r} (known: {', '.join(sorted(known))})")
    bad = [key for key, value in override.items() if isinstance(value, bool) or not isinstance(value, int | float)]
    if bad:
        raise UserError("E_THRESHOLDS_INVALID", detail=f"{source}: {bad[0]} must be a number")
    updated = replace(base, **{key: float(value) for key, value in override.items()})
    _check(updated)
    return updated


def resolve_thresholds(override: dict | None = None, environment: dict | None = None) -> ReliabilityThresholds:
    """Seuils par défaut, surchargés par le YAML de MOLONARI_THRESHOLDS puis par `override` (E_THRESHOLDS_INVALID si
    une clé est inconnue, une valeur n'est pas un nombre positif, ou un seuil vert est moins exigeant que l'orange)."""
    environment = os.environ if environment is None else environment
    thresholds = ReliabilityThresholds()
    path = environment.get(ENVIRONMENT_VARIABLE)
    if path:
        import yaml

        try:
            document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise UserError("E_THRESHOLDS_INVALID", detail=f"{ENVIRONMENT_VARIABLE} ({path}): {error}") from None
        thresholds = _apply(thresholds, {} if document is None else document, ENVIRONMENT_VARIABLE)
    if override:
        thresholds = _apply(thresholds, override, "thresholds")
    return thresholds


def grade_below(value: float, green: float, orange: float) -> str:
    """Niveau d'une grandeur qui doit être petite (R̂, RMSE/σT, écart, trous) ; non finie : rouge."""
    if not math.isfinite(value):
        return "red"
    return "green" if value <= green else "orange" if value <= orange else "red"


def grade_above(value: float, green: float, orange: float) -> str:
    """Niveau d'une grandeur qui doit être grande (ESS) ; non finie : rouge."""
    if not math.isfinite(value):
        return "red"
    return "green" if value >= green else "orange" if value >= orange else "red"


LEVELS = ("green", "orange", "red")
