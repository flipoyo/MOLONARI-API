"""`molonari-twin plot temperature|fields` (F2) : les figures de l'admin d'un twin calculé, écrites hors des publications.

Ordre des contrôles : (1) le niveau (dev ou opx), (2) le dossier de sortie, (3) la lecture des réponses par `LeveledAPI`,
(4) la création du dossier et l'écriture (par un nom temporaire, puis renommé : jamais de fichier partiel). Rien n'est créé
avant la quatrième étape. Les figures sont dans `report/plots.py` ; elles montrent les mesures et ne sont jamais publiées.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from ..errors import Issue, UserError

FORBIDDEN_FOLDERS = ("public_reports", "results")  # noms de dossier exacts, pas des sous-chaînes
ACTIONS = ("temperature", "fields")


@dataclass(frozen=True)
class PlotFile:
    path: Path
    sha256: str
    bytes: int


@dataclass(frozen=True)
class PlotResult:
    files: list[PlotFile]
    warnings: list[Issue]


def read_period(start: str | None, end: str | None):
    """(début, fin) en `datetime` (None : sans borne), lus par `PeriodParams` comme `ask --params` : une date invalide, avec
    fuseau, ou une fin avant le début est une E_REQUEST_INVALID (un problème par champ), jamais un ValueError."""
    from ..contract.params import PeriodParams

    try:
        period = PeriodParams.model_validate({name: value for name, value in (("start", start), ("end", end)) if value is not None})
    except ValidationError as error:
        issues = [Issue(".".join(map(str, e["loc"])) or None, "E_PARAM_INVALID", prefixed=True, message=e["msg"]) for e in error.errors()]
        raise UserError("E_REQUEST_INVALID", issues=issues) from None
    return period.start, period.end


def check_output(output: Path) -> None:
    """E_PLOT_OUTPUT_FORBIDDEN si le chemin résolu passe par un dossier nommé `public_reports` ou `results`."""
    for folder in FORBIDDEN_FOLDERS:
        if folder in output.resolve().parts:
            raise UserError("E_PLOT_OUTPUT_FORBIDDEN", path=str(output), folder=folder)


def default_output(ref: str) -> Path:
    return Path.cwd() / f"{ref}_plots"


def build_figures(api, ref: str, action: str, start: str | None, end: str | None) -> tuple[list[tuple[str, bytes]], list[Issue]]:
    """((nom de fichier, octets SVG), avertissements) : les réponses sont lues au niveau de `api` (un LeveledAPI)."""
    from ..report import plots

    first, last = read_period(start, end)
    describe = api.ask(ref, "describe")
    if action == "temperature":
        sensors = api.ask(ref, "sensor_intervals", {"level": 0.9})
        return [("temperature.svg", plots.temperature_figure(sensors, describe, first, last))], list(sensors.warnings)
    params = {"variables": ["temperature", "water_flux"], **{name: value for name, value in (("start", start), ("end", end)) if value is not None}}
    fields = api.ask(ref, "fields", params)
    return plots.fields_figures(fields, describe, plots.role_of(describe)), list(fields.warnings)


def write_figures(folder: Path, figures: list[tuple[str, bytes]]) -> list[PlotFile]:
    """Écrit les figures dans `folder` (créé s'il manque), chacune par un nom temporaire puis renommée."""
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for name, content in figures:
        target = (folder / name).resolve()
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(content)
        partial.replace(target)
        written.append(PlotFile(target, hashlib.sha256(content).hexdigest(), len(content)))
    return written


def run_plot(api, ref: str, action: str, output: Path | None, start: str | None, end: str | None) -> PlotResult:
    """Les quatre contrôles, dans l'ordre. `api` : LeveledAPI au niveau demandé."""
    api.require_dev(f"plot {action}")
    folder = output if output is not None else default_output(ref)
    check_output(folder)
    figures, warnings = build_figures(api, ref, action, start, end)
    return PlotResult(write_figures(folder, figures), warnings)
