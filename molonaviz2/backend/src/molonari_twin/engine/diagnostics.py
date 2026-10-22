"""Diagnostic a posteriori d'une inversion (ArviZ), sans dépendance à l'interface."""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from ..domain.compute import INVERTED_PARAMETERS, PARAMETER_LABELS
from .results import InversionResult


def inference_data(result: InversionResult, layer_names: list[str]):
    """Échantillons au format ArviZ : une variable par paramètre, dimension « couche »."""
    samples = np.moveaxis(result.samples, 1, 0)  # (chaînes, itérations, couches, paramètres)
    posterior = {name: samples[..., p] for p, name in enumerate(INVERTED_PARAMETERS)}
    stats = {"energy": np.moveaxis(result.energies, 1, 0)}
    return az.from_dict(
        {"posterior": posterior, "sample_stats": stats},
        coords={"layer": layer_names},
        dims={name: ["layer"] for name in INVERTED_PARAMETERS},
    )


def summary_table(result: InversionResult, layer_names: list[str]) -> pd.DataFrame:
    """Moyenne, écart-type, intervalle HDI à 90 %, ESS et R-hat par paramètre et par couche."""
    data = inference_data(result, layer_names)
    table = az.summary(data, kind="all", ci_prob=0.9, ci_kind="hdi", round_to=None)
    best = {
        f"{name}[{layer}]": result.best_parameters[i, p]
        for i, layer in enumerate(layer_names)
        for p, name in enumerate(INVERTED_PARAMETERS)
    }
    table.insert(0, "best", [best.get(index, np.nan) for index in table.index])
    keep = ["best", "mean", "sd", "hdi90_lb", "hdi90_ub", "ess_bulk", "r_hat"]
    table = table[[c for c in keep if c in table.columns]]
    table.index = [_label(index) for index in table.index]
    return table


def _label(index: str) -> str:
    name, _, layer = index.partition("[")
    return f"{PARAMETER_LABELS.get(name, name)} · {layer.rstrip(']')}"


def export_netcdf(result: InversionResult, layer_names: list[str], path) -> None:
    inference_data(result, layer_names).to_netcdf(str(path))
