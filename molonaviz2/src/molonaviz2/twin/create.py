"""Création d'un twin (dossier twin_xx avec twin.yaml et fichier de mesures)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..domain.compute import INVERTED_PARAMETERS, ForwardConfig, InversionConfig
from ..errors import UserError
from ..i18n import t
from ..processing.cleanup import write_cleaned_csv
from .config import SCHEMA, write_twin_yaml
from .run import next_twin_folder

MEASUREMENTS_NAME = "measurements.csv"


def _number(value: float):
    """Entier si la valeur est entière (YAML plus lisible), sinon flottant."""
    return int(value) if float(value).is_integer() and abs(value) < 1e15 else float(value)


def twin_document(name: str, point: str, description: str, depths, offset: float, sigma_t: float, sigma_p: float,
                  config: ForwardConfig | InversionConfig) -> dict:
    mode = "mcmc" if isinstance(config, InversionConfig) else "direct"
    document = {
        "schema": SCHEMA,
        "name": name,
        "point": point,
        "description": description,
        "mode": mode,
        "data": {"measurements": MEASUREMENTS_NAME},
        "sensors": {
            "depths": [_number(d) for d in depths],
            "offset": _number(offset),
            "sigma_temperature": _number(sigma_t),
            "sigma_pressure": _number(sigma_p),
        },
    }
    if mode == "direct":
        document["layers"] = [
            {"name": layer.name, "z_low": _number(layer.z_low), **{k: _number(getattr(layer, k)) for k in INVERTED_PARAMETERS}}
            for layer in config.layers
        ]
    else:
        document["layers"] = [{"name": layer.name, "z_low": _number(layer.z_low)} for layer in config.layers]
    document["numerics"] = {"nb_cells": config.nb_cells, "initial_condition": "linear"}
    if mode == "mcmc":
        document["mcmc"] = {
            "priors": [{k: [_number(p[k].low), _number(p[k].high)] for k in INVERTED_PARAMETERS} for p in config.priors],
            **{k: getattr(config, k) for k in ("nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr")},
            **{k: _number(getattr(config, k)) for k in ("c", "c_star", "gr_threshold")},
            "quantiles": [_number(q) for q in config.quantiles],
            **{k: getattr(config, k) for k in ("n_field_samples", "space_step", "time_step")},
        }
        document["seed"] = config.seed
    return document


def create_twin(twins_folder: Path, measurements: pd.DataFrame, point: str, description: str, depths, offset: float, sigma_t: float,
                sigma_p: float, config: ForwardConfig | InversionConfig) -> Path:
    """Crée le prochain twin_xx du dossier. Retourne le dossier créé."""
    twins_folder = Path(twins_folder)
    if not twins_folder.is_dir():
        raise UserError(t("Twin folder not found: {path}").format(path=twins_folder))
    if measurements.empty:
        raise UserError(t("No cleaned measurements yet: apply a cleaning first."))
    folder = next_twin_folder(twins_folder)
    folder.mkdir()
    write_cleaned_csv(measurements, folder / MEASUREMENTS_NAME)
    write_twin_yaml(folder / "twin.yaml", twin_document(folder.name, point, description, depths, offset, sigma_t, sigma_p, config))
    return folder
