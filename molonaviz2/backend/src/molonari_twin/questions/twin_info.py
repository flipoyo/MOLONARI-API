"""Ce que contient un twin (HN-02) et d'où vient son résultat (HN-43)."""

from __future__ import annotations

import numpy as np

from ..twin.config import TwinConfig
from ..twin.results import TwinResults
from .answer import Answer, Quantity


def _date(value):
    return None if value is None else np.datetime64(value, "s")


def describe(config: TwinConfig) -> Answer:
    """Couches, capteurs, période, mode et réglages principaux, lus dans la configuration (sans calcul)."""
    data = {
        "name": Quantity(config.name, "1"),
        "point": Quantity(config.point, "1"),
        "mode": Quantity(config.mode, "1"),
        "description": Quantity(config.description, "1"),
        "start": Quantity(_date(config.start), "datetime64[s]"),
        "end": Quantity(_date(config.end), "datetime64[s]"),
        "layers": Quantity(np.array([layer.name for layer in config.layers]), "1"),
        "layer_bottoms": Quantity(np.array([layer.z_low for layer in config.layers], dtype=float), "m"),
        "sensor_depths": Quantity(np.array(config.depths, dtype=float), "m"),
        "offset": Quantity(float(config.offset), "m"),
        "nb_cells": Quantity(int(config.nb_cells), "1"),
        "sigma_temperature": Quantity(float(config.sigma_temperature), "°C"),
    }
    if config.mcmc is not None:
        mcmc = config.mcmc
        data.update({
            "nb_chains": Quantity(int(mcmc.nb_chains), "1"),
            "nb_iter": Quantity(int(mcmc.nb_iter), "1"),
            "burn_in_max": Quantity(int(mcmc.burn_in_max), "1"),
            "thin": Quantity(int(mcmc.thin), "1"),
            "store_fields": Quantity(str(mcmc.store_fields), "1"),
        })
    return Answer(data=data, provenance={"question": "describe", "parameters": {}})


def provenance(results: TwinResults) -> Answer:
    """Versions, empreinte des données, plateforme et durée du calcul, et la commande pour le refaire. Le détail
    des dépôts (commits, remotes, état) est dans la provenance de la réponse."""
    metadata = results.metadata
    data_info = metadata.get("data", {})
    data = {
        "twin_id": Quantity(str(metadata.get("twin_id", "")), "1"),
        "code_version": Quantity(str(metadata.get("code_version", "")), "1"),
        "results_format": Quantity(results.results_format, "1"),
        "created_at": Quantity(str(metadata.get("created_at", "")), "1"),
        "device": Quantity(str(metadata.get("device", "")), "1"),
        "duration": Quantity(float(metadata.get("duration_s", float("nan"))), "s"),
        "data_file": Quantity(str(data_info.get("file", "")), "1"),
        "data_sha256": Quantity(str(data_info.get("sha256", "")), "1"),
        "reproduce": Quantity(f"molonari-twin run {metadata.get('name', '<twin>')}", "1"),
    }
    return Answer(data=data, provenance={"question": "provenance", "parameters": {},
                                         "repositories": metadata.get("provenance")},
                  warnings=list(results.warnings))


def describe_sampling(config: TwinConfig) -> Answer:
    """Comment un twin mcmc est (ou sera) échantillonné (BN-36) : chaînes, itérations, burn-in, `thin`, total
    d'échantillons, champs par chaîne et intervalle entre eux, pas d'espace et de temps, `store_fields` et taille
    stockée. Lu dans la configuration et le nombre de mesures de la période, sans calcul."""
    from ..engine.layout import sampling_layout

    mcmc = config.inversion_config()
    layout = sampling_layout(mcmc, len(config.load_measurements()))
    stored = layout.stored_bytes(mcmc.store_fields)
    data = {
        "nb_chains": Quantity(layout.nb_chains, "1"),
        "burn_in_max": Quantity(layout.burn_in_max, "1"),
        "nb_iter": Quantity(layout.nb_iter, "1"),
        "thin": Quantity(layout.thin, "1"),
        "kept_per_chain": Quantity(layout.kept, "1"),
        "samples": Quantity(layout.parameter_sets, "1"),
        "fields_per_chain": Quantity(layout.fields_per_chain, "1"),
        "fields_total": Quantity(layout.fields_total, "1"),
        "field_interval": Quantity(layout.interval, "1"),
        "nb_cells": Quantity(layout.nb_cells, "1"),
        "field_cells": Quantity(layout.field_cells, "1"),
        "nb_times": Quantity(layout.nb_times, "1"),
        "field_times": Quantity(layout.field_times, "1"),
        "space_step": Quantity(int(mcmc.space_step), "1"),
        "time_step": Quantity(int(mcmc.time_step), "1"),
        "store_fields": Quantity(str(mcmc.store_fields), "1"),
        "stored_bytes": Quantity(stored, "B"),
    }
    return Answer(data=data, provenance={"question": "sampling_layout", "parameters": {}})
