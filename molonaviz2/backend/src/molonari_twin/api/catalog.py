"""Catalogue des questions de l'API /v1 (lot C2 : questions sur la configuration et les résultats d'un twin).

Chaque question a un identifiant (celui de l'URL, le nom du catalogue : `exchanged_volume`), une référence du
plan (`HN-12` pour le haut niveau, `BN-34` pour le bas niveau), un libellé à traduire, les modes de twin où
elle a un sens (D : direct, M : mcmc), la version d'API qui l'a introduite, le modèle de ses paramètres, et ce
qu'elle exige du twin (sa configuration, ou ses résultats). Le catalogue complet, avec les lots suivants, est
décrit dans docs/CATALOGUE_QUESTIONS.md (espace cgs).

Ce qu'une question exige : `none` (aucun twin), `folder` (le dossier d'un twin, même à la configuration invalide),
`config` (une configuration valide) ou `results` (des résultats calculés).

Ajouter une question : une fonction pure dans questions/, un modèle de paramètres dans contract/params.py, une
entrée ici. Le serveur, la documentation (docs/API_V1.md) et les tests la prennent alors en compte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..contract import params as p
from ..domain.compute import INVERTED_PARAMETERS
from ..i18n import N_
from ..questions import (
    Answer,
    chain_data,
    describe,
    describe_sampling,
    provenance,
    residuals,
    sensor_report,
    temperature_at,
    twin_exchanged_volume,
    water_flux,
)


@dataclass
class TwinContext:
    """Ce qu'une question peut lire d'un twin : son dossier, sa configuration, et ses résultats s'il est calculé.

    Une question « folder » ne reçoit que le dossier (`config` est None) : elle lit elle-même la configuration,
    qui peut être invalide. Une question « none » n'a pas de twin : TwinAPI.ask_global ne crée pas de contexte."""

    ref: str
    config: object | None  # TwinConfig
    results: object | None = None  # TwinResults
    folder: Path | None = None


@dataclass(frozen=True)
class Question:
    id: str
    ref: str
    category: str
    label: str
    description: str
    modes: tuple[str, ...]
    params: type[p.Params]
    requires: Literal["none", "folder", "config", "results"]
    run: Callable[[TwinContext, p.Params], Answer] = field(repr=False)
    since: str = "1.0.0"
    outputs: tuple[str, ...] = ()  # noms des grandeurs de la réponse (certaines selon le mode ou les données)

    @property
    def level(self) -> str:
        return "high" if self.ref.startswith("HN-") else "low"


# Bandes du modèle, en mode M avec des champs stockés (questions.sensors.model_bands).
BANDS = ("field_time", "credible", "predictive", "coverage", "coverage_per_sensor", "points")

CATALOG: tuple[Question, ...] = (
    Question(
        "describe", "HN-02", "discovery", N_("What does this twin contain?"),
        "Layers, sensors, period, mode and main settings, read from the configuration (no computation needed).",
        ("D", "M"), p.NoParams, "config",
        lambda twin, _: describe(twin.config),
        outputs=(
            "name", "point", "mode", "description", "start", "end", "layers", "layer_bottoms", "sensor_depths", "offset",
            "nb_cells", "sigma_temperature", "seed", "nb_chains", "nb_iter", "burn_in_max", "thin", "store_fields",
        ),
    ),
    Question(
        "water_flux", "HN-10", "water", N_("What is the water flux through the riverbed?"),
        "Mean, median, 5th and 95th percentiles of the flux q at the riverbed (z = 0) over the period, in m/s, "
        "positive downwards (infiltration: the river feeds the aquifer).",
        ("D", "M"), p.PeriodParams, "results",
        lambda twin, a: water_flux(twin.results, a.start, a.end),
        outputs=("mean", "median", "p5", "p95", "points", "start", "end"),
    ),
    Question(
        "exchanged_volume", "HN-12", "water", N_("What volume of water crossed 1 m² of riverbed?"),
        "Infiltrated, exfiltrated and net volumes over the period, integrated by trapezoids with the sign change cut "
        "at the zero crossing: m³/m², or m³ when an area is given.",
        ("D", "M"), p.VolumeParams, "results",
        lambda twin, a: twin_exchanged_volume(twin.results, a.start, a.end, a.area),
        outputs=("infiltration", "exfiltration", "net"),
    ),
    Question(
        "temperature", "HN-30", "temperature", N_("What is the temperature at a depth and a date?"),
        "Simulated temperature in °C, linearly interpolated: one value (z and t), a profile T(z) (t alone) or a "
        "series T(t) (z alone).",
        ("D", "M"), p.TemperatureParams, "results",
        lambda twin, a: temperature_at(twin.results, a.z, a.t),
        outputs=("temperature", "depth", "time"),
    ),
    Question(
        "sensor_intervals", "HN-33", "temperature", N_("What uncertainty surrounds each measurement point?"),
        "Per sensor: the measurement with its instrument interval (± z·σ), the simulation and, in mcmc mode with "
        "stored fields, the credible band (model) and the predictive band (a new measurement), with the share of "
        "measurements the predictive band covers. Arrays are sensor first: (3, T).",
        ("D", "M"), p.IntervalParams, "results",
        lambda twin, a: sensor_report(twin.results, a.level, a.sigma, a.seed),
        outputs=("time", "measured", "simulated", "level", "z", "sigma") + BANDS,
    ),
    Question(
        "provenance", "HN-43", "reliability", N_("Where does this result come from?"),
        "Versions, data fingerprint, platform and duration of the computation, the command that reproduces it; "
        "the commits and state of each repository are in the provenance of the answer.",
        ("D", "M"), p.NoParams, "results",
        lambda twin, _: provenance(twin.results),
        outputs=(
            "twin_id", "code_version", "results_format", "created_at", "device", "duration", "data_file", "data_sha256",
            "reproduce",
        ),
    ),
    Question(
        "residuals", "BN-31", "raw_results", N_("Residuals at each sensor"),
        "Measured, simulated and residual (simulated − measured) temperatures per sensor, with the RMSE; option "
        "`intervals`: the credible and predictive bands.",
        ("D", "M"), p.ResidualParams, "results",
        lambda twin, a: residuals(twin.results, a.intervals, a.level, a.sigma, a.seed),
        outputs=("time", "measured", "simulated", "residual", "rmse") + BANDS,
    ),
    Question(
        "chain_data", "BN-34", "raw_results", N_("Everything about one or more chains"),
        "Parameters, energies, acceptance and burn-in resets of the chosen chains, with the computed iteration of "
        "each kept sample; read thinning and iteration range as in the computation.",
        ("M",), p.ChainParams, "results",
        lambda twin, a: chain_data(twin.results, a.chains, a.thin, a.iterations),
        outputs=(
            "iteration", "chain", *(f"samples.{name}" for name in INVERTED_PARAMETERS), "energy", "acceptance", "resets",
            "count",
        ),
    ),
    Question(
        "sampling_layout", "BN-36", "raw_results", N_("How was this twin sampled?"),
        "Chains, iterations per chain, burn-in, thin, total samples, fields per chain and the interval between "
        "them, space and time steps, store_fields and stored size; from the configuration, before or after the run.",
        ("M",), p.NoParams, "config",
        lambda twin, _: describe_sampling(twin.config),
        outputs=(
            "nb_chains", "burn_in_max", "nb_iter", "thin", "kept_per_chain", "samples", "fields_per_chain",
            "fields_total", "field_interval", "nb_cells", "field_cells", "nb_times", "field_times", "space_step",
            "time_step", "store_fields", "stored_bytes",
        ),
    ),
)
