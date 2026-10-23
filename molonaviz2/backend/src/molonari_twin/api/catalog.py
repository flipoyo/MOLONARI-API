"""Catalogue des questions de l'API /v1 (lot C2 : questions sur la configuration et les résultats d'un twin).

Chaque question a un identifiant (celui de l'URL, le nom du catalogue : `exchanged_volume`), une référence du
plan (`HN-12` pour le haut niveau, `BN-34` pour le bas niveau), un libellé à traduire, les modes de twin où
elle a un sens (D : direct, M : mcmc), la version d'API qui l'a introduite, le modèle de ses paramètres, et ce
qu'elle exige du twin (sa configuration, ou ses résultats). Le catalogue complet, avec les lots suivants, est
décrit dans docs/CATALOGUE_QUESTIONS.md (espace cgs).

Ajouter une question : une fonction pure dans questions/, un modèle de paramètres dans contract/params.py, une
entrée ici. Le serveur, la documentation (docs/API_V1.md) et les tests la prennent alors en compte.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from ..contract import params as p
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
    """Ce qu'une question peut lire d'un twin : sa configuration, et ses résultats s'il est calculé."""

    ref: str
    config: object  # TwinConfig
    results: object | None = None  # TwinResults


@dataclass(frozen=True)
class Question:
    id: str
    ref: str
    category: str
    label: str
    description: str
    modes: tuple[str, ...]
    params: type[p.Params]
    requires: Literal["config", "results"]
    run: Callable[[TwinContext, p.Params], Answer] = field(repr=False)
    since: str = "1.0.0"

    @property
    def level(self) -> str:
        return "high" if self.ref.startswith("HN-") else "low"


CATALOG: tuple[Question, ...] = (
    Question(
        "describe", "HN-02", "discovery", N_("What does this twin contain?"),
        "Layers, sensors, period, mode and main settings, read from the configuration (no computation needed).",
        ("D", "M"), p.NoParams, "config",
        lambda twin, _: describe(twin.config),
    ),
    Question(
        "water_flux", "HN-10", "water", N_("What is the water flux through the riverbed?"),
        "Mean, median, 5th and 95th percentiles of the flux q at the riverbed (z = 0) over the period, in m/s, "
        "positive downwards (infiltration: the river feeds the aquifer).",
        ("D", "M"), p.PeriodParams, "results",
        lambda twin, a: water_flux(twin.results, a.start, a.end),
    ),
    Question(
        "exchanged_volume", "HN-12", "water", N_("What volume of water crossed 1 m² of riverbed?"),
        "Infiltrated, exfiltrated and net volumes over the period, integrated by trapezoids with the sign change cut "
        "at the zero crossing: m³/m², or m³ when an area is given.",
        ("D", "M"), p.VolumeParams, "results",
        lambda twin, a: twin_exchanged_volume(twin.results, a.start, a.end, a.area),
    ),
    Question(
        "temperature", "HN-30", "temperature", N_("What is the temperature at a depth and a date?"),
        "Simulated temperature in °C, linearly interpolated: one value (z and t), a profile T(z) (t alone) or a "
        "series T(t) (z alone).",
        ("D", "M"), p.TemperatureParams, "results",
        lambda twin, a: temperature_at(twin.results, a.z, a.t),
    ),
    Question(
        "sensor_intervals", "HN-33", "temperature", N_("What uncertainty surrounds each measurement point?"),
        "Per sensor: the measurement with its instrument interval (± z·σ), the simulation and, in mcmc mode with "
        "stored fields, the credible band (model) and the predictive band (a new measurement), with the share of "
        "measurements the predictive band covers. Arrays are sensor first: (3, T).",
        ("D", "M"), p.IntervalParams, "results",
        lambda twin, a: sensor_report(twin.results, a.level, a.sigma, a.seed),
    ),
    Question(
        "provenance", "HN-43", "reliability", N_("Where does this result come from?"),
        "Versions, data fingerprint, platform and duration of the computation, the command that reproduces it; "
        "the commits and state of each repository are in the provenance of the answer.",
        ("D", "M"), p.NoParams, "results",
        lambda twin, _: provenance(twin.results),
    ),
    Question(
        "residuals", "BN-31", "raw_results", N_("Residuals at each sensor"),
        "Measured, simulated and residual (simulated − measured) temperatures per sensor, with the RMSE; option "
        "`intervals`: the credible and predictive bands.",
        ("D", "M"), p.ResidualParams, "results",
        lambda twin, a: residuals(twin.results, a.intervals, a.level, a.sigma, a.seed),
    ),
    Question(
        "chain_data", "BN-34", "raw_results", N_("Everything about one or more chains"),
        "Parameters, energies, acceptance and burn-in resets of the chosen chains, with the computed iteration of "
        "each kept sample; read thinning and iteration range as in the computation.",
        ("M",), p.ChainParams, "results",
        lambda twin, a: chain_data(twin.results, a.chains, a.thin, a.iterations),
    ),
    Question(
        "sampling_layout", "BN-36", "raw_results", N_("How was this twin sampled?"),
        "Chains, iterations per chain, burn-in, thin, total samples, fields per chain and the interval between "
        "them, space and time steps, store_fields and stored size; from the configuration, before or after the run.",
        ("M",), p.NoParams, "config",
        lambda twin, _: describe_sampling(twin.config),
    ),
)
