"""Catalogue des questions de l'API /v1 (lot C2 : questions sur la configuration et les résultats d'un twin).

Chaque question a un identifiant (celui de l'URL, le nom du catalogue : `exchanged_volume`), une référence du
plan (`HN-12` pour le haut niveau, `BN-34` pour le bas niveau), un libellé à traduire, les modes de twin où
elle a un sens (D : direct, M : mcmc), la version d'API qui l'a introduite, le modèle de ses paramètres, et ce
qu'elle exige du twin (sa configuration, ou ses résultats). Le catalogue complet, avec les lots suivants, est
décrit dans docs/CATALOGUE_QUESTIONS.md (espace cgs).

Ce qu'une question exige : `none` (aucun twin), `folder` (le dossier d'un twin, même à la configuration invalide),
`config` (une configuration valide) ou `results` (des résultats calculés).

Ce qu'une question expose (`exposes`, colonne « Accès » du catalogue) : les éléments de visibilité que sa réponse
**révèle** (`questions/visibility.py`), et non ceux dont elle dépend. `()` : des agrégats, qui n'ont rien de privé
(accès « rapport ») ; une liste : la réponse n'entre dans le rapport public que si ces éléments sont publics (accès
« rapport si public ») ; `DEV` : réservée au développeur, jamais publiée (accès « dév. »). Le champ est obligatoire :
une question sans déclaration ne se construit pas. Le contrôle est dans api/publication.py.

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
    PUBLIC_DE_FAIT,
    Answer,
    chain_data,
    check_twin,
    convergence,
    describe,
    describe_sampling,
    energy_breakdown,
    exchange_direction,
    fit_quality,
    heat_budget,
    parameters_catalog,
    posterior_samples,
    processing_preview,
    provenance,
    public_convergence,
    raw_series,
    reliability,
    residuals,
    result_fields,
    sensor_report,
    soil_properties,
    temperature_at,
    twin_exchanged_volume,
    water_flux,
)
from ..twin.config import VISIBILITY_ELEMENTS


@dataclass
class TwinContext:
    """Ce qu'une question peut lire d'un twin : son dossier, sa configuration, et ses résultats s'il est calculé.

    Une question « folder » ne reçoit que le dossier (`config` est None) : elle lit elle-même la configuration,
    qui peut être invalide. Une question « none » n'a pas de twin : TwinAPI.ask_global ne crée pas de contexte."""

    ref: str
    config: object | None  # TwinConfig
    results: object | None = None  # TwinResults
    folder: Path | None = None


DEV = "dev"
"""Valeur de `Question.exposes` d'une question réservée au développeur : jamais publiée."""

Exposure = tuple[str, ...] | Literal["dev"]


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
    exposes: Exposure = field(kw_only=True)  # éléments de visibilité révélés par la réponse (obligatoire), ou DEV

    @property
    def level(self) -> str:
        return "high" if self.ref.startswith("HN-") else "low"

    @property
    def access(self) -> Literal["report", "report_if_public", "dev"]:
        """Colonne « Accès » du catalogue : `report` (rien de privé), `report_if_public` (selon la visibilité) ou `dev`."""
        if self.exposes == DEV:
            return "dev"
        return "report" if set(self.exposes) <= PUBLIC_DE_FAIT else "report_if_public"


# Sorties de soil_properties : une par propriété ; médiane et identifiabilité en mode M seulement.
SOIL_OUTPUTS = tuple(
    f"{name}{suffix}" for name in ("k", "K", "n", "lambda_s", "rhos_cs") for suffix in ("", "_median", "_identifiability"))

# Bandes du modèle, en mode M avec des champs stockés (questions.sensors.model_bands).
BANDS = ("field_time", "credible", "predictive", "coverage", "coverage_per_sensor", "points")

CATALOG: tuple[Question, ...] = (
    Question(
        "describe", "HN-02", "discovery", N_("What does this twin contain?"),
        "Layers, sensors, period, mode and main settings, read from the configuration (no computation needed), with the "
        "publication keys: site label, monitoring status, visibility of each element and the report asked for (the default "
        "report when the twin has none: `report_declared` is then false).",
        ("D", "M"), p.NoParams, "config",
        lambda twin, _: describe(twin.config),
        outputs=(
            "name", "point", "mode", "description", "start", "end", "layers", "layer_bottoms", "sensor_depths", "offset",
            "nb_cells", "sigma_temperature", "nb_chains", "nb_iter", "burn_in_max", "thin", "store_fields",
            "site_label", "monitoring_status", *(f"visibility.{element}" for element in VISIBILITY_ELEMENTS),
            "report_declared", "report_questions", "report_parameters", "report_forward", "report_inversion",
        ),
        exposes=("inversion_settings_public",),  # nb_chains, nb_iter et thin, que les échantillons publics révèlent ; la projection publique retire les réglages privés
    ),
    Question(
        "water_flux", "HN-10", "water", N_("What is the water flux through the riverbed?"),
        "Mean flux q at the riverbed (z = 0) over the period, in m/s, positive downwards (infiltration: the river "
        "feeds the aquifer), with the median and the 5th and 95th percentiles of q over time. In mcmc mode `mean` is "
        "the posterior mean flux and its interval, at `level` (0.9 by default), is the posterior interval of that "
        "mean, drawn from the series q(0, t) stored for every kept sample (`mean_best`: the mean of the best sample); "
        "in direct mode `mean` is a single value, without interval (`uncertainty` is none).",
        ("D", "M"), p.FluxParams, "results",
        lambda twin, a: water_flux(twin.results, a.start, a.end, a.level),
        outputs=("mean", "median", "p5", "p95", "points", "start", "end", "level", "model", "uncertainty", "mean_best", "samples"),
        exposes=(),
    ),
    Question(
        "exchange_direction", "HN-11", "water", N_("Does the river feed the aquifer?"),
        "Share of the time in infiltration (q ≥ 0, the river feeds the aquifer) and in exfiltration, weighted by the "
        "time intervals, dominant regime and number of reversals of q at the riverbed over the period. In mcmc mode "
        "with stored water-flux fields (`store_fields: full`), the probability that q ≥ 0 over the posterior samples, "
        "as a series and averaged over time.",
        ("D", "M"), p.PeriodParams, "results",
        lambda twin, a: exchange_direction(twin.results, a.start, a.end),
        outputs=(
            "infiltration_share", "exfiltration_share", "dominant", "reversals", "points", "start", "end",
            "p_infiltration", "p_infiltration_series", "field_time", "samples",
        ),
        exposes=(),
    ),
    Question(
        "exchanged_volume", "HN-12", "water", N_("What volume of water crossed 1 m² of riverbed?"),
        "Infiltrated, exfiltrated and net volumes over the period, integrated by trapezoids with the sign change cut "
        "at the zero crossing: m³/m², or m³ when an area is given.",
        ("D", "M"), p.VolumeParams, "results",
        lambda twin, a: twin_exchanged_volume(twin.results, a.start, a.end, a.area),
        outputs=("infiltration", "exfiltration", "net"),
        exposes=(),
    ),
    Question(
        "heat_budget", "HN-20", "heat", N_("Which processes carry the heat?"),
        "Mean advective, conductive and total heat fluxes (W/m², positive downwards) at a depth over the period, "
        "their shares (absolute value of the mean) and the dominant process. The lateral exchange is not modelled: "
        "it is zero, with a warning. Needs results of format 2.0.0.",
        ("D", "M"), p.HeatBudgetParams, "results",
        lambda twin, a: heat_budget(twin.results, a.z, a.start, a.end),
        outputs=(
            "advective", "conductive", "lateral", "total", "advective_share", "conductive_share", "lateral_share",
            "dominant", "depth", "points", "start", "end",
        ),
        exposes=("states",),
    ),
    Question(
        "temperature", "HN-30", "temperature", N_("What is the temperature at a depth and a date?"),
        "Simulated temperature in °C, linearly interpolated: one value (z and t), a profile T(z) (t alone) or a "
        "series T(t) (z alone).",
        ("D", "M"), p.TemperatureParams, "results",
        lambda twin, a: temperature_at(twin.results, a.z, a.t),
        outputs=("temperature", "depth", "time"),
        exposes=("states",),
    ),
    Question(
        "sensor_intervals", "HN-33", "temperature", N_("What uncertainty surrounds each measurement point?"),
        "Per sensor: the measurement with its instrument interval (± z·σ), the simulation and, in mcmc mode with "
        "stored fields, the credible band (model) and the predictive band (a new measurement), with the share of "
        "measurements the predictive band covers. Arrays are sensor first: (3, T).",
        ("D", "M"), p.IntervalParams, "results",
        lambda twin, a: sensor_report(twin.results, a.level, a.sigma, a.seed),
        outputs=("time", "measured", "simulated", "level", "z", "sigma") + BANDS,
        exposes=("raw_data", "processed_data", "states"),  # mesures et températures simulées aux capteurs
    ),
    Question(
        "fit_quality", "HN-40", "reliability", N_("Does the model reproduce the measurements?"),
        "Per sensor: RMSE, bias (simulated − measured), RMSE/σT, lag-1 autocorrelation of the residuals and, in mcmc "
        "mode with stored fields, the share of measurements the predictive band covers; then the total RMSE (root "
        "mean square of the three). Arrays are sensor first: (3,). A direct twin is the forward model (`model` "
        "forward, `uncertainty` none); an mcmc twin is the inversion model, represented by its best sample, and "
        "`median_rmse` and `median_rmse_total` give the RMSE of the posterior median in addition. The direct "
        "difference forward − inversion is `TwinAPI.model_difference`, which needs two twins.",
        ("D", "M"), p.IntervalParams, "results",
        lambda twin, a: fit_quality(twin.results, a.level, a.sigma, a.seed),
        outputs=(
            "rmse", "bias", "rmse_over_sigma", "lag1_autocorrelation", "sigma", "level", "coverage_per_sensor", "rmse_total",
            "sensor", "model", "uncertainty", "median_rmse", "median_rmse_total",
        ),
        exposes=("raw_data", "processed_data"),  # RMSE et biais : des agrégats, obligatoires au rapport (SPEC_RAPPORT_PUBLIC, section 4)
    ),
    Question(
        "reliability", "HN-41", "reliability", N_("Can I trust this result?"),
        "A green, orange or red verdict, the worst of the criteria, each with its level and its reason: convergence "
        "(R̂, ESS) in mcmc mode, fit (RMSE/σT), identifiability and predictive coverage in mcmc mode, gaps in the "
        "data and provenance. **The thresholds are provisional and not validated by a hydrogeologist** "
        "(`thresholds_validated` is false): the response gives those it used, and they can be overridden.",
        ("D", "M"), p.ReliabilityParams, "results",
        lambda twin, a: reliability(twin.results, a.level, a.sigma, a.seed, a.thresholds),
        outputs=(
            "verdict", "criteria", "levels", "reasons", "threshold_names", "thresholds", "thresholds_validated", "level",
        ),
        exposes=(),
    ),
    Question(
        "public_convergence", "HN-45", "reliability", N_("Has the sampling converged?"),
        "The convergence of the algorithm for the public report, without a verdict: largest R̂ and smallest ESS over all "
        "parameters, mean acceptance, whether the burn-in converged, chains and kept samples per chain, with the "
        "reference thresholds (the defaults of the reliability thresholds, not their overrides) and "
        "`thresholds_validated`, false until a hydrogeologist validates them. No colour and no verdict; the "
        "developer's own R̂ threshold is private and not given. mcmc mode only.",
        ("M",), p.NoParams, "results",
        lambda twin, _: public_convergence(twin.results),
        outputs=(
            "rhat_max", "ess_min", "acceptance_mean", "burn_in_converged", "burn_in_iterations", "chains", "samples_per_chain",
            "rhat_green", "rhat_orange", "ess_green", "ess_orange", "thresholds_validated",
        ),
        exposes=(),
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
        exposes=(),  # la projection publique ne garde que versions, plateforme et durée
    ),
    Question(
        "residuals", "BN-31", "raw_results", N_("Residuals at each sensor"),
        "Measured, simulated and residual (simulated − measured) temperatures per sensor, with the RMSE; option "
        "`intervals`: the credible and predictive bands.",
        ("D", "M"), p.ResidualParams, "results",
        lambda twin, a: residuals(twin.results, a.intervals, a.level, a.sigma, a.seed),
        outputs=("time", "measured", "simulated", "residual", "rmse") + BANDS,
        exposes=DEV,
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
        exposes=("posterior", "inversion_settings_public"),
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
        exposes=DEV,
    ),
    Question(
        "soil_properties", "HN-42", "reliability", N_("Which soil properties were estimated?"),
        "Per layer: permeability k (m²), hydraulic conductivity K (m/s, k·ρw·g/μ with water at 20 °C: an "
        "approximation), porosity n, solid thermal conductivity λs and heat capacity ρs·cs. In mcmc mode the best "
        "value, the median, the highest-density interval (computed in the space of the priors, −log₁₀ k for k) and the "
        "identifiability (posterior standard deviation over that of a uniform prior: near 1, the data teach nothing). "
        "In direct mode, the values of the configuration.",
        ("D", "M"), p.SoilParams, "results",
        lambda twin, a: soil_properties(twin.results, a.level, [layer.name for layer in twin.config.layers]),
        outputs=("layers", "level", *SOIL_OUTPUTS),
        exposes=("posterior", "forward_parameters"),  # mode M : la PDF ; mode D : les paramètres écrits dans la configuration
    ),
    Question(
        "posterior_samples", "BN-32", "raw_results", N_("Posterior samples of the parameters"),
        "The posterior samples as an array (chain, iteration, layer, parameter) in the space of the priors "
        "(−log₁₀ k, n, λs, ρs·cs; k = 10^−x in m²), with its axes. Chains, range of computed iterations, burn "
        "(kept iterations skipped at the start) and thin choose the samples.",
        ("M",), p.PosteriorParams, "results",
        lambda twin, a: posterior_samples(twin.results, a.chains, a.iter_range, a.burn, a.thin),
        outputs=("samples", "iteration", "chain", "layer", "parameter", "parameter_unit", "space"),
        exposes=("posterior", "inversion_settings_public"),
    ),
    Question(
        "convergence", "BN-40", "diagnostics", N_("Have the chains converged?"),
        "Per parameter and layer: R̂ (rank-normalized), bulk and tail ESS and MCSE of the mean. Per chain: "
        "acceptance, mean energy, bulk ESS and the largest R̂ without that chain. The statistics follow the "
        "selection (chains, burn, thin); `rhat_max_all` uses everything, to show the effect of the selection.",
        ("M",), p.ConvergenceParams, "results",
        lambda twin, a: convergence(twin.results, a.chains, a.burn, a.thin),
        outputs=(
            "rhat", "ess_bulk", "ess_tail", "mcse_mean", "rhat_max", "rhat_max_all", "chain", "acceptance",
            "mean_energy", "chain_ess_bulk", "rhat_without_chain", "layer", "parameter", "samples_per_chain",
            "burn_in_converged", "burn_in_iterations",
        ),
        exposes=DEV,
    ),
    Question(
        "energy_breakdown", "BN-43", "diagnostics", N_("What is the energy made of?"),
        "The terms of the energy (negative Gaussian log-likelihood) of the recorded fit: N·ln σ², the residual term "
        "Σr²/(2σ²), the implicit RMSE and the reference N/2, the value the residual term takes when σ is right.",
        ("D", "M"), p.SigmaParams, "results",
        lambda twin, a: energy_breakdown(twin.results, a.sigma),
        outputs=("n_points", "sigma", "log_term", "residual_term", "energy", "implicit_rmse", "reference", "ratio"),
        exposes=DEV,
    ),
    Question(
        "parameters_catalog", "BN-10", "build", N_("Which parameters can be set?"),
        "The settings with their bounds, defaults and units, the values of `store_fields`, the default priors and the "
        "physical bounds of the inverted parameters, and the keys of the layers and of the mcmc section of twin.yaml. "
        "It is about no twin.",
        ("D", "M"), p.NoParams, "none",
        lambda twin, _: parameters_catalog(),
        outputs=(
            "settings.name", "settings.minimum", "settings.maximum", "settings.default", "settings.unit",
            "settings.description", "store_fields.values", "store_fields.default", "store_fields.description",
            "priors.parameter", "priors.label", "priors.low", "priors.high", "priors.physical_low", "priors.physical_high",
            "layer_keys.name", "layer_keys.description", "mcmc_keys.name", "mcmc_keys.description",
        ),
        exposes=DEV,
    ),
    Question(
        "check", "BN-11", "build", N_("Is this configuration valid, and what will it cost?"),
        "Whether the configuration of a twin folder is valid, with its problems (also given as warnings); if so, the "
        "twin_id expected for the published code, the state (new, computed, duplicate or modified) and, in mcmc mode, "
        "the number of samples, the evaluations and the stored size, which depends on `store_fields`. An invalid "
        "configuration is an answer (`valid` is false), not an error.",
        ("D", "M"), p.NoParams, "folder",
        lambda twin, _: check_twin(twin.folder),
        outputs=(
            "valid", "name", "mode", "twin_id", "twin_id_local", "state", "duplicate_of", "samples", "max_evaluations",
            "stored_bytes", "store_fields", "issue_codes", "issue_fields", "issue_messages",
        ),
        exposes=DEV,
    ),
    Question(
        "fields", "BN-30", "raw_results", N_("Complete fields"),
        "Temperature (°C), water flux (m/s) and advective and conductive heat fluxes (W/m², positive downwards) as "
        "arrays (depth, time) of the best fit, with their axes. Choose the variables, period, depth range and time "
        "step. They are large: prefer Accept: application/x-npz.",
        ("D", "M"), p.FieldsParams, "results",
        lambda twin, a: result_fields(twin.results, a.variables, a.start, a.end, a.depth_range, a.time_step),
        outputs=("depth", "time", "temperature", "water_flux", "heat_flux_advective", "heat_flux_conductive"),
        exposes=("states",),
    ),
    Question(
        "raw_series", "BN-03", "data", N_("Raw measurements of a point"),
        "The raw measurements of a point folder (CSV exports of the logger software), between two dates: thermometer "
        "temperatures (°C), voltage (V) and temperature of the pressure sensor (°C), and the number of file lines "
        "left out (logger events, empty or unreadable values, duplicate dates). It is about no twin. Formats that are "
        "not supported are refused with a coded error, never read wrongly.",
        ("D", "M"), p.RawSeriesParams, "none",
        lambda twin, a: raw_series(a.point_folder, a.start, a.end),
        outputs=(
            "temperature_time", "temperature", "temperature_dropped", "pressure_time", "voltage", "pressure_temperature",
            "pressure_dropped",
        ),
        exposes=DEV,
    ),
    Question(
        "processing_preview", "BN-04", "data", N_("Effect of a cleaning"),
        "What a cleaning would do to a point, without saving anything: every date common to the temperature and pressure "
        "files, with the measurements (the pressure converted to a differential head, in m, with the calibration of the "
        "laboratory), the reason each date is left out (`outside_window`, `manual`, `missing`, `iqr:<variable>` or "
        "`zscore:<variable>`; empty when it is kept), the count per reason, the dates found in only one file, and the "
        "uncertainties σ of the instruments (Sigma_Meas_P, Sigma_Meas_T). It is about no twin.",
        ("D", "M"), p.ProcessingPreviewParams, "none",
        lambda twin, a: processing_preview(a.point_folder, a.lab_folder, a.start, a.end, a.methods, a.excluded),
        outputs=(
            "time", "temperature", "river_temperature", "pressure", "kept", "reason", "reasons", "reason_counts",
            "kept_count", "total_count", "unmatched_temperature", "unmatched_pressure", "sigma_pressure",
            "sigma_temperature",
        ),
        exposes=DEV,
    ),
)
