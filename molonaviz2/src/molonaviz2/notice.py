"""Notice des paramètres de calcul : une fiche par réglage (Aide › Notice des paramètres).

Les textes sont en anglais (langue source) et traduits à l'affichage avec `t()`. Les
valeurs par défaut et les bornes ne sont pas recopiées : elles sont lues dans le
domaine (`domain.compute`) et le nettoyage (`processing.cleanup`), pour que la notice
reste juste quand ces constantes changent.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields

from .domain.compute import (
    DEFAULT_PRIORS,
    PARAMETER_LABELS,
    PARAMETER_UNITS,
    PHYSICAL_BOUNDS,
    SETTING_LIMITS,
    ForwardConfig,
    InversionConfig,
    LayerSpec,
)
from .i18n import N_, t
from .processing.cleanup import IQR_FACTOR, ZSCORE_LIMIT
from .twin.config import TwinConfig

CLEANING = N_("Cleaning")
LAYERS = N_("Layers")
DIRECT = N_("Direct model")
INVERSION = N_("Inversion")
ADVANCED = N_("Advanced inversion settings")
RESULTS = N_("Results")
TWINS = N_("Twins (twin.yaml)")
SECTIONS = (CLEANING, LAYERS, DIRECT, INVERSION, ADVANCED, RESULTS, TWINS)


def number(value: float) -> str:
    """Comme l'interface : notation scientifique pour les très grandes ou très petites valeurs."""
    if value != 0 and (abs(value) >= 1e5 or abs(value) < 1e-3):
        return f"{value:.4g}"
    return f"{value:.6g}"


def limits_text(low: float | None, high: float | None) -> str:
    if low is not None and high is not None:
        return f"{number(low)} – {number(high)}"
    return f"≥ {number(low)}" if low is not None else f"≤ {number(high)}"


@dataclass(frozen=True)
class Entry:
    key: str
    section: str
    label: str  # nom affiché dans l'interface (traduit à l'affichage ; un symbole reste tel quel)
    summary: str  # une phrase : sert aussi d'infobulle au champ
    details: str = ""  # définition et conseils ; peut contenir des {champs} remplis par `values`
    unit: str = ""
    default: Callable[[], str] | None = None
    limits: Callable[[], str] | None = None
    twin_key: str = ""
    values: Callable[[], dict] = field(default=dict)
    alias: str = ""  # nom en clair à côté d'un symbole (n → porosité)

    def title(self) -> str:
        return t(self.label) + (f" — {t(self.alias)}" if self.alias else "")

    def text(self, which: str) -> str:
        """summary ou details, traduit puis complété par les valeurs du code."""
        source = getattr(self, which)
        return t(source).format(**self.values()) if source else ""


def _inversion_default(name: str) -> Callable[[], str]:
    return lambda: number(getattr(InversionConfig(layers=[]), name))


def _setting_limits(name: str) -> Callable[[], str]:
    return lambda: limits_text(*SETTING_LIMITS[name])


def _layer_default(name: str) -> Callable[[], str]:
    def text() -> str:
        prior = DEFAULT_PRIORS[name]
        return t("{value} (direct model) · prior {low} – {high} (inversion)").format(
            value=number(getattr(LayerSpec("", 1.0), name)), low=number(prior.low), high=number(prior.high))
    return text


def _twin_default(name: str) -> Callable[[], str]:
    return lambda: str(next(f.default for f in fields(TwinConfig) if f.name == name))


def _physical(name: str) -> Callable[[], str]:
    return lambda: t("physical range {range}").format(range=limits_text(*PHYSICAL_BOUNDS[name]))


ENTRIES: tuple[Entry, ...] = (
    # Nettoyage -----------------------------------------------------------------------------
    Entry("cleaning.period", CLEANING, N_("Period (start, end)"),
          N_("Measurements outside this period are left out of every computation."),
          N_("Exclude the installation and the removal of the device: while the probe is out of the riverbed, its temperatures do not "
             "describe the sediment column. “Whole period” resets the start and the end."),
          default=lambda: t("Whole period"), twin_key="data.start, data.end"),
    Entry("cleaning.outliers", CLEANING, N_("Outlier detection"),
          N_("Method that flags outliers, chosen for each variable (pressure, temperatures)."),
          N_("IQR: a value is an outlier outside [Q1 − {iqr}·IQR, Q3 + {iqr}·IQR], with IQR = Q3 − Q1. Z-score: a value is an outlier "
             "when it is more than {z} standard deviations from the mean. Flagged values are removed before the computations; check them "
             "on the preview before applying."),
          default=lambda: t("None"), values=lambda: {"iqr": number(IQR_FACTOR), "z": number(ZSCORE_LIMIT)}),
    Entry("cleaning.exclusions", CLEANING, N_("Manual exclusion"),
          N_("Periods removed by hand: select a period on the graph, adjust the band over the dates, then exclude it."),
          N_("For isolated errors that the automatic methods do not catch, such as a probe that was touched or a logger event.")),
    # Couches -------------------------------------------------------------------------------
    Entry("layers.name", LAYERS, N_("Name"),
          N_("Name of the layer, for display only."),
          N_("It does not enter the twin identifier: two twins that differ only by their layer names are duplicates."),
          twin_key="layers[].name"),
    Entry("layers.z_low", LAYERS, N_("Bottom (m)"),
          N_("Depth of the bottom of the layer, below the riverbed."),
          N_("Bottoms increase from top to bottom. The last layer ends at the column depth (deepest thermometer + offset): its bottom "
             "cannot be edited."),
          unit="m", twin_key="layers[].z_low"),
    Entry("layers.neg_log10_k", LAYERS, PARAMETER_LABELS["neg_log10_k"],
          N_("Intrinsic permeability of the sediment, entered as −log₁₀ k."),
          N_("k is in m²: k = 10⁻¹² m² is entered as 12. A larger value means a less permeable layer and a smaller water flux. Indicative "
             "orders of magnitude: gravel 7–9, sand 9–12, silt 12–15, clay above 15."),
          unit=PARAMETER_UNITS["neg_log10_k"], default=_layer_default("neg_log10_k"), limits=_physical("neg_log10_k"),
          twin_key="layers[].neg_log10_k, mcmc.priors[].neg_log10_k", alias=N_("permeability")),
    Entry("layers.porosity", LAYERS, PARAMETER_LABELS["porosity"],
          N_("Porosity: fraction of the volume occupied by water."),
          N_("It sets how much water the sediment stores, and the share of water in the heat capacity and the conductivity of the medium."),
          unit=PARAMETER_UNITS["porosity"], default=_layer_default("porosity"), limits=_physical("porosity"),
          twin_key="layers[].porosity, mcmc.priors[].porosity", alias=N_("porosity")),
    Entry("layers.lambda_s", LAYERS, PARAMETER_LABELS["lambda_s"],
          N_("Thermal conductivity of the solid grains."),
          N_("The conductivity of the saturated medium is λm = (n·√λw + (1 − n)·√λs)², where λw is that of water."),
          unit=PARAMETER_UNITS["lambda_s"], default=_layer_default("lambda_s"), limits=_physical("lambda_s"),
          twin_key="layers[].lambda_s, mcmc.priors[].lambda_s", alias=N_("grain conductivity")),
    Entry("layers.rhos_cs", LAYERS, PARAMETER_LABELS["rhos_cs"],
          N_("Volumetric heat capacity of the solid grains."),
          N_("The heat capacity of the saturated medium is ρm·cm = n·ρw·cw + (1 − n)·ρs·cs, where ρw·cw is that of water."),
          unit=PARAMETER_UNITS["rhos_cs"], default=_layer_default("rhos_cs"), limits=_physical("rhos_cs"),
          twin_key="layers[].rhos_cs, mcmc.priors[].rhos_cs", alias=N_("grain heat capacity")),
    Entry("layers.priors", LAYERS, N_("Prior ranges (min, max)"),
          N_("In the inversion, each parameter is drawn uniformly between its min and its max."),
          N_("The ranges must stay within the physical range of the parameter. A narrow range speeds up the inversion but excludes the "
             "values outside it: if the posterior piles up against a bound, widen the range."),
          twin_key="mcmc.priors"),
    # Modèle direct ---------------------------------------------------------------------------
    Entry("numerics.nb_cells", DIRECT, N_("Discretisation"),
          N_("Number of cells of the column, from the riverbed to the column depth."),
          N_("Cell size = column depth / number of cells. More cells give a finer solution and a slower computation. The same setting is "
             "in the advanced inversion settings."),
          unit=N_("cells"), default=lambda: number(ForwardConfig(layers=[]).nb_cells), limits=_setting_limits("nb_cells"),
          twin_key="numerics.nb_cells"),
    # Inversion -------------------------------------------------------------------------------
    Entry("inversion.sigma_t", INVERSION, N_("Measurement uncertainty σT"),
          N_("Standard deviation of the temperature measurement error. It sets how closely the model must fit the data."),
          N_("It enters the likelihood: the energy is the sum of the squared residuals at t₁–t₃ divided by 2·σT², plus a constant term. "
             "Too small: few proposals are accepted and the chains get stuck. Too large: wide posteriors that fit the data loosely. By "
             "default, the uncertainty of the thermometer in the laboratory (0.1 °C if none is given). It also sets the measurement "
             "interval in the Results tab."),
          unit="°C", default=lambda: t("thermometer uncertainty, otherwise 0.1 °C"), limits=_setting_limits("sigma_t"),
          twin_key="sensors.sigma_temperature"),
    Entry("inversion.nb_chains", INVERSION, N_("Number of chains"),
          N_("Number of Markov chains run in parallel."),
          N_("DREAM needs at least 2·δ + 1 chains. More chains explore the parameter space better and give more samples, at the cost of "
             "computing time. The Per chain box shows what the settings give."),
          unit=N_("chains"), default=_inversion_default("nb_chains"), limits=_setting_limits("nb_chains"), twin_key="mcmc.nb_chains"),
    Entry("inversion.nb_iter", INVERSION, N_("Iterations kept"),
          N_("Iterations per chain kept after the burn-in: they form the posterior sample."),
          N_("Chains × iterations kept gives the number of parameter sets. Successive iterations are correlated: the ESS of the results "
             "table gives the number of effectively independent samples."),
          unit=N_("iterations"), default=_inversion_default("nb_iter"), limits=_setting_limits("nb_iter"), twin_key="mcmc.nb_iter"),
    Entry("inversion.burn_in_max", INVERSION, N_("Maximum burn-in"),
          N_("Maximum number of iterations per chain before the sampling starts."),
          N_("The burn-in stops early as soon as R-hat is below the Gelman-Rubin threshold for every parameter. Its iterations are not "
             "kept. During the burn-in, the crossover probabilities adapt and outlier chains are moved onto the best chain."),
          unit=N_("iterations"), default=_inversion_default("burn_in_max"), limits=_setting_limits("burn_in_max"),
          twin_key="mcmc.burn_in_max"),
    Entry("inversion.quantiles", INVERSION, N_("Quantiles"),
          N_("Quantiles of the temperature and flux fields, computed over the kept fields."),
          N_("Numbers strictly between 0 and 1, separated by commas. The lowest and the highest quantiles bound the grey credible band "
             "in the Results tab."),
          default=lambda: ", ".join(number(q) for q in InversionConfig(layers=[]).quantiles), twin_key="mcmc.quantiles"),
    Entry("inversion.seed", INVERSION, N_("Random seed"),
          N_("Seed of the random number generator."),
          N_("Same seed, settings, data and code: same result, bit for bit. “New” draws another seed."),
          default=_inversion_default("seed"), limits=_setting_limits("seed"), twin_key="seed"),
    # Réglages avancés ------------------------------------------------------------------------
    Entry("inversion.delta", ADVANCED, N_("Pairs of chains per jump (δ)"),
          N_("Number of pairs of other chains whose differences build a proposal."),
          N_("Jump = (1 + λ)·γ·Σ (xa − xb) + ζ over δ pairs, with γ = 2.38 / √(2·d*·δ), d* being the number of parameters updated; "
             "γ = 1 one time in five, to jump between modes. Requires at least 2·δ + 1 chains."),
          unit=N_("pairs"), default=_inversion_default("delta"), limits=_setting_limits("delta"), twin_key="mcmc.delta"),
    Entry("inversion.n_cr", ADVANCED, N_("Crossover values (nCR)"),
          N_("Number of crossover values CR = 1/nCR, 2/nCR, …, 1."),
          N_("At each jump, each parameter of a layer is updated with probability CR (at least one parameter). During the burn-in, the "
             "probability of each CR adapts to the values that move the chains most."),
          default=_inversion_default("n_cr"), limits=_setting_limits("n_cr"), twin_key="mcmc.n_cr"),
    Entry("inversion.c", ADVANCED, "c",
          N_("Width of the random scaling of a jump: λ is drawn uniformly in [−c, c]."),
          N_("It varies the length of the jumps around the scale γ chosen by DREAM."),
          default=_inversion_default("c"), limits=_setting_limits("c"), twin_key="mcmc.c", alias=N_("jump scaling")),
    Entry("inversion.c_star", ADVANCED, "c*",
          N_("Standard deviation of the small Gaussian noise ζ added to a jump."),
          N_("It keeps the chains from collapsing onto each other. It is expressed in the units of the prior space (−log₁₀ k, n, λs, "
             "ρs·cs)."),
          default=_inversion_default("c_star"), limits=_setting_limits("c_star"), twin_key="mcmc.c_star", alias=N_("jump noise")),
    Entry("inversion.gr_threshold", ADVANCED, N_("Gelman-Rubin threshold"),
          N_("The burn-in stops when R-hat is below this value for every parameter."),
          N_("R-hat compares the variance within the chains with the variance between them (split R-hat). 1.2 is permissive; 1.1 or 1.05 "
             "is stricter. In the results table, R-hat ≤ 1.01 indicates well-mixed chains."),
          default=_inversion_default("gr_threshold"), limits=_setting_limits("gr_threshold"), twin_key="mcmc.gr_threshold"),
    Entry("inversion.n_field_samples", ADVANCED, N_("Fields kept for quantiles"),
          N_("Total number of temperature and flux fields (T, q) kept to compute the quantiles."),
          N_("It is rounded to a whole number per chain, taken at regular intervals among the kept iterations: the Per chain box gives "
             "the exact numbers. More fields give smoother quantiles and use more memory."),
          unit=N_("fields"), default=_inversion_default("n_field_samples"), limits=_setting_limits("n_field_samples"),
          twin_key="mcmc.n_field_samples"),
    Entry("inversion.space_step", ADVANCED, N_("Spatial sub-sampling step"),
          N_("Keep one cell out of k in the stored fields."),
          N_("It reduces the memory used for fine meshes; the quantile fields are coarser. The best fit is always computed and shown on "
             "the full grid."),
          default=_inversion_default("space_step"), limits=_setting_limits("space_step"), twin_key="mcmc.space_step"),
    Entry("inversion.time_step", ADVANCED, N_("Temporal sub-sampling step"),
          N_("Keep one date out of k in the stored fields."),
          N_("It reduces the memory used for long series; the credible bands are drawn on fewer dates."),
          default=_inversion_default("time_step"), limits=_setting_limits("time_step"), twin_key="mcmc.time_step"),
    Entry("inversion.per_chain", ADVANCED, N_("Per chain"),
          N_("What the settings give for each chain: iterations, fields kept, memory and model evaluations."),
          N_("It is computed exactly as the engine does, and updated at every change of a setting.")),
    # Résultats -------------------------------------------------------------------------------
    Entry("results.measurement_level", RESULTS, N_("Measurement interval"),
          N_("Level of the band measurement ± z·σT drawn around the measured temperatures."),
          N_("1σ (68 %): z = 1; 90 %: z = 1.645; 95 %: z = 1.960. σT is that of the inversion, or the thermometer uncertainty for a "
             "direct model. The residual lines of a twin use the same level."),
          default=lambda: "90 %"),
    # Twins -----------------------------------------------------------------------------------
    Entry("twin.mode", TWINS, "mode",
          N_("direct: the soil parameters are fixed in the layers. mcmc: they are estimated by DREAM within the prior ranges "
             "(experimental)."),
          twin_key="mode"),
    Entry("twin.measurements", TWINS, "data.measurements",
          N_("Measurement file of the twin: date, temp_bed, t1, t2, t3, t4, pressure (°C, m)."),
          N_("Its fingerprint enters the twin identifier: changing one value gives another twin."),
          twin_key="data.measurements"),
    Entry("twin.depths", TWINS, "sensors.depths",
          N_("Depths of the four thermometers (m)."),
          N_("t₁–t₃ are compared with the model; the deepest one gives the temperature at the bottom of the column (aquifer)."),
          unit="m", twin_key="sensors.depths"),
    Entry("twin.offset", TWINS, "sensors.offset",
          N_("Offset of the thermometers relative to the riverbed (m), added to every depth."),
          unit="m", default=_twin_default("offset"), twin_key="sensors.offset"),
    Entry("twin.sigma_pressure", TWINS, "sensors.sigma_pressure",
          N_("Uncertainty of the pressure measurement (m)."),
          N_("Stored with the twin. The likelihood uses only the temperatures, so it does not change the result."),
          unit="m", default=_twin_default("sigma_pressure"), twin_key="sensors.sigma_pressure"),
    Entry("twin.initial_condition", TWINS, "numerics.initial_condition",
          N_("Initial temperature profile of the column."),
          N_("Only “linear” is available: linear interpolation of the measurements at the first date."),
          default=_twin_default("initial_condition"), twin_key="numerics.initial_condition"),
    Entry("twin.column_depth", TWINS, "numerics.column_depth",
          N_("Optional check of the column depth (m)."),
          N_("If given, it must equal the deepest thermometer depth plus the offset."),
          unit="m", twin_key="numerics.column_depth"),
    Entry("twin.code_version", TWINS, "code_version",
          N_("Optional: version of the code the twin was defined for."),
          N_("If given and different from the running code, the twin is refused: create a new twin to use this code."),
          twin_key="code_version"),
)

BY_KEY = {entry.key: entry for entry in ENTRIES}
