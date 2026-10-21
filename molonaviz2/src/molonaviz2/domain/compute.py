"""Configuration des calculs : couches, modèle direct, inversion bayésienne (DREAM).

L'utilisateur saisit la perméabilité sous forme de −log₁₀ k (k en m²), comme dans
Molonaviz ; le moteur reçoit k. Les valeurs par défaut reprennent celles de
Molonaviz (frontend/dialogCompute.py).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..errors import Issue  # aussi importable d'ici (ancien emplacement)
from ..i18n import N_

# Paramètres estimés par l'inversion, dans l'espace où les lois a priori sont uniformes.
INVERTED_PARAMETERS = ("neg_log10_k", "porosity", "lambda_s", "rhos_cs")

PARAMETER_LABELS = {
    "neg_log10_k": "−log₁₀ k",
    "porosity": "n",
    "lambda_s": "λs",
    "rhos_cs": "ρs·cs",
}
PARAMETER_UNITS = {
    "neg_log10_k": N_("k in m²"),
    "porosity": "–",
    "lambda_s": "W·m⁻¹·K⁻¹",
    "rhos_cs": "J·m⁻³·K⁻¹",
}


@dataclass
class LayerSpec:
    name: str
    z_low: float  # profondeur du bas de la couche (m)
    neg_log10_k: float = 12.0
    porosity: float = 0.15
    lambda_s: float = 3.4
    rhos_cs: float = 5e6

    @property
    def permeability(self) -> float:
        return 10.0 ** (-self.neg_log10_k)

    def physical(self) -> tuple[float, float, float, float]:
        """Paramètres dans l'ordre attendu par le moteur : (k, n, λs, ρs·cs)."""
        return (self.permeability, self.porosity, self.lambda_s, self.rhos_cs)

    @classmethod
    def from_dict(cls, data: dict) -> LayerSpec:
        """Couche relue d'un projet. Les clés q_s et t_lateral des projets plus anciens
        (échange latéral, pas pris en charge dans cette version) sont ignorées."""
        return cls(**{k: v for k, v in data.items() if k not in ("q_s", "t_lateral")})


@dataclass
class PriorRange:
    low: float
    high: float


DEFAULT_PRIORS = {
    "neg_log10_k": PriorRange(11.0, 15.0),
    "porosity": PriorRange(0.01, 0.25),
    "lambda_s": PriorRange(1.0, 5.0),
    "rhos_cs": PriorRange(1e6, 1e7),
}

# Bornes physiques admissibles (au-delà, le modèle n'a pas de sens).
PHYSICAL_BOUNDS = {
    "neg_log10_k": (3.0, 20.0),
    "porosity": (1e-4, 0.9999),
    "lambda_s": (1e-3, 100.0),
    "rhos_cs": (1e4, 1e8),
}


# Bornes admises pour les réglages saisis dans l'interface : (minimum, maximum), None = sans borne.
# Les onglets de calcul et la notice des paramètres les lisent ici.
SETTING_LIMITS: dict[str, tuple[float | None, float | None]] = {
    "nb_cells": (10, 2000),
    "sigma_t": (1e-6, None),
    "nb_chains": (3, 200),
    "nb_iter": (1, 1_000_000),
    "burn_in_max": (2, 1_000_000),
    "seed": (0, None),
    "delta": (1, 20),
    "n_cr": (1, 20),
    "c": (1e-9, 0.999),
    "c_star": (1e-15, None),
    "gr_threshold": (1.0001, None),
    "n_field_samples": (2, 5000),
    "space_step": (1, None),
    "time_step": (1, None),
}


def setting_limits(name: str) -> dict[str, float | None]:
    """Bornes d'un réglage, sous la forme attendue par NumberField (minimum=…, maximum=…)."""
    low, high = SETTING_LIMITS[name]
    return {"minimum": low, "maximum": high}


def default_priors() -> dict[str, PriorRange]:
    return {name: PriorRange(p.low, p.high) for name, p in DEFAULT_PRIORS.items()}


def _validate_layers(layers: list[LayerSpec], column_depth: float | None) -> list[Issue]:
    issues: list[Issue] = []
    if not layers:
        return [Issue("layers", "E_LAYERS_REQUIRED")]
    previous = 0.0
    for i, layer in enumerate(layers):
        prefix = f"layers[{i}]"
        if layer.z_low <= previous:
            issues.append(Issue(f"{prefix}.z_low", "E_LAYERS_ORDER"))
        previous = layer.z_low
        for name in INVERTED_PARAMETERS:
            low, high = PHYSICAL_BOUNDS[name]
            value = getattr(layer, name)
            if not low <= value <= high:
                issues.append(Issue(f"{prefix}.{name}", "E_VALUE_OUT_OF_RANGE", low=low, high=high))
    if column_depth is not None and abs(layers[-1].z_low - column_depth) > 1e-9:
        issues.append(Issue(f"layers[{len(layers) - 1}].z_low", "E_LAYERS_COLUMN_DEPTH", depth=column_depth))
    return issues


@dataclass
class ForwardConfig:
    layers: list[LayerSpec]
    nb_cells: int = 100

    def validate(self, column_depth: float | None = None) -> list[Issue]:
        issues = _validate_layers(self.layers, column_depth)
        if not 10 <= self.nb_cells <= 2000:
            issues.append(Issue("nb_cells", "E_NB_CELLS_RANGE"))
        return issues

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ForwardConfig:
        return cls(layers=[LayerSpec.from_dict(layer) for layer in data["layers"]], nb_cells=data["nb_cells"])


@dataclass
class InversionConfig:
    layers: list[LayerSpec]  # géométrie (z_low) ; les autres valeurs sont estimées
    priors: list[dict[str, PriorRange]] = field(default_factory=list)
    nb_cells: int = 100
    sigma_t: float = 0.1  # incertitude des mesures de température (°C), fixe la vraisemblance
    nb_chains: int = 10
    nb_iter: int = 1000  # itérations conservées après le burn-in
    burn_in_max: int = 1000  # le burn-in s'arrête plus tôt si Gelman-Rubin < seuil
    delta: int = 3  # nombre de paires de chaînes utilisées pour la proposition
    n_cr: int = 3  # nombre de valeurs de crossover
    c: float = 0.1
    c_star: float = 1e-6
    gr_threshold: float = 1.2
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    n_field_samples: int = 200  # nombre total de champs (T, q) conservés pour les quantiles
    space_step: int = 1  # sous-échantillonnage spatial des champs conservés
    time_step: int = 1  # sous-échantillonnage temporel des champs conservés
    seed: int = 0

    def __post_init__(self):
        if not self.priors:
            self.priors = [default_priors() for _ in self.layers]

    def validate(self, column_depth: float | None = None) -> list[Issue]:
        issues = _validate_layers(self.layers, column_depth)
        if len(self.priors) != len(self.layers):
            issues.append(Issue("priors", "E_PRIORS_PER_LAYER"))
        for i, priors in enumerate(self.priors):
            for name in INVERTED_PARAMETERS:
                prior = priors.get(name)
                key = f"priors[{i}].{name}"
                if prior is None:
                    issues.append(Issue(key, "E_PRIOR_MISSING"))
                    continue
                low, high = PHYSICAL_BOUNDS[name]
                if not prior.low < prior.high:
                    issues.append(Issue(key, "E_PRIOR_ORDER"))
                elif prior.low < low or prior.high > high:
                    issues.append(Issue(key, "E_PRIOR_OUT_OF_BOUNDS", low=low, high=high))
        if not 10 <= self.nb_cells <= 2000:
            issues.append(Issue("nb_cells", "E_NB_CELLS_RANGE"))
        if self.sigma_t <= 0:
            issues.append(Issue("sigma_t", "E_SIGMA_T_POSITIVE"))
        if self.delta < 1:
            issues.append(Issue("delta", "E_MCMC_DELTA"))
        if self.nb_chains < 2 * self.delta + 1:
            issues.append(Issue("nb_chains", "E_MCMC_NB_CHAINS", n=2 * self.delta + 1))
        if self.nb_iter < 1:
            issues.append(Issue("nb_iter", "E_MCMC_NB_ITER"))
        if self.burn_in_max < 2:
            issues.append(Issue("burn_in_max", "E_MCMC_BURN_IN"))
        if self.n_cr < 1:
            issues.append(Issue("n_cr", "E_MCMC_N_CR"))
        if not 0 < self.c < 1:
            issues.append(Issue("c", "E_MCMC_C"))
        if self.c_star <= 0:
            issues.append(Issue("c_star", "E_MCMC_C_STAR"))
        if self.gr_threshold <= 1:
            issues.append(Issue("gr_threshold", "E_MCMC_GR_THRESHOLD"))
        if not self.quantiles or any(not 0 < q < 1 for q in self.quantiles):
            issues.append(Issue("quantiles", "E_MCMC_QUANTILES"))
        if self.n_field_samples < 2:
            issues.append(Issue("n_field_samples", "E_MCMC_FIELD_SAMPLES"))
        if self.space_step < 1 or self.time_step < 1:
            issues.append(Issue("space_step", "E_MCMC_SUBSAMPLING"))
        return issues

    def to_dict(self) -> dict:
        data = asdict(self)
        data["quantiles"] = list(self.quantiles)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> InversionConfig:
        data = dict(data)
        data["layers"] = [LayerSpec.from_dict(layer) for layer in data["layers"]]
        data["priors"] = [{name: PriorRange(**p) for name, p in priors.items()} for priors in data["priors"]]
        data["quantiles"] = tuple(data["quantiles"])
        return cls(**data)
