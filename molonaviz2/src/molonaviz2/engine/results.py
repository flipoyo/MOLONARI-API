"""Résultats des calculs, sérialisables en tableaux numpy (stockage .npz)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np


@dataclass
class ForwardResult:
    depths: np.ndarray  # profondeur du centre des cellules (m), (C,)
    times: np.ndarray  # datetime64[s], (T,)
    temperature: np.ndarray  # °C, (C, T)
    water_flux: np.ndarray  # débit spécifique vertical (m·s⁻¹), positif vers le bas, (C, T)
    advective_flux: np.ndarray  # flux de chaleur advectif ρw·cw·q·(T − T_ref) (W·m⁻²), positif vers le bas, (C, T)
    conductive_flux: np.ndarray  # flux de chaleur conductif λm·∂T/∂z (W·m⁻²), positif vers le haut comme pyheatmy, (C, T)
    sensor_index: np.ndarray  # indice de cellule des 3 capteurs, (3,)
    observed: np.ndarray  # températures mesurées aux 3 capteurs (°C), (T, 3)
    rmse: np.ndarray  # RMSE par capteur puis total (°C), (4,)

    @property
    def sensor_depths(self) -> np.ndarray:
        return self.depths[self.sensor_index]

    def simulated_at_sensors(self) -> np.ndarray:
        """Températures simulées aux cellules des capteurs (°C), (T, 3)."""
        return self.temperature[self.sensor_index].T

    @property
    def conductive_flux_down(self) -> np.ndarray:
        """Flux conductif −λm·∂T/∂z, positif vers le bas comme le flux advectif (W·m⁻²).

        `conductive_flux` garde le signe de pyheatmy (λm·∂T/∂z, positif vers le haut)."""
        return -self.conductive_flux

    @property
    def total_heat_flux(self) -> np.ndarray:
        return self.advective_flux + self.conductive_flux

    def to_arrays(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {prefix + f.name: np.asarray(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_arrays(cls, arrays, prefix: str = "") -> ForwardResult:
        return cls(**{f.name: np.asarray(arrays[prefix + f.name]) for f in fields(cls)})


@dataclass
class QuantileResult:
    quantile: float
    temperature: np.ndarray  # °C, (C', T')
    water_flux: np.ndarray  # m·s⁻¹, (C', T')
    rmse: np.ndarray  # (4,)


@dataclass
class InversionResult:
    best: ForwardResult  # modèle direct avec les paramètres d'énergie minimale
    best_parameters: np.ndarray  # (L, 4) dans l'espace des lois a priori (−log₁₀ k, n, λs, ρs·cs)
    samples: np.ndarray  # paramètres des itérations gardées après burn-in, (I, N, L, 4)
    energies: np.ndarray  # énergie des itérations gardées, (I, N)
    acceptance: np.ndarray  # taux d'acceptation par chaîne sur les itérations gardées, (N,)
    burn_in_iterations: int
    burn_in_converged: bool
    rhat: np.ndarray  # R-hat final par paramètre (L, 4)
    quantiles: list[QuantileResult] = field(default_factory=list)
    field_depths: np.ndarray | None = None  # profondeurs des champs sous-échantillonnés (C',)
    field_times: np.ndarray | None = None  # dates des champs sous-échantillonnés (T',)
    # Ajouts de 0.4.0 : absents (None, thin = 1) dans un calcul enregistré avant.
    thin: int = 1  # samples[i] est l'itération calculée (i + 1)·thin après le burn-in
    field_chain: np.ndarray | None = None  # chaîne d'origine de chaque champ, (S,)
    field_iteration: np.ndarray | None = None  # itération calculée d'origine, comptée à partir de 1 après le burn-in, (S,)
    resets: np.ndarray | None = None  # réinitialisations de chaque chaîne pendant le burn-in, (N,)
    sensor_temperature: np.ndarray | None = None  # °C, float64, aux 3 capteurs, (S, 3, T')
    field_temperature: np.ndarray | None = None  # °C, float32, (S, C', T')
    field_water_flux: np.ndarray | None = None  # m·s⁻¹, float32, (S, C', T')

    def to_arrays(self) -> dict[str, np.ndarray]:
        arrays = self.best.to_arrays("best.")
        arrays.update(
            best_parameters=self.best_parameters,
            samples=self.samples,
            energies=self.energies,
            acceptance=self.acceptance,
            burn_in_iterations=np.array(self.burn_in_iterations),
            burn_in_converged=np.array(self.burn_in_converged),
            rhat=self.rhat,
            quantile_values=np.array([q.quantile for q in self.quantiles]),
            quantile_temperature=np.stack([q.temperature for q in self.quantiles]),
            quantile_water_flux=np.stack([q.water_flux for q in self.quantiles]),
            quantile_rmse=np.stack([q.rmse for q in self.quantiles]),
            field_depths=self.field_depths,
            field_times=self.field_times,
            thin=np.array(self.thin),
            **{name: getattr(self, name) for name in OPTIONAL_ARRAYS},
        )
        return arrays

    @classmethod
    def from_arrays(cls, arrays) -> InversionResult:
        quantiles = [
            QuantileResult(float(q), np.asarray(temp), np.asarray(flux), np.asarray(rmse))
            for q, temp, flux, rmse in zip(
                arrays["quantile_values"],
                arrays["quantile_temperature"],
                arrays["quantile_water_flux"],
                arrays["quantile_rmse"],
            )
        ]
        return cls(
            best=ForwardResult.from_arrays(arrays, "best."),
            best_parameters=np.asarray(arrays["best_parameters"]),
            samples=np.asarray(arrays["samples"]),
            energies=np.asarray(arrays["energies"]),
            acceptance=np.asarray(arrays["acceptance"]),
            burn_in_iterations=int(arrays["burn_in_iterations"]),
            burn_in_converged=bool(arrays["burn_in_converged"]),
            rhat=np.asarray(arrays["rhat"]),
            quantiles=quantiles,
            field_depths=np.asarray(arrays["field_depths"]),
            field_times=np.asarray(arrays["field_times"]),
            thin=int(arrays.get("thin", 1)),
            **{name: None if arrays.get(name) is None else np.asarray(arrays[name]) for name in OPTIONAL_ARRAYS},
        )


# Tableaux facultatifs d'InversionResult : le stockage n'écrit pas ceux qui valent None.
OPTIONAL_ARRAYS = ("field_chain", "field_iteration", "resets", "sensor_temperature", "field_temperature", "field_water_flux")
