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
    advective_flux: np.ndarray  # flux de chaleur advectif vertical (W·m⁻²), (C, T)
    conductive_flux: np.ndarray  # flux de chaleur conductif (W·m⁻²), (C, T)
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
    samples: np.ndarray  # paramètres après burn-in, (I, N, L, 4)
    energies: np.ndarray  # énergie après burn-in, (I, N)
    acceptance: np.ndarray  # taux d'acceptation par chaîne après burn-in, (N,)
    burn_in_iterations: int
    burn_in_converged: bool
    rhat: np.ndarray  # R-hat final par paramètre (L, 4)
    quantiles: list[QuantileResult] = field(default_factory=list)
    field_depths: np.ndarray | None = None  # profondeurs des champs sous-échantillonnés (C',)
    field_times: np.ndarray | None = None  # dates des champs sous-échantillonnés (T',)

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
        )
