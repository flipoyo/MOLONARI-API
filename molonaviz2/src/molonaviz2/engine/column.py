"""Données d'entrée d'une colonne (point de prélèvement), indépendantes du moteur.

Reprend la construction de `Compute.set_column` de Molonaviz : les 3 premiers
thermomètres sont les capteurs, le 4ᵉ (le plus profond) fixe la température de
l'aquifère, la température du capteur de pression fixe celle de la rivière.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..domain.models import CLEANED_COLUMNS
from ..errors import UserError
from ..i18n import t

ZERO_CELSIUS = 273.15


@dataclass(frozen=True)
class ColumnInputs:
    times: np.ndarray  # datetime64[s], (T,)
    head: np.ndarray  # charge différentielle rivière − aquifère (m), (T,)
    t_river: np.ndarray  # °C, (T,)
    t_sensors: np.ndarray  # °C, (T, 3)
    t_aquifer: np.ndarray  # °C, (T,)
    sensor_depths: tuple[float, float, float, float]  # profondeurs des 4 thermomètres (m)
    offset: float  # décalage des thermomètres par rapport au lit (m)
    sigma_pressure: float
    sigma_temperature: float

    @property
    def real_depths(self) -> np.ndarray:
        """Profondeurs [rivière, capteur 1..3, aquifère] après application de l'offset (m)."""
        return np.array([0.0, *self.sensor_depths]) + np.array([0.0, *([self.offset] * 4)])

    @property
    def column_depth(self) -> float:
        return float(self.real_depths[-1])

    @property
    def nb_times(self) -> int:
        return len(self.times)

    def time_steps(self) -> np.ndarray:
        """Pas de temps (s), (T−1,)."""
        return np.diff(self.times.astype("datetime64[s]").astype(np.int64)).astype(np.float64)

    @classmethod
    def from_cleaned(
        cls,
        cleaned: pd.DataFrame,
        sensor_depths,
        offset: float,
        sigma_pressure: float,
        sigma_temperature: float,
    ) -> ColumnInputs:
        missing = [c for c in CLEANED_COLUMNS if c not in cleaned.columns]
        if missing:
            raise UserError(t("Cleaned measures are missing columns: {cols}.").format(cols=", ".join(missing)))
        df = cleaned.dropna(subset=CLEANED_COLUMNS).sort_values("date")
        if len(df) < 3:
            raise UserError(t("At least 3 cleaned measurement times are required to run a model."))
        times = df["date"].to_numpy().astype("datetime64[s]")
        if np.any(np.diff(times.astype(np.int64)) <= 0):
            raise UserError(t("Measurement times must be strictly increasing (duplicate dates found)."))
        depths = tuple(float(d) for d in sensor_depths)
        if len(depths) != 4 or any(b <= a for a, b in zip(depths, depths[1:])) or depths[0] <= 0:
            raise UserError(t("The shaft must define 4 strictly increasing, positive sensor depths."))
        return cls(
            times=times,
            head=df["pressure"].to_numpy(np.float64),
            t_river=df["temp_bed"].to_numpy(np.float64),
            t_sensors=df[["t1", "t2", "t3"]].to_numpy(np.float64),
            t_aquifer=df["t4"].to_numpy(np.float64),
            sensor_depths=depths,  # type: ignore[arg-type]
            offset=float(offset),
            sigma_pressure=float(sigma_pressure),
            sigma_temperature=float(sigma_temperature),
        )
