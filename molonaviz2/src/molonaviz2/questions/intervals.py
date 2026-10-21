"""Intervalle de mesure ± z·σ autour des températures mesurées (déplacé de ui/units.py, qui le réexporte)."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from .answer import Answer, Quantity

MEASUREMENT_LEVELS = (0.6827, 0.90, 0.95)  # 1σ, 90 %, 95 %


def z_score(level: float) -> float:
    """Demi-largeur, en σ, de l'intervalle centré qui contient `level` d'une loi normale (90 % → 1,645)."""
    return float(norm.ppf(0.5 + level / 2))


def measurement_interval(observed_celsius, sigma_celsius: float, level: float):
    """Bornes (°C) de l'intervalle mesure ± z·σT, z venant du niveau."""
    observed = np.asarray(observed_celsius, dtype=float)
    half = z_score(level) * float(sigma_celsius)
    return observed - half, observed + half


def sensor_intervals(observed, sigma: float, level: float) -> Answer:
    """Mesures aux capteurs (°C) avec leur intervalle de mesure ± z·σ au niveau `level`."""
    observed = np.asarray(observed, dtype=float)
    return Answer(
        data={
            "temperature": Quantity(observed, "°C", measurement_interval(observed, sigma, level)),
            "level": Quantity(float(level), "1"),
            "z": Quantity(z_score(level), "1"),
        },
        provenance={"question": "sensor_intervals", "parameters": {"sigma": float(sigma), "level": float(level)}},
    )
