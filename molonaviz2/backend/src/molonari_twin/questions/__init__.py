"""Questions : réponses calculées à partir d'un résultat (volumes, intervalles, bande prédictive, chaînes, flux,
températures, capteurs, description et provenance d'un twin).

Partie du cœur : ni Qt, ni HTTP, ni phrase traduite. Chaque question renvoie une `Answer`.
"""

from .answer import Answer, Quantity
from .chains import chain_data, posterior_samples
from .convergence import convergence
from .energy import energy_breakdown
from .fit import fit_quality, lag1_autocorrelation
from .flux import exchange_direction, twin_exchanged_volume, water_flux
from .heat import heat_budget
from .intervals import (
    MEASUREMENT_LEVELS,
    measurement_interval,
    sensor_intervals,
    z_score,
)
from .predictive import coverage, observed_at, predictive_band, sensor_samples
from .sensors import residuals, sensor_report
from .soil import soil_properties
from .temperature import temperature_at
from .twin_info import describe, describe_sampling, provenance
from .volumes import exchanged_volume

__all__ = [
    "MEASUREMENT_LEVELS",
    "Answer",
    "Quantity",
    "chain_data",
    "convergence",
    "coverage",
    "describe",
    "describe_sampling",
    "energy_breakdown",
    "exchange_direction",
    "exchanged_volume",
    "fit_quality",
    "heat_budget",
    "lag1_autocorrelation",
    "measurement_interval",
    "observed_at",
    "posterior_samples",
    "predictive_band",
    "provenance",
    "residuals",
    "sensor_intervals",
    "sensor_report",
    "sensor_samples",
    "soil_properties",
    "temperature_at",
    "twin_exchanged_volume",
    "water_flux",
    "z_score",
]
