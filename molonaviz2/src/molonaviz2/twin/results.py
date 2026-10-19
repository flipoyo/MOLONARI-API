"""Écriture et relecture des résultats d'un twin (`results/<twin_id>/`).

Fichiers :
- temperature.csv   θ(z, t) en °C : une ligne par profondeur (m), une colonne par date ;
- water_flux.csv    q(t) en m/s au fond de la rivière (positif vers le bas : infiltration,
                    la rivière alimente la nappe ; négatif : exfiltration), moyenne sur la
                    colonne, et quantiles pour le mode mcmc ;
- residuals.csv     t₁–t₃ mesurés, simulés et résidus (simulé − mesuré), en °C ;
- metadata.json     identité, config canonique, empreinte des données, version du code,
                    appareil, durée, RMSE, diagnostic mcmc ;
- mcmc uniquement : posterior_summary.csv et posterior_samples.csv.

Les fichiers de données ne dépendent que du calcul : deux exécutions d'un même twin
donnent des fichiers identiques octet pour octet (metadata.json contient en plus la
date et la durée de l'exécution).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.compute import INVERTED_PARAMETERS
from ..engine.results import ForwardResult, InversionResult

FLOAT_FORMAT = "%.17g"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _dates(values) -> list[str]:
    return pd.to_datetime(values).strftime(DATE_FORMAT).tolist()


def write_results(folder: Path, forward: ForwardResult, inversion: InversionResult | None, layer_names: list[str], metadata: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    dates = _dates(forward.times)
    pd.DataFrame(forward.temperature, index=pd.Index(forward.depths, name="depth_m"), columns=dates).to_csv(
        folder / "temperature.csv", float_format=FLOAT_FORMAT)

    flux = pd.DataFrame({
        "date": dates,
        "q_riverbed_m_per_s": forward.water_flux[0],
        "q_column_mean_m_per_s": forward.water_flux.mean(axis=0),
    })
    flux["direction"] = np.where(flux["q_riverbed_m_per_s"] >= 0, "infiltration", "exfiltration")
    if inversion is not None and inversion.quantiles:
        field_dates = _dates(inversion.field_times)
        for quantile in inversion.quantiles:
            series = pd.Series(quantile.water_flux[0], index=field_dates)
            flux[f"q_riverbed_q{quantile.quantile:g}"] = flux["date"].map(series)
    flux.to_csv(folder / "water_flux.csv", index=False, float_format=FLOAT_FORMAT)

    simulated = forward.simulated_at_sensors()
    residuals = pd.DataFrame({"date": dates})
    for i in range(3):
        residuals[f"t{i + 1}_measured"] = forward.observed[:, i]
        residuals[f"t{i + 1}_simulated"] = simulated[:, i]
        residuals[f"t{i + 1}_residual"] = simulated[:, i] - forward.observed[:, i]
    residuals.to_csv(folder / "residuals.csv", index=False, float_format=FLOAT_FORMAT)

    if inversion is not None:
        from ..engine.diagnostics import summary_table

        summary_table(inversion, layer_names).to_csv(folder / "posterior_summary.csv", float_format=FLOAT_FORMAT)
        flat = inversion.samples.reshape(-1, inversion.samples.shape[2], inversion.samples.shape[3])
        samples = {f"{name}[{layer}]": flat[:, i, k] for i, layer in enumerate(layer_names) for k, name in enumerate(INVERTED_PARAMETERS)}
        pd.DataFrame(samples).to_csv(folder / "posterior_samples.csv", index=False, float_format=FLOAT_FORMAT)

    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


@dataclass
class TwinResults:
    folder: Path
    depths: np.ndarray
    times: np.ndarray
    temperature: np.ndarray  # (profondeurs, dates), °C
    flux: pd.DataFrame
    residuals: pd.DataFrame
    metadata: dict

    @property
    def sensor_depths(self) -> list[float]:
        return self.metadata.get("sensor_cell_depths", [])


def load_results(folder: Path) -> TwinResults:
    folder = Path(folder)
    temperature = pd.read_csv(folder / "temperature.csv", index_col=0)
    flux = pd.read_csv(folder / "water_flux.csv", parse_dates=["date"])
    residuals = pd.read_csv(folder / "residuals.csv", parse_dates=["date"])
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    return TwinResults(
        folder=folder,
        depths=temperature.index.to_numpy(float),
        times=pd.to_datetime(temperature.columns).to_numpy(),
        temperature=temperature.to_numpy(float),
        flux=flux,
        residuals=residuals,
        metadata=metadata,
    )
