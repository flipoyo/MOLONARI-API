"""Écriture et relecture des résultats d'un twin (`results/<twin_id>/`).

Fichiers du format 1.x (inchangés en 2.0.0) :
- temperature.csv   θ(z, t) en °C : une ligne par profondeur (m), une colonne par date ;
- water_flux.csv    q(t) en m/s au fond de la rivière (positif vers le bas : infiltration,
                    la rivière alimente la nappe ; négatif : exfiltration), moyenne sur la
                    colonne, et quantiles pour le mode mcmc ;
- residuals.csv     t₁–t₃ mesurés, simulés et résidus (simulé − mesuré), en °C ;
- metadata.json     identité, config canonique, empreinte des données, version du code,
                    appareil, durée, RMSE, diagnostic mcmc ;
- mcmc uniquement : posterior_summary.csv et posterior_samples.csv.

Ajouts du format 2.0.0 (metadata.json : results_format, instruments, conventions) :
- water_flux_field.csv, heat_flux_advective.csv, heat_flux_conductive.csv : q(z, t) en m/s
  et flux de chaleur en W/m², au format de temperature.csv ; la première cellule donne
  la profondeur et l'unité (depth_m|m_per_s, depth_m|W_per_m2). Tous positifs vers le bas.
- mcmc uniquement, tableaux .npz sans objet Python (lisibles avec allow_pickle=False) :
  quantiles.npz (quantiles des champs), posterior.npz (échantillons, énergies, acceptation,
  réinitialisations, numéro d'itération calculée), et selon store_fields fields_temperature.npz
  et fields_water_flux.npz (champs avec leur chaîne et leur itération d'origine).

Les fichiers de données ne dépendent que du calcul : deux exécutions d'un même twin
donnent des fichiers identiques octet pour octet (metadata.json contient en plus la
date et la durée de l'exécution).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.compute import INVERTED_PARAMETERS
from ..engine.results import ForwardResult, InversionResult
from ..errors import Issue
from .provenance import provenance_warnings

FLOAT_FORMAT = "%.17g"
RESULTS_FORMAT = "2.0.0"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _dates(values) -> list[str]:
    return pd.to_datetime(values).strftime(DATE_FORMAT).tolist()


def _write_field(path: Path, depths, dates: list[str], values, index_name: str) -> None:
    pd.DataFrame(values, index=pd.Index(depths, name=index_name), columns=dates).to_csv(path, float_format=FLOAT_FORMAT)


def write_results(folder: Path, forward: ForwardResult, inversion: InversionResult | None, layer_names: list[str], metadata: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    dates = _dates(forward.times)
    _write_field(folder / "temperature.csv", forward.depths, dates, forward.temperature, "depth_m")
    _write_field(folder / "water_flux_field.csv", forward.depths, dates, forward.water_flux, "depth_m|m_per_s")
    _write_field(folder / "heat_flux_advective.csv", forward.depths, dates, forward.advective_flux, "depth_m|W_per_m2")
    _write_field(folder / "heat_flux_conductive.csv", forward.depths, dates, forward.conductive_flux_down, "depth_m|W_per_m2")

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
        _write_arrays(folder, forward, inversion, layer_names)

    (folder / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _write_arrays(folder: Path, forward: ForwardResult, inversion: InversionResult, layer_names: list[str]) -> None:
    """Tableaux du format 2.0.0 (mcmc) : np.savez, chaînes en <U et dates en datetime64[s], sans pickle."""
    times = np.asarray(inversion.field_times).astype("datetime64[s]")
    depths = np.asarray(inversion.field_depths, np.float64)
    quantiles = inversion.quantiles
    np.savez(
        folder / "quantiles.npz",
        quantile=np.array([q.quantile for q in quantiles], np.float64),
        depth_m=depths,
        time=times,
        temperature_C=np.stack([q.temperature for q in quantiles]),
        water_flux_m_per_s=np.stack([q.water_flux for q in quantiles]),
    )
    kept, chains = inversion.energies.shape
    np.savez(
        folder / "posterior.npz",
        samples=inversion.samples,
        parameter=np.array(INVERTED_PARAMETERS),
        layer=np.array(layer_names),
        energy=inversion.energies,
        acceptance=inversion.acceptance,
        resets=inversion.resets if inversion.resets is not None else np.zeros(chains, np.int64),
        iteration=(np.arange(kept) + 1) * inversion.thin,  # itération calculée après le burn-in
    )
    if inversion.field_chain is None:
        return
    provenance = {"chain": inversion.field_chain, "iteration": inversion.field_iteration, "time": times}
    if inversion.sensor_temperature is not None:
        np.savez(folder / "fields_temperature.npz", values=inversion.sensor_temperature, depth_m=forward.sensor_depths, **provenance)
    if inversion.field_temperature is not None:
        np.savez(folder / "fields_temperature.npz", values=inversion.field_temperature, depth_m=depths, **provenance)
    if inversion.field_water_flux is not None:
        np.savez(folder / "fields_water_flux.npz", values=inversion.field_water_flux, depth_m=depths, **provenance)


@dataclass
class TwinResults:
    folder: Path
    depths: np.ndarray
    times: np.ndarray
    temperature: np.ndarray  # (profondeurs, dates), °C
    flux: pd.DataFrame
    residuals: pd.DataFrame
    metadata: dict
    # Format 2.0.0 : None (ou vide) pour un résultat 1.x, qui porte alors l'avertissement W_FIELD_UNAVAILABLE.
    results_format: str = "1"
    water_flux_field: np.ndarray | None = None  # (profondeurs, dates), m/s, positif vers le bas
    heat_flux_advective: np.ndarray | None = None  # (profondeurs, dates), W/m², positif vers le bas
    heat_flux_conductive: np.ndarray | None = None  # (profondeurs, dates), W/m², positif vers le bas
    quantiles: dict[str, np.ndarray] | None = None  # quantiles.npz (mcmc)
    posterior: dict[str, np.ndarray] | None = None  # posterior.npz (mcmc)
    fields: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)  # "temperature", "water_flux" : fields_*.npz
    warnings: list[Issue] = field(default_factory=list)

    @property
    def sensor_depths(self) -> list[float]:
        return self.metadata.get("sensor_cell_depths", [])


def load_results(folder: Path) -> TwinResults:
    folder = Path(folder)
    temperature = pd.read_csv(folder / "temperature.csv", index_col=0)
    flux = pd.read_csv(folder / "water_flux.csv", parse_dates=["date"])
    residuals = pd.read_csv(folder / "residuals.csv", parse_dates=["date"])
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    results_format = str(metadata.get("results_format", "1"))
    fields = {kind: _read_arrays(folder / f"fields_{kind}.npz") for kind in ("temperature", "water_flux")}
    return TwinResults(
        folder=folder,
        depths=temperature.index.to_numpy(float),
        times=pd.to_datetime(temperature.columns).to_numpy(),
        temperature=temperature.to_numpy(float),
        flux=flux,
        residuals=residuals,
        metadata=metadata,
        results_format=results_format,
        water_flux_field=_read_field(folder / "water_flux_field.csv"),
        heat_flux_advective=_read_field(folder / "heat_flux_advective.csv"),
        heat_flux_conductive=_read_field(folder / "heat_flux_conductive.csv"),
        quantiles=_read_arrays(folder / "quantiles.npz"),
        posterior=_read_arrays(folder / "posterior.npz"),
        fields={kind: arrays for kind, arrays in fields.items() if arrays is not None},
        warnings=([] if results_format.startswith("2.") else [Issue(None, "W_FIELD_UNAVAILABLE")])
        + provenance_warnings(metadata.get("provenance")),
    )


def _read_field(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return pd.read_csv(path, index_col=0, float_precision="round_trip").to_numpy(float)


def _read_arrays(path: Path) -> dict[str, np.ndarray] | None:
    """Tableaux d'un .npz, copiés en mémoire ; allow_pickle=False refuse tout objet Python."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
