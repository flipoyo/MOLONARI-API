"""Configuration d'un twin (`twin.yaml`) : lecture, validation, forme canonique, identifiant.

Un twin est la copie numérique d'un point de mesure : une colonne 1D de sédiments
entre le fond de la rivière et le thermomètre le plus profond. Il est défini
entièrement par son `twin.yaml` (données d'entrée, paramètres du sol, réglages
numériques, graine, version du code). Même config + mêmes données + même code
donnent le même résultat.

Règles :
- `twin_id` = empreinte (SHA-256) de la config canonique, qui contient l'empreinte
  du fichier de mesures et la version du code. Les libellés (nom du twin,
  description, nom du point, noms des couches) n'y entrent pas : deux twins qui ne
  diffèrent que par leur nom sont des doublons.
- Les clés inconnues sont refusées : une faute de frappe ne doit pas être ignorée en
  silence, sinon le twin ne serait pas celui que l'on croit.

Exemple minimal (mode direct) :

    schema: molonaviz-twin/1
    name: twin_01
    point: Point034
    mode: direct
    data:
      measurements: measurements.csv   # date, temp_bed, t1, t2, t3, t4, pressure (°C, m)
    sensors:
      depths: [0.1, 0.2, 0.3, 0.4]
      offset: 0.0
    layers:
      - {name: couche 1, z_low: 0.4, neg_log10_k: 12, porosity: 0.15, lambda_s: 3.4, rhos_cs: 5.0e6}
    numerics:
      nb_cells: 100
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from ..domain.compute import (
    DEFAULT_PRIORS,
    INVERTED_PARAMETERS,
    ForwardConfig,
    InversionConfig,
    LayerSpec,
    PriorRange,
)
from ..engine.column import ColumnInputs
from ..errors import Issue, UserError
from ..i18n import t

SCHEMA = "molonaviz-twin/1"
CONFIG_NAME = "twin.yaml"
MODES = ("direct", "mcmc")
ID_LENGTH = 16

TOP_KEYS = {"schema", "name", "description", "point", "mode", "data", "sensors", "layers", "numerics", "mcmc", "seed", "code_version"}
REQUIRED_TOP = ("schema", "name", "mode", "data", "sensors", "layers")
DATA_KEYS = {"measurements", "start", "end"}
SENSOR_KEYS = {"depths", "offset", "sigma_temperature", "sigma_pressure"}
LAYER_KEYS = {"name", "z_low", "neg_log10_k", "porosity", "lambda_s", "rhos_cs"}
NUMERIC_KEYS = {"nb_cells", "initial_condition", "column_depth"}
MCMC_KEYS = {"priors", "nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr", "c", "c_star", "gr_threshold", "quantiles",
             "n_field_samples", "space_step", "time_step"}
INITIAL_CONDITIONS = ("linear",)


class TwinConfigError(UserError):
    """Config invalide. `issues` liste les problèmes, chacun préfixé par le chemin de la clé."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("\n".join(issues))


@dataclass
class TwinConfig:
    path: Path  # chemin du twin.yaml
    name: str
    mode: str
    measurements: Path
    depths: tuple[float, float, float, float]
    offset: float = 0.0
    sigma_temperature: float = 0.1
    sigma_pressure: float = 0.01
    layers: list[LayerSpec] = field(default_factory=list)
    nb_cells: int = 100
    initial_condition: str = "linear"
    start: datetime | None = None
    end: datetime | None = None
    mcmc: InversionConfig | None = None
    seed: int = 0
    description: str = ""
    point: str = ""
    declared_code_version: str | None = None

    @property
    def folder(self) -> Path:
        return self.path.parent

    @property
    def column_depth(self) -> float:
        return float(self.depths[-1] + self.offset)

    @property
    def experimental(self) -> bool:
        return self.mode == "mcmc"

    # Données ---------------------------------------------------------------------------
    def load_measurements(self) -> pd.DataFrame:
        from ..processing.cleanup import read_cleaned_csv

        frame = read_cleaned_csv(self.measurements)
        if self.start is not None:
            frame = frame[frame["date"] >= pd.Timestamp(self.start)]
        if self.end is not None:
            frame = frame[frame["date"] <= pd.Timestamp(self.end)]
        return frame.reset_index(drop=True)

    def column_inputs(self) -> ColumnInputs:
        return ColumnInputs.from_cleaned(self.load_measurements(), self.depths, self.offset, self.sigma_pressure, self.sigma_temperature)

    def forward_config(self) -> ForwardConfig:
        return ForwardConfig(self.layers, self.nb_cells)

    def inversion_config(self) -> InversionConfig:
        assert self.mcmc is not None
        return self.mcmc

    # Identité --------------------------------------------------------------------------
    def canonical(self, code_version: str) -> dict:
        """Forme canonique : tout ce qui détermine le résultat, et rien d'autre."""
        data = {
            "measurements_sha256": file_digest(self.measurements),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
        }
        sensors = {"depths": [float(d) for d in self.depths], "offset": float(self.offset), "sigma_pressure": float(self.sigma_pressure)}
        canonical = {
            "schema": SCHEMA,
            "mode": self.mode,
            "data": data,
            "sensors": sensors,
            "numerics": {"nb_cells": int(self.nb_cells), "initial_condition": self.initial_condition},
            "code_version": self.declared_code_version or code_version,
        }
        if self.mode == "direct":
            canonical["layers"] = [
                {k: float(getattr(layer, k)) for k in ("z_low", "neg_log10_k", "porosity", "lambda_s", "rhos_cs")}
                for layer in self.layers
            ]
        else:
            sensors["sigma_temperature"] = float(self.sigma_temperature)
            mcmc = self.mcmc
            canonical["layers"] = [{"z_low": float(layer.z_low)} for layer in self.layers]
            canonical["mcmc"] = {
                "priors": [{name: [float(p[name].low), float(p[name].high)] for name in INVERTED_PARAMETERS} for p in mcmc.priors],
                **{key: getattr(mcmc, key) for key in ("nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr", "n_field_samples",
                                                         "space_step", "time_step")},
                **{key: float(getattr(mcmc, key)) for key in ("c", "c_star", "gr_threshold")},
                "quantiles": [float(q) for q in mcmc.quantiles],
            }
            canonical["seed"] = int(self.seed)
        return canonical

    def twin_id(self, code_version: str) -> str:
        text = json.dumps(self.canonical(code_version), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(text.encode("ascii")).hexdigest()[:ID_LENGTH]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def config_path(path: str | Path) -> Path:
    """Accepte le dossier twin_xx ou directement son twin.yaml."""
    path = Path(path)
    return path / CONFIG_NAME if path.is_dir() else path


# Validation ---------------------------------------------------------------------------


class _Checker:
    def __init__(self):
        self.issues: list[str] = []

    def add(self, where: str, message: str) -> None:
        self.issues.append(f"{where}: {message}" if where else message)

    def mapping(self, value, where: str, allowed: set[str], required=()) -> dict:
        if not isinstance(value, dict):
            self.add(where, t("expected a mapping (key: value)."))
            return {}
        for key in sorted(set(value) - allowed, key=str):
            self.add(f"{where}.{key}" if where else str(key), t("unknown key (allowed: {keys}).").format(keys=", ".join(sorted(allowed))))
        for key in required:
            if key not in value:
                self.add(f"{where}.{key}" if where else key, t("required."))
        return value

    def number(self, value, where: str, integer: bool = False, positive: bool = False, minimum: float | None = None):
        if isinstance(value, bool) or not isinstance(value, int | float):
            self.add(where, t("expected a whole number.") if integer else t("expected a number."))
            return None
        if integer and value != int(value):
            self.add(where, t("expected a whole number."))
            return None
        if positive and value <= 0:
            self.add(where, t("must be positive."))
            return None
        if minimum is not None and value < minimum:
            self.add(where, t("must be at least {v}.").format(v=minimum))
            return None
        return int(value) if integer else float(value)

    def date(self, value, where: str):
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        from ..io.dates import parse_date

        try:
            return parse_date(str(value))
        except UserError:
            self.add(where, t("unreadable date \"{value}\".").format(value=value))
            return None


def load_twin(path: str | Path, check_data: bool = True) -> TwinConfig:
    """Lit et valide un twin. Lève TwinConfigError avec la liste complète des problèmes."""
    path = config_path(path)
    if not path.is_file():
        raise TwinConfigError([t("File not found: {path}").format(path=path)])
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TwinConfigError([t("Invalid YAML: {error}").format(error=exc)]) from None
    check = _Checker()
    doc = check.mapping(raw, "", TOP_KEYS, REQUIRED_TOP)
    if check.issues and not doc:
        raise TwinConfigError(check.issues)

    if doc.get("schema") not in (None, SCHEMA):
        check.add("schema", t("unsupported schema \"{value}\" (expected {expected}).").format(value=doc.get("schema"), expected=SCHEMA))
    name = str(doc.get("name", "")).strip()
    if "name" in doc and not name:
        check.add("name", t("must not be empty."))
    mode = doc.get("mode")
    if "mode" in doc and mode not in MODES:
        check.add("mode", t("must be \"direct\" or \"mcmc\"."))

    data = check.mapping(doc.get("data", {}), "data", DATA_KEYS, ("measurements",))
    measurements = None
    if "measurements" in data:
        measurements = (path.parent / str(data["measurements"])).resolve()
        if not measurements.is_file():
            check.add("data.measurements", t("file not found: {path}").format(path=measurements))
            measurements = None
    start, end = check.date(data.get("start"), "data.start"), check.date(data.get("end"), "data.end")
    if start and end and start >= end:
        check.add("data.end", t("must be after data.start."))

    sensors = check.mapping(doc.get("sensors", {}), "sensors", SENSOR_KEYS, ("depths",))
    depths = sensors.get("depths")
    if "depths" in sensors:
        values = [check.number(d, "sensors.depths", positive=True) for d in depths] if isinstance(depths, list) else []
        if len(values) != 4 or None in values or any(b <= a for a, b in zip(values, values[1:])):
            check.add("sensors.depths", t("expected 4 strictly increasing positive depths (m)."))
            depths = None
        else:
            depths = tuple(values)
    offset = check.number(sensors.get("offset", 0.0), "sensors.offset")
    sigma_t = check.number(sensors.get("sigma_temperature", 0.1), "sensors.sigma_temperature", positive=True)
    sigma_p = check.number(sensors.get("sigma_pressure", 0.01), "sensors.sigma_pressure", positive=True)

    numerics = check.mapping(doc.get("numerics", {}) or {}, "numerics", NUMERIC_KEYS)
    nb_cells = check.number(numerics.get("nb_cells", 100), "numerics.nb_cells", integer=True, positive=True)
    initial = numerics.get("initial_condition", "linear")
    if initial not in INITIAL_CONDITIONS:
        check.add("numerics.initial_condition", t("only \"linear\" is supported (linear interpolation of the measurements at t0)."))
    column_depth = None
    if depths is not None and offset is not None:
        column_depth = depths[-1] + offset
        if "column_depth" in numerics:
            declared = check.number(numerics["column_depth"], "numerics.column_depth", positive=True)
            if declared is not None and abs(declared - column_depth) > 1e-9:
                check.add("numerics.column_depth", t("must equal the deepest sensor depth + offset ({depth:g} m).").format(depth=column_depth))

    layers: list[LayerSpec] = []
    raw_layers = doc.get("layers")
    if "layers" in doc:
        if not isinstance(raw_layers, list) or not raw_layers:
            check.add("layers", t("expected a non-empty list of layers."))
            raw_layers = []
        for i, raw_layer in enumerate(raw_layers):
            where = f"layers[{i}]"
            layer = check.mapping(raw_layer, where, LAYER_KEYS, ("z_low",))
            values = {"name": str(layer.get("name", f"layer {i + 1}"))}
            for key in ("z_low", "neg_log10_k", "porosity", "lambda_s", "rhos_cs"):
                if key in layer:
                    number = check.number(layer[key], f"{where}.{key}")
                    if number is not None:
                        values[key] = number
            if mode == "direct":
                missing = [k for k in INVERTED_PARAMETERS if k not in layer]
                if missing:
                    check.add(where, t("direct mode needs {keys}.").format(keys=", ".join(missing)))
            if "z_low" in values:
                layers.append(LayerSpec(**values))

    seed = check.number(doc.get("seed", 0), "seed", integer=True, minimum=0)
    mcmc = None
    if mode == "mcmc":
        raw_mcmc = check.mapping(doc.get("mcmc", {}) or {}, "mcmc", MCMC_KEYS)
        priors = []
        raw_priors = raw_mcmc.get("priors")
        if raw_priors is None:
            priors = [{k: PriorRange(v.low, v.high) for k, v in DEFAULT_PRIORS.items()} for _ in layers]
        elif not isinstance(raw_priors, list) or len(raw_priors) != len(layers):
            check.add("mcmc.priors", t("expected one entry per layer ({n}).").format(n=len(layers)))
        else:
            for i, raw_prior in enumerate(raw_priors):
                where = f"mcmc.priors[{i}]"
                entry = check.mapping(raw_prior, where, set(INVERTED_PARAMETERS))
                prior = {}
                for param in INVERTED_PARAMETERS:
                    bounds = entry.get(param, [DEFAULT_PRIORS[param].low, DEFAULT_PRIORS[param].high])
                    if not isinstance(bounds, list) or len(bounds) != 2:
                        check.add(f"{where}.{param}", t("expected [min, max]."))
                        bounds = [DEFAULT_PRIORS[param].low, DEFAULT_PRIORS[param].high]
                    low, high = (check.number(b, f"{where}.{param}") for b in bounds)
                    prior[param] = PriorRange(low if low is not None else 0.0, high if high is not None else 0.0)
                priors.append(prior)
        settings = {}
        for key in ("nb_chains", "nb_iter", "burn_in_max", "delta", "n_cr", "n_field_samples", "space_step", "time_step"):
            if key in raw_mcmc:
                value = check.number(raw_mcmc[key], f"mcmc.{key}", integer=True, positive=True)
                if value is not None:
                    settings[key] = value
        for key in ("c", "c_star", "gr_threshold"):
            if key in raw_mcmc:
                value = check.number(raw_mcmc[key], f"mcmc.{key}", positive=True)
                if value is not None:
                    settings[key] = value
        if "quantiles" in raw_mcmc:
            quantiles = raw_mcmc["quantiles"]
            if not isinstance(quantiles, list) or not quantiles:
                check.add("mcmc.quantiles", t("expected a list of numbers between 0 and 1."))
            else:
                settings["quantiles"] = tuple(check.number(q, "mcmc.quantiles") or 0.0 for q in quantiles)
        if layers and len(priors) == len(layers):
            mcmc = InversionConfig(layers=layers, priors=priors, nb_cells=nb_cells or 100, sigma_t=sigma_t or 0.1, seed=seed or 0, **settings)
    elif "mcmc" in doc:
        check.add("mcmc", t("only allowed in mcmc mode."))

    # Validation métier commune avec l'application (bornes physiques, couches, DREAM).
    if layers and column_depth is not None and nb_cells:
        engine_config = mcmc if mcmc is not None else ForwardConfig(layers, nb_cells)
        for issue in engine_config.validate(column_depth):
            check.issues.append(Issue.from_error(issue.field, issue, prefixed=True))

    declared_version = doc.get("code_version")
    if declared_version is not None:
        from ..engine import api

        if str(declared_version) != api.engine_version():
            check.add("code_version", t("this twin was defined for \"{declared}\" but the code is \"{current}\". Create a new twin to use this code.").format(
                declared=declared_version, current=api.engine_version()))

    config = None
    if not check.issues:
        config = TwinConfig(
            path=path, name=name, mode=mode, measurements=measurements, depths=depths, offset=offset,
            sigma_temperature=sigma_t, sigma_pressure=sigma_p, layers=layers, nb_cells=nb_cells, initial_condition=initial,
            start=start, end=end, mcmc=mcmc, seed=seed, description=str(doc.get("description", "")), point=str(doc.get("point", "")),
            declared_code_version=str(declared_version) if declared_version is not None else None,
        )
        if check_data:
            try:
                frame = config.load_measurements()
                if len(frame) < 3:
                    check.add("data", t("fewer than 3 measurements in the selected period."))
            except (UserError, ValueError, OSError) as exc:
                check.add("data.measurements", str(exc))
    if check.issues:
        raise TwinConfigError(check.issues)
    return config


def write_twin_yaml(path: Path, document: dict) -> None:
    """Écrit un twin.yaml lisible (ordre des clés conservé)."""
    header = "# Config figée d'un twin Molonaviz. Pour changer un paramètre, créez un nouveau twin.\n"
    path.write_text(header + yaml.safe_dump(document, sort_keys=False, allow_unicode=True, default_flow_style=None, width=120), encoding="utf-8")
