"""Nettoyage des mesures brutes d'un point (portage de frontend/dialogsCleanup.py).

1. Les mesures de température (tige) et de pression sont associées par date ;
   seules les dates présentes dans les deux fichiers sont gardées.
2. La tension du capteur est convertie en charge différentielle :
   ΔH = (U − T_capteur·dU/dT − intercept) / (dU/dH).
3. Une fenêtre de dates restreint la période étudiée.
4. Pour chaque variable, une méthode de détection des valeurs aberrantes, calculée
   sur la fenêtre retenue : aucune, IQR (hors [Q1 − 1,5·IQR, Q3 + 1,5·IQR]) ou
   Z-score (|z| > 3). Une date est écartée dès qu'une variable y est aberrante.
5. Des dates peuvent être écartées à la main.

Le résultat garde, pour chaque date, la raison de son exclusion : l'interface
peut ainsi montrer exactement ce qui sera retiré avant d'appliquer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from ..domain.models import CLEANED_COLUMNS, PressureSensor
from ..errors import UserError

VARIABLES = ["temp_bed", "t1", "t2", "t3", "t4", "pressure"]
METHODS = ("none", "iqr", "zscore")
ZSCORE_LIMIT = 3.0
IQR_FACTOR = 1.5


@dataclass
class CleanupSettings:
    start: datetime | None = None
    end: datetime | None = None
    methods: dict[str, str] = field(default_factory=lambda: {v: "none" for v in VARIABLES})
    excluded: list[datetime] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "methods": dict(self.methods),
            "excluded": [d.isoformat() for d in self.excluded],
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> CleanupSettings:
        if not data or "methods" not in data:
            return cls()
        methods = {v: data["methods"].get(v, "none") for v in VARIABLES}
        return cls(
            start=datetime.fromisoformat(data["start"]) if data.get("start") else None,
            end=datetime.fromisoformat(data["end"]) if data.get("end") else None,
            methods={v: m if m in METHODS else "none" for v, m in methods.items()},
            excluded=[datetime.fromisoformat(d) for d in data.get("excluded", [])],
        )


@dataclass
class CleanupResult:
    measures: pd.DataFrame  # toutes les dates communes : date, temp_bed, t1..t4, pressure
    reason: pd.Series  # raison d'exclusion par date ("" si gardée)
    unmatched_temperature: int  # dates de la tige sans mesure de pression
    unmatched_pressure: int  # dates de pression sans mesure de la tige

    @property
    def kept(self) -> pd.Series:
        return self.reason == ""

    @property
    def cleaned(self) -> pd.DataFrame:
        return self.measures.loc[self.kept, CLEANED_COLUMNS].reset_index(drop=True)

    def counts(self) -> dict[str, int]:
        return self.reason[self.reason != ""].value_counts().to_dict()


def merge_raw(raw_temperature: pd.DataFrame, raw_pressure: pd.DataFrame, sensor: PressureSensor) -> tuple[pd.DataFrame, int, int]:
    """Associe température et pression par date et calcule la charge différentielle."""
    if raw_temperature.empty or raw_pressure.empty:
        raise UserError("E_CLEANUP_MISSING_MEASURES")
    if sensor.du_dh == 0:
        raise UserError("E_CLEANUP_PRESSURE_CALIBRATION", name=sensor.name)
    raw_temperature = raw_temperature.assign(date=pd.to_datetime(raw_temperature["date"]).astype("datetime64[ns]"))
    raw_pressure = raw_pressure.assign(date=pd.to_datetime(raw_pressure["date"]).astype("datetime64[ns]"))
    merged = raw_temperature.merge(raw_pressure, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if merged.empty:
        raise UserError("E_CLEANUP_NO_COMMON_DATE")
    merged["pressure"] = sensor.differential_head(merged["voltage"], merged["temp_bed"])
    unmatched_t = len(raw_temperature) - len(merged)
    unmatched_p = len(raw_pressure) - len(merged)
    return merged[CLEANED_COLUMNS], unmatched_t, unmatched_p


def iqr_outliers(values: pd.Series) -> pd.Series:
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    spread = q3 - q1
    return (values < q1 - IQR_FACTOR * spread) | (values > q3 + IQR_FACTOR * spread)


def zscore_outliers(values: pd.Series) -> pd.Series:
    std = values.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=values.index)
    return ((values - values.mean()) / std).abs() > ZSCORE_LIMIT


def clean(raw_temperature: pd.DataFrame, raw_pressure: pd.DataFrame, sensor: PressureSensor, settings: CleanupSettings) -> CleanupResult:
    measures, unmatched_t, unmatched_p = merge_raw(raw_temperature, raw_pressure, sensor)
    reason = pd.Series("", index=measures.index, dtype=object)

    def mark(mask: pd.Series, label: str) -> None:
        reason[(reason == "") & mask] = label

    if settings.start is not None:
        mark(measures["date"] < pd.Timestamp(settings.start), "outside_window")
    if settings.end is not None:
        mark(measures["date"] > pd.Timestamp(settings.end), "outside_window")
    if settings.excluded:
        mark(measures["date"].isin(pd.to_datetime(settings.excluded)), "manual")
    mark(measures[VARIABLES].isna().any(axis=1), "missing")

    window = reason != "outside_window"
    for variable in VARIABLES:
        method = settings.methods.get(variable, "none")
        if method == "none":
            continue
        values = measures.loc[window & measures[variable].notna(), variable]
        detector = iqr_outliers if method == "iqr" else zscore_outliers
        flagged = detector(values).reindex(measures.index, fill_value=False)
        mark(flagged, f"{method}:{variable}")
    return CleanupResult(measures, reason, unmatched_t, unmatched_p)


def read_cleaned_csv(path, exact: bool = False) -> pd.DataFrame:
    """Importe des mesures déjà nettoyées (export de cette application ou de Molonaviz).

    `exact` : analyseur de flottants sans perte (`processed_data`, écrit à 17 chiffres) ; l'analyseur rapide de pandas peut
    se tromper d'un ulp, ce qui changerait l'empreinte des mesures."""
    from ..io.dates import parse_dates

    frame = pd.read_csv(path, float_precision="round_trip" if exact else None)
    lower = {c.lower().strip(): c for c in frame.columns}
    aliases = {"date": "date", "tempbed": "temp_bed", "temp_bed": "temp_bed", "temp1": "t1", "t1": "t1", "temp2": "t2", "t2": "t2",
               "temp3": "t3", "t3": "t3", "temp4": "t4", "t4": "t4", "pressure": "pressure"}
    renamed = {lower[k]: v for k, v in aliases.items() if k in lower}
    frame = frame.rename(columns=renamed)
    missing = [c for c in CLEANED_COLUMNS if c not in frame.columns]
    if missing:
        raise UserError("E_CLEANED_FILE_COLUMNS", cols=", ".join(missing))
    dates, _, chronological = parse_dates(frame["date"].astype(str).tolist())
    if not chronological:
        raise UserError("E_CLEANED_FILE_NOT_CHRONOLOGICAL")
    frame["date"] = dates.to_numpy().astype("datetime64[ns]")
    return frame[CLEANED_COLUMNS].dropna().sort_values("date").reset_index(drop=True)


def write_cleaned_csv(cleaned: pd.DataFrame, path) -> None:
    out = cleaned[CLEANED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(path, index=False, float_format="%.6g")
