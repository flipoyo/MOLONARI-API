"""Paramètres des questions (corps de POST /v1/twins/{ref}/questions/{id}), avec leurs bornes et descriptions.

Ils font partie du contrat : le client peut valider une requête avant de l'envoyer, et le schéma JSON de chaque
modèle est publié par GET /v1/questions. Un paramètre inconnu est refusé.

Dates : ISO 8601 sans fuseau, dans le repère des données du twin. Un fuseau est refusé tant que la convention
de dates n'est pas fixée (avec le pipeline brut) : il serait sinon ignoré sans le dire.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NoParams(Params):
    """Aucun paramètre."""


def _naive(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is not None:
        raise ValueError("give the date without a time zone, in the time frame of the twin's data")
    return value


class PeriodParams(Params):
    start: datetime | None = Field(default=None, description="First date of the period, included (ISO 8601, no time zone). Default: start of the results.")
    end: datetime | None = Field(default=None, description="Last date of the period, included (ISO 8601, no time zone). Default: end of the results.")

    _check_dates = field_validator("start", "end")(_naive)

    @model_validator(mode="after")
    def _ordered(self):
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end must not be before start")
        return self


class VolumeParams(PeriodParams):
    area: float | None = Field(default=None, gt=0, description="Riverbed area represented by the point (m²). Without it, volumes are per m² (m³/m²).")


class TemperatureParams(Params):
    z: float | None = Field(default=None, ge=0, description="Depth (m, positive downwards) between the centres of the first and last "
                            "computed cells, e.g. 0.004 to 0.396 m for 50 cells over 0.4 m (the error gives the range). "
                            "Alone: the series T(t).")
    t: datetime | None = Field(default=None, description="Date (ISO 8601, no time zone). Alone: the profile T(z).")

    _check_date = field_validator("t")(_naive)

    @model_validator(mode="after")
    def _point(self):
        if self.z is None and self.t is None:
            raise ValueError("give a depth z, a date t, or both")
        return self


class IntervalParams(Params):
    level: float = Field(default=0.9, gt=0, lt=1, description="Level of the intervals, e.g. 0.9 for 90 %.")
    sigma: float | None = Field(default=None, ge=0, description="Measurement uncertainty of the thermometers (°C). Default: the one recorded with the results.")
    seed: int = Field(default=0, ge=0, description="Seed of the measurement noise drawn for the predictive band (same seed, same band).")


class ResidualParams(IntervalParams):
    intervals: bool = Field(default=False, description="Add the credible and predictive bands (mcmc mode, stored fields).")


class ChainParams(Params):
    chains: list[int] | None = Field(default=None, description="Chains to read, numbered from 0. Default: all.")
    thin: int = Field(default=1, ge=1, description="Keep one kept iteration out of `thin`.")
    iterations: tuple[int | None, int | None] | None = Field(default=None, description="(first, last) computed iteration, included; null: no bound.")

    @field_validator("chains")
    @classmethod
    def _chains(cls, value):
        if value is not None and any(c < 0 for c in value):
            raise ValueError("chains are numbered from 0")
        return value
