"""Fichier projet `.molo` : une base SQLite autonome.

- Un seul fichier, copiable et partageable. Les images et notices y sont incluses.
- Écritures en transaction : une erreur ou une interruption laisse le fichier dans
  son état précédent.
- Les mesures sont stockées ligne à ligne (dates en secondes, heure locale du
  logger) ; les résultats de calcul sont stockés en tableaux numpy compressés
  (un blob par calcul), avec la configuration complète, le moteur et l'appareil
  pour la reproductibilité.
- Le schéma est versionné (PRAGMA user_version) et migré à l'ouverture.

Toutes les méthodes s'appellent depuis le thread de l'interface.
"""

from __future__ import annotations

import io
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..domain.models import (
    CLEANED_COLUMNS,
    RAW_PRESSURE_COLUMNS,
    RAW_TEMPERATURE_COLUMNS,
    Lab,
    PointStage,
    PressureSensor,
    SamplingPoint,
    Shaft,
    Study,
    Thermometer,
)
from ..errors import UserError
from ..i18n import t

APPLICATION_ID = 0x4D4F4C4F  # « MOLO »
SCHEMA_VERSION = 1
SUFFIX = ".molo"

_SCHEMA_V1 = """
CREATE TABLE labs (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE);
CREATE TABLE thermometers (
    id INTEGER PRIMARY KEY, lab_id INTEGER NOT NULL REFERENCES labs(id) ON DELETE CASCADE,
    name TEXT NOT NULL, manufacturer TEXT NOT NULL, reference TEXT NOT NULL, sigma REAL NOT NULL,
    UNIQUE (lab_id, name));
CREATE TABLE pressure_sensors (
    id INTEGER PRIMARY KEY, lab_id INTEGER NOT NULL REFERENCES labs(id) ON DELETE CASCADE,
    name TEXT NOT NULL, datalogger TEXT NOT NULL, calibration_date TEXT, intercept REAL NOT NULL,
    du_dh REAL NOT NULL, du_dt REAL NOT NULL, sigma REAL NOT NULL, thermometer TEXT NOT NULL,
    UNIQUE (lab_id, name));
CREATE TABLE shafts (
    id INTEGER PRIMARY KEY, lab_id INTEGER NOT NULL REFERENCES labs(id) ON DELETE CASCADE,
    name TEXT NOT NULL, datalogger TEXT NOT NULL, thermometer TEXT NOT NULL,
    depth1 REAL NOT NULL, depth2 REAL NOT NULL, depth3 REAL NOT NULL, depth4 REAL NOT NULL,
    UNIQUE (lab_id, name));
CREATE TABLE studies (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE,
    lab_id INTEGER NOT NULL REFERENCES labs(id) ON DELETE RESTRICT);
CREATE TABLE points (
    id INTEGER PRIMARY KEY, study_id INTEGER NOT NULL REFERENCES studies(id) ON DELETE CASCADE,
    name TEXT NOT NULL, pressure_sensor TEXT NOT NULL, shaft TEXT NOT NULL,
    setup_date TEXT, measure_date TEXT, river_bed REAL NOT NULL, sensor_offset REAL NOT NULL,
    notice TEXT NOT NULL DEFAULT '', scheme BLOB, created_at TEXT NOT NULL,
    UNIQUE (study_id, name));
CREATE TABLE raw_temperature (
    point_id INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    ts INTEGER NOT NULL, t1 REAL, t2 REAL, t3 REAL, t4 REAL);
CREATE INDEX raw_temperature_point ON raw_temperature (point_id, ts);
CREATE TABLE raw_pressure (
    point_id INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    ts INTEGER NOT NULL, voltage REAL, temp_bed REAL);
CREATE INDEX raw_pressure_point ON raw_pressure (point_id, ts);
CREATE TABLE cleaned (
    point_id INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    ts INTEGER NOT NULL, temp_bed REAL NOT NULL, t1 REAL NOT NULL, t2 REAL NOT NULL,
    t3 REAL NOT NULL, t4 REAL NOT NULL, pressure REAL NOT NULL);
CREATE INDEX cleaned_point ON cleaned (point_id, ts);
CREATE TABLE cleanup_settings (
    point_id INTEGER PRIMARY KEY REFERENCES points(id) ON DELETE CASCADE,
    settings TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE computations (
    id INTEGER PRIMARY KEY, point_id INTEGER NOT NULL REFERENCES points(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('direct', 'inversion')),
    created_at TEXT NOT NULL, duration_s REAL NOT NULL, config TEXT NOT NULL,
    engine TEXT NOT NULL, device TEXT NOT NULL, results BLOB NOT NULL);
CREATE INDEX computations_point ON computations (point_id, kind, created_at);
"""

# Migrations : version de départ → script SQL. Ajouter ici les évolutions du schéma.
MIGRATIONS: dict[int, str] = {0: _SCHEMA_V1}


@dataclass
class ComputationInfo:
    id: int
    kind: str
    created_at: datetime
    duration_s: float
    config: dict
    engine: str
    device: str


def _date_to_text(value: datetime | None) -> str | None:
    return value.isoformat(sep=" ") if value else None


def _text_to_date(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _to_seconds(dates: pd.Series) -> np.ndarray:
    return pd.to_datetime(dates).to_numpy().astype("datetime64[s]").astype(np.int64)


def _from_seconds(seconds) -> np.ndarray:
    return np.asarray(seconds, np.int64).astype("datetime64[s]").astype("datetime64[ns]")


class Project:
    def __init__(self, path: Path, connection: sqlite3.Connection):
        self.path = Path(path)
        self._con = connection

    # Ouverture ------------------------------------------------------------------------
    @classmethod
    def create(cls, path: str | Path) -> Project:
        path = Path(path)
        if path.suffix != SUFFIX:
            path = path.with_suffix(SUFFIX)
        if path.exists():
            raise UserError(t("A file already exists at {path}.").format(path=path))
        connection = cls._connect(path)
        connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
        project = cls(path, connection)
        project._migrate()
        return project

    @classmethod
    def open(cls, path: str | Path) -> Project:
        path = Path(path)
        if not path.is_file():
            raise UserError(t("File not found: {path}").format(path=path))
        try:
            connection = cls._connect(path)
            app_id = connection.execute("PRAGMA application_id").fetchone()[0]
        except sqlite3.DatabaseError:
            raise UserError(t("{name} is not a Molonaviz project.").format(name=path.name)) from None
        if app_id != APPLICATION_ID:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            connection.close()
            if "Labo" in tables and "SamplingPoint" in tables:
                raise UserError(
                    t("{name} is a database from the previous Molonaviz. Use File › Import Molonaviz database to convert it.").format(name=path.name)
                )
            raise UserError(t("{name} is not a Molonaviz project.").format(name=path.name))
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            connection.close()
            raise UserError(t("{name} was created by a newer version of Molonaviz. Please update the application.").format(name=path.name))
        project = cls(path, connection)
        project._migrate()
        return project

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(path, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _migrate(self) -> None:
        version = self._con.execute("PRAGMA user_version").fetchone()[0]
        while version < SCHEMA_VERSION:
            # Un seul script transactionnel par migration (executescript validerait une transaction ouverte).
            script = f"BEGIN; {MIGRATIONS[version]} PRAGMA user_version = {version + 1}; COMMIT;"
            try:
                self._con.executescript(script)
            except BaseException:
                if self._con.in_transaction:
                    self._con.execute("ROLLBACK")
                raise
            version += 1

    def close(self) -> None:
        self._con.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self._con.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._con.execute("ROLLBACK")
            raise
        self._con.execute("COMMIT")

    def _one(self, sql: str, params=()) -> tuple | None:
        return self._con.execute(sql, params).fetchone()

    # Laboratoires ---------------------------------------------------------------------
    def lab_names(self) -> list[str]:
        return [row[0] for row in self._con.execute("SELECT name FROM labs ORDER BY name COLLATE NOCASE")]

    def _lab_id(self, name: str) -> int:
        row = self._one("SELECT id FROM labs WHERE name = ?", (name,))
        if row is None:
            raise UserError(t("Laboratory \"{name}\" does not exist.").format(name=name))
        return row[0]

    def add_lab(self, lab: Lab) -> None:
        if not lab.name.strip():
            raise UserError(t("The laboratory needs a name."))
        if self._one("SELECT 1 FROM labs WHERE name = ?", (lab.name,)):
            raise UserError(t("A laboratory named \"{name}\" already exists.").format(name=lab.name))
        with self._transaction():
            lab_id = self._con.execute("INSERT INTO labs (name) VALUES (?)", (lab.name,)).lastrowid
            for item in [*lab.thermometers, *lab.pressure_sensors, *lab.shafts]:
                self._insert_lab_item(lab_id, item)

    def add_lab_item(self, lab_name: str, item: Thermometer | PressureSensor | Shaft) -> None:
        lab_id = self._lab_id(lab_name)
        table = {Thermometer: "thermometers", PressureSensor: "pressure_sensors", Shaft: "shafts"}[type(item)]
        if self._one(f"SELECT 1 FROM {table} WHERE lab_id = ? AND name = ?", (lab_id, item.name)):
            raise UserError(t("\"{name}\" already exists in this laboratory.").format(name=item.name))
        with self._transaction():
            self._insert_lab_item(lab_id, item)

    def _insert_lab_item(self, lab_id: int, item) -> None:
        if isinstance(item, Thermometer):
            self._con.execute(
                "INSERT INTO thermometers (lab_id, name, manufacturer, reference, sigma) VALUES (?, ?, ?, ?, ?)",
                (lab_id, item.name, item.manufacturer, item.reference, item.sigma),
            )
        elif isinstance(item, PressureSensor):
            self._con.execute(
                "INSERT INTO pressure_sensors (lab_id, name, datalogger, calibration_date, intercept, du_dh, du_dt, sigma, thermometer)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (lab_id, item.name, item.datalogger, _date_to_text(item.calibration_date), item.intercept, item.du_dh, item.du_dt, item.sigma, item.thermometer),
            )
        else:
            self._con.execute(
                "INSERT INTO shafts (lab_id, name, datalogger, thermometer, depth1, depth2, depth3, depth4) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (lab_id, item.name, item.datalogger, item.thermometer, *item.depths),
            )

    def get_lab(self, name: str) -> Lab:
        lab_id = self._lab_id(name)
        lab = Lab(name)
        for row in self._con.execute("SELECT name, manufacturer, reference, sigma FROM thermometers WHERE lab_id = ? ORDER BY name", (lab_id,)):
            lab.thermometers.append(Thermometer(*row))
        for row in self._con.execute(
            "SELECT name, datalogger, calibration_date, intercept, du_dh, du_dt, sigma, thermometer FROM pressure_sensors WHERE lab_id = ? ORDER BY name",
            (lab_id,),
        ):
            lab.pressure_sensors.append(PressureSensor(row[0], row[1], _text_to_date(row[2]), *row[3:]))
        for row in self._con.execute(
            "SELECT name, datalogger, thermometer, depth1, depth2, depth3, depth4 FROM shafts WHERE lab_id = ? ORDER BY name", (lab_id,)
        ):
            lab.shafts.append(Shaft(row[0], row[1], row[2], tuple(row[3:])))  # type: ignore[arg-type]
        return lab

    def delete_lab(self, name: str) -> None:
        lab_id = self._lab_id(name)
        users = [row[0] for row in self._con.execute("SELECT name FROM studies WHERE lab_id = ?", (lab_id,))]
        if users:
            raise UserError(t("This laboratory is used by: {studies}. Delete these studies first.").format(studies=", ".join(users)))
        with self._transaction():
            self._con.execute("DELETE FROM labs WHERE id = ?", (lab_id,))

    # Études ---------------------------------------------------------------------------
    def studies(self) -> list[Study]:
        return [
            Study(name, lab)
            for name, lab in self._con.execute(
                "SELECT s.name, l.name FROM studies s JOIN labs l ON l.id = s.lab_id ORDER BY s.name COLLATE NOCASE"
            )
        ]

    def add_study(self, study: Study) -> None:
        if not study.name.strip():
            raise UserError(t("The study needs a name."))
        lab_id = self._lab_id(study.lab)
        if self._one("SELECT 1 FROM studies WHERE name = ?", (study.name,)):
            raise UserError(t("A study named \"{name}\" already exists.").format(name=study.name))
        with self._transaction():
            self._con.execute("INSERT INTO studies (name, lab_id) VALUES (?, ?)", (study.name, lab_id))

    def _study_id(self, name: str) -> int:
        row = self._one("SELECT id FROM studies WHERE name = ?", (name,))
        if row is None:
            raise UserError(t("Study \"{name}\" does not exist.").format(name=name))
        return row[0]

    def delete_study(self, name: str) -> None:
        study_id = self._study_id(name)
        with self._transaction():
            self._con.execute("DELETE FROM studies WHERE id = ?", (study_id,))

    def study_lab(self, study: str) -> Lab:
        row = self._one("SELECT l.name FROM studies s JOIN labs l ON l.id = s.lab_id WHERE s.name = ?", (study,))
        if row is None:
            raise UserError(t("Study \"{name}\" does not exist.").format(name=study))
        return self.get_lab(row[0])

    # Points ---------------------------------------------------------------------------
    def _point_id(self, study: str, name: str) -> int:
        row = self._one(
            "SELECT p.id FROM points p JOIN studies s ON s.id = p.study_id WHERE s.name = ? AND p.name = ?", (study, name)
        )
        if row is None:
            raise UserError(t("Point \"{name}\" does not exist in study \"{study}\".").format(name=name, study=study))
        return row[0]

    def point_names(self, study: str) -> list[str]:
        study_id = self._study_id(study)
        return [row[0] for row in self._con.execute("SELECT name FROM points WHERE study_id = ? ORDER BY name COLLATE NOCASE", (study_id,))]

    def add_point(self, point: SamplingPoint, raw_temperature: pd.DataFrame, raw_pressure: pd.DataFrame) -> None:
        study_id = self._study_id(point.study)
        if not point.name.strip():
            raise UserError(t("The point needs a name."))
        if self._one("SELECT 1 FROM points WHERE study_id = ? AND name = ?", (study_id, point.name)):
            raise UserError(t("A point named \"{name}\" already exists in this study.").format(name=point.name))
        lab = self.study_lab(point.study)
        if point.pressure_sensor not in {p.name for p in lab.pressure_sensors}:
            raise UserError(t("Pressure sensor \"{name}\" is not in laboratory \"{lab}\".").format(name=point.pressure_sensor, lab=lab.name))
        if point.shaft not in {s.name for s in lab.shafts}:
            raise UserError(t("Shaft \"{name}\" is not in laboratory \"{lab}\".").format(name=point.shaft, lab=lab.name))
        with self._transaction():
            point_id = self._con.execute(
                "INSERT INTO points (study_id, name, pressure_sensor, shaft, setup_date, measure_date, river_bed, sensor_offset, notice, scheme, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    study_id, point.name, point.pressure_sensor, point.shaft, _date_to_text(point.setup_date),
                    _date_to_text(point.measure_date), point.river_bed, point.offset, point.notice, point.scheme,
                    datetime.now().isoformat(sep=" ", timespec="seconds"),
                ),
            ).lastrowid
            self._insert_rows("raw_temperature", RAW_TEMPERATURE_COLUMNS, point_id, raw_temperature)
            self._insert_rows("raw_pressure", RAW_PRESSURE_COLUMNS, point_id, raw_pressure)

    def _insert_rows(self, table: str, columns: list[str], point_id: int, frame: pd.DataFrame) -> None:
        values = frame[columns[1:]].to_numpy(np.float64)
        seconds = _to_seconds(frame["date"])
        placeholders = ", ".join("?" * (len(columns) + 1))
        rows = ((point_id, int(s), *map(float, v)) for s, v in zip(seconds, values))
        self._con.executemany(f"INSERT INTO {table} (point_id, ts, {', '.join(columns[1:])}) VALUES ({placeholders})", rows)

    def _read_rows(self, table: str, columns: list[str], point_id: int) -> pd.DataFrame:
        rows = self._con.execute(f"SELECT ts, {', '.join(columns[1:])} FROM {table} WHERE point_id = ? ORDER BY ts", (point_id,)).fetchall()
        frame = pd.DataFrame(rows, columns=columns, dtype="float64") if rows else pd.DataFrame(columns=columns, dtype="float64")
        frame["date"] = _from_seconds(frame["date"].to_numpy()) if rows else pd.Series(dtype="datetime64[ns]")
        return frame

    def get_point(self, study: str, name: str) -> SamplingPoint:
        point_id = self._point_id(study, name)
        row = self._one(
            "SELECT name, pressure_sensor, shaft, setup_date, measure_date, river_bed, sensor_offset, notice, scheme FROM points WHERE id = ?",
            (point_id,),
        )
        return SamplingPoint(
            name=row[0], study=study, pressure_sensor=row[1], shaft=row[2], setup_date=_text_to_date(row[3]),
            measure_date=_text_to_date(row[4]), river_bed=row[5], offset=row[6], notice=row[7], scheme=row[8],
        )

    def delete_point(self, study: str, name: str) -> None:
        point_id = self._point_id(study, name)
        with self._transaction():
            self._con.execute("DELETE FROM points WHERE id = ?", (point_id,))

    def raw_temperature(self, study: str, name: str) -> pd.DataFrame:
        return self._read_rows("raw_temperature", RAW_TEMPERATURE_COLUMNS, self._point_id(study, name))

    def raw_pressure(self, study: str, name: str) -> pd.DataFrame:
        return self._read_rows("raw_pressure", RAW_PRESSURE_COLUMNS, self._point_id(study, name))

    # Mesures nettoyées ----------------------------------------------------------------
    def set_cleaned(self, study: str, name: str, cleaned: pd.DataFrame, settings: dict) -> None:
        """Remplace les mesures nettoyées. Les calculs existants, devenus obsolètes, sont supprimés."""
        point_id = self._point_id(study, name)
        frame = cleaned[CLEANED_COLUMNS]
        if frame[CLEANED_COLUMNS[1:]].isna().any().any():
            raise UserError(t("Cleaned measures must not contain empty values."))
        with self._transaction():
            self._con.execute("DELETE FROM cleaned WHERE point_id = ?", (point_id,))
            self._con.execute("DELETE FROM computations WHERE point_id = ?", (point_id,))
            self._insert_rows("cleaned", CLEANED_COLUMNS, point_id, frame)
            self._con.execute(
                "INSERT OR REPLACE INTO cleanup_settings (point_id, settings, updated_at) VALUES (?, ?, ?)",
                (point_id, json.dumps(settings), datetime.now().isoformat(sep=" ", timespec="seconds")),
            )

    def cleaned(self, study: str, name: str) -> pd.DataFrame:
        return self._read_rows("cleaned", CLEANED_COLUMNS, self._point_id(study, name))

    def cleanup_settings(self, study: str, name: str) -> dict | None:
        row = self._one("SELECT settings FROM cleanup_settings WHERE point_id = ?", (self._point_id(study, name),))
        return json.loads(row[0]) if row else None

    def reset_point(self, study: str, name: str) -> None:
        """Supprime mesures nettoyées et calculs : le point revient à ses mesures brutes."""
        point_id = self._point_id(study, name)
        with self._transaction():
            for table in ("cleaned", "cleanup_settings", "computations"):
                self._con.execute(f"DELETE FROM {table} WHERE point_id = ?", (point_id,))

    # Calculs --------------------------------------------------------------------------
    def add_computation(
        self, study: str, name: str, kind: str, config: dict, arrays: dict[str, np.ndarray], duration_s: float, engine: str, device: str
    ) -> int:
        point_id = self._point_id(study, name)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{k: v for k, v in arrays.items() if v is not None})
        with self._transaction():
            return self._con.execute(
                "INSERT INTO computations (point_id, kind, created_at, duration_s, config, engine, device, results) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (point_id, kind, datetime.now().isoformat(sep=" ", timespec="seconds"), duration_s, json.dumps(config), engine, device, buffer.getvalue()),
            ).lastrowid

    def computations(self, study: str, name: str, kind: str | None = None) -> list[ComputationInfo]:
        point_id = self._point_id(study, name)
        sql = "SELECT id, kind, created_at, duration_s, config, engine, device FROM computations WHERE point_id = ?"
        params: tuple = (point_id,)
        if kind:
            sql += " AND kind = ?"
            params += (kind,)
        rows = self._con.execute(sql + " ORDER BY created_at DESC, id DESC", params).fetchall()
        return [ComputationInfo(r[0], r[1], datetime.fromisoformat(r[2]), r[3], json.loads(r[4]), r[5], r[6]) for r in rows]

    def load_result(self, computation_id: int) -> dict[str, np.ndarray]:
        row = self._one("SELECT results FROM computations WHERE id = ?", (computation_id,))
        if row is None:
            raise UserError(t("This computation no longer exists."))
        with np.load(io.BytesIO(row[0]), allow_pickle=False) as data:
            return {key: data[key] for key in data.files}

    def delete_computation(self, computation_id: int) -> None:
        with self._transaction():
            self._con.execute("DELETE FROM computations WHERE id = ?", (computation_id,))

    def stage(self, study: str, name: str) -> PointStage:
        point_id = self._point_id(study, name)
        kinds = {row[0] for row in self._con.execute("SELECT DISTINCT kind FROM computations WHERE point_id = ?", (point_id,))}
        if "inversion" in kinds:
            return PointStage.INVERSION
        if "direct" in kinds:
            return PointStage.DIRECT_MODEL
        if self._one("SELECT 1 FROM cleaned WHERE point_id = ? LIMIT 1", (point_id,)):
            return PointStage.CLEANED
        return PointStage.RAW
