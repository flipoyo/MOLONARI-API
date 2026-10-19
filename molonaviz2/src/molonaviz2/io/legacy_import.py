"""Conversion d'une base de l'ancien Molonaviz (`Molonari.sqlite`) en projet `.molo`.

Sont repris : laboratoires et capteurs, études, points (avec notice et schéma
s'ils sont retrouvés), mesures brutes et mesures nettoyées. Les résultats de
calcul ne sont pas repris : ils dépendaient de l'ancienne version de pyheatmy et
doivent être recalculés.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..domain.models import Lab, PressureSensor, SamplingPoint, Shaft, Study, Thermometer
from ..errors import UserError
from ..i18n import t
from ..storage.project import Project
from .lab_import import read_text

LEGACY_DATE_FORMAT = "%Y/%m/%d %H:%M:%S"


@dataclass
class LegacyReport:
    labs: int = 0
    studies: int = 0
    points: int = 0
    cleaned_points: int = 0
    warnings: list[str] = field(default_factory=list)


def _date(value) -> datetime | None:
    if not value:
        return None
    for fmt in (LEGACY_DATE_FORMAT, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(str(value).strip(), fmt)
        except ValueError:
            continue
    return None


def _find_file(stored: str | None, base: Path, subfolder: str) -> Path | None:
    if not stored:
        return None
    candidates = [Path(stored), base / subfolder / Path(stored).name]
    return next((p for p in candidates if p.is_file()), None)


def import_legacy_database(database: Path, project: Project) -> LegacyReport:
    database = Path(database)
    if not database.is_file():
        raise UserError(t("File not found: {path}").format(path=database))
    con = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    try:
        tables = {row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"Labo", "Study", "SamplingPoint", "RawMeasuresTemp"} <= tables:
            raise UserError(t("{name} is not a database from the previous Molonaviz.").format(name=database.name))
        return _import(con, database.parent, project)
    finally:
        con.close()


def _import(con: sqlite3.Connection, base: Path, project: Project) -> LegacyReport:
    report = LegacyReport()
    thermo_names = dict(con.execute("SELECT ID, Name FROM Thermometer"))
    lab_names = dict(con.execute("SELECT ID, Name FROM Labo"))
    existing_labs = set(project.lab_names())
    for lab_id, lab_name in lab_names.items():
        if lab_name in existing_labs:
            report.warnings.append(t("Laboratory \"{name}\" already exists: kept as is.").format(name=lab_name))
            continue
        lab = Lab(lab_name)
        for name, manu, ref, error in con.execute("SELECT Name, ManuName, ManuRef, Error FROM Thermometer WHERE Labo = ?", (lab_id,)):
            lab.thermometers.append(Thermometer(name, manu or "", ref or "", float(error)))
        for row in con.execute(
            "SELECT Name, Datalogger, Calibration, Intercept, DuDH, DuDT, Error, ThermoModel FROM PressureSensor WHERE Labo = ?", (lab_id,)
        ):
            lab.pressure_sensors.append(
                PressureSensor(row[0], row[1] or "", _date(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6]), thermo_names.get(row[7], ""))
            )
        for row in con.execute("SELECT Name, Datalogger, Depth1, Depth2, Depth3, Depth4, ThermoModel FROM Shaft WHERE Labo = ?", (lab_id,)):
            lab.shafts.append(Shaft(row[0], row[1] or "", thermo_names.get(row[6], ""), (float(row[2]), float(row[3]), float(row[4]), float(row[5]))))
        project.add_lab(lab)
        report.labs += 1

    psensor_names = dict(con.execute("SELECT ID, Name FROM PressureSensor"))
    shaft_names = dict(con.execute("SELECT ID, Name FROM Shaft"))
    existing_studies = {s.name for s in project.studies()}
    for study_id, study_name, lab_id in con.execute("SELECT ID, Name, Labo FROM Study").fetchall():
        if study_name in existing_studies:
            report.warnings.append(t("Study \"{name}\" already exists: skipped.").format(name=study_name))
            continue
        project.add_study(Study(study_name, lab_names[lab_id]))
        report.studies += 1
        points = con.execute(
            'SELECT ID, Name, Notice, Setup, LastTransfer, "Offset", RiverBed, Shaft, PressureSensor, Scheme FROM SamplingPoint WHERE Study = ?',
            (study_id,),
        ).fetchall()
        for pid, name, notice_path, setup, last, offset, river_bed, shaft, psensor, scheme_path in points:
            notice_file = _find_file(notice_path, base, "Notices")
            scheme_file = _find_file(scheme_path, base, "Schemes")
            if notice_path and notice_file is None:
                report.warnings.append(t("{point}: notice not found ({path}).").format(point=name, path=notice_path))
            if scheme_path and scheme_file is None:
                report.warnings.append(t("{point}: scheme not found ({path}).").format(point=name, path=scheme_path))
            point = SamplingPoint(
                name=name, study=study_name, pressure_sensor=psensor_names.get(psensor, ""), shaft=shaft_names.get(shaft, ""),
                setup_date=_date(setup), measure_date=_date(last), river_bed=float(river_bed or 0.0), offset=float(offset or 0.0),
                notice=read_text(notice_file) if notice_file else "",
                scheme=scheme_file.read_bytes() if scheme_file and scheme_file.suffix.lower() in (".png", ".jpg", ".jpeg") else None,
            )
            raw_t = pd.read_sql_query("SELECT Date AS date, Temp1 AS t1, Temp2 AS t2, Temp3 AS t3, Temp4 AS t4 FROM RawMeasuresTemp WHERE SamplingPoint = ? ORDER BY Date", con, params=(pid,))
            raw_p = pd.read_sql_query("SELECT Date AS date, Voltage AS voltage, TempBed AS temp_bed FROM RawMeasuresPress WHERE SamplingPoint = ? ORDER BY Date", con, params=(pid,))
            for frame in (raw_t, raw_p):
                frame["date"] = pd.to_datetime(frame["date"], format=LEGACY_DATE_FORMAT)
            try:
                project.add_point(point, raw_t, raw_p)
            except UserError as exc:
                report.warnings.append(f"{name} : {exc}")
                continue
            report.points += 1
            cleaned = pd.read_sql_query(
                "SELECT d.Date AS date, c.TempBed AS temp_bed, c.Temp1 AS t1, c.Temp2 AS t2, c.Temp3 AS t3, c.Temp4 AS t4, c.Pressure AS pressure"
                " FROM CleanedMeasures c JOIN Date d ON d.ID = c.Date JOIN Point pt ON pt.ID = c.PointKey"
                " WHERE pt.SamplingPoint = ? ORDER BY d.Date",
                con,
                params=(pid,),
            )
            if len(cleaned):
                cleaned["date"] = pd.to_datetime(cleaned["date"], format=LEGACY_DATE_FORMAT)
                project.set_cleaned(study_name, name, cleaned, {"source": "legacy"})
                report.cleaned_points += 1
    return report
