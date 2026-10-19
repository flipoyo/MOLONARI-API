"""Explorateur du projet (dock gauche) : laboratoires et capteurs, études et points."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from ..i18n import t
from ..storage.project import Project
from .point_view.context import STAGE_LABELS
from .widgets import format_number

KIND = QtCore.Qt.ItemDataRole.UserRole
PAYLOAD = QtCore.Qt.ItemDataRole.UserRole + 1


class ProjectExplorer(QtWidgets.QTreeWidget):
    point_activated = QtCore.Signal(str, str)  # étude, point
    selection_described = QtCore.Signal(str)  # description HTML de l'élément sélectionné
    context_requested = QtCore.Signal(str, object, QtCore.QPoint)  # type, données, position globale

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderHidden(True)
        self.setColumnCount(2)
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.project: Project | None = None
        self.itemDoubleClicked.connect(self._activated)
        self.itemActivated.connect(self._activated)
        self.currentItemChanged.connect(self._describe)
        self.customContextMenuRequested.connect(self._context_menu)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.refresh()

    def _item(self, parent, text: str, kind: str, payload=None, detail: str = "") -> QtWidgets.QTreeWidgetItem:
        item = QtWidgets.QTreeWidgetItem(parent, [text, detail])
        item.setData(0, KIND, kind)
        item.setData(0, PAYLOAD, payload)
        return item

    def refresh(self) -> None:
        expanded = self._expanded_keys()
        self.clear()
        if self.project is None:
            return
        labs = self._item(self, t("Laboratories"), "labs")
        for lab_name in self.project.lab_names():
            lab = self.project.get_lab(lab_name)
            lab_item = self._item(labs, lab.name, "lab", lab.name)
            group = self._item(lab_item, t("Thermometers"), "group", None, str(len(lab.thermometers)))
            for item in lab.thermometers:
                self._item(group, item.name, "thermometer", item)
            group = self._item(lab_item, t("Pressure sensors"), "group", None, str(len(lab.pressure_sensors)))
            for item in lab.pressure_sensors:
                self._item(group, item.name, "pressure_sensor", item)
            group = self._item(lab_item, t("Shafts"), "group", None, str(len(lab.shafts)))
            for item in lab.shafts:
                self._item(group, item.name, "shaft", item)
        studies = self._item(self, t("Studies"), "studies")
        for study in self.project.studies():
            study_item = self._item(studies, study.name, "study", study.name, study.lab)
            for point in self.project.point_names(study.name):
                stage = self.project.stage(study.name, point)
                self._item(study_item, point, "point", (study.name, point), t(STAGE_LABELS[stage]))
        labs.setExpanded(True)
        studies.setExpanded(True)
        self._restore_expanded(expanded)

    def _key(self, item: QtWidgets.QTreeWidgetItem) -> tuple:
        path = []
        while item is not None:
            path.append(item.text(0))
            item = item.parent()
        return tuple(reversed(path))

    def _expanded_keys(self) -> set[tuple]:
        keys = set()
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if item.isExpanded():
                keys.add(self._key(item))
            iterator += 1
        return keys

    def _restore_expanded(self, keys: set[tuple]) -> None:
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if self._key(item) in keys:
                item.setExpanded(True)
            iterator += 1

    def select_point(self, study: str, point: str) -> None:
        iterator = QtWidgets.QTreeWidgetItemIterator(self)
        while iterator.value():
            item = iterator.value()
            if item.data(0, KIND) == "point" and item.data(0, PAYLOAD) == (study, point):
                self.setCurrentItem(item)
                return
            iterator += 1

    def _activated(self, item: QtWidgets.QTreeWidgetItem) -> None:
        if item.data(0, KIND) == "point":
            study, point = item.data(0, PAYLOAD)
            self.point_activated.emit(study, point)

    def _context_menu(self, position: QtCore.QPoint) -> None:
        item = self.itemAt(position)
        if item is not None:
            self.context_requested.emit(item.data(0, KIND), item.data(0, PAYLOAD), self.viewport().mapToGlobal(position))

    def _describe(self, item: QtWidgets.QTreeWidgetItem | None) -> None:
        self.selection_described.emit(describe(item.data(0, KIND), item.data(0, PAYLOAD), self.project) if item else "")


def _table(rows: list[tuple[str, str]]) -> str:
    return "<table cellspacing='3'>" + "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows) + "</table>"


def describe(kind: str, payload, project: Project | None) -> str:
    if kind == "thermometer":
        return _table([(t("Thermometer"), payload.name), (t("Manufacturer"), payload.manufacturer), (t("Reference"), payload.reference),
                       (t("Uncertainty σ"), f"{format_number(payload.sigma)} °C")])
    if kind == "pressure_sensor":
        date = payload.calibration_date.strftime("%Y-%m-%d") if payload.calibration_date else "—"
        return _table([
            (t("Pressure sensor"), payload.name), (t("Datalogger"), payload.datalogger), (t("Calibration date"), date),
            (t("Intercept"), f"{format_number(payload.intercept)} V"), ("dU/dH", f"{format_number(payload.du_dh)} V/m"),
            ("dU/dT", f"{format_number(payload.du_dt)} V/°C"), (t("Uncertainty σ"), f"{format_number(payload.sigma)} m"),
            (t("Thermometer"), payload.thermometer),
        ])
    if kind == "shaft":
        return _table([(t("Shaft"), payload.name), (t("Datalogger"), payload.datalogger), (t("Thermometer"), payload.thermometer)]
                      + [(t("Depth of sensor {n}").format(n=i + 1), f"{format_number(d)} m") for i, d in enumerate(payload.depths)])
    if kind == "study" and project is not None:
        study = next((s for s in project.studies() if s.name == payload), None)
        if study:
            return _table([(t("Study"), study.name), (t("Laboratory"), study.lab), (t("Points"), str(len(project.point_names(study.name))))])
    if kind == "point" and project is not None:
        study, name = payload
        point = project.get_point(study, name)
        stage = project.stage(study, name)
        return _table([
            (t("Point"), point.name), (t("Study"), study), (t("Stage"), t(STAGE_LABELS[stage])),
            (t("Pressure sensor"), point.pressure_sensor), (t("Shaft"), point.shaft),
            (t("River bed depth"), f"{format_number(point.river_bed)} m"), (t("Sensor offset (Δh)"), f"{format_number(point.offset)} m"),
        ]) + "<p>" + t("Double-click to open.") + "</p>"
    if kind == "lab" and project is not None:
        lab = project.get_lab(payload)
        return _table([(t("Laboratory"), lab.name), (t("Thermometers"), str(len(lab.thermometers))),
                       (t("Pressure sensors"), str(len(lab.pressure_sensors))), (t("Shafts"), str(len(lab.shafts)))])
    return ""
