"""Préférences persistantes de l'utilisateur (QSettings)."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore

from .i18n import DEFAULT_LANGUAGE, LANGUAGES

MAX_RECENT_PROJECTS = 8
TEMPERATURE_UNITS = ("°C", "K", "°F")


class Settings:
    def __init__(self, qsettings: QtCore.QSettings | None = None):
        self._s = qsettings if qsettings is not None else QtCore.QSettings()

    # Langue
    @property
    def language(self) -> str:
        value = self._s.value("language", DEFAULT_LANGUAGE)
        return value if value in LANGUAGES else DEFAULT_LANGUAGE

    @language.setter
    def language(self, value: str) -> None:
        self._s.setValue("language", value)

    # Unité de température affichée
    @property
    def temperature_unit(self) -> str:
        value = self._s.value("temperature_unit", "°C")
        return value if value in TEMPERATURE_UNITS else "°C"

    @temperature_unit.setter
    def temperature_unit(self, value: str) -> None:
        self._s.setValue("temperature_unit", value)

    # Projets récents
    def recent_projects(self) -> list[str]:
        value = self._s.value("recent_projects", [])
        if isinstance(value, str):  # QSettings renvoie une chaîne s'il n'y a qu'un élément
            value = [value]
        return [p for p in value if Path(p).exists()]

    def add_recent_project(self, path: str | Path) -> None:
        path = str(Path(path).resolve())
        recent = [p for p in self.recent_projects() if p != path]
        self._s.setValue("recent_projects", [path, *recent][:MAX_RECENT_PROJECTS])

    # Dossier des twins (twin_xx.molo)
    @property
    def twin_folder(self) -> Path | None:
        value = self._s.value("twin_folder", "")
        return Path(value) if value else None

    @twin_folder.setter
    def twin_folder(self, value: str | Path | None) -> None:
        self._s.setValue("twin_folder", str(Path(value).resolve()) if value else "")

    # Disposition de la fenêtre
    def window_state(self) -> tuple[QtCore.QByteArray | None, QtCore.QByteArray | None]:
        return self._s.value("window/geometry"), self._s.value("window/state")

    def save_window_state(self, geometry: QtCore.QByteArray, state: QtCore.QByteArray) -> None:
        self._s.setValue("window/geometry", geometry)
        self._s.setValue("window/state", state)
