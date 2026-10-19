"""Journal de l'application : fichier tournant + diffusion vers le panneau « Journal »."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from PySide6 import QtCore

LOGGER_NAME = "molonaviz2"
log = logging.getLogger(LOGGER_NAME)


def log_directory() -> Path:
    base = QtCore.QStandardPaths.writableLocation(QtCore.QStandardPaths.StandardLocation.AppDataLocation)
    path = Path(base or Path.home() / ".molonaviz2") / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


class LogBridge(QtCore.QObject):
    """Relaie les enregistrements du journal vers l'interface, quel que soit le thread émetteur."""

    record = QtCore.Signal(str, str)  # niveau, message


class _BridgeHandler(logging.Handler):
    def __init__(self, bridge: LogBridge):
        super().__init__(level=logging.INFO)
        self._bridge = bridge
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._bridge.record.emit(record.levelname, self.format(record))
        except RuntimeError:
            # Le pont Qt a été détruit (fermeture de l'application) : on ignore.
            pass


bridge = LogBridge()


def setup_logging(directory: Path | None = None) -> Path:
    """Configure le journal. Retourne le chemin du fichier de journal."""
    directory = directory or log_directory()
    path = directory / "molonaviz2.log"
    log.setLevel(logging.DEBUG)
    if not any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        file_handler = RotatingFileHandler(path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        log.addHandler(file_handler)
    if not any(isinstance(h, _BridgeHandler) for h in log.handlers):
        log.addHandler(_BridgeHandler(bridge))
    return path
