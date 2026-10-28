"""Rapport public d'un space (P2c.3) : le dossier `Public Report/` que la gate sert aux users.

    from molonari_twin.report import generate_report
    generate_report("twins/twin_03")

Voir generator.py (entrées, visibilité), writer.py (JSON, npz, écriture atomique) et figures.py (SVG).
"""

from .generator import (
    REPORT_FORMAT,
    Report,
    ReportResult,
    Space,
    build_report,
    generate_report,
    resolve_space,
)
from .writer import FOLDER_NAME, INDEX_NAME, NPZ_THRESHOLD

__all__ = [
    "FOLDER_NAME",
    "INDEX_NAME",
    "NPZ_THRESHOLD",
    "REPORT_FORMAT",
    "Report",
    "ReportResult",
    "Space",
    "build_report",
    "generate_report",
    "resolve_space",
]
