"""Apparence : style Fusion, palette du système (clair/sombre), une seule couleur d'accent.

Volontairement sobre : pas de dégradés ni d'effets. La QSS se limite à l'espacement
et à la hiérarchie typographique.
"""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets

ACCENT = "#1f6feb"

_QSS = """
QGroupBox { font-weight: 600; margin-top: 14px; }
QGroupBox::title { subcontrol-origin: margin; left: 2px; padding: 0 2px; }
QLabel[role="heading"] { font-size: 15px; font-weight: 600; }
QLabel[role="hint"] { color: palette(placeholder-text); }
QLabel[role="error"] { color: #c0392b; }
QLineEdit[invalid="true"], QDoubleSpinBox[invalid="true"], QSpinBox[invalid="true"] { border: 1px solid #c0392b; }
QTabWidget::pane { border-top: 1px solid palette(mid); }
QDockWidget::title { padding: 4px 6px; font-weight: 600; }
"""


def apply_theme(app: QtWidgets.QApplication) -> None:
    app.setStyle("Fusion")
    palette = app.palette()
    accent = QtGui.QColor(ACCENT)
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, accent)
    palette.setColor(QtGui.QPalette.ColorRole.Link, accent)
    app.setPalette(palette)
    app.setStyleSheet(_QSS)


def set_invalid(widget: QtWidgets.QWidget, invalid: bool) -> None:
    """Marque visuellement un champ invalide (bordure rouge via la QSS)."""
    if widget.property("invalid") == invalid:
        return
    widget.setProperty("invalid", invalid)
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def muted_color() -> str:
    """Couleur du texte secondaire (pour le texte enrichi, où palette() n'existe pas)."""
    return QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.PlaceholderText).name()
