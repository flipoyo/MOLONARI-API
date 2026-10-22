"""Gestion centralisée des erreurs.

Toute exception non rattrapée (y compris dans un slot Qt) est journalisée et
présentée à l'utilisateur ; l'application continue de fonctionner. Sans ce
crochet, PySide6 interromprait le processus.
"""

from __future__ import annotations

import sys
import traceback
from collections.abc import Callable

from PySide6 import QtWidgets

from molonaviz2.facade import UserError

from ..i18n import render, t
from ..logs import log

__all__ = ["UserError", "error_text", "install_excepthook", "report_exception", "set_presenter"]


Presenter = Callable[[str, str, str], None]


def error_text(obj) -> str:
    """Texte traduit d'une erreur ou d'un problème du cœur (à partir de son code), ou str(obj)."""
    return render(obj, t)


def _message_box(title: str, message: str, details: str) -> None:
    box = QtWidgets.QMessageBox(QtWidgets.QApplication.activeWindow())
    box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
    box.setWindowTitle(title)
    box.setText(message)
    if details:
        box.setDetailedText(details)
    box.exec()


_presenter: Presenter = _message_box
_presenting = False


def set_presenter(presenter: Presenter) -> Presenter:
    """Remplace la fonction d'affichage des erreurs (utile pour les tests). Retourne l'ancienne."""
    global _presenter
    previous, _presenter = _presenter, presenter
    return previous


def report_exception(exc: BaseException, context: str = "") -> None:
    """Journalise et affiche une exception."""
    global _presenting
    details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if isinstance(exc, UserError):
        log.warning("%s%s", f"{context} : " if context else "", exc)
        title, message, details = t("Action not completed"), error_text(exc), ""
    else:
        log.error("Unexpected error%s\n%s", f" ({context})" if context else "", details)
        title = t("Unexpected error")
        message = t("The last action could not be completed. Molonaviz is still running and your data has not been modified.")
        if context:
            message = f"{context}\n\n{message}"
        message += f"\n\n{type(exc).__name__}: {exc}"
    if _presenting:  # Une erreur pendant l'affichage d'une erreur : journal uniquement.
        return
    _presenting = True
    try:
        _presenter(title, message, details)
    finally:
        _presenting = False


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    report_exception(exc_value.with_traceback(exc_tb))


def install_excepthook() -> None:
    sys.excepthook = _excepthook
