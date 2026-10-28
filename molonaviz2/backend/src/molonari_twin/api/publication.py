"""Publier une réponse dans le rapport public : le contrôle de visibilité et la projection (P2c.2b).

    check_publishable(question, visibility, "M")                 # lève E_REPORT_DEV_ONLY ou E_REPORT_PRIVATE
    entry = publishable_answer(question, answer, visibility, "M")  # contrôle, puis projection publique

Le générateur de rapport (P2c.3) appelle `publishable_answer` pour chaque réponse qu'il écrit dans `Public Report/` :
une réponse qui expose un élément privé n'est jamais écrite. La règle porte sur ce que la réponse **expose**
(`Question.exposes`), et non sur ce dont elle dépend (SPEC_RAPPORT_PUBLIC, section 5). La visibilité est une entrée
(un objet `Visibility`) : la clé `visibility` du twin s'y branchera sans changer ce module.
"""

from __future__ import annotations

from ..errors import UserError
from ..questions import Answer, Visibility, public_answer
from ..questions.visibility import applicable
from .catalog import DEV, Question

MODES = {"direct": "D", "mcmc": "M"}


def exposed_elements(question: Question, mode: str) -> tuple[str, ...]:
    """Les éléments de visibilité que `question` expose à un twin de ce mode (« D »/« M », ou « direct »/« mcmc »).
    Un élément d'un autre mode n'existe pas dans ce twin : la PDF d'un twin direct, les paramètres du sol d'un mcmc."""
    return () if question.exposes == DEV else tuple(applicable(question.exposes, MODES.get(mode, mode)))


def check_publishable(question: Question, visibility: Visibility, mode: str) -> None:
    """Refuse la question si elle est réservée au développeur (E_REPORT_DEV_ONLY) ou si elle expose un élément privé
    (E_REPORT_PRIVATE, qui nomme les éléments). Silencieux sinon."""
    if question.exposes == DEV:
        raise UserError("E_REPORT_DEV_ONLY", question=question.id)
    private = visibility.private_elements(question.exposes, MODES.get(mode, mode))
    if private:
        raise UserError("E_REPORT_PRIVATE", question=question.id, elements=", ".join(private))


def publishable_answer(question: Question, answer: Answer, visibility: Visibility, mode: str,
                       instance: str | None = None) -> Answer:
    """La réponse telle que le rapport la publie : `check_publishable`, puis la projection publique (liste blanche).
    `instance` est la référence de l'instance du space, ajoutée à la provenance."""
    check_publishable(question, visibility, mode)
    return public_answer(question.id, answer, instance)
