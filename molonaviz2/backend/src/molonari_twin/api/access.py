"""Niveaux d'accès aux questions : `user`, `dev`, `opx`.

    api = LeveledAPI(TwinAPI("twins"), "user")
    api.questions()                         # les 5 questions du niveau user
    api.ask("twin_03", "water_flux")        # réponse réduite à la liste blanche
    api.ask("twin_03", "residuals")         # E_ACCESS_USER_ONLY

Le USER n'interroge que les échanges d'eau, la chaleur et la température ; tout le reste est réservé à dev et opx
(`dev`, `opx` : l'API complète, telle que la rend TwinAPI). La couche est au-dessus de TwinAPI et du générateur de rapport, qui
ne changent pas. **Ce n'est pas une sécurité** : sans authentification, le niveau oriente l'usage (un guide pour la
ligne de commande et pour `serve --local --level user`). Aucune identité n'est contrôlée.

Au niveau `user`, une réponse est réduite à une liste blanche de grandeurs et à la provenance publique
(`PUBLIC_PROVENANCE` : la question, sa référence, ses paramètres) : ni graine, ni chemin, ni `twin_id`, ni nom de
point, ni échantillons. Les trois questions d'eau ont la projection publique du rapport (`questions.public`). La chaleur
et la température ont leur liste ici, **distincte** de `PUBLIC_DATA` : elles restent refusées dans le rapport public
(états du twin, tension T3), dont le contenu ne change donc pas.

`admin` reste un alias de `opx` pour la compatibilité.
"""

from __future__ import annotations

from ..errors import UserError
from ..questions import Answer
from ..questions.public import PUBLIC_DATA, project_answer, public_answer
from .catalog import Question
from .service import TwinAPI

USER = "user"
DEV = "dev"
OPX = "opx"
ADMIN = "admin"  # alias for opx
LEVELS = (USER, DEV, OPX)

# Les questions du niveau user, sur les twins direct et mcmc. `sensor_intervals` (HN-33) est exclue : elle révèle les mesures.
WATER_QUESTIONS = ("water_flux", "exchange_direction", "exchanged_volume")
USER_QUESTIONS = (*WATER_QUESTIONS, "heat_budget", "temperature")

# Grandeurs que le niveau user reçoit de la chaleur et de la température (liste blanche, distincte de PUBLIC_DATA).
USER_DATA: dict[str, tuple[str, ...]] = {
    "heat_budget": ("advective", "conductive", "lateral", "total", "advective_share", "conductive_share", "lateral_share",
                    "dominant", "depth", "points", "start", "end"),
    "temperature": ("temperature", "depth", "time"),
}

assert set(WATER_QUESTIONS) <= set(PUBLIC_DATA) and set(USER_QUESTIONS) == set(WATER_QUESTIONS) | set(USER_DATA)


class LeveledAPI:
    """TwinAPI vue depuis un niveau d'accès. Aux niveaux `dev` et `opx`, elle ne change rien."""

    def __init__(self, api: TwinAPI, level: str = OPX):
        # Normalize admin (alias) to opx for backward compatibility
        if level == ADMIN:
            level = OPX
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r} (expected one of {', '.join(LEVELS)}, or 'admin' as alias for 'opx')")
        self.api = api
        self.level = level

    @property
    def restricted(self) -> bool:
        """True if the level is user (restricted access); False for dev and opx."""
        return self.level == USER

    def question(self, question_id: str) -> Question:
        return self.api.question(question_id)

    def questions(self, twin: str | None = None) -> list[Question]:
        """Le catalogue de ce niveau (pour un twin, les questions qui ont un sens pour son mode)."""
        listed = self.api.questions(twin)
        return listed if not self.restricted else [q for q in listed if q.id in USER_QUESTIONS]

    def require_admin(self, name: str) -> None:
        """E_ACCESS_USER_ONLY au niveau user : pour ce qui n'est pas une question (lancer un twin, suivre les tâches)."""
        if self.restricted:
            raise UserError("E_ACCESS_USER_ONLY", name=name)

    def ask(self, ref: str, question_id: str, params: dict | None = None) -> Answer:
        question = self.api.question(question_id)
        self.allow(question)
        return self.restrict(question, self.api.ask(ref, question_id, params))

    def ask_state(self, folder, question_id: str, params: dict | None = None) -> Answer:
        """Comme `ask`, pour un état généré (`generate`) : un dossier de résultats, sans twin.yaml ni registre."""
        question = self.api.question(question_id)
        self.allow(question)
        return self.restrict(question, self.api.ask_state(folder, question_id, params))

    def ask_global(self, question_id: str, params: dict | None = None) -> Answer:
        question = self.api.question(question_id)
        self.allow(question)
        return self.restrict(question, self.api.ask_global(question_id, params))

    def allow(self, question: Question) -> None:
        """E_ACCESS_USER_ONLY si la question n'est pas du niveau user (avant tout autre contrôle : paramètres, twin)."""
        if self.restricted and question.id not in USER_QUESTIONS:
            raise UserError("E_ACCESS_USER_ONLY", name=question.id)

    def restrict(self, question: Question, answer: Answer) -> Answer:
        """La réponse telle que ce niveau la reçoit : complète pour dev et opx, sa liste blanche pour le user."""
        if not self.restricted:
            return answer
        if question.id in WATER_QUESTIONS:
            return public_answer(question.id, answer)
        return project_answer(answer, USER_DATA[question.id])
