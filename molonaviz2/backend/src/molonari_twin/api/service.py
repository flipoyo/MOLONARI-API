"""TwinAPI : poser une question du catalogue à un twin d'un dossier de twins.

    api = TwinAPI("examples/twins")
    answer = api.ask("twin_03", "exchanged_volume", {"start": "2016-06-28T00:00:00"})

Étapes : la question existe (E_QUESTION_UNKNOWN), ses paramètres sont valides (E_QUESTION_PARAMS, avec un
problème par champ), la question s'adresse à un twin (E_QUESTION_NO_TWIN sinon : voir ask_global), le twin existe
(E_TWIN_NOT_FOUND), la question a un sens pour son mode (E_QUESTION_NOT_AVAILABLE), et ses résultats existent si
elle en a besoin (E_TWIN_NOT_COMPUTED, ou E_TWIN_FROZEN si la configuration a changé après le calcul). Une question
sur le dossier (`folder`) saute la lecture de la configuration : elle la lit elle-même, valide ou non. Les résultats relus sont gardés en cache, tant que leur
metadata.json ne change pas. La provenance de la réponse nomme le twin, son twin_id et ses résultats.
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections import OrderedDict
from pathlib import Path

from pydantic import ValidationError

from ..errors import Issue, UserError
from ..questions import Answer, model_difference
from .catalog import CATALOG, Question, TwinContext

TWIN_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MODES = {"direct": "D", "mcmc": "M"}


class TwinAPI:
    def __init__(self, twins_folder: str | Path, cache_size: int = 8):
        self.twins_folder = Path(twins_folder)
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, int], object] = OrderedDict()
        self._lock = threading.Lock()
        self._by_key = {key: q for q in CATALOG for key in (q.id, q.ref)}

    # Catalogue ---------------------------------------------------------------------------------

    def question(self, question_id: str) -> Question:
        """La question d'identifiant `question_id` (« exchanged_volume ») ou de référence (« HN-12 »)."""
        if question_id not in self._by_key:
            raise UserError("E_QUESTION_UNKNOWN", question=question_id)
        return self._by_key[question_id]

    def questions(self, twin: str | None = None) -> list[Question]:
        """Le catalogue ; pour un twin donné, les seules questions qui ont un sens pour son mode."""
        if twin is None:
            return list(CATALOG)
        mode = MODES[self._config(twin).mode]
        return [q for q in CATALOG if q.requires != "none" and mode in q.modes]

    # Twins -------------------------------------------------------------------------------------

    def twin_folder(self, ref: str) -> Path:
        """Dossier du twin `ref` (un nom de dossier `twin_NN`, sans chemin). E_TWIN_NOT_FOUND sinon."""
        folder = self.twins_folder / ref
        if not TWIN_REF.match(ref) or ".." in ref or not (folder / "twin.yaml").is_file():
            raise UserError("E_TWIN_NOT_FOUND", ref=ref, folder=str(self.twins_folder))
        return folder

    def _config(self, ref: str):
        from ..twin.config import load_twin

        return load_twin(self.twin_folder(ref))

    def _results(self, folder: Path):
        from ..twin.results import load_results

        key = (str(folder), (folder / "metadata.json").stat().st_mtime_ns)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        results = load_results(folder)
        with self._lock:
            self._cache[key] = results
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return results

    # Questions ---------------------------------------------------------------------------------

    def parameters(self, question: Question, params: dict | None):
        try:
            return question.params.model_validate(params or {})
        except ValidationError as error:
            issues = [Issue(".".join(map(str, e["loc"])) or None, "E_PARAM_INVALID", prefixed=True, message=e["msg"])
                      for e in error.errors()]
            raise UserError("E_QUESTION_PARAMS", question=question.id, issues=issues) from None

    def ask(self, ref: str, question_id: str, params: dict | None = None) -> Answer:
        question = self.question(question_id)
        arguments = self.parameters(question, params)
        if question.requires == "none":
            raise UserError("E_QUESTION_NO_TWIN", question=question.id)
        folder = self.twin_folder(ref)
        origin = {"twin": ref}
        if question.requires == "folder":
            answer = question.run(TwinContext(ref=ref, config=None, folder=folder), arguments)
            return self._with_provenance(answer, question, origin)
        config = self._config(ref)
        mode = MODES[config.mode]
        if mode not in question.modes:
            raise UserError("E_QUESTION_NOT_AVAILABLE", question=question.id, ref=ref, mode=config.mode)
        twin = TwinContext(ref=ref, config=config, folder=folder)
        if question.requires == "results":
            twin.results, details = self._computed(ref, config)
            origin.update(details)
        return self._with_provenance(question.run(twin, arguments), question, origin)

    def _computed(self, ref: str, config) -> tuple[object, dict]:
        """Résultats du twin `ref` et leur origine (twin_id, dossier) : E_TWIN_FROZEN, E_TWIN_NOT_COMPUTED."""
        from ..twin.run import twin_status

        status = twin_status(config)
        if status.state == "modified":
            raise UserError("E_TWIN_FROZEN", name=config.name, previous=status.previous_id)
        if not status.has_results:
            raise UserError("E_TWIN_NOT_COMPUTED", ref=ref)
        return self._results(status.results_dir), {"twin_id": status.twin_id, "results": str(status.results_dir)}

    def model_difference(self, forward_ref: str, inversion_ref: str) -> Answer:
        """Différence directe forward − inversion aux trois capteurs (P2c.1d) entre deux twins calculés du dossier :
        `forward_ref` en mode direct, `inversion_ref` en mode mcmc (E_QUESTION_NOT_AVAILABLE sinon). Ce n'est pas une
        question du catalogue, qui porte sur un seul twin : le générateur de rapport l'appelle pour un space de deux
        instances. La provenance nomme les deux twins."""
        loaded = {}
        for role, ref, mode in (("forward", forward_ref, "direct"), ("inversion", inversion_ref, "mcmc")):
            config = self._config(ref)
            if config.mode != mode:
                raise UserError("E_QUESTION_NOT_AVAILABLE", question="model_difference", ref=ref, mode=config.mode)
            loaded[role] = self._computed(ref, config)
        answer = model_difference(loaded["forward"][0], loaded["inversion"][0])
        origin = {"forward": forward_ref, "inversion": inversion_ref,
                  "forward_twin_id": loaded["forward"][1]["twin_id"], "inversion_twin_id": loaded["inversion"][1]["twin_id"]}
        return dataclasses.replace(answer, provenance={**answer.provenance, **origin})

    def ask_global(self, question_id: str, params: dict | None = None) -> Answer:
        """Pose une question qui n'a pas besoin de twin (`requires="none"`) : E_QUESTION_TWIN_REQUIRED sinon."""
        question = self.question(question_id)
        arguments = self.parameters(question, params)
        if question.requires != "none":
            raise UserError("E_QUESTION_TWIN_REQUIRED", question=question.id)
        return self._with_provenance(question.run(TwinContext(ref="", config=None), arguments), question, {})

    @staticmethod
    def _with_provenance(answer: Answer, question: Question, origin: dict) -> Answer:
        provenance = {**answer.provenance, "question": question.id, "ref": question.ref, **origin}
        return dataclasses.replace(answer, provenance=provenance)
