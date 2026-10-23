"""TwinAPI : poser une question du catalogue à un twin d'un dossier de twins.

    api = TwinAPI("examples/twins")
    answer = api.ask("twin_03", "exchanged_volume", {"start": "2016-06-28T00:00:00"})

Étapes : la question existe (E_QUESTION_UNKNOWN), ses paramètres sont valides (E_QUESTION_PARAMS, avec un
problème par champ), le twin existe (E_TWIN_NOT_FOUND), la question a un sens pour son mode
(E_QUESTION_NOT_AVAILABLE), et ses résultats existent si elle en a besoin (E_TWIN_NOT_COMPUTED, ou E_TWIN_FROZEN
si la configuration a changé après le calcul). Les résultats relus sont gardés en cache, tant que leur
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
from ..questions import Answer
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
        return [q for q in CATALOG if mode in q.modes]

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
        from ..twin.run import twin_status

        question = self.question(question_id)
        arguments = self.parameters(question, params)
        config = self._config(ref)
        mode = MODES[config.mode]
        if mode not in question.modes:
            raise UserError("E_QUESTION_NOT_AVAILABLE", question=question.id, ref=ref, mode=config.mode)
        twin = TwinContext(ref=ref, config=config)
        origin = {"twin": ref}
        if question.requires == "results":
            status = twin_status(config)
            if status.state == "modified":
                raise UserError("E_TWIN_FROZEN", name=config.name, previous=status.previous_id)
            if not status.has_results:
                raise UserError("E_TWIN_NOT_COMPUTED", ref=ref)
            twin.results = self._results(status.results_dir)
            origin.update(twin_id=status.twin_id, results=str(status.results_dir))
        answer = question.run(twin, arguments)
        provenance = {**answer.provenance, "question": question.id, "ref": question.ref, **origin}
        return dataclasses.replace(answer, provenance=provenance)
