# molonari-twin

Backend des twins MOLONARI : lecture des données de terrain, nettoyage, calcul (moteur JAX du modèle direct, inversion DREAM), twins versionnés et questions posées à un résultat. Il ne dépend ni de Qt, ni d'une bibliothèque graphique, ni de l'application [`molonaviz2`](../frontend/).

Vue d'ensemble, installation dans l'espace cgs, format des twins et des résultats : [README de l'espace](../README.md).

## Installation seule

Depuis `molonaviz2/` (l'espace uv) :

```bash
uv sync --package molonari-twin            # backend et outils de test, sans PySide6
uv sync --package molonari-twin --no-dev   # backend seul : ni pyheatmy (référence des tests), ni matplotlib
```

## Ligne de commande

```bash
uv run molonari-twin validate examples/twins/twin_01
uv run molonari-twin id       examples/twins/twin_01
uv run molonari-twin run      examples/twins/twin_01   # --strict : code publié et dépôts propres exigés
uv run molonari-twin list     examples/twins
python -m molonari_twin …                              # équivalent
```

Messages en français par défaut ; `MOLONARI_LANG=en` les affiche en anglais (l'ancien nom `MOLONAVIZ_LANG` est encore lu).

## API HTTP locale

```bash
uv sync --package molonari-twin --extra server   # FastAPI et uvicorn (extra optionnel)
uv run molonari-twin serve --local --twins-folder examples/twins   # http://127.0.0.1:8765
curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/run         # 202 + task_id
curl -N http://127.0.0.1:8765/v1/tasks/<task_id>/events         # progression (SSE)
curl -X DELETE http://127.0.0.1:8765/v1/tasks/<task_id>         # annulation
```

L'API ne calcule jamais dans son processus. Elle confie chaque calcul à un `TaskRunner` (`molonari_twin.tasks`). En local, `LocalTaskRunner` exécute chaque tâche dans un sous-processus. Son état est gardé dans une base SQLite, par défaut `$XDG_CACHE_HOME/molonari-twin/tasks.sqlite`.

L'annulation est d'abord coopérative : le DREAM la lit à chaque bloc. Si elle n'est pas honorée, le worker est terminé après un délai de grâce. Un timeout dur s'applique de la même façon.

Le flux SSE reprend après `Last-Event-ID`. Seule l'adresse `127.0.0.1` est acceptée tant que l'authentification n'existe pas (phase 5, avec Celery). Schéma OpenAPI : `/openapi.json`.

### Questions

```bash
curl http://127.0.0.1:8765/v1/questions?twin=twin_03                       # catalogue et schémas des paramètres
curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/questions/exchanged_volume \
     -H 'Content-Type: application/json' -d '{"start": "2016-06-28T00:00:00"}'
curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/questions/HN-12 -H 'Accept: application/x-npz' -o answer.npz
```

Les questions lisent la configuration ou les résultats d'un twin : elles répondent tout de suite (200), sans tâche et sans charger JAX. Référence de chaque question (paramètres, sorties, modes, erreurs) : [docs/API_V1.md](docs/API_V1.md), générée par `python tools/gen_api_doc.py`.

Sans HTTP, par exemple dans un notebook :

```python
from molonari_twin.api import TwinAPI

answer = TwinAPI("examples/twins").ask("twin_03", "exchanged_volume", {"start": "2016-06-28T00:00:00"})
answer.data["net"]   # Quantity(value, unit="m³/m²", interval=None)
```

Vingt questions, dont des diagnostics (`fit_quality`, `convergence`, `energy_breakdown`), un verdict de fiabilité (`reliability`) et des accès bruts (`fields`, `posterior_samples`). Deux sortes de questions n'ont pas besoin d'un twin calculé : `check` lit le dossier d'un twin, même invalide (la réponse dit `valid: false`), et `parameters_catalog` ne porte sur aucun twin (`POST /v1/questions/parameters_catalog`).

**Les seuils de `reliability` sont provisoires et n'ont pas été validés par un hydrogéologue** : la réponse donne `thresholds_validated: false` et les seuils utilisés. Ils se surchargent par le paramètre `thresholds` ou par un fichier YAML dont le chemin est dans `MOLONARI_THRESHOLDS`. Le flux de chaleur latéral n'est pas modélisé : `heat_budget` le dit par un avertissement. Les champs complets sont volumineux : demander `Accept: application/x-npz`.

Le contrat (`molonari_twin.contract`) contient les modèles pydantic de l'API, les modèles des paramètres et le codec des réponses (JSON, npz). Il ne contient aucune logique métier, et n'importe ni FastAPI ni JAX. Tout changement du contrat se voit dans `tests/fixtures/openapi_v1.json`.

## Façade

Le frontend n'importe que `molonari_twin.facade`. Elle réexporte tels quels les types, constantes et fonctions dont il a besoin (modèles, projet `.molo`, imports de fichiers, nettoyage, calcul, twins, questions, erreurs, traduction pure) ; l'importer ne charge ni JAX, ni ArviZ, ni Qt. Aucun autre module du backend ne l'importe.

## Tests

```bash
cd backend
uv run pytest -p no:pytest-qt        # suite rapide, sans Qt
uv run pytest -p no:pytest-qt -m slow
```

`tests/test_architecture.py` vérifie que le backend s'importe et exécute ses commandes quand PySide6, pyqtgraph, matplotlib et `molonaviz2` sont absents, qu'il ne produit aucune phrase traduite, et que la façade reste légère.

## Version

Version dans `pyproject.toml`, historique dans [CHANGELOG.md](CHANGELOG.md), tags `molonari-twin-vX.Y.Z`. La version entre dans `engine_version()` (`molonari-twin X.Y.Z · jax …`), donc dans chaque `twin_id`.
