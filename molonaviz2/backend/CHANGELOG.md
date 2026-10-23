# Changelog de molonari-twin

Toutes les évolutions notables du backend des twins MOLONARI sont notées ici.
Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions : [SemVer 2.0.0](https://semver.org/lang/fr/).

Jusqu'à 0.6.0-alpha.1, le backend faisait partie du paquet molonaviz2 : son historique est dans [../frontend/CHANGELOG.md](../frontend/CHANGELOG.md).

## Règles de version

- La version est écrite une seule fois, dans `pyproject.toml`. Le code la lit avec `importlib.metadata`.
- Pré-versions admises : `-alpha.N`, `-beta.N` et `-rc.N`, qui ont aussi un sens en PEP 440.
- Commits conventionnels : `fix` fait monter le CORRECTIF, `feat` le MINEUR, `!` ou `BREAKING CHANGE` le MAJEUR. Tant que la version est en `0.y.z`, un changement incompatible fait monter le MINEUR.
- Chaque version a sa section ici. `python tools/check_version.py` le vérifie, comme `tests/test_version.py`.
- Tags : `molonari-twin-vX.Y.Z`, posés sur `CGS-MOLO` après la fusion. La CI (`molonaviz2-version`) vérifie que le tag correspond à `pyproject.toml`.
- La version entre dans `engine_version()` (`molonari-twin X.Y.Z · jax …`), donc dans chaque `twin_id`. Une nouvelle version du backend donne de nouveaux identifiants ; une version du seul frontend n'en change aucun.

## [0.7.0-alpha.2] - 2026-10-23

Lot C2, suite de la phase 3 du plan twin : le contrat de l'API et les questions par HTTP. Les questions lisent la configuration ou les résultats d'un twin ; elles répondent tout de suite, sans tâche et sans charger JAX.

### Ajouté
- Le paquet `molonari_twin.contract` : modèles pydantic de l'API (erreurs, tâches, réponses, catalogue), modèles des paramètres des questions, `API_VERSION = 1.0.0-alpha.2`, et le codec des réponses (`Answer` ⇄ JSON et ⇄ npz).
- Le paquet `molonari_twin.api` : le catalogue des questions et `TwinAPI`, utilisable sans HTTP (par exemple dans un notebook). 9 questions du catalogue 1.0.0 :
  - `describe` (HN-02), `water_flux` (HN-10), `exchanged_volume` (HN-12), `temperature` (HN-30), `sensor_intervals` (HN-33), `provenance` (HN-43) ;
  - `residuals` (BN-31), `chain_data` (BN-34), `sampling_layout` (BN-36).
- Les endpoints `GET /v1/questions` (catalogue, avec le schéma JSON des paramètres) et `POST /v1/twins/{ref}/questions/{id}` (réponse en JSON, ou en npz avec `Accept: application/x-npz`).
- Les fonctions pures `water_flux`, `twin_exchanged_volume`, `temperature_at`, `sensor_report`, `residuals`, `describe`, `describe_sampling` et `provenance` dans `questions`. `sensor_report` reprend le calcul de la bande prédictive de l'application (même graine).
- La référence des questions, générée à partir du catalogue : [docs/API_V1.md](docs/API_V1.md) (`python tools/gen_api_doc.py`, vérifiée par les tests).
- Le contrat OpenAPI figé dans `tests/fixtures/openapi_v1.json` : tout changement du contrat fait échouer un test.
- Codes d'erreur : `E_QUESTION_PARAMS`, `E_PARAM_INVALID`, `E_QUESTION_UNKNOWN`, `E_QUESTION_NOT_AVAILABLE`, `E_TWIN_NOT_COMPUTED`, `E_REQUEST_INVALID`, `E_QUESTION_PERIOD_EMPTY`, `E_QUESTION_OUT_OF_RANGE`, `E_QUESTION_Z_OR_T` et `E_QUESTION_SIGMA_REQUIRED`, traduits.

### Modifié
- `engine_version()` lit la version de jax dans les métadonnées du paquet, sans importer jax. La chaîne est la même : aucun `twin_id` ne change pour cette raison.
- `SamplingLayout` et `sampling_layout` passent dans `engine/layout.py`, sans JAX ; `engine/jax/dream.py` les réimporte (mêmes objets).
- Les modèles de `server/schemas.py` passent dans le contrat ; le fichier est supprimé.
- Une requête mal formée répond aussi `{code, field, params, message_en}` (`E_REQUEST_INVALID`), au lieu du `{"detail": …}` de FastAPI.
- pydantic devient une dépendance du backend.

### Corrigé
- `LocalTaskRunner` nettoie une tâche (fichiers partiels) avant de la passer à son état terminal, et non après : un client qui voit `cancelled` ou `failed` ne trouve plus de fichiers partiels. Le test de l'annulation forcée échouait par intermittence sous la charge de la suite complète.
- Les identifiants de référence de 0.7.0-alpha.1 ci-dessous avaient été relevés sur un arbre modifié : ils portaient le suffixe local `+g….dirty`. Ce sont maintenant ceux de la version taguée.

### Identifiants de référence (0.7.0-alpha.2)
- `twin_01` et `twin_02` : `0492305d15af546b` ; `twin_03` : `0dcf7eee6055596f` ; `twin_04` : `f0346f838ce08236`.
- Calculés avec la version publiée (`release_version()`, sans suffixe local), donc ceux de la version taguée.

## [0.7.0-alpha.1] - 2026-10-23

Lot C1, première partie de la phase 3 du plan twin. L'API confie les calculs à un `TaskRunner` et ne calcule jamais dans son processus. En phase 5, Celery prendra le relais derrière la même interface.

### Ajouté
- Le paquet `molonari_twin.tasks` contient :
  - le protocole `TaskRunner` et la clé d'idempotence `task_key` ;
  - le magasin SQLite des tâches et de leurs événements ;
  - `LocalTaskRunner` : un worker par tâche dans un sous-processus, une annulation coopérative puis forcée, un timeout dur, la reprise après un arrêt brutal et le nettoyage des résultats partiels ;
  - les jobs `twin.run` et `diag.sleep`.
- Le paquet `molonari_twin.server`, en extra optionnel `server` (FastAPI, uvicorn), expose :
  - `GET /v1/info` et `POST /v1/twins/{ref}/run` ;
  - `GET` et `DELETE /v1/tasks/{id}`, et `GET /v1/tasks/{id}/result` ;
  - `GET /v1/tasks/{id}/events`, en SSE avec reprise par `Last-Event-ID`.

  Les erreurs ont la forme `{code, field, params, message_en}`.
- La commande `molonari-twin serve --local` lance l'API, sur `127.0.0.1` uniquement.
- Les codes d'erreur `E_TASK_FINISHED`, `E_TASK_INTERRUPTED`, `E_TASK_KIND_UNKNOWN`, `E_TASK_NOT_FINISHED`, `E_TASK_NOT_FOUND`, `E_TASK_TIMEOUT`, `E_TWIN_NOT_FOUND`, `E_SERVER_EXTRA_MISSING` et `E_DIAG_FAILURE`, avec leur traduction.

### Identifiants de référence (0.7.0-alpha.1)
- `twin_01` et `twin_02` : `65cd2ff96603f343` ; `twin_03` : `3a5cfa6b1d158dc5` ; `twin_04` : `b67597158b4e52ba` (corrigés en 0.7.0-alpha.2 ; ceux d'abord notés ici portaient un suffixe local).

## [0.6.0] - 2026-10-22

Premier numéro stable du backend en paquet séparé : clôt la séparation du backend et du frontend (#37, lots B1 à B3) et rassemble la pré-version 0.6.0-alpha.1 ci-dessous.

### Ajouté
- Le catalogue français du backend ne contient plus que ses chaînes : modèles des messages d'erreur, ligne de commande, marqueurs `N_` (145 entrées). Il est vérifié par `tests/test_catalog.py` : toutes les chaînes traduites, aucune entrée inutilisée (#37).
- `MOLONARI_LANG` choisit la langue de la ligne de commande (#37).

### Modifié
- `MOLONAVIZ_LANG`, ancien nom de la variable, reste lue quand `MOLONARI_LANG` n'est pas définie (#37).
- `tools/extract_strings.py` relève les chaînes paquet par paquet ; `tests/test_messages.py` ne lit plus que le code du backend (#37).

### Identifiants de référence (0.6.0)
- `twin_01` et `twin_02` : `1fe696e072167811` ; `twin_03` : `5e5be1abb4833551` ; `twin_04` : `01fd14746ddae774`.

## [0.6.0-alpha.1] - 2026-10-22

Premier numéro du backend en paquet séparé (#37, lot B2). Le code est celui de molonaviz2 0.6.0-alpha.1, déplacé dans `molonaviz2/backend/`.

### Ajouté
- Paquet `molonari_twin` (distribution `molonari-twin`) : données, nettoyage, moteur JAX et DREAM, twins, questions et façade. Il ne dépend ni de PySide6, ni de pyqtgraph, ni de matplotlib (#37).
- Commande `molonari-twin validate|id|run|list`, aussi lancée par `python -m molonari_twin` (#37).
- La façade exporte aussi la traduction pure (`N_`, `LANGUAGES`, `load_catalog`, `translate`, `render`) et `backend_version` (#37).

### Modifié
- `engine_version()` vaut `molonari-twin X.Y.Z[+g…] · jax …`. Le suffixe local se calcule à partir des tags `molonari-twin-v*` et du seul dossier `backend/`, et la provenance du code porte sur ce dossier (#37).
- `t()` et `install_language()` quittent l'i18n du backend : ils sont dans le frontend (#37).
- Le schéma canonique des twins reste `molonaviz-twin/1` : les `twin_id` ne dépendent pas du nom du paquet, seulement de sa version.

### Identifiants de référence (molonari-twin 0.6.0-alpha.1)
- `twin_01` et `twin_02` : `1edb6a9a3f307ba0` ; `twin_03` : `6aa2aaf841f74f74` ; `twin_04` : `3535bbcdb0adda21`.
