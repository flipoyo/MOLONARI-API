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

## [Non publié]

Phase 2 du plan twin v3 : issues P2.1 (pipeline du brut, avec BN-03 et BN-04) et P2c.1 (questions du rapport public). Les questions ne chargent toujours pas JAX (23 au total).

- P2.1 : le pipeline du brut au résultat, avec BN-03 et BN-04 (sortis du lot C4, car ils ne dépendent que du pipeline brut).
- P2c.1 : les questions du rapport public. Pas de nouvelle version ici : la version est celle de l'alpha.3, donc **aucun identifiant de référence ne change** (le `twin_id` dépend de la configuration et de la version du code, pas du contenu des résultats).
- P2c.2 : visibilité des questions et projections publiques. Chaque question déclare ce qu'elle **expose** (`Question.exposes`, colonne « Accès » du catalogue ; obligatoire). `questions/visibility.py` (`Visibility`, défaut : tout privé sauf la PDF), `questions/public.py` (liste blanche de HN-02 et HN-43 et des agrégats publiables), `api/publication.py` (refus : `E_REPORT_DEV_ONLY`, `E_REPORT_PRIVATE`, `E_REPORT_NO_PROJECTION`). La graine sort de HN-02 et de HN-43 (patch « accès » : elle est réservée au niveau ORX). **`API_VERSION` passe à 1.0.0-alpha.5** : le champ `seed` disparaît de la réponse de HN-02 ; le reste de `/v1` ne change pas.

### Ajouté
- `twin/pipeline.py` : `read_raw_point`, `resolve_instruments`, `load_point`, `clean_point`, `column_inputs` et `run_pipeline`. Il enchaîne les lecteurs (`io/lab_import.py`, `io/point_import.py`), `clean(…, CleanupSettings)` et `ColumnInputs.from_cleaned`, sans dupliquer leur code. Le brut de Point034 redonne exactement les mesures nettoyées de `examples/twins/twin_01/measurements.csv` (1436 dates, ou 1430 dans la fenêtre du twin).
- σ des instruments : `Sigma_Meas_P` du capteur de pression et `Sigma_Meas_T` du thermomètre de la tige entrent dans `ColumnInputs` ; `sigma_pressure` et `sigma_temperature` peuvent les remplacer.
- Questions sans twin `raw_series` (BN-03, mesures brutes d'un point) et `processing_preview` (BN-04, effet d'un nettoyage : mesures associées, raison d'exclusion de chaque date, décompte, σ). Elles lisent les dossiers d'un point et d'un laboratoire (`point_folder`, `lab_folder`) et répondent sans charger JAX. Nouvelle catégorie « Données et traitement » dans `docs/API_V1.md`.
- Refus explicites, par une erreur structurée : `E_PIPELINE_FORMAT_UNSUPPORTED` (exports de tableur, fichiers `.hobo`…), `E_PIPELINE_NO_TEMPERATURE_FILE`, `E_PIPELINE_NO_PRESSURE_FILE`, `E_PIPELINE_NO_INFO`, `E_PIPELINE_NO_PRESSURE_SENSOR`, `E_PIPELINE_NO_SHAFT`, `E_PIPELINE_THERMOMETER_UNKNOWN`, `E_PIPELINE_SIGMA_POSITIVE` et `E_MEASURES_UNIT_UNSUPPORTED` (pression en kPa, psi, bar… au lieu de la tension du capteur). Tous traduits.
- `fields_riverbed_flux.npz` (twins `mcmc`) : q(z = 0, t) en m/s de chacun des `n_field_samples` champs conservés, en float64, **quel que soit `store_fields`** (P2c.1b). C'est un état privé : le rapport n'en publie que l'agrégat. `InversionResult.field_riverbed_flux` et `TwinResults.riverbed_flux`. Un résultat écrit avant n'a pas ce fichier (avertissement `W_FLUX_NO_SAMPLES` dans HN-10).
- HN-10 `water_flux` : paramètre `level` (0,9 par défaut, hors de ]0, 1[ : `E_QUESTION_PARAMS`) ; en mode `mcmc`, `mean` est le flux moyen a posteriori et son `interval` celui de cette moyenne, tiré des séries stockées ; `mean_best` garde la moyenne du meilleur échantillon ; en mode `direct`, valeur seule, sans `interval`. Nouvelles sorties : `level`, `model`, `uncertainty`, `samples`, `mean_best` (P2c.1a).
- HN-40 `fit_quality` : `rmse_total` (RMSE totale), `sensor`, `model` (`forward` ou `inversion`), `uncertainty` (`none` ou `posterior`) ; en mode `mcmc`, `median_rmse` et `median_rmse_total` de la médiane a posteriori, en complément du meilleur échantillon (avertissement `W_FIT_NO_MEDIAN` sans le quantile 0,5) (P2c.1c, P2c.1e).
- `model_difference` (fonction pure et `TwinAPI.model_difference`) : différence directe forward − inversion aux trois capteurs, sans passer par les mesures : écart moyen et RMSE par capteur, puis RMSE totale. Ce n'est pas une question du catalogue (elle porte sur deux twins) ; `E_QUESTION_MODELS_INCOMPARABLE` si les dates de simulation diffèrent (P2c.1d).
- HN-45 `public_convergence` : R̂ maximal, ESS minimal, acceptation moyenne, convergence du burn-in, chaînes et échantillons par chaîne, seuils de référence (ceux par défaut de `thresholds.py`) et `thresholds_validated`, **sans verdict** ; mode `mcmc` seulement (un twin `direct` reçoit `E_QUESTION_NOT_AVAILABLE`) (P2c.1f). 21 questions au total.

### Modifié
- `API_VERSION = 1.0.0-alpha.4` (deux questions de plus) ; le contrat OpenAPI figé et `docs/API_V1.md` suivent.
- Le laboratoire lit `Sigma_Meas_T` et `Sigma_Meas_P` par leur nom (sans tenir compte de la casse), puis par position à défaut : les fichiers existants se lisent comme avant.
- `SamplingLayout.stored_bytes` (BN-36, BN-11, interface) compte q(0, t) : `none` ne vaut plus 0.
- `convergence_extremes` partage le calcul de R̂ maximal et d'ESS minimal entre HN-41 et HN-45 (mêmes valeurs).
- La version du paquet ne change pas : à monter au moment de la publication du lot.

### Pas encore fait
- P2.1f, les dates avec fuseau, attend la convention de dates (P0.5b). Un fuseau en paramètre reste refusé : `E_PIPELINE_TIMEZONE` pour le pipeline, `E_QUESTION_PARAMS` pour les questions.

## [0.7.0-alpha.3] - 2026-10-23

Lot C3, suite de la phase 3 du plan twin : fiabilité, diagnostics et accès bruts aux résultats. 11 questions de plus (20 au total), sans charger JAX ni ArviZ. **Les seuils du verdict de fiabilité sont provisoires : ils doivent être validés par un hydrogéologue avant l'API 1.0.0 (choix C9).**

### Ajouté
- `engine/stats.py`, en numpy et scipy : R̂ (split, puis rank-normalized), ESS bulk et tail, MCSE de la moyenne et HDI. Égaux à ArviZ à 1e-6 près (les tests comparent aux deux).
- Questions de lecture des résultats :
  - `exchange_direction` (HN-11) : parts d'infiltration et d'exfiltration, régime dominant, inversions ; en mode M avec `store_fields: full`, P(q ≥ 0) ;
  - `heat_budget` (HN-20) : flux de chaleur advectif, conductif et total à une profondeur, parts, processus dominant ; le flux latéral n'est pas modélisé : il vaut 0 (`W_LATERAL_NOT_MODELLED`) ;
  - `fit_quality` (HN-40), `reliability` (HN-41), `soil_properties` (HN-42) ;
  - `fields` (BN-30), `posterior_samples` (BN-32), `convergence` (BN-40), `energy_breakdown` (BN-43).
- Questions de configuration : `parameters_catalog` (BN-10, sans twin) et `check` (BN-11, sur le dossier d'un twin, même invalide : une configuration invalide est une réponse avec `valid: false`).
- `requires` accepte `none` et `folder`. `POST /v1/questions/{id}` pose une question sans twin ; `TwinAPI.ask_global`.
- `reliability` : verdict vert, orange ou rouge (le pire de six critères : convergence, ajustement, identifiabilité, couverture, données, provenance), avec une raison `W_RELIABILITY_*` par critère. Seuils dans `questions/thresholds.py`, surchargeables par le paramètre `thresholds` et par la variable `MOLONARI_THRESHOLDS` (un YAML) ; la réponse donne les seuils utilisés et `thresholds_validated: false`.
- `GET /v1/info` donne aussi `twin_schema`, `results_format`, `data_release` (null jusqu'au lot C4) et `device`. `serve` lance au démarrage la tâche `diag.device` : un worker importe JAX, pas le processus de l'API ; `device` vaut null tant qu'elle n'a pas fini.
- Codes : `W_LATERAL_NOT_MODELLED`, `W_RELIABILITY_CONVERGENCE`, `_FIT`, `_IDENTIFIABILITY`, `_COVERAGE`, `_DATA`, `_PROVENANCE`, `E_QUESTION_VARIABLE`, `E_QUESTION_NO_TWIN`, `E_QUESTION_TWIN_REQUIRED` et `E_THRESHOLDS_INVALID`, traduits.

### Modifié
- `API_VERSION = 1.0.0-alpha.3`. `InfoOut` et `QuestionOut.requires` s'élargissent.
- `split_rhat` passe de `engine/jax/dream.py` à `engine/stats.py`, que `dream.py` réimporte (même fonction).
- `chain_data` partage sa sélection des chaînes et des itérations avec `posterior_samples` (`select_samples`), sans changer de comportement.
- La conductivité hydraulique K de `soil_properties` suppose l'eau à 20 °C (μ = 1,0e-3 Pa·s) : c'est une approximation.

### Identifiants de référence (0.7.0-alpha.3)
- `twin_01` et `twin_02` : `7501fd11570bbe5c` ; `twin_03` : `ab5eaa6782784ab4` ; `twin_04` : `50cde3108f8f26ea`.
- Calculés avec la version publiée (`release_version()`, sans suffixe local).

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
