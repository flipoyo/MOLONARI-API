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

### T11 : niveaux d'accès `user` et `admin`, commandes `questions` et `ask` ; T10 : test de `report --output`

- `api/access.py` (`LeveledAPI`), une couche au-dessus de `TwinAPI` et du générateur de rapport, qui ne changent pas. Le **user** n'interroge que les échanges d'eau, la chaleur et la température (`water_flux`, `exchange_direction`, `exchanged_volume`, `heat_budget`, `temperature`, sur les twins direct et mcmc ; `sensor_intervals` en est exclue, elle révèle les mesures) ; tout le reste est réservé à l'**admin**. Sans authentification, le niveau oriente l'usage : ce n'est pas une sécurité, et aucun contrôle d'identité n'existe.
- Réponses du user par liste blanche, provenance réduite à `PUBLIC_PROVENANCE` (question, référence, paramètres) : ni graine, ni chemin, ni `twin_id`, ni nom de point, ni échantillons. Eau : la projection publique du rapport (`public_answer`). Chaleur et température : une liste distincte de `PUBLIC_DATA` (`USER_DATA`), de sorte que le contenu du rapport public ne change pas (ces deux questions y restent refusées). `questions/public.py` gagne `project_answer`, le noyau commun. `temperature` exige toujours `z` ou `t`.
- Nouveau code `E_ACCESS_ADMIN_ONLY` (traduit), HTTP **403**. Il est levé avant tout autre contrôle (paramètres, twin).
- `molonari-twin questions [<twin>]` et `molonari-twin ask <twin> <question> [--params '<json>']`, avec `--level {user,admin}` (`user` par défaut) et `--json`. Une couche mince sur `TwinAPI`, sans FastAPI : `--json` donne le JSON de `GET /v1/questions` et de `POST /v1/twins/{ref}/questions/{id}` ; une erreur est écrite comme celle de `/v1` sur la sortie standard (code de sortie 1) ; `--params` a le contenu du corps du POST (un JSON invalide est l'erreur codée `E_REQUEST_INVALID`, sans trace). Sans `--json`, texte lisible : libellé traduit, valeurs, unités, avertissements, et un résumé (nombre, minimum, maximum) pour une longue série.
- `serve --local --level {user,admin}` (`user` par défaut) ; `create_app(level=…)` reste à `admin` par défaut. Au niveau `user` : `/v1/questions` ne liste que les cinq questions ; les autres questions, `POST /v1/twins/{ref}/run`, `/v1/tasks/*` et les questions globales sont refusés en 403 ; `/v1/info` reste accessible ; la réponse npz est filtrée comme la réponse JSON. Le niveau est écrit au démarrage sur la sortie d'erreur.
- `question_out` passe de `server/app.py` à `contract/catalog.py` (avec `questions_out`) : le catalogue se liste sans FastAPI. `/v1` ne change pas.
- T10 : `tests/test_report_output.py` (`slow`) génère le rapport d'un space de twins 2.0.0 avec `--output <point>/public_reports` et balaie les octets écrits, entrées des `.npz` décompressées comprises : ni graine (texte et binaire), ni chemin absolu. Aucun défaut révélé, aucun changement de code du générateur.
- `API_VERSION` ne change pas (1.0.0-alpha.7) : aucune route ni aucun modèle de sortie nouveau, le schéma OpenAPI est identique. Le 403 ne se produit qu'à un niveau que le serveur choisit au lancement.
### T5 à T7 : virtual lab, `lab build` et `processed_data`

- `lab.yaml` (`twin/lab.py`, `schema: molonari-lab/1.0.0`) : le laboratoire (`laboratory`), les points couverts (`points.<point>.folder`) et leur nettoyage (`points.<point>.processing`, mêmes clés que `processing` d'un twin 2.0.0), et la version du code (`code_version`, facultative : une différence avec le paquet est l'avertissement `W_LAB_CODE_VERSION`, pas un refus). Les clés inconnues sont refusées, les chemins sont relatifs à la racine des données (`MOLONARI_DATA`, sinon le `data/` de l'espace cgs qui contient le lab) et n'en sortent pas.
- `molonari-twin lab build <dossier du lab> --point <point>` : enchaîne `pipeline.load_point` et `pipeline.clean_point` (aucun calcul dupliqué) et écrit `<racine des données>/processed_data/<point>/measurements.csv` (`CLEANED_COLUMNS`) et `manifest.json` (`molonari-processed/1.0.0`) : nom du lab, version du code, empreintes des fichiers bruts du point et des fichiers du laboratoire utilisés (capteur de pression, tige, thermomètre), profondeurs, décalage, sigma_P, sigma_T, `utc_offset`, `processing`, lignes, début, fin et `cleaned_sha256`. Écriture atomique (`<point>.partial`, puis remplacement). **Aucun horodatage** : un second build des mêmes entrées avec le même code donne les mêmes octets. Un point absent de `lab.yaml` est refusé.
- Le CSV est écrit comme `frame_digest` compte les mesures (dates à la seconde, flottants à 17 chiffres) et relu par `read_cleaned_csv(exact=True)` : l'analyseur de flottants rapide de pandas se trompe d'un ulp sur certaines valeurs, ce qui changerait l'empreinte. Le comportement par défaut de `read_cleaned_csv` (twins 1.x, import) ne change pas.
- Twin 2.0.0 : `data: {processed: processed_data/Point034}` (chemin relatif à la racine des données) lit ces données sans second nettoyage. Exclusif avec `data.lab`, `data.point` et `processing` (`E_TWIN_PROCESSED_EXCLUSIVE`). Profondeurs, décalage et sigma viennent du manifeste (`sensors.sigma_*` les remplace), l'empreinte des mesures relues doit être `cleaned_sha256` (`E_PROCESSED_DIGEST`). **Le `twin_id` est celui du même twin depuis le brut, et le résultat aussi** : `canonical()` utilisait déjà `cleaned_sha256`. `metadata.json` écrit `data.processed`, `data.lab`, `data.code_version` et `data.cleaned_sha256`.
- Aucun `twin_id` existant ne change (twins 1.x, ni 2.0.0 depuis le brut : `tests/test_twin_schema.py`). `API_VERSION` ne change pas.
- Nouveaux codes, traduits : `E_LAB_CONFIG_INVALID`, `E_LAB_DATA_ROOT`, `E_LAB_POINT_UNKNOWN`, `E_LAB_BUILD_EMPTY`, `W_LAB_CODE_VERSION`, `E_PROCESSED_SCHEMA`, `E_PROCESSED_MANIFEST_KEY`, `E_PROCESSED_MANIFEST_UNREADABLE`, `E_PROCESSED_DIGEST`, `E_TWIN_PROCESSED_EXCLUSIVE`.

### P2c.3 : générateur de rapport public et `molonari-twin report`

- `molonari_twin/report/` : `generate_report(twin)` écrit `Public Report/` dans le dossier du twin (`--output` pour un autre dossier). Le **space** (provisoire, jusqu'aux operating spaces, P2b.1) est celui que désigne la clé `report` du twin donné : `report.models.forward` (twin direct) et `report.models.inversion` (twin mcmc), dossiers frères ; sans `report.models`, le twin seul. Le générateur tourne au niveau **dev** : il ne lit ni `orx.yaml` ni la graine, et pose ses questions par `TwinAPI`.
- Entrées, dans un ordre fixe : par modèle `describe`, `provenance` et les `report.entries` (flux d'eau et RMSE au minimum) ; convergence publique (inversion) ; `compare` (forward − inversion) si les deux modèles sont là ; par défaut, si la visibilité le permet, HN-42 (propriétés du sol) et, pour l'inversion, BN-32 et BN-34 (toujours en npz). Une entrée obligatoire refusée fait échouer le rapport (rien n'est écrit) ; une entrée par défaut refusée est notée dans `skipped` avec son code. Un rapport sans entrée est refusé (`E_REPORT_EMPTY`).
- `report.json` (`report_format` 1.0.0) : `space`, `twin_semver` (null : SemVer composé en attente de P0.4), `generated_at` (date du calcul, jamais l'horloge : deux générations donnent les mêmes octets), `generator_version`, `platform`, `visibility`, `entries[]` (`file`, `format`, `question`, `ref`, `model`, `uncertainty`, `parameters`, `sha256`), `skipped[]`, `figures[]` (`file`, `from_entry`, `sha256`), `signature` et `license` (null). `recipe_sha256` est l'empreinte de la recette dev, sans graine ni sel (T12).
- Étiquettes : `model` (`forward`, `inversion`, ou `both` pour `compare`) et `uncertainty` (`none`, `posterior`). Un résultat forward n'a aucun intervalle et porte `W_NO_UNCERTAINTY` (nouveau code, traduit).
- Une entrée est la réponse `Answer` en JSON, ou en npz au-delà de `NPZ_THRESHOLD` (1 Mio, provisoire : décision P0.5f). Figures : SVG écrits sans matplotlib, sur des primitives communes (`report/svgplot.py` : graduations 1-2-5, axes avec unités, légende, palette Okabe-Ito) ; flux d'eau et RMSE ; les données de chaque figure sont celles de l'entrée dont elle dérive.
- Visibilité appliquée : la plus stricte des twins du space, élément par élément (un élément n'est comparé qu'aux twins où il existe).
- Écriture atomique (`Public Report.partial/`, puis remplacement) : un échec laisse l'ancien rapport intact et aucun dossier partiel ; un fichier étranger est retiré à la régénération.
- Projections publiques de HN-40 (RMSE, biais, RMSE totale ; ni σ ni RMSE/σ), HN-42, BN-32, BN-34 et de la comparaison (`questions/public.py`). `sigma` est retiré de la provenance publique. Les questions d'états ou de mesures (HN-20, HN-30, HN-33, BN-30) restent sans projection (T3).
- Nouveaux codes : `E_REPORT_EMPTY`, `E_REPORT_MODEL_MODE`, `W_NO_UNCERTAINTY` (traduits). `API_VERSION` ne change pas (aucune route ni sortie de `/v1` nouvelle).

## [0.8.0-alpha.1] - 2026-10-28

Phase 2 du plan twin v3 : issues P2.1 (pipeline du brut, avec BN-03 et BN-04) et P2c.1 (questions du rapport public). Les questions ne chargent toujours pas JAX (23 au total).

- P2.1 : le pipeline du brut au résultat, avec BN-03 et BN-04 (sortis du lot C4, car ils ne dépendent que du pipeline brut).
- P2c.1 : les questions du rapport public. Le contenu des résultats n'entre pas dans le `twin_id` (configuration et version du code seulement) : ce lot n'a changé aucun identifiant. Ceux de cette version sont en bas de la section.
- P2c.2 : visibilité des questions et projections publiques. Chaque question déclare ce qu'elle **expose** (`Question.exposes`, colonne « Accès » du catalogue ; obligatoire). `questions/visibility.py` (`Visibility`, défaut : tout privé sauf la PDF), `questions/public.py` (liste blanche de HN-02 et HN-43 et des agrégats publiables), `api/publication.py` (refus : `E_REPORT_DEV_ONLY`, `E_REPORT_PRIVATE`, `E_REPORT_NO_PROJECTION`). La graine sort de HN-02 et de HN-43 (patch « accès » : elle est réservée au niveau ORX). **`API_VERSION` passe à 1.0.0-alpha.5** : le champ `seed` disparaît de la réponse de HN-02 ; le reste de `/v1` ne change pas.

### P2.1f : dates avec fuseau

> **NOTE PROVISOIRE, à régler les années suivantes (`TODO(fuseau)`).** La convention appliquée aux bruts est **UTC+2 toute l'année** (`io.dates.PROVISIONAL_UTC_OFFSET`), décidée par l'équipe à titre provisoire. L'heure réelle des horloges des loggers (CET en hiver, CEST en été) doit être **confirmée auprès des opérateurs terrain**, puis la gestion CET/CEST traitée proprement à la place de ce décalage fixe.

- **Ce que déclarent les bruts** : tous les exports HOBOware du dépôt de données (Point034, 047, 051-1, 051-2, 52, 53) écrivent dans l'en-tête de la colonne des dates un décalage fixe, « Date Heure, GMT+01:00 », en été comme en hiver. Cet en-tête ne reflète pas l'heure d'été réelle.
- **Convention provisoire** : un en-tête qui déclare `GMT+01:00` est lu comme `UTC+2`, avec l'avertissement `W_MEASURES_TIMEZONE_OVERRIDDEN` (pas une erreur). Un en-tête `GMT+02:00` est gardé tel quel, sans avertissement. Tout autre décalage d'en-tête garde le comportement décrit plus bas (conflit ou erreur). Une date qui porte elle-même son décalage (`2016-06-27T12:00:00+01:00`) est exacte et n'est jamais remplacée.
- **Les dates ne bougent pas** : elles restent des dates sans fuseau, écrites telles quelles. Seul `utc_offset` (sur `ParsedMeasures`, `RawPoint`, `LoadedPoint` et `PipelineResult`) change : UTC = date - `utc_offset`. Les mesures nettoyées de `Demo_twin_0.molo` (Point034) ne changent pas.
- Point2024-1 à 3 (pression `GMT+02:00`, température `GMT+01:00`) : les deux sont lus en UTC+2, il n'y a donc plus de conflit ni de décalage à faire.
- Si la pression et la température déclarent tout de même deux décalages différents (par exemple des dates à décalage explicite), les dates de la pression sont ramenées exactement dans le repère de la température, avec l'avertissement `W_PIPELINE_TIMEZONE_SHIFTED`.
- `timezone` (pipeline) n'est plus refusé : c'est une **vérification** (`+02:00`, `UTC+2`, `GMT+2`) contre le `utc_offset` appliqué. Il ne choisit rien.
- Rien n'est supposé : sans déclaration, les dates restent telles qu'écrites, `utc_offset` est `None` et `W_MEASURES_TIMEZONE_UNDECLARED` le dit.
- Erreurs : `E_PIPELINE_TIMEZONE` est supprimé et remplacé par `E_PIPELINE_TIMEZONE_INVALID` (nom de zone comme `Europe/Paris`, décalage hors de ±14 h), `E_PIPELINE_TIMEZONE_CONFLICT`, `E_PIPELINE_TIMEZONE_UNDECLARED`, `E_PIPELINE_TIMEZONE_AMBIGUOUS` (un seul des deux fichiers déclare), `E_MEASURES_TIMEZONE_CONFLICT` (une date contredit le fichier), `E_MEASURES_TIMEZONE_MIXED`, `E_MEASURES_TIMEZONE_PARTIAL`, `E_DATE_OFFSET_INVALID`. Ces codes ne sont pas dans le contrat `/v1` : `API_VERSION` ne change pas.

### P2.2 : schéma molonari-twin/2.0.0

- Lecture du schéma `molonari-twin/2.0.0` : le twin part du brut. `data: {lab, point}` désigne le laboratoire et le point dans le dépôt de données (chemins relatifs à sa racine, qui ne peuvent en sortir) ; `processing: {start, end, methods, excluded}` règle le nettoyage. Le pipeline du brut (`twin/pipeline.py`) fournit les profondeurs, le décalage et les σ. `sensors.sigma_*` peut les remplacer.
- Racine du dépôt de données : variable `MOLONARI_DATA`, sinon le dossier `data/` de l'espace de travail cgs (là où se trouve `molonari.cgs`). Le twin.yaml n'en dit pas plus.
- ~~`mcmc.seed` obligatoire dans `twin.yaml`~~ : remplacé par la couche `orx` (P2.2c, ci-dessous). La graine entre dans le `twin_id`, car elle détermine le résultat.
- Clés retirées du 2.0.0, refusées avec leur nouvelle place : `E_TWIN_KEY_MOVED`, `E_TWIN_KEY_FROM_DATA`, `E_TWIN_CODE_VERSION_REMOVED`.
- `legacy_twin_id` : étiquette d'un twin migré depuis un 1.x (hors du `twin_id`).
- Identifiant d'un twin 2.0.0 : empreinte de la forme canonique, avec l'empreinte des mesures nettoyées à la place de celle du fichier, la graine dans `mcmc.seed` et la version du code. Le chemin du dépôt n'y entre pas. Les twins 1.x gardent leur forme canonique : `twin_01`, `twin_02`, `twin_03` et `twin_04` ont les mêmes identifiants qu'avant.
- La graine n'apparaît plus dans HN-02 (`describe`) ni dans `metadata.json["config"]`, quel que soit le schéma (P0.7). Elle reste dans le `twin_id`, qui est une empreinte.
- `metadata.json` : `files`, la SHA-256 de chaque fichier de résultats (P2.2f, absent de A3). Pour un twin 2.0.0, `data` donne le laboratoire, le point et l'empreinte des mesures nettoyées.
- `API_VERSION = 1.0.0-alpha.5` : HN-02 ne donne plus `seed`. Le snapshot OpenAPI et `docs/API_V1.md` sont régénérés.

### P2.3 : clés de publication du schéma 2.0.0

- Quatre clés nouvelles, lues et validées dans `twin.yaml` 2.0.0 seulement (les twins 1.x les refusent comme clés inconnues, et sont lus sans changement) :
  - `visibility` : un booléen par élément (`inversion_settings`, `inversion_settings_public`, `forward_parameters`, `posterior`, `states`, `raw_data`, `processed_data`). Défaut : tout privé, sauf la PDF (`posterior`) et les réglages que ses échantillons révèlent (`inversion_settings_public`), qui ne peuvent pas rester privés tant que `posterior` est public (`E_TWIN_VISIBILITY_REVEALED`). **La graine n'est pas un élément** : `visibility.seed` est refusée (`E_TWIN_VISIBILITY_SEED`).
  - `monitoring: {status: actif | inactif}`.
  - `site_label` : libellé public du site, une ligne de 120 caractères au plus, qui ne reprend ni le nom du twin ni celui du point (`E_TWIN_SITE_LABEL_DISTINCT`).
  - `report: {models: {forward, inversion}, entries: [{question, parameters}]}`. Chaque question doit exister et lire une configuration ou des résultats ; ses paramètres sont validés par le modèle de la question ; pas d'entrée en double ; `models` nomme deux dossiers de twins différents. Sans la clé, le rapport par défaut : `water_flux` à 0,9 et `fit_quality`.
- Contrôles du rapport : un rapport déclaré vide ou sans `water_flux` ni `fit_quality` est refusé (`E_TWIN_REPORT_EMPTY`, `E_TWIN_REPORT_MINIMUM`), de même qu'un twin mcmc sans flux d'eau à 90 % (`E_TWIN_REPORT_LEVEL`) ; `mcmc.quantiles` doit contenir 0,05 et 0,95 dès qu'une entrée, ou le rapport par défaut, demande un intervalle à 90 % (`E_TWIN_REPORT_QUANTILES`).
- Aucune de ces clés n'entre dans le `twin_id` : elles ne changent pas le résultat. Aucun identifiant de référence ne change.
- HN-02 (`describe`) donne `site_label`, `monitoring_status`, `visibility.<élément>` (défauts compris) et le rapport demandé (`report_declared`, `report_questions`, `report_parameters`, `report_forward`, `report_inversion`), donc le rapport par défaut quand la clé `report` est absente (`report_declared` vaut alors faux).
- `API_VERSION = 1.0.0-alpha.6` : sorties de HN-02. Snapshot OpenAPI et `docs/API_V1.md` régénérés.
- `Visibility.from_config(config)` (`questions/visibility.py`, P2c.2) construit la déclaration à partir de la clé ; `ELEMENTS` y vient de `twin/config.py` (`VISIBILITY_ELEMENTS`), sans second jeu de noms. La projection publique de HN-02 ajoute `site_label` et `monitoring_status` (la carte d'identité) ; ni `visibility.*` ni `report_*` n'y entrent (ils sont dans `report.json`).
- Reste à faire : refuser dans le générateur (P2c.3) les questions réservées au développeur, lire `report.entries` pour écrire le rapport, et contrôler la comparabilité des deux twins de `report.models` (operating spaces).
### P2.2c : la graine dans orx.yaml (couche orx)

- Un twin 2.0.0 lit sa graine dans `orx.yaml`, à côté de `twin.yaml` (couche `orx`, décision P0.7d du 2026-10-28) : `mcmc.seed` et la plateforme de référence (`platform` : `python`, `platform`, `processor`, `device`, `jax`, `jaxlib`, `numpy`, étiquette qui n'entre pas dans le `twin_id`). Nouveau module `twin/orx.py` ; `schema: molonari-orx/1.0.0` est facultatif ; les clés inconnues sont refusées (chemins `orx.mcmc.seed`…).
- **Sans `orx.yaml` (ou sans graine), un twin 2.0.0 reste lisible** : `load_twin`, HN-02, les questions de résultats et BN-11 répondent. `TwinConfig.seed` vaut `None` (`has_seed` est faux) et `twin_id` lève `E_TWIN_REQUIRED` (`orx.mcmc.seed`). Le dev retrouve les résultats par le registre (nom du twin → `twin_id`), et une recette modifiée après le calcul est toujours détectée (E_TWIN_FROZEN) par comparaison avec `metadata.json["config"]`, qui est la recette sans graine. Seul un changement de la graine seule échappe au dev.
- **`run_twin` refuse un twin sans graine** avant tout calcul, y compris pour renvoyer un résultat existant : `TwinConfigError` (`E_TWIN_CONFIG_INVALID`) dont le problème est `orx.mcmc.seed` / `E_TWIN_REQUIRED`. Il en va de même pour la commande `run` et la tâche `twin.run`.
- Un `twin.yaml` 2.0.0 qui contient une graine (`mcmc.seed`, ou `seed` au niveau supérieur) est refusé : `E_TWIN_SEED_ORX`, avec la place du fichier. Avant, `seed` donnait `E_TWIN_KEY_MOVED`.
- Les twins 1.x gardent leur `seed` dans `twin.yaml` et sont lus sans changement : un `orx.yaml` à côté d'eux est ignoré, et leurs identifiants de référence ne bougent pas (`twin_01` et `twin_02` 7501fd11570bbe5c, `twin_03` ab5eaa6782784ab4, `twin_04` 50cde3108f8f26ea).
- Le `twin_id` d'un twin 2.0.0 ne change pas : la forme canonique garde la graine dans `mcmc.seed`, donc un même twin donne le même identifiant que lorsqu'il l'avait dans `twin.yaml`. La graine reste absente de HN-02, de HN-43 et de `metadata.json["config"]`.
- BN-11 (`check`) : `twin_id` est vide pour un twin 2.0.0 sans graine, avec l'avertissement `W_TWIN_NO_SEED`. `API_VERSION = 1.0.0-alpha.6`. Nouveaux codes : `E_TWIN_SEED_ORX`, `E_TWIN_EXPECTED_TEXT`, `W_TWIN_NO_SEED` (traduits). Le snapshot OpenAPI et `docs/API_V1.md` sont régénérés.
### P2.4 : tests de déterminisme et `--strict`

- `tests/test_determinism.py` : du brut aux fichiers de résultats, `thin = k` garde exactement une itération sur k (même trajectoire que `thin = 1`, au bit près) ; les chaînes et itérations des champs sont celles des paramètres (recalcul du modèle direct) ; même graine, `posterior.npz` et tous les fichiers de résultats identiques au bit près (aussi entre deux processus) ; deux graines, des chaînes toutes différentes ; `--strict` refuse un dépôt modifié (twin ou code, fichier modifié ou non suivi) et un build `+g…`, sans rien écrire. Aucun changement du code, aucune version.
- `acceptance` (posterior.npz) est comptée sur les itérations gardées : elle dépend de `thin`. Déterministe, donc non corrigée ici.

### Corrigé

- Après la fusion de P2.1f, P2.2c et P2.3 : clé `E_TWIN_EXPECTED_TEXT` en double dans `errors.py` retirée ; `TOP_KEYS_2` de `config.py` définie deux fois (la seconde faisait refuser `site_label`, `monitoring`, `visibility` et `report`) ; tests de publication et de déterminisme (les `slow`, que la CI lance) alignés sur `orx.yaml` ; `API_VERSION = 1.0.0-alpha.7` (P2.2c et P2.3 visaient toutes deux alpha.6) ; snapshot OpenAPI et `docs/API_V1.md` régénérés.

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
- P2.2 : `create_twin` écrit encore du 1.1.0 (mesures nettoyées). La création d'un twin 2.0.0 depuis le brut, la migration d'un 1.x et la graine dans `orx.yaml` (P0.7d) ne sont pas faites.
- Les questions refusent encore un fuseau dans leurs paramètres de date (`E_QUESTION_PARAMS`) : leurs dates sont dans le repère des données du twin, que le pipeline expose désormais (`utc_offset`). À reprendre avec le contrat de dates de l'API.

### Identifiants de référence (0.8.0-alpha.1)
- `twin_01` et `twin_02` : `8c7952b3efe34b53` ; `twin_03` : `8724404bf1d92706` ; `twin_04` : `f7d4db454041be34`.
- Ils changent par rapport à 0.7.0-alpha.3 parce que la version du code entre dans `engine_version()`, pas parce que le calcul a changé : avec la version `0.7.0-alpha.3`, le code de cette version redonne `7501fd11570bbe5c`, `ab5eaa6782784ab4` et `50cde3108f8f26ea`.
- Calculés avec la version publiée (`release_version()`, sans suffixe local).

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
