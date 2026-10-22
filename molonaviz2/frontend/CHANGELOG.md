# Changelog de Molonaviz 2

Toutes les évolutions notables de Molonaviz 2 sont notées ici. Depuis 0.6.0-alpha.2, ce fichier suit l'application (`frontend/`) ; le backend a le sien : [../backend/CHANGELOG.md](../backend/CHANGELOG.md).
Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions : [SemVer 2.0.0](https://semver.org/lang/fr/).

## Règles de version

- La version est écrite une seule fois, dans `pyproject.toml`. Le code la lit avec `importlib.metadata`.
- Pré-versions admises : `-alpha.N`, `-beta.N` et `-rc.N`, qui ont aussi un sens en PEP 440.
- Commits conventionnels : `fix` fait monter le CORRECTIF, `feat` le MINEUR, `!` ou `BREAKING CHANGE` le MAJEUR. Tant que la version est en `0.y.z`, un changement incompatible fait monter le MINEUR.
- Chaque version a sa section ici. `python tools/check_version.py` le vérifie, comme `tests/test_version.py`.
- Tags : `molonaviz2-vX.Y.Z`, posés sur `CGS-MOLO` après la fusion. La CI (`molonaviz2-version`) vérifie que le tag correspond à `pyproject.toml`.
- Jusqu'à 0.6.0-alpha.1, la version entrait dans `engine_version()`, donc dans chaque `twin_id`. Depuis 0.6.0-alpha.2, c'est la version du backend (`molonari-twin`) qui y entre : une version de la seule application ne change aucun identifiant.

## [0.6.0-alpha.2] - 2026-10-22

Deux dossiers, deux paquets : le backend `molonari-twin` et l'application `molonaviz2` (#37, lot B2). Rien ne change à l'écran ni dans les calculs.

### Modifié
- `molonaviz2/` devient un espace uv à deux membres, avec un seul `uv.lock`. `backend/` contient le paquet `molonari_twin`, avec son propre CHANGELOG ; `frontend/` contient l'application `molonaviz2` (#37).
- L'application n'appelle le backend que par `molonari_twin.facade`. Elle ne dépend plus directement de JAX, scipy ni ArviZ (#37).
- La version du code d'un twin (`code_version`, et donc le `twin_id`) est celle du backend : `molonari-twin X.Y.Z · jax …`. Une nouvelle version de la seule application ne change plus les identifiants (#37).
- `molonaviz2 twin …` reste disponible, comme alias de `molonari-twin …` (#37).
- Les tests sont répartis en deux suites (`backend/tests`, `frontend/tests`), chacune avec son test d'architecture ; la CI a un job backend sans Qt et un job frontend (#37).

### Identifiants de référence (0.6.0-alpha.2, code molonari-twin 0.6.0-alpha.1)
- `twin_01` et `twin_02` : `1edb6a9a3f307ba0` ; `twin_03` : `6aa2aaf841f74f74` ; `twin_04` : `3535bbcdb0adda21`.

## [0.6.0-alpha.1] - 2026-10-22

Façade publique du cœur : premier lot de la séparation du backend et du frontend (#37, lot B1). Rien ne change pour l'utilisateur : écrans, textes, calculs et résultats écrits restent les mêmes.

### Ajouté
- `molonaviz2.facade` : seul module du cœur que l'interface importe, avec la racine du paquet (version) et `i18n` (#37).
  - Il réexporte tels quels les types, constantes et fonctions dont l'interface a besoin : modèles et configurations, projet `.molo`, imports de fichiers, nettoyage, calcul, twins, questions et erreurs.
  - `summary_table` et `export_netcdf` chargent ArviZ seulement à l'appel ; `twin_cli(argv)` lance la ligne de commande des twins ; `twin_defaults()` donne les valeurs par défaut d'un twin.
  - L'importer ne charge ni Qt, ni JAX, ni ArviZ.

### Modifié
- `Cancelled` (annulation d'un calcul) est défini dans `errors`. `engine.jax.dream.Cancelled` reste importable : c'est la même classe (#37).
- L'interface n'importe plus `domain`, `io`, `processing`, `storage`, `engine`, `twin` ni `questions` directement. Les 38 imports « tolérés » depuis A1 passent par la façade, et le test d'architecture n'admet plus aucune exception (#37).
- L'application ne charge plus JAX ni ArviZ au démarrage : ils sont chargés au premier calcul ou au premier diagnostic (#37).
- `ui/plots/posterior` ne réexporte plus `summary_table`, `export_netcdf` ni `inference_data` (#37).

### Identifiants de référence (0.6.0-alpha.1)
- `twin_01` et `twin_02` : `801fcd52aad54165` ; `twin_03` : `7c37332fa1df829c` ; `twin_04` : `5f020c2ec719d907`.

## [0.5.0] - 2026-10-21

Bande prédictive, couverture et vue Chaînes dans l'interface (#28). Clôt le jalon 0.5 et rassemble les pré-versions 0.5.0-alpha.1 (#26) et 0.5.0-alpha.2 (#27) ci-dessous.

### Ajouté
- **Résultats › Températures** : bande prédictive aux 3 capteurs (#28).
  - Son niveau est celui de l'intervalle de mesure, et son σ celui de l'inversion.
  - Sa légende donne la couverture des mesures, par exemple « Prédictive 90 % — 91 % des mesures (n = 1800) ».
  - Les valeurs viennent seulement de `questions` (`sensor_samples`, `predictive_band`, `observed_at`, `coverage`), avec une graine fixe (`ui/predictive.py`).
  - Sans champs stockés (`store_fields: none`, ou calcul antérieur à 0.4.0), une mention l'explique ; la bande crédible et l'intervalle de mesure restent affichés.
- **Résultats › Paramètres › Chaînes** : traces des paramètres et de l'énergie en fonction de l'itération calculée, avec le choix des chaînes, l'éclaircissement d'affichage, la plage d'itérations, l'acceptation et les réinitialisations par chaîne, et le nombre d'échantillons affichés. Tout passe par `questions.chain_data` (#28).
- **Fenêtre d'un twin** : onglet « Capteurs t₁–t₃ », avec les mesures, la simulation, l'intervalle de mesure et, pour un twin mcmc dont les champs existent, la bande prédictive et sa couverture. Sinon, une mention traduite : `W_FIELD_UNAVAILABLE` (résultat 1.x) ou `E_QUESTION_NO_FIELDS` (mcmc sans champs) (#28).
- Notice : fiches « Bande prédictive » et « Chaînes » (#28).

### Modifié
- Résultats › Paramètres : les histogrammes et les chaînes sont deux vues d'un même onglet. Les anciennes traces, dont l'abscisse ignorait le `thin` du calcul, sont remplacées par la vue Chaînes (#28).

### Écarts au plan
- La bande prédictive est dessinée en contour léger (violet), pour laisser voir l'intervalle de mesure et la bande crédible.
- La graine de la bande est fixe (`PREDICTIVE_SEED = 0`) : la bande ne change pas d'un affichage à l'autre, et les tests la recalculent à l'identique.
- Les chaînes sont numérotées à partir de 0, comme dans `posterior.npz` et `chain_data`.

### Identifiants de référence (0.5.0)
- `twin_01` et `twin_02` : `136ab511459b7996` ; `twin_03` : `065d48abf5224659` ; `twin_04` : `53462d6e7119aa9c`.

## [0.5.0-alpha.2] - 2026-10-21

Cœur des questions : volumes, intervalles, bande prédictive et données par chaîne (#27).

### Ajouté
- Paquet `questions/` dans le cœur, sans Qt ni HTTP. Chaque question renvoie une `Answer(data, provenance, warnings)`, dont `data` associe un nom à une `Quantity(value, unit, interval)`. Les deux sont des dataclasses figées (#27).
- `exchanged_volume(times, q, area=None)` : volume d'eau échangé par la méthode des trapèzes, sur des pas de temps irréguliers. L'infiltration (≥ 0) et l'exfiltration (≤ 0, signée comme q) sont intégrées séparément, en coupant chaque intervalle au point où q change de signe. Le résultat est en m³/m², ou en m³ avec `area` (#27).
- `sensor_intervals(observed, sigma, level)` : intervalle de mesure ± z·σ (#27).
- `sensor_samples(result)` : échantillons de température aux 3 capteurs (S, 3, T′), depuis un résultat d'inversion ou les résultats relus d'un twin, avec des champs stockés aux capteurs ou en entier (#27).
- `predictive_band(sensor_samples, sigma, level, seed)` : bande prédictive, c'est-à-dire les quantiles des échantillons bruités par N(0, σ²), tirés avec une graine. `coverage(observed, band)` : part des mesures dans une bande, NaN ignorés, avec le nombre de points, au total et par capteur. `observed_at(times, observed, at_times)` : mesures alignées sur les dates des champs (#27).
- `chain_data(result, chains=None, thin=1, iterations=None)` : paramètres, énergies, acceptation et réinitialisations par chaîne, avec un `thin` d'affichage cohérent avec celui du calcul. Avec toutes les chaînes, `count` = `nb_chains × ⌊nb_iter / (thin_calcul·thin)⌋` (BN-36) (#27).
- Codes `E_QUESTION_CHAINS`, `E_QUESTION_LEVEL`, `E_QUESTION_NO_FIELDS`, `E_QUESTION_NO_POSTERIOR`, `E_QUESTION_SHAPES`, `E_QUESTION_SIGMA`, `E_QUESTION_THIN`, `E_QUESTION_TIMES` et `W_COVERAGE_NO_POINTS`, traduits (#27).

### Modifié
- `MEASUREMENT_LEVELS`, `z_score` et `measurement_interval` passent de `ui/units.py` à `questions/intervals.py`. `ui/units.py` les réexporte, sans changement de comportement (#27).

### Écarts au plan
- `sensor_intervals` n'existait pas sous ce nom : on déplace les fonctions de l'intervalle de mesure (B1), et `sensor_intervals` les enveloppe dans une `Answer`. Le « mode D » du plan consolidé est lu comme cet intervalle de mesure.
- Ajouts pour A6 : `sensor_samples` et `observed_at`, pour que l'interface n'ait rien à extraire ni à aligner. `chain_data` accepte aussi les résultats relus d'un twin.
- La `provenance` d'une réponse ne contient que la question et ses paramètres. La provenance du calcul (code, twin, environnement) reste dans `metadata.json`.
- Test de couverture (`slow`) : les paramètres de la colonne synthétique sont corrélés le long d'une crête, et R-hat reste au-dessus de 1,2 après 4 000 itérations, alors que les 8 chaînes sont dans le même mode (le test le vérifie). La couverture, dominée par le bruit, n'en dépend presque pas : 89,4 % avec 600 itérations, 89,3 % avec 4 000.

### Identifiants de référence (0.5.0-alpha.2)
- `twin_01` et `twin_02` : `31c13214a9932979` ; `twin_03` : `8177890967a15401` ; `twin_04` : `9b10240b767a1f0c`.

## [0.5.0-alpha.1] - 2026-10-21

Provenance des twins et mode `--strict` (#26).

### Ajouté
- `metadata.json["provenance"]` (#26) :
  - **dépôt du code et dépôt du twin** : URL distante sans identifiants, commit, tags exacts, branche ou HEAD détaché, état propre ou modifié limité au chemin ;
  - **environnement** : Python, plateforme, processeur, appareil, versions de jax, jaxlib et numpy ;
  - **empreinte SHA-256** du `.gts` cgs le plus récent ;
  - **avertissements** `W_PROVENANCE_NO_GIT`, `W_PROVENANCE_NOT_A_REPOSITORY` et `W_PROVENANCE_NO_REMOTE` quand une information manque. La valeur vaut alors `null`, et l'exécution continue. `twin run` et la fenêtre du twin affichent ces avertissements.
- `molonaviz2 twin run --strict` (`run_twin(strict=True)`) refuse de calculer, avant tout calcul et en citant tous les motifs : `E_STRICT_PRERELEASE`, `E_STRICT_LOCAL_BUILD`, `E_STRICT_DIRTY`, `E_STRICT_UNTAGGED` et `E_STRICT_NO_PROVENANCE` (#26).
- `gitinfo.py` : seule partie du code qui appelle git (#26).
- `release_version()` : version publiée du code chargé, sans suffixe local (#26).

### Modifié
- `engine_version()` porte un suffixe local quand le code chargé vient d'un dépôt git et n'est pas exactement sa version publiée. Elle est calculée une fois par processus (#26).
  - **Suffixes** : `+g<commit>` si le tag `molonaviz2-vX.Y.Z` est absent ou si `molonaviz2/` en diffère, et `.dirty` en plus si `molonaviz2/` a des modifications non commitées. L'exécutable, ou un code sans git, n'a pas de suffixe.
  - **Conséquence** : sur une branche de travail, les `twin_id` diffèrent de ceux de la version publiée.

### Écarts au plan
- pyheatmy n'est pas enregistré dans la provenance : le moteur est un portage JAX qui ne l'importe pas. pyheatmy ne sert qu'aux tests de parité (`engine/reference.py`).
- Les tags exacts sont enregistrés en liste (`tags`), car un commit peut en porter plusieurs. Une branche absente vaut `null`, avec `detached_head: true`.
- Les fichiers non suivis (et non ignorés) rendent un dépôt « modifié ».
- `tests/test_version.py` admet le suffixe local dans `engine_version()`.
- Les identifiants de référence ci-dessous sont calculés avec la version sans suffixe (`release_version()`), celle du code une fois tagué. Sur la branche, `twin id` affiche d'autres identifiants.

### Identifiants de référence (0.5.0-alpha.1)
- `twin_01` et `twin_02` : `e651e9b35210336c` ; `twin_03` : `e65140bfbf71d29c` ; `twin_04` : `d8d2e8b837c9f8c6`.

## [0.4.0] - 2026-10-21

Résultats de twin au format 2.0.0 et conventions des flux de chaleur (#25). Rassemble la pré-version 0.4.0-alpha.1 ci-dessous (#24).

### Ajouté
- Format de résultats `2.0.0`. `metadata.json` déclare :
  - `results_format` ;
  - les incertitudes des instruments (`instruments`) ;
  - les conventions physiques (`conventions`) : z, q et flux de chaleur positifs vers le bas, et flux advectif rapporté à T_ref = 0 °C (#25).
- Champs de flux en CSV : `water_flux_field.csv`, `heat_flux_advective.csv` et `heat_flux_conductive.csv`, au format de `temperature.csv`, avec la profondeur et l'unité dans la première cellule (#25).
- En mcmc, des tableaux `.npz` lisibles avec `allow_pickle=False` :
  - `quantiles.npz` ;
  - `posterior.npz` : échantillons, énergies, acceptation, réinitialisations et numéro d'itération calculée ;
  - selon `store_fields`, `fields_temperature.npz` et `fields_water_flux.npz`, avec la chaîne et l'itération de chaque champ (#25).
- `load_results` lit les formats 1.x et 2.0.0. Un résultat 1.x porte l'avertissement `W_FIELD_UNAVAILABLE`, affiché dans la fenêtre du twin (#25).
- `ForwardResult.conductive_flux_down` (−λm·∂T/∂z, positif vers le bas) et la constante `HEAT_REFERENCE_C` = 0 °C (#25).

### Modifié
- Les fichiers 1.x (`temperature.csv`, `water_flux.csv`, `residuals.csv`, `posterior_*.csv`) restent identiques octet pour octet. Un dossier de résultats contient en plus les fichiers 2.0.0 (#25).

### Corrigé
- Onglet Résultats : le flux de chaleur total additionnait le flux advectif (positif vers le bas) et le flux conductif hérité de pyheatmy (positif vers le haut). Le total et la carte du flux conductif sont maintenant positifs vers le bas. L'export CSV écrit `conductive_heat_flux_down_W_m2.csv` à la place de `conductive_heat_flux_W_m2.csv` : le nom change avec le signe (#25).

### Écarts au plan
- Première cellule des CSV de champs : `depth_m|<unité>` (séparateur `|`), pour garder une seule ligne d'en-tête relue par `index_col=0`.
- La convention « positif vers le bas » s'applique aussi à la carte du flux conductif et à l'export CSV de l'onglet Résultats (décision du propriétaire), pas seulement au total.
- `posterior.npz["iteration"]` donne le numéro d'itération calculée (`(i + 1)·thin`), comme `field_iteration`.

### Identifiants de référence (0.4.0)
- `twin_01` et `twin_02` : `c126c779567c5bbb` ; `twin_03` : `e97b626a80613ebb` ; `twin_04` : `a2ab3b5bb10ec53f`.

## [0.4.0-alpha.1] - 2026-10-21

Chaînes DREAM, éclaircissement, champs par chaîne, `store_fields` et schéma de twin 1.1.0 (#24) : l'inversion garde de quoi alimenter les lots suivants (résultats de twin 2.0.0, bande prédictive, vue des chaînes). Avec les réglages par défaut, les résultats sont identiques à ceux de 0.3.1, octet pour octet.

### Ajouté
- Réglage `thin` (pas d'éclaircissement) : toutes les itérations sont calculées, mais seules les itérations multiples de `thin`, comptées après le burn-in à partir de 1, sont gardées (paramètres, énergies, acceptation, champs). À graine égale, `thin = k` donne exactement un sous-ensemble du résultat de `thin = 1` (#24).
- Réglage `fields_per_chain` : nombre de champs par chaîne, à la place de `n_field_samples`. Clé de twin seulement : l'onglet Inversion garde `n_field_samples` (#24).
- Réglage `store_fields` : ce que l'inversion garde des champs échantillonnés en plus des quantiles. `none` ; `sensors` (par défaut) : la température aux 3 capteurs ; `full` : température et flux d'eau sur la grille sous-échantillonnée (#24).
- `InversionResult` : `thin`, provenance de chaque champ (`field_chain`, `field_iteration`), réinitialisations de chaque chaîne pendant le burn-in (`resets`), et champs stockés (`sensor_temperature`, `field_temperature`, `field_water_flux`). Les calculs `.molo` enregistrés avant se relisent sans ces tableaux (#24).
- Schéma de twin `molonari-twin/1.1.0`, écrit par les nouveaux twins. Sont lus `molonaviz-twin/1` (= 1.0.0), `molonari-twin/1.0.0` et `molonari-twin/1.1.0` ; les clés `mcmc.thin`, `mcmc.fields_per_chain` et `mcmc.store_fields` ne sont admises qu'en 1.1.0 (#24).
- Codes `E_MCMC_THIN`, `E_MCMC_FIELDS_PER_CHAIN`, `E_MCMC_STORE_FIELDS` et `E_MCMC_FIELDS_BOTH` (#24).
- Onglet Inversion : réglages avancés « Pas d'éclaircissement (thin) » et « Champs stockés » ; l'encadré « Par chaîne » montre les itérations gardées sur les itérations calculées et la mémoire des champs stockés. Notice : fiches `thin`, `fields_per_chain` et `store_fields` (#24).

### Modifié
- Un twin créé depuis l'application déclare `molonari-twin/1.1.0` (il n'est pas lisible par 0.3.x). La forme canonique garde `molonaviz-twin/1` et n'inclut les nouvelles clés que si elles s'écartent de leur défaut : les `twin_id("v1")` existants ne changent pas (#24).

### Écarts au plan
- `field_iteration` compte les itérations calculées, pas les itérations gardées : l'indice dans `samples` est `field_iteration // thin − 1` (la formule `field_iteration − 1` du plan n'est juste que pour `thin = 1`).
- `sensor_temperature` est en float64 : la vérification à 1e-12 près est impossible en float32.
- Cette vérification se fait en écart relatif (1e-12) : JAX en `vmap` et le modèle direct seul ne fusionnent pas les opérations de la même façon.
- Les clés 1.1.0 sont refusées dans un twin 1.0.0, sinon la version du schéma ne servirait à rien.
- Codes ajoutés au plan : `E_MCMC_THIN` et `E_MCMC_FIELDS_PER_CHAIN`, pour valider ces réglages comme les autres.

### Identifiants de référence (0.4.0-alpha.1)
- `twin_01` et `twin_02` : `6d3dc56c5b0c5598` ; `twin_03` : `71dc5343e5b8118d` ; `twin_04` : `a42e9c298aa44a94`.

## [0.3.1] - 2026-10-21

Avertissements ruff existants (#29) : les 37 avertissements de `ruff check` (ruff 0.17.1, règles par défaut) sont corrigés, règle par règle. Aucun comportement ne change : résultats, textes affichés et tests identiques.

### Corrigé
- `RUF100` : directives `noqa` de règles non activées retirées ; la raison reste en commentaire (#29).
- `I001` : imports triés ; aucun import à effet de bord (backend matplotlib, `sys.path`) ne change de place (#29).
- `RUF007` : `itertools.pairwise` pour vérifier l'ordre des profondeurs (#29).
- `B008` : l'index racine Qt par défaut de `DataFrameModel` devient une constante de module, `NO_PARENT` (#29).
- `PLW1510`, `ISC004`, `PLR0402`, `RUF046`, `C408`, `PIE808` : une correction chacun (#29).

### Identifiants de référence (0.3.1)
- `twin_01` et `twin_02` : `cb91dd196c280674` ; `twin_03` : `a65838cf829d8a0b` ; `twin_04` : `150b1f2b0066f7d5`.

## [0.3.0] - 2026-10-21

Cœur sans Qt et codes d'erreur (#23) : première étape de la séparation entre le futur backend twin
et l'interface PySide6. Les textes affichés, en français comme en anglais, ne changent pas.

### Ajouté
- Codes d'erreur stables : chaque erreur attendue (`UserError`) et chaque problème de saisie ou de validation (`Issue`) porte un code `E_…` (erreur) ou `W_…` (avertissement), ses paramètres et le champ concerné. `errors.MESSAGES` associe chaque code à son modèle anglais, qui est aussi la clé de sa traduction dans `fr.json` (#23).
- `i18n.render(obj, tr)` : traduction d'une erreur ou d'un problème au bord (interface, ligne de commande), à partir de son code (#23).
- Ligne de commande des twins sans Qt : la langue vient de la variable d'environnement `MOLONAVIZ_LANG` (`fr`, `en`), sinon `fr` (#23).
- `tests/test_architecture.py` : le cœur s'importe et s'exécute sans PySide6 ni interface, ne produit aucune phrase traduite, et l'interface n'ajoute pas d'import du cœur hors façade. `tests/test_messages.py` : chaque code a un modèle et une traduction, et chaque appel fournit les paramètres de son modèle (#23).
- CI `molonaviz2-tests` : suite rapide, suite `slow`, et job `core-without-qt` (#23).

### Modifié
- `UserError(code, field=None, **params)` prend un code ; un texte libre reste accepté par l'API Python (code de repli `E_TEXT`), mais pas dans le code de molonaviz2 (#23).
- `i18n` est scindé : la partie pure (`translate`, `load_catalog`, `N_`) ne dépend plus de Qt ; `t` et `install_language` passent par `i18n/qt.py` (#23).
- Le script `molonaviz2` et `python -m molonaviz2` aiguillent `twin …` vers la ligne de commande avant tout import de Qt. La ligne de commande ne lit plus les préférences de l'application (#23).
- `domain.compute.Issue` est désormais `errors.Issue` (toujours importable à l'ancien emplacement) (#23).
- La description de `molonaviz2 twin -h` est traduite (#23).

### Identifiants de référence (0.3.0)
- `twin_01` et `twin_02` : `a0c8e1b5444c31ae` ; `twin_03` : `9eb09cb9985362fb` ; `twin_04` : `c99f719be178b74f`.

## [0.2.0] - 2026-10-19

Rassemble les pré-versions 0.2.0-alpha.1 et 0.2.0-alpha.2 ci-dessous.

### Ajouté
- Notice des paramètres de calcul : Aide › Notice des paramètres (Maj+F1), une fiche par réglage (définition, unité, valeur par défaut, valeurs admises, clé de `twin.yaml`, conseils), avec recherche. Les infobulles des champs en reprennent le résumé, et Maj+F1 sur un champ ouvre sa fiche (#19).
- `SETTING_LIMITS` : bornes des réglages, une seule source pour les onglets de calcul et la notice (#19).
- README : installation de l'ensemble des dépôts par cgs (branche `CGS-MOLO`), et installation sans pyheatmy (#19).

### Corrigé
- README : `twin_04` est en mode mcmc (#19).

### Identifiants de référence (0.2.0)
- `twin_01` et `twin_02` : `45ec70ecf74b2f33` ; `twin_03` : `2d80e9fb04492ce3` ; `twin_04` : `5109a4ba2ac2ed32`.

## [0.2.0-alpha.2] - 2026-10-19

### Ajouté
- Onglet Inversion, encadré « Par chaîne », mis à jour à chaque réglage : burn-in maximal et itérations conservées par chaîne, champs (T, q) par chaîne et intervalle entre deux champs (avec le nombre demandé quand l'arrondi le change), grille des champs après sous-échantillonnage et mémoire estimée, nombre maximal d'évaluations du modèle direct (#18).
- `sampling_layout()` (moteur) : seule source de ces nombres. `DreamSampler` parcourt ses blocs, sans changer les résultats (#18).
- Suivi en direct : « itération i / N (par chaîne) » ; onglet Résultats : burn-in, itérations conservées et champs par chaîne dans la ligne de provenance ; axe des traces « par chaîne » (#18).

## [0.2.0-alpha.1] - 2026-10-19

### Ajouté
- Intervalle d'incertitude de mesure autour des températures mesurées (onglet Résultats › Températures) : bande mesure ± z·σT, au niveau 1σ (68 %), 90 % (par défaut) ou 95 %. σT est celui de l'inversion, ou l'incertitude du thermomètre pour un modèle direct (#17).
- Fenêtre d'un twin, onglet Résidus : lignes à ± z·σT, au même niveau (#17).

### Corrigé
- Fenêtre d'un twin en mode direct : les lignes d'incertitude des résidus n'apparaissaient pas, faute de σT dans les métadonnées ; σT est maintenant lu dans `twin.yaml` (#17).
- La bande crédible du modèle a sa propre entrée dans la légende des graphes des capteurs (#17).

## [0.1.0] - 2026-10-19

Première version de Molonaviz 2. Elle rassemble les pré-versions 0.1.0-alpha.1 à 0.1.0-beta.1 ci-dessous, et remplace Molonaviz (PyQt5) sans le modifier.

### Ajouté
- Exécutable PyInstaller : dossier autonome, plus un `.app` sous macOS. La version est lue dans `pyproject.toml`, et les métadonnées de `molonaviz2` sont embarquées pour que `engine_version()` y soit la même que depuis les sources (#7).
- README : identifiants de référence des twins d'exemple en 0.1.0, réinitialisation de la démo, organisation du code (#7).

### Identifiants de référence (0.1.0)
- `twin_01` et `twin_02` : `7b98a3053d8ddc44` ; `twin_03` : `c193219ee1831cc1` ; `twin_04` : `603275e2d2c924af`.

## [0.1.0-beta.1] - 2026-10-19

### Ajouté
- Interface PySide6 : fenêtre principale, explorateur des études et des points, import d'un laboratoire et d'un point avec un aperçu avant toute écriture (#6).
- Onglets d'un point : Infos, Mesures, Nettoyage, Modèle direct, Inversion et Résultats (cartes θ(z, t) et des flux, profils, lois a posteriori, diagnostics ArviZ, exports CSV, NetCDF, PNG, SVG et PDF) (#6).
- Fenêtre d'un twin et section Twins de l'écran d'accueil ; **Créer un twin…** à partir d'un calcul de l'application (#6).
- Calculs en tâche de fond, avec suivi en direct et annulation par Échap (#6).
- Français et anglais ; températures en °C, K ou °F ; `make strings` liste les chaînes non traduites (#6).

## [0.1.0-alpha.5] - 2026-10-19

### Ajouté
- Twins versionnés. `twin.yaml` (schéma `molonaviz-twin/1`) est une config figée, en mode `direct` ou `mcmc` (ce dernier expérimental) (#5).
- `twin_id` : SHA-256, tronqué à 16 caractères, de la config canonique. Il comprend l'empreinte des mesures et `engine_version()`, mais pas les libellés (#5).
- Registre `registry.csv` avec les états nouveau, calculé, doublon et modifié. Les résultats sont écrits de façon atomique et sont reproductibles octet pour octet (#5).
- Ligne de commande `molonaviz2 twin validate|id|run|list` (points d'entrée `app.py` et `__main__.py`) (#5).
- Exemples `examples/twins/twin_01` à `twin_04`, tous sur Point034 (#5).

## [0.1.0-alpha.4] - 2026-10-19

### Ajouté
- Modèle direct en JAX (float64) : écoulement puis chaleur en 1D, schéma semi-implicite, portage de pyheatmy (`lax.scan`, `tridiagonal_solve`). Identique à pyheatmy passé en float64, à 1e-9 près en relatif (#4).
- Inversion DREAM vectorisée sur les chaînes (`vmap`), avec correction des chaînes aberrantes, R-hat split et diagnostics ArviZ (#4).
- `engine_version()` : `molonaviz2 <version> · jax <version>` (#4).

### Retiré
- L'échange latéral d'eau (`q_s`, `T_lat`) n'est pas pris en charge dans cette version. Il sera réintroduit par une PR dédiée (#4).

## [0.1.0-alpha.3] - 2026-10-19

### Ajouté
- Import d'un laboratoire (`temperature_sensors/`, `pressure_sensors/`, `shafts/`), au même format que Molonaviz (#3).
- Lecture des mesures HOBO (export HOBOware) et du format propre. Les colonnes sont repérées par l'en-tête, °F et K sont convertis en °C, et les lignes d'événements du logger sont écartées (#3).
- Une soixantaine d'écritures de dates. Un fichier dont les dates ne sont pas chronologiques est refusé, avec la ligne fautive (#3).
- Conversion d'une ancienne base Molonaviz (`Molonari.sqlite`) en projet `.molo` (#3).
- Nettoyage des mesures : période, valeurs aberrantes (IQR, Z-score) et exclusions manuelles (#3).

## [0.1.0-alpha.2] - 2026-10-19

### Ajouté
- Versionnage SemVer : version unique lue par `importlib.metadata`, ce CHANGELOG, `tools/check_version.py`, `tests/test_version.py` et le workflow CI `molonaviz2-version` (#9).

## [0.1.0-alpha.1] - 2026-10-19

### Ajouté
- Socle : projet uv, traduction (i18n), erreurs, objets du domaine et configurations de calcul, format de projet `.molo` (#2).
