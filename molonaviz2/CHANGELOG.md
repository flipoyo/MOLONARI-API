# Changelog de Molonaviz 2

Toutes les évolutions notables de Molonaviz 2 sont notées ici.
Format : [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versions : [SemVer 2.0.0](https://semver.org/lang/fr/).

## Règles de version

- La version est écrite une seule fois, dans `pyproject.toml`. Le code la lit avec `importlib.metadata`.
- Pré-versions admises : `-alpha.N`, `-beta.N` et `-rc.N`, qui ont aussi un sens en PEP 440.
- Commits conventionnels : `fix` fait monter le CORRECTIF, `feat` le MINEUR, `!` ou `BREAKING CHANGE` le MAJEUR. Tant que la version est en `0.y.z`, un changement incompatible fait monter le MINEUR.
- Chaque version a sa section ici. `python tools/check_version.py` le vérifie, comme `tests/test_version.py`.
- Tags : `molonaviz2-vX.Y.Z`, posés sur `CGS-MOLO` après la fusion. La CI (`molonaviz2-version`) vérifie que le tag correspond à `pyproject.toml`.
- La version entre dans `engine_version()`, donc dans chaque `twin_id`. Une nouvelle version donne de nouveaux identifiants, et un twin déjà calculé passe à l'état « modifié » : il faut créer un nouveau twin pour utiliser le nouveau code.

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
