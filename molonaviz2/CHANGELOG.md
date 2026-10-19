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
