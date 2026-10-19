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

## [0.1.0-alpha.2] - 2026-10-19

### Ajouté
- Versionnage SemVer : version unique lue par `importlib.metadata`, ce CHANGELOG, `tools/check_version.py`, `tests/test_version.py` et le workflow CI `molonaviz2-version` (#9).

## [0.1.0-alpha.1] - 2026-10-19

### Ajouté
- Socle : projet uv, traduction (i18n), erreurs, objets du domaine et configurations de calcul, format de projet `.molo` (#2).
