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
