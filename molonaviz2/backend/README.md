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
