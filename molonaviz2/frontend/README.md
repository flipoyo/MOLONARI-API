# Molonaviz 2 (application)

Application de bureau PySide6 pour l'inversion bayésienne des mesures MOLONARI : laboratoires et capteurs, études, points, nettoyage, modèle direct, inversion, résultats, fenêtre des twins. Elle ne calcule rien elle-même : elle appelle le backend [`molonari-twin`](../backend/) par sa façade `molonari_twin.facade`, dans le même processus.

Installation, utilisation pas à pas et twins : [README de l'espace](../README.md).

## Lancer

Depuis `molonaviz2/` (l'espace uv) :

```bash
uv sync
uv run molonaviz2                  # application
uv run molonaviz2 --selftest       # petit calcul sans fenêtre
uv run molonaviz2 twin …           # alias de molonari-twin …, sans Qt
```

## Traduction

Les textes de l'application sont dans `src/molonaviz2/i18n/fr.json` ; les messages d'erreur du backend restent dans le catalogue du backend. `molonaviz2.i18n.load_catalog` réunit les deux, et Qt traduit l'interface avec ce catalogue réuni. `make strings` (depuis l'espace) liste les textes sans traduction.

## Tests

```bash
cd frontend
uv run pytest                      # interface hors écran (pytest-qt)
```

`tests/test_architecture.py` vérifie que l'application n'importe du backend que sa façade, n'importe aucune bibliothèque de calcul (JAX, ArviZ, scipy), et que `molonaviz2 twin …` ne charge pas Qt.

## Distribution

```bash
make package             # depuis molonaviz2/ : PyInstaller, dist/Molonaviz2/ (et .app sur macOS)
make selftest-package
```

La spec (`packaging/molonaviz2.spec`) embarque le backend avec ses données (`fr.json`) et ses métadonnées : la version du code, et donc les `twin_id`, sont les mêmes depuis les sources et depuis l'exécutable.

## Version

Version dans `pyproject.toml`, historique dans [CHANGELOG.md](CHANGELOG.md) (celui de Molonaviz 2 depuis 0.1), tags `molonaviz2-vX.Y.Z`.
