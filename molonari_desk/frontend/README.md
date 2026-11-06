# MOLONARI Desk (application)

Application de bureau PySide6 pour l'inversion bayésienne des mesures MOLONARI : laboratoires et capteurs, études, points, nettoyage, modèle direct, inversion, résultats, fenêtre des twins. Elle ne calcule rien elle-même : elle appelle le backend [`molonari-twin`](../backend/) par sa façade `molonari_twin.facade`, dans le même processus.

Installation, utilisation pas à pas et twins : [README de l'espace](../README.md).

Notices : [utilisation](../docs/NOTICE_UTILISATEUR.md) et [scientifique](../docs/NOTICE_SCIENTIFIQUE.md). La table des paramètres de la notice scientifique est générée depuis `molonari_desk.notice` par `python tools/gen_notice_tables.py` (depuis `molonari_desk/`), et vérifiée par `tests/test_notice_doc.py`.

## Lancer

Depuis `molonari_desk/` (l'espace uv) :

```bash
uv sync
uv run molonari-desk                  # application
uv run molonari-desk --selftest       # petit calcul sans fenêtre
uv run molonari-desk twin …           # alias de molonari-twin …, sans Qt
```

## Traduction

Les textes de l'application sont dans `src/molonari_desk/i18n/fr.json` ; les messages d'erreur du backend restent dans le catalogue du backend. `molonari_desk.i18n.load_catalog` réunit les deux, et Qt traduit l'interface avec ce catalogue réuni. `make strings` (depuis l'espace) liste les textes sans traduction.

## Tests

```bash
cd frontend
uv run pytest                      # interface hors écran (pytest-qt)
```

`tests/test_architecture.py` vérifie que l'application n'importe du backend que sa façade, n'importe aucune bibliothèque de calcul (JAX, ArviZ, scipy), et que `molonari-desk twin …` ne charge pas Qt.

## Distribution

```bash
make package             # depuis molonari_desk/ : PyInstaller, dist/molonari_desk/ (et .app sur macOS)
make selftest-package
```

La spec (`packaging/molonari_desk.spec`) embarque le backend avec ses données (`fr.json`) et ses métadonnées : la version du code, et donc les `twin_id`, sont les mêmes depuis les sources et depuis l'exécutable.

## Version

Version dans `pyproject.toml`, historique dans [CHANGELOG.md](CHANGELOG.md) (celui de l'application depuis 0.1, sous le nom molonaviz2 jusqu'à 0.6.0), tags `molonari-desk-vX.Y.Z`.
