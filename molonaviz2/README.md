# Molonaviz 2

Application de bureau pour l'inversion bayésienne des mesures de température et de pression dans la zone hyporhéique (projet MOLONARI). Elle remplace Molonaviz (PyQt5) avec les mêmes fonctions : laboratoires et capteurs, études, points de prélèvement, nettoyage des mesures, modèle direct et inversion MCMC (DREAM).

## Installation

Prérequis : [uv](https://docs.astral.sh/uv/), Python 3.12 ou 3.13, et [ComplexGitSync](https://github.com/flipoyo/ComplexGitSync) (`cgitsync`).

Molonaviz 2 s'installe avec l'ensemble des dépôts MOLONARI, que cgs monte côte à côte à partir du manifeste `molonari.cgs` du dépôt MOLONARI1D (branche `CGS-MOLO`) :

```bash
cgitsync bootstrap molonari.cgs            # MOLONARI1D, MOLONARI-API, MOLONARI-data, MOLONARI-computo…
export CGSHOME=<dossier indiqué à la fin du bootstrap>
cd "$CGSHOME/Molonaviz/molonaviz2"
uv sync          # installe l'application et les outils de développement
uv run molonaviz2
```

Le manifeste doit installer les dépôts sur la branche `CGS-MOLO` (`project = { name = "MOLONARI1D", default_branch = "CGS-MOLO" }`) : sur `main`, MOLONARI-API ne contient pas encore Molonaviz 2.

Le projet utilise `../../pyheatmy` (dépôt MOLONARI-computo, monté par cgs à la racine de l'espace) comme moteur de référence pour les tests uniquement. Il n'est pas nécessaire pour utiliser l'application, et l'exécutable distribué ne l'embarque pas. Sans lui, par exemple dans un clone seul de MOLONARI-API, l'application s'installe et se lance avec `uv sync --frozen --no-default-groups` puis `uv run --frozen --no-default-groups molonaviz2`.

## Utilisation

1. **Fichier › Nouveau projet.** Un projet est un seul fichier `.molo`, qui contient tout : capteurs, mesures, notices, schémas et résultats.
2. **Laboratoire › Importer un laboratoire.** C'est le dossier qui contient `temperature_sensors/`, `pressure_sensors/` et `shafts/`, au même format que Molonaviz.
3. **Étude › Nouvelle étude**, liée à ce laboratoire.
4. **Point › Importer un point** (Ctrl+I). Choisissez le dossier du point : les fichiers sont repérés, les champs remplis, et un aperçu montre ce qui sera importé avant toute écriture.
5. **Onglet Nettoyage.** Choisissez la période, les méthodes de détection des valeurs aberrantes (IQR, Z-score), et les exclusions manuelles éventuelles, puis appliquez.
6. **Onglet Modèle direct.** Définissez les couches, puis lancez le calcul (F5).
7. **Onglet Inversion.** Réglez les bornes a priori, puis lancez DREAM (F5). La convergence se suit en direct, et Échap annule.
8. **Onglet Résultats.** On y trouve les cartes de température et de flux, les profils, les distributions a posteriori et le tableau de diagnostic (ArviZ). Exports possibles : CSV, NetCDF, figures en PNG, SVG ou PDF.

Une ancienne base Molonaviz (`Molonari.sqlite`) se convertit avec **Fichier › Importer une base Molonaviz** : laboratoires, études, points, mesures brutes et nettoyées. Les anciens résultats de calcul ne sont pas repris ; il faut relancer les modèles.

L'interface est en français par défaut, avec l'anglais disponible (**Affichage › Langue**). Les températures s'affichent en °C, K ou °F (**Affichage › Unité de température**).

**Aide › Notice des paramètres** (Maj+F1) explique chaque réglage des calculs : définition, unité, valeur par défaut, valeurs admises, clé de `twin.yaml` et conseils. Maj+F1 sur un champ ouvre directement sa fiche, et l'infobulle de chaque champ en reprend le résumé.

### Formats de fichiers acceptés

- **Mesures HOBO (export HOBOware).**
  - La ligne de titre et la colonne `#` sont acceptées.
  - Les colonnes sont repérées d'après l'en-tête. Les températures en °F ou en K sont converties en °C.
  - Les lignes d'événements du logger (valeurs vides) sont écartées, et l'aperçu donne leurs numéros.
- **Format propre** : date suivie de 2 colonnes (pression) ou de 4 colonnes (températures), séparées par `,`, `;` ou une tabulation.
- **Dates.** L'application reconnaît une soixantaine d'écritures, et même un mélange d'écritures dans un même fichier. **Un fichier dont les dates ne sont pas chronologiques est refusé**, avec le numéro de la ligne fautive. C'est en général le signe qu'un tableur a reconverti des dates (par exemple `09/01/24` devenu `2009/1/24`). Il faut alors réexporter le fichier depuis le logiciel du logger.

## Twins (jumeaux numériques)

Un twin est la copie numérique d'un point de mesure : une colonne 1D de sédiments, du fond de la rivière au thermomètre le plus profond. On lui donne les mesures du capteur. Il simule la température θ(z, t) dans le sol et en déduit le flux d'eau q(t) entre la rivière et la nappe. q > 0 signifie une infiltration (la rivière alimente la nappe), q < 0 une exfiltration (la nappe alimente la rivière). Les thermomètres t₁–t₃, qui ne sont pas imposés au modèle, servent de contrôle : ce sont les résidus.

Un twin est défini entièrement par son `twin.yaml` : données, paramètres du sol, réglages numériques, graine et version du code. Même config, mêmes données et même code donnent les mêmes fichiers de résultats, octet pour octet.

```
<racine>/
  twins/                      ← dossier choisi dans l'application (Fichier › Twins)
    registry.csv              ← une ligne par twin exécuté ou reconnu comme doublon
    twin_01/twin.yaml + measurements.csv
    twin_02/…
  results/<twin_id>/          ← temperature.csv, water_flux.csv, residuals.csv, metadata.json
```

Trois règles s'appliquent :
- **Identifiant.** `twin_id` est le SHA-256 de la config canonique, qui inclut l'empreinte du fichier de mesures et la version du code. Les libellés (nom, description, point, noms des couches) n'en font pas partie.
- **Config figée.** Un twin modifié après son exécution est refusé. Pour changer un paramètre, on crée un nouveau twin.
- **Doublons.** Une config identique relancée, sous le même nom ou sous un autre, renvoie au résultat existant, sans recalcul.

Le mode `direct` fixe les paramètres du sol. Le mode `mcmc` les estime par inversion ; il est **expérimental**.

**Dans l'application**
- L'écran d'accueil et **Fichier › Twins** listent les twins du dossier choisi, avec leur état : pas encore exécuté, calculé, doublon, modifié (figé) ou invalide.
- Un double-clic ouvre la fenêtre du twin : config, identifiant, Exécuter (F5), frise θ(z, t), flux q(t), résidus et métadonnées.
- **Créer un twin…**, dans l'onglet Résultats d'un point, fige un calcul de l'application en twin.

**En ligne de commande (démo)**

```bash
uv run molonaviz2 twin validate examples/twins/twin_01   # étape 2 : validation
uv run molonaviz2 twin id       examples/twins/twin_01   # étape 3 : twin_id et état dans le registre
uv run molonaviz2 twin run      examples/twins/twin_01   # étapes 4-5 : exécution, résultats dans examples/results/<id>
uv run molonaviz2 twin run      examples/twins/twin_02   # étape 6 : même config sous un autre nom → doublon
uv run molonaviz2 twin list     examples/twins
```

Les exemples fournis portent tous sur Point034. Leurs identifiants valent pour la version 0.2.0 : ils changent avec la version du code, qui entre dans le `twin_id`.

| Twin | Mode | `twin_id` (0.2.0) | Contenu |
|---|---|---|---|
| `twin_01` | direct | `45ec70ecf74b2f33` | monocouche, −log₁₀ k = 12 |
| `twin_02` | direct | `45ec70ecf74b2f33` | copie renommée de `twin_01`, pour montrer la détection de doublon |
| `twin_03` | mcmc | `2d80e9fb04492ce3` | inversion courte (8 chaînes, 300 itérations, graine 1) |
| `twin_04` | mcmc | `5109a4ba2ac2ed32` | créé depuis l'application, à partir d'une inversion |

Pour réinitialiser la démo : fermer l'application, puis `rm examples/twins/registry.csv; rm -rf examples/results`.

Le schéma complet de `twin.yaml`, avec toutes les clés acceptées, est documenté dans `src/molonaviz2/twin/config.py`.

## Calcul

- **Modèle direct** : écoulement puis chaleur en 1D, avec un schéma semi-implicite. C'est un portage en JAX du modèle de pyheatmy (`linear_system.py`), compilé avec `jit` et `lax.scan`.
- **Inversion** : DREAM, avec toutes les chaînes calculées en parallèle (`vmap`). Le burn-in s'arrête dès que le critère de Gelman-Rubin passe sous le seuil, et les chaînes aberrantes y sont corrigées. Chaque calcul enregistre sa configuration, sa graine, la version du moteur et l'appareil utilisé ; « Réutiliser cette configuration » relance un calcul à l'identique.
- **Précision** : le calcul se fait toujours en float64. Le GPU n'est utilisé qu'avec CUDA sous Linux, car JAX n'offre pas le float64 sur Metal (Mac) ni de GPU natif sous Windows. La barre d'état indique l'appareil utilisé.
- **Perméabilité** : elle se saisit en −log₁₀ k, avec k en m², comme dans Molonaviz.
- **Échange latéral d'eau (q_s)** : pas pris en charge dans cette version. Les clés `q_s` et `t_lateral` sont refusées dans un `twin.yaml`, et ignorées à la relecture d'un ancien projet `.molo`.

## Tests

```bash
uv run pytest            # suite rapide (~1 min) : moteur, données, interface (hors écran)
uv run pytest -m slow    # inversion DREAM sur une colonne synthétique aux paramètres connus
```

Le moteur JAX est validé contre pyheatmy. Avec pyheatmy passé en float64, les résultats concordent à 1e-9 près en relatif, sur les températures comme sur les flux, en monocouche, multicouche et pas de temps variable, y compris sur les mesures réelles de Point034. Avec pyheatmy tel quel, qui calcule en float32, l'écart reste sous 0,05 K.

## Distribution

```bash
make package             # PyInstaller : dist/Molonaviz2/ (et dist/Molonaviz2.app sur macOS)
make selftest-package    # lance un petit modèle direct dans l'exécutable construit
```

La version de l'exécutable est lue dans `pyproject.toml`, comme celle de l'application : les `twin_id` sont les mêmes depuis les sources et depuis l'exécutable.

Il faut un build par système d'exploitation, réalisé sur ce système. Un exécutable CUDA se construit sous Linux, avec `jax[cuda12]` installé.

## Versions

Molonaviz 2 suit [SemVer 2.0.0](https://semver.org/lang/fr/). La version n'est écrite que dans `pyproject.toml`, et chaque version a sa section dans [CHANGELOG.md](CHANGELOG.md), qui donne aussi les règles. `python tools/check_version.py` vérifie que les deux sont cohérents. Les tags sont de la forme `molonaviz2-vX.Y.Z`.

## Organisation du code

```
src/molonaviz2/
  domain/      objets du domaine et configurations de calcul (valeurs par défaut, validation)
  io/          lecture des fichiers de terrain, import des anciennes bases Molonaviz
  storage/     fichier projet .molo (SQLite, schéma versionné, écritures transactionnelles)
  processing/  nettoyage des mesures
  engine/      moteur JAX (modèle direct, DREAM), adaptateur pyheatmy pour les tests
  twin/        twins versionnés : twin.yaml, twin_id, registre, exécution, ligne de commande
  workers/     calculs en arrière-plan (annulation, progression)
  ui/          interface PySide6 : fenêtre, explorateur, onglets d'un point, graphiques
  i18n/        traduction française (fr.json) ; `make strings` liste les chaînes non traduites
```
