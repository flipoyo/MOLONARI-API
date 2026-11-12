# MOLONARI Desk — notice d'utilisation

Cette notice s'adresse à la personne qui utilise l'application : technicien, étudiant ou chercheur qui revient du
terrain avec les fichiers d'un point de mesure. Elle explique comment installer MOLONARI Desk, importer les mesures,
les nettoyer, lancer les calculs et lire les résultats. Le modèle physique, l'inversion et la lecture critique des
résultats sont traités dans la [notice scientifique](NOTICE_SCIENTIFIQUE.md).

MOLONARI Desk s'appelait molonaviz2 jusqu'à la version 0.6.0. Voir [Ce qui change depuis molonaviz2](#ce-qui-change-depuis-molonaviz2).

## Sommaire

1. [À quoi sert MOLONARI Desk](#à-quoi-sert-molonari-desk)
2. [Installer et lancer](#installer-et-lancer)
3. [Vue d'ensemble de la fenêtre](#vue-densemble-de-la-fenêtre)
4. [Le parcours type, pas à pas](#le-parcours-type-pas-à-pas)
5. [Les twins](#les-twins)
6. [Aide intégrée, langue et unités](#aide-intégrée-langue-et-unités)
7. [Raccourcis clavier](#raccourcis-clavier)
8. [Formats de fichiers](#formats-de-fichiers)
9. [Dépannage](#dépannage)
10. [Ce qui change depuis molonaviz2](#ce-qui-change-depuis-molonaviz2)

## À quoi sert MOLONARI Desk

Un point MOLONARI est une tige plantée dans le lit d'une rivière. Elle porte quatre thermomètres à différentes
profondeurs et un capteur de pression différentielle entre la rivière et la nappe. À partir de ces mesures,
MOLONARI Desk estime **le flux d'eau qui traverse le lit de la rivière** : la rivière alimente-t-elle la nappe
(infiltration) ou la nappe alimente-t-elle la rivière (exfiltration), et avec quel débit ?

Deux calculs sont proposés :

- le **modèle direct** : vous fixez les propriétés du sol, l'application simule les températures et vous les
  comparez aux mesures ;
- l'**inversion** : l'application cherche elle-même les propriétés du sol qui expliquent les mesures (méthode
  bayésienne DREAM) et donne leur incertitude.

Tout se fait sur votre ordinateur, sans serveur ni connexion.

## Installer et lancer

### Prérequis

- [uv](https://docs.astral.sh/uv/) ;
- Python 3.12 ou 3.13 (uv peut l'installer pour vous) ;
- [ComplexGitSync](https://github.com/flipoyo/ComplexGitSync) (`cgitsync`), qui installe l'ensemble des dépôts MOLONARI.

### Installation

MOLONARI Desk se trouve dans le dépôt MOLONARI-API, dossier `molonari_desk/`. On l'installe avec les autres dépôts
MOLONARI, à partir du manifeste `molonari.cgs` du dépôt MOLONARI1D (branche `CGS-MOLO`) :

```bash
cgitsync bootstrap molonari.cgs
export CGSHOME=<dossier indiqué à la fin du bootstrap>
cd "$CGSHOME/Molonaviz/molonari_desk"
uv sync
```

Dans un clone seul de MOLONARI-API, sans les autres dépôts, utilisez
`uv sync --frozen --no-default-groups`.

### Lancer

```bash
uv run molonari-desk              # ouvre l'application
uv run molonari-desk --selftest   # petit calcul de contrôle, sans fenêtre
uv run molonari-desk --version
```

Une version autonome (sans Python) peut aussi être construite par `make package`. Elle produit
`dist/molonari_desk/molonari_desk`, et `dist/molonari_desk.app` sur macOS.

Les calculs se font toujours en double précision (float64). Ils utilisent le processeur (CPU), sauf sous Linux avec
une carte graphique NVIDIA (CUDA). La barre d'état, en bas à droite, indique l'appareil utilisé.

## Vue d'ensemble de la fenêtre

- **Panneau Projet** (à gauche). Il contient l'arborescence du projet : Laboratoires (thermomètres, capteurs de
  pression, tiges), puis Études et leurs points. Double-cliquez sur un point pour l'ouvrir. Un clic droit propose
  les actions sur l'élément : supprimer un laboratoire, une étude ou un point, ou revenir aux mesures brutes.
- **Zone centrale**. Le point ouvert y apparaît avec six onglets : **Infos**, **Mesures**, **Nettoyage**,
  **Modèle direct**, **Inversion** et **Résultats**.
- **Panneaux Détails et Journal**. Détails décrit l'élément sélectionné. Journal affiche les messages de
  l'application. Le menu **Affichage** les montre ou les masque.
- **Barre d'état**. Elle affiche le projet ouvert et l'appareil de calcul.

Un **projet** est un fichier unique `.molo`. Il contient tout : capteurs, mesures, notices, schémas d'installation
et résultats. Pour transmettre votre travail, il suffit de transmettre ce fichier.

## Le parcours type, pas à pas

**Aide › Prise en main** (F1) résume ces étapes dans l'application.

### 1. Créer ou ouvrir un projet

**Fichier › Nouveau projet…** (Ctrl+N) crée le fichier `.molo`. **Fichier › Ouvrir un projet…** (Ctrl+O) en ouvre
un existant. **Fichier › Projets récents** donne les derniers projets utilisés.

### 2. Importer un laboratoire

Un laboratoire est l'ensemble des capteurs d'une équipe. **Laboratoire › Importer un laboratoire…** (Ctrl+L)
attend un dossier qui contient trois sous-dossiers :

| Sous-dossier | Un fichier par… | Contenu |
|---|---|---|
| `temperature_sensors/` | thermomètre | nom, référence, fabricant, incertitude σ |
| `pressure_sensors/` | capteur de pression | nom, enregistreur, date d'étalonnage, étalonnage (ordonnée à l'origine, dU/dH, dU/dT), incertitude σ, thermomètre associé |
| `shafts/` | tige | nom, enregistreur, thermomètre, profondeurs des 4 capteurs |

L'aperçu liste ce qui sera importé et les fichiers écartés, avec la raison. **Rien n'est écrit dans le projet
avant de cliquer sur Importer.** On peut aussi ajouter un capteur à la main : **Laboratoire › Ajouter un capteur…**.
Ajoutez d'abord un thermomètre, car les capteurs de pression et les tiges y font référence.

L'**incertitude σ des thermomètres** compte : elle sert de valeur par défaut à l'incertitude de mesure de
l'inversion. Renseignez-la.

### 3. Créer une étude

**Étude › Nouvelle étude…** (Ctrl+Maj+N). Une étude regroupe des points mesurés avec les capteurs d'un même
laboratoire.

### 4. Importer un point

**Point › Importer un point…** (Ctrl+I). Le plus simple est de cliquer sur **Remplir depuis un dossier de point…**
et de choisir le dossier du point. L'application repère les fichiers et remplit les champs :

- **fichier info** : nom du point, capteur de pression, tige, dates d'installation et de mesure, profondeur du lit,
  décalage des capteurs (Δh) ;
- **mesures de pression** et **mesures de température** ;
- **notice** (texte) et **schéma d'installation** (image), facultatifs.

Si le fichier info ne nomme pas le capteur ou la tige, ou s'il ne donne pas la profondeur du lit, l'aperçu le
signale et vous les saisissez. L'**Aperçu des mesures** montre ce qui sera importé :

- le nombre de mesures gardées ;
- la période couverte ;
- le format de date reconnu ;
- les colonnes utilisées ;
- les conversions d'unités ;
- les lignes écartées, avec leurs numéros.

Vérifiez-le, car rien n'est écrit avant **Importer**.

### 5. Regarder les mesures (onglets Infos et Mesures)

L'onglet **Infos** reprend l'installation : étude, laboratoire, dates, profondeur du lit, décalage, capteurs,
notice et schéma. L'onglet **Mesures** montre les mesures **brutes** et **nettoyées** en tableau et en graphe :

- la température de la rivière, lue par le capteur de pression ;
- les capteurs 1 à 4 ;
- la tension du capteur de pression ;
- la charge différentielle ΔH, en mètres.

La tension est convertie en charge par l'étalonnage du capteur de pression :
ΔH = (U − T·dU/dT − ordonnée à l'origine) / (dU/dH).

### 6. Nettoyer les mesures (onglet Nettoyage)

Les calculs ne travaillent que sur les **mesures nettoyées**. Trois outils sont disponibles :

1. **Période.** Retirez l'installation et le retrait de la tige : tant que la sonde n'est pas dans le lit, ses
   températures ne décrivent pas le sédiment. **Toute la période** remet les bornes à zéro.
2. **Valeurs aberrantes.** Choisissez une méthode pour chaque variable :
   - **Aucune** ;
   - **IQR** : valeur hors de [Q1 − 1,5·IQR ; Q3 + 1,5·IQR] ;
   - **Z-score** : valeur à plus de 3 écarts-types de la moyenne.

   Le calcul se fait sur la période choisie. Une date est retirée dès qu'une variable y est aberrante.
3. **Exclusion manuelle.** Cliquez sur **Sélectionner une période sur le graphe**, placez la bande sur les dates à
   retirer, puis cliquez sur **Exclure la période sélectionnée**. Les exclusions se rétablissent avec
   **Rétablir les dates exclues à la main**.

Sur le graphe, les mesures gardées sont tracées en trait et les mesures retirées en points rouges. Le résumé donne
le nombre de dates gardées. **Rien n'est enregistré avant Appliquer le nettoyage.** Il faut au moins 3 dates pour
lancer un modèle.

Attention : appliquer un nouveau nettoyage **supprime les calculs** du point, qui reposaient sur les anciennes
mesures nettoyées. L'application demande confirmation. On peut aussi **importer** des mesures nettoyées ailleurs, ou
**exporter** celles du point.

### 7. Le modèle direct (onglet Modèle direct)

1. Définissez les **couches** de haut en bas : nom, profondeur du bas (m) et les quatre propriétés du sol :
   - −log₁₀ k, la perméabilité (12 signifie k = 10⁻¹² m²) ;
   - n, la porosité ;
   - λs, la conductivité thermique des grains ;
   - ρs·cs, la capacité thermique des grains.

   La dernière couche s'arrête au fond de la colonne, c'est-à-dire au thermomètre le plus profond plus le décalage.
2. Réglez la **discrétisation** (nombre de cellules).
3. Cliquez sur **Lancer le modèle direct** (F5).

L'onglet affiche la RMSE à chaque capteur et la RMSE globale : c'est l'écart moyen entre températures simulées et
mesurées. **Ouvrir les résultats détaillés** mène à l'onglet Résultats.

### 8. L'inversion (onglet Inversion)

1. **Couches et bornes a priori.** Pour chaque couche et chaque propriété, donnez un minimum et un maximum.
   L'inversion cherche la valeur entre ces bornes. Des bornes trop étroites excluent peut-être la bonne valeur ;
   des bornes trop larges ralentissent le calcul.
2. **Échantillonnage** :
   - incertitude de mesure σT (par défaut, celle du thermomètre) ;
   - nombre de chaînes ;
   - itérations conservées ;
   - burn-in maximal ;
   - quantiles ;
   - graine aléatoire (**Nouvelle** en tire une autre).
3. **Réglages avancés** : à laisser par défaut, sauf besoin précis. **Valeurs par défaut** les rétablit.
4. L'encadré **Par chaîne** traduit vos réglages en clair : itérations, nombre de jeux de paramètres, champs gardés,
   mémoire et nombre d'évaluations du modèle. Il se met à jour à chaque modification.
5. Cliquez sur **Lancer l'inversion** (F5).

Pendant le calcul, l'onglet suit la convergence en direct :

- l'énergie de chaque chaîne ;
- le taux d'acceptation ;
- le R-hat maximal ;
- le temps écoulé et le temps restant estimé.

Le burn-in s'arrête dès que R-hat passe sous le seuil de Gelman-Rubin. **Annuler** (Échap) arrête le calcul à la
fin du bloc en cours : rien n'est enregistré. À la fin, l'onglet indique si les chaînes ont **convergé**, ou si le
burn-in a atteint son maximum sans converger.

Une inversion prend de quelques dizaines de secondes à quelques minutes, selon la longueur des mesures, le nombre
de cellules, de chaînes et d'itérations.

### 9. Lire les résultats (onglet Résultats)

La liste en haut de l'onglet choisit le calcul à afficher : modèle direct ou inversion, avec sa date. Le résumé donne :

- le moteur et l'appareil ;
- la durée ;
- les couches et les cellules ;
- pour une inversion : la graine, les chaînes, le burn-in (convergé ou non) et les itérations conservées.

| Vue | Ce qu'elle montre |
|---|---|
| **Températures** | En haut, la carte profondeur × temps de la température simulée. En bas, les capteurs 1 à 3 : mesures et températures simulées. Après une inversion, s'y ajoutent l'intervalle de crédibilité et la **bande prédictive**, avec la part des mesures qu'elle contient. **Intervalle de mesure** règle la bande ± z·σT autour des mesures. |
| **Profils** | Profil de température en profondeur à la date choisie avec le curseur (points : mesures). |
| **Flux** | En haut, le flux d'eau à la profondeur choisie avec le curseur (m/s, **positif vers le bas** : infiltration). En bas, la carte du flux de chaleur total, advectif ou conductif (W/m², positif vers le bas). L'échelle de couleur est automatique : ± le 95ᵉ percentile des valeurs absolues, pour que le pic de l'après-midi près du lit n'écrase pas le reste ; les valeurs au-delà saturent, ce que montrent les flèches de la barre de couleur. Décochez **Échelle auto** pour régler une échelle ± W/m² commune aux trois cartes. |
| **Paramètres** (inversion) | Tableau par couche : meilleur, moyenne, écart-type, HDI 90 %, ESS et R-hat. **Histogrammes** des distributions a posteriori et **Chaînes** : trace de chaque chaîne, avec choix des chaînes, éclaircissement d'affichage et plage d'itérations. |

**Lire le signe du flux.** q > 0 : la rivière alimente la nappe (infiltration). q < 0 : la nappe alimente la rivière
(exfiltration).

**Actions** :

- **Réutiliser cette configuration** recharge les réglages dans l'onglet Modèle direct ou Inversion, pour relancer
  le même calcul ;
- **Exporter › Données en fichiers CSV…** écrit les séries et les champs ;
- **Exporter › Échantillons a posteriori (NetCDF, ArviZ)…** écrit les échantillons de l'inversion ;
- **Exporter la figure…**, sous chaque figure, l'enregistre en PNG, SVG ou PDF ;
- **Créer un twin…** fige ce calcul en twin (voir plus bas) ;
- **Supprimer** efface définitivement le calcul.

### 10. Reprendre une ancienne base Molonaviz

**Fichier › Importer une base Molonaviz…** convertit une base `Molonari.sqlite` de l'ancien logiciel en projet
`.molo` : laboratoires, études, points, mesures brutes et nettoyées. Les anciens résultats de calcul ne sont pas
repris : relancez les modèles.

## Les twins

Un **twin** (jumeau numérique) est un calcul figé et reproductible d'un point. Il tient dans un dossier `twin_xx/`
qui contient :

- un `twin.yaml` (données, couches, réglages, graine) ;
- un fichier de mesures ;
- un registre des exécutions ;
- les résultats.

Même config, mêmes données et même version du code donnent les mêmes fichiers, octet pour octet. Chaque twin a un
identifiant (`twin_id`).

Dans MOLONARI Desk :

1. **Fichier › Twins › Choisir le dossier des twins…** indique le dossier qui contient vos `twin_xx/`. L'écran
   d'accueil et le menu **Fichier › Twins** listent alors les twins avec leur état :
   - pas encore exécuté ;
   - calculé ;
   - doublon ;
   - modifié après exécution (figé) ;
   - invalide.
2. **Créer un twin…** (onglet Résultats d'un point) fige un calcul de l'application en twin.
3. Un double-clic ouvre la **fenêtre du twin** :
   - sa config et son identifiant ;
   - **Exécuter** (F5), **Annuler** (Échap), **Recharger**, qui relit `twin.yaml` après une modification à la main ;
   - **Afficher le dossier des résultats** ;
   - les onglets : température θ(z, t), flux d'eau q(t), capteurs t₁–t₃ (avec la bande prédictive pour un twin
     mcmc), résidus t₁–t₃ et métadonnées.

Trois règles :

- **Config figée.** Une config ne se modifie qu'avant sa première exécution. Pour changer un paramètre, créez un
  nouveau twin.
- **Doublon.** Une config identique, même sous un autre nom, renvoie aux résultats existants sans recalcul.
- **Mode mcmc** : il est encore **expérimental**. Le mode direct, lui, fixe les paramètres du sol.

Les twins se pilotent aussi en ligne de commande, sans interface : `uv run molonari-desk twin …` est un alias de
`molonari-twin …` (par exemple `molonari-twin validate`, `run`, `list`). Voir le README de l'espace.

## Aide intégrée, langue et unités

- **Aide › Notice des paramètres** (Maj+F1) : une fiche par réglage de calcul, avec la définition, l'unité, la
  valeur par défaut, les valeurs admises, la clé de `twin.yaml` et des conseils. **Maj+F1 sur un champ** ouvre
  directement sa fiche, et l'infobulle de chaque champ en reprend le résumé. Les valeurs y sont lues dans le code :
  elles sont toujours justes pour votre version.
- **Aide › Prise en main** (F1) : les étapes du parcours type.
- **Aide › Ouvrir le dossier du journal** : les fichiers du journal, à joindre à un signalement de problème.
- **Affichage › Langue** : français (par défaut) ou anglais.
- **Affichage › Unité de température** : °C, K ou °F. L'unité ne change que l'affichage : les calculs et les
  fichiers restent en °C.

## Raccourcis clavier

| Raccourci | Action |
|---|---|
| Ctrl+N / Ctrl+O | Nouveau projet / ouvrir un projet |
| Ctrl+W / Ctrl+Q | Fermer le projet / quitter |
| Ctrl+L | Importer un laboratoire |
| Ctrl+Maj+N | Nouvelle étude |
| Ctrl+I | Importer un point |
| F5 | Lancer le calcul de l'onglet (modèle direct, inversion, twin) |
| Échap | Annuler le calcul en cours |
| F1 | Prise en main |
| Maj+F1 | Notice des paramètres (sur un champ : sa fiche) |

Sur macOS, Ctrl correspond à la touche Cmd.

## Formats de fichiers

- **Mesures HOBO (export HOBOware).**
  - La ligne de titre et la colonne `#` sont acceptées.
  - Les colonnes sont repérées d'après l'en-tête. Les températures en °F ou en K sont converties en °C.
  - Les lignes d'événements du logger (valeurs vides) sont écartées, et l'aperçu donne leurs numéros.
- **Format propre** : une date suivie de 2 colonnes (pression : tension, température) ou de 4 colonnes
  (températures). Séparateur `,`, `;` ou tabulation ; virgule décimale si le séparateur n'est pas `,`.
- **Pression.** Le fichier doit donner la **tension** du capteur (V). Un fichier en kPa, psi ou bar est refusé :
  le lire donnerait des charges fausses.
- **Dates.** Une soixantaine d'écritures sont reconnues, même mélangées dans un fichier.
  - **Un fichier dont les dates ne sont pas chronologiques est refusé**, avec le numéro de la ligne fautive. C'est
    souvent un tableur qui a reconverti des dates (`09/01/24` devenu `2009/1/24`) : réexportez le fichier depuis
    le logiciel du logger.
- **Fuseau horaire.** Le décalage est lu dans l'en-tête de la colonne des dates (« Date Heure, GMT+01:00 »).
  Convention provisoire : un en-tête en GMT+01:00 est lu comme UTC+2 toute l'année, avec un avertissement, en
  attendant que l'heure réelle des loggers soit confirmée.
- **Laboratoire** : fichiers CSV « clé,valeur », lus par position comme dans l'ancien Molonaviz. La virgule décimale
  est acceptée.

## Dépannage

| Message ou symptôme | Que faire |
|---|---|
| « Les dates de … ne sont pas chronologiques : la ligne … suit la ligne … » | Réexporter le fichier depuis le logiciel du logger, sans passer par un tableur. |
| Unité de pression refusée (kPa, psi, bar) | Exporter la tension du capteur, pas une pression convertie. |
| « Le fichier info indique le capteur de pression (ou la tige) « … », absent du laboratoire » | Importer d'abord le bon laboratoire, ou choisir le capteur dans la liste. |
| « Il faut au moins 3 dates pour lancer un modèle » | Élargir la période ou alléger le nettoyage. |
| « Les bas de couche doivent croître… et rester dans la colonne » | Vérifier les profondeurs. La dernière couche doit finir au fond de la colonne (thermomètre le plus profond + décalage). |
| Valeur hors du domaine physique | Une propriété ou une borne a priori sort des valeurs admises : voir la notice des paramètres (Maj+F1). |
| L'inversion finit « non convergé (maximum atteint) » | Augmenter le burn-in maximal ou le nombre de chaînes, revoir les bornes a priori et σT. Voir la notice scientifique avant d'interpréter le résultat. |
| « Action non effectuée » ou « Erreur inattendue » | L'application continue et vos données ne sont pas modifiées. Joindre le journal à votre signalement. |

**Journaux** : `~/.molonari_desk/logs/molonari_desk.log`, ou **Aide › Ouvrir le dossier du journal**.

## Ce qui change depuis molonaviz2

- La commande s'appelle `molonari-desk` (`uv run molonari-desk`, `molonari-desk twin …`). `molonaviz2` n'existe plus.
- Le dossier s'appelle `molonari_desk/` dans le dépôt MOLONARI-API.
- Les réglages de l'application (langue, unité, projets récents, dossier des twins) et les journaux sont à un
  nouvel emplacement. **Les anciens réglages ne sont pas repris** : choisissez à nouveau la langue, l'unité et le
  dossier des twins au premier lancement.
- Les projets `.molo`, les twins et leurs résultats ne changent pas.
