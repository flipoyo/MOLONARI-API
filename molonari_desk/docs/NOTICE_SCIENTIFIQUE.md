# MOLONARI Desk — notice scientifique

Cette notice s'adresse à l'hydrogéologue qui utilise MOLONARI Desk pour estimer les échanges entre une rivière et
sa nappe à partir de profils de température. Elle décrit :

- le modèle physique résolu ;
- l'inversion bayésienne ;
- les sorties et la façon de juger leur fiabilité ;
- les limites de la méthode.

L'usage de l'interface est décrit dans la [notice d'utilisation](NOTICE_UTILISATEUR.md).

Les équations et les constantes de cette notice sont celles du code :

- `backend/src/molonari_twin/engine/jax/forward.py`, le modèle direct, portage en JAX de pyheatmy 0.9.7 ;
- `backend/src/molonari_twin/engine/jax/dream.py`, l'inversion ;
- `backend/src/molonari_twin/domain/compute.py`, les réglages et les bornes.

La [table des paramètres](#5-paramètres-et-réglages) est générée à partir du code.

## Sommaire

1. [Principe de la méthode](#1-principe-de-la-méthode)
2. [Dispositif et données d'entrée](#2-dispositif-et-données-dentrée)
3. [Modèle direct](#3-modèle-direct)
4. [Inversion bayésienne](#4-inversion-bayésienne)
5. [Paramètres et réglages](#5-paramètres-et-réglages)
6. [Sorties et conventions de signe](#6-sorties-et-conventions-de-signe)
7. [Juger la fiabilité d'un résultat](#7-juger-la-fiabilité-dun-résultat)
8. [Limites et bonnes pratiques](#8-limites-et-bonnes-pratiques)
9. [Reproductibilité](#9-reproductibilité)
10. [Références](#10-références)

## 1. Principe de la méthode

La chaleur est un traceur naturel des échanges nappe-rivière (Stallman, 1965 ; Anderson, 2005). Sous le lit, la
température varie au rythme de la rivière, journalier et saisonnier. Ce signal se propage en profondeur par
conduction, et il est transporté par l'eau qui traverse le lit (advection). Un flux descendant (infiltration)
enfonce le signal et l'atténue moins vite. Un flux montant (exfiltration) le comprime vers la surface.

MOLONARI Desk résout ce problème sur une **colonne verticale 1D** de sédiments, du lit de la rivière au thermomètre
le plus profond. Les forçages sont mesurés : charge différentielle entre rivière et nappe, température de la rivière
et température au fond de la colonne. Les températures aux profondeurs intermédiaires servent à contraindre les
propriétés du sol, donc le flux.

- **Modèle direct** : on fixe les propriétés de chaque couche, on simule T(z, t) et q(z, t), et on compare aux
  mesures.
- **Inversion** : on estime les propriétés de chaque couche, et leur incertitude, par échantillonnage bayésien
  (DREAM). On en déduit la loi a posteriori du flux.

## 2. Dispositif et données d'entrée

Un point MOLONARI comporte :

- un **capteur de pression différentielle**, qui mesure la tension U et la température de la rivière T_riv ;
- une **tige** à quatre thermomètres, aux profondeurs z₁ < z₂ < z₃ < z₄.

La tension est convertie en charge différentielle (rivière − nappe, en m) par l'étalonnage du capteur :

$$\Delta H = \frac{U - T\cdot \mathrm{d}U/\mathrm{d}T - U_0}{\mathrm{d}U/\mathrm{d}H}$$

où U₀ est l'ordonnée à l'origine (`processing/cleanup.py`, `domain/models.py`).

Rôle de chaque mesure dans le modèle (`engine/column.py`) :

| Mesure | Profondeur | Rôle |
|---|---|---|
| ΔH (capteur de pression) | z = 0 | condition à la limite haute de l'écoulement |
| T_riv (capteur de pression) | z = 0 | condition à la limite haute de la chaleur |
| t₁, t₂, t₃ | z₁ + Δh, z₂ + Δh, z₃ + Δh | **observations** : comparées aux températures simulées |
| t₄ | z₄ + Δh = L | condition à la limite basse de la chaleur (température de la nappe) |

Δh est le **décalage des capteurs** (offset) par rapport au lit, ajouté à chaque profondeur. La colonne a pour
hauteur L = z₄ + Δh. z est compté **positivement vers le bas** à partir du lit.

Les calculs n'utilisent que les **mesures nettoyées** (période, valeurs aberrantes, exclusions manuelles). Il faut au
moins 3 dates, strictement croissantes. Le pas de temps peut varier.

## 3. Modèle direct

### 3.1 Écoulement

La charge hydraulique H(z, t) suit une équation de diffusion :

$$S_s\,\frac{\partial H}{\partial t} = \frac{\partial}{\partial z}\left(K\,\frac{\partial H}{\partial z}\right),
\qquad K = \frac{\rho_w\, g\, k}{\mu}, \qquad S_s = \frac{n}{L}$$

- k est la perméabilité intrinsèque (m²), saisie sous la forme −log₁₀ k, et n la porosité ;
- K est la conductivité hydraulique (m/s) ;
- le coefficient d'emmagasinement S_s est pris égal à n/L, comme dans pyheatmy.

Conditions aux limites :

- H(0, t) = ΔH(t), la charge mesurée ;
- H(L, t) = 0 : la charge est comptée par rapport à la nappe.

Condition initiale :

- **une couche** : profil linéaire entre ΔH(t₀) et 0 ;
- **plusieurs couches** : profil linéaire par morceaux, avec un flux continu aux interfaces (régime permanent
  initial).

Le flux de Darcy s'en déduit, en m/s, positif vers le bas :

$$q = -K\,\frac{\partial H}{\partial z}$$

Comme S_s est faible, la charge s'équilibre vite : q suit presque instantanément ΔH(t). En première approximation,
q ≈ K·ΔH/L pour une couche, et q ≈ ΔH / Σ(eᵢ/Kᵢ) en série pour des couches d'épaisseurs eᵢ.

### 3.2 Chaleur

La température T(z, t) suit l'équation d'advection-conduction, avec l'eau et les grains à l'équilibre thermique :

$$\rho_m c_m\,\frac{\partial T}{\partial t} = \lambda_m\,\frac{\partial^2 T}{\partial z^2} - \rho_w c_w\, q\,\frac{\partial T}{\partial z}$$

Les propriétés du milieu saturé sont des mélanges de celles de l'eau et des grains :

$$\rho_m c_m = n\,\rho_w c_w + (1-n)\,\rho_s c_s, \qquad
\lambda_m = \left(n\sqrt{\lambda_w} + (1-n)\sqrt{\lambda_s}\right)^2$$

Les propriétés sont prises par cellule. L'équation discrétisée n'a pas de terme en ∂λm/∂z : une interface thermique
entre couches est donc traitée cellule par cellule, et non sous forme conservative.

Conditions aux limites :

- T(0, t) = T_riv(t) ;
- T(L, t) = t₄(t).

Condition initiale : interpolation linéaire, à la première date, des températures mesurées aux profondeurs
[0, z₁, z₂, z₃, z₄] + Δh (seule option, `numerics.initial_condition: linear`). Les premières heures simulées
dépendent de ce profil : écartez-les du jugement, ou choisissez une période qui commence après la mise en équilibre.

### 3.3 Constantes

| Symbole | Valeur | Signification |
|---|---|---|
| ρw | 1000 kg·m⁻³ | masse volumique de l'eau |
| cw | 4185 J·kg⁻¹·K⁻¹ | capacité thermique massique de l'eau |
| λw | 0,6071 W·m⁻¹·K⁻¹ | conductivité thermique de l'eau |
| g | 9,81 m·s⁻² | gravité |
| μ | 10⁻³ Pa·s | viscosité dynamique de l'eau |

Ces constantes sont fixes : elles ne dépendent pas de la température de l'eau.

### 3.4 Schéma numérique

- **Maillage** : N cellules de taille Δz = L/N (`numerics.nb_cells`, 100 par défaut), de centres zᵢ = Δz/2 + i·Δz.
  Les capteurs t₁–t₃ sont comparés à la cellule la plus proche.
- **Temps** : les pas de temps sont ceux des mesures. On résout d'abord la charge, puis la température.
- **Schéma** : θ-schéma semi-implicite avec θ = 0,6 (poids de l'instant suivant), donc un peu plus implicite que
  Crank-Nicolson. Les systèmes tridiagonaux sont résolus à chaque pas.
- **Bords** : les conditions aux limites s'appliquent à une demi-cellule des centres extrêmes, par des coefficients
  décentrés d'ordre 2 (facteurs 4/3 et 8/3).
- **Interfaces entre couches** : la conductivité hydraulique est une moyenne harmonique pondérée par la position de
  l'interface entre deux centres de cellules.
- **Précision** : tout est calculé en float64. Le moteur JAX est validé contre pyheatmy passé en float64 : écart
  relatif inférieur à 10⁻⁹ en monocouche, multicouche et pas de temps variable, y compris sur Point034.

Le maillage doit résoudre les gradients thermiques. Si le résultat change sensiblement quand vous doublez N, le
maillage est trop grossier.

### 3.5 Erreur d'ajustement

Pour chaque capteur i ∈ {1, 2, 3} :

$$\mathrm{RMSE}_i = \sqrt{\frac{1}{N_t}\sum_t \left(T^{sim}_i(t) - T^{obs}_i(t)\right)^2}, \qquad
\mathrm{RMSE}_{global} = \sqrt{\frac{1}{3}\sum_i \mathrm{RMSE}_i^2}$$

## 4. Inversion bayésienne

### 4.1 Modèle statistique

Les inconnues sont, pour chaque couche ℓ, θℓ = (−log₁₀ k, n, λs, ρs·cs). Les profondeurs des couches sont fixées par
l'utilisateur.

**Lois a priori.** Elles sont uniformes et indépendantes entre les bornes [min, max] de chaque paramètre, elles-mêmes
comprises dans le domaine physique (voir la table). La perméabilité est uniforme en −log₁₀ k, donc log-uniforme en k.

**Vraisemblance.** Les erreurs sont gaussiennes, indépendantes et de même écart-type σT, sur les températures des
trois capteurs à toutes les dates. L'**énergie** minimisée est l'opposé du logarithme de la vraisemblance :

$$E(\theta) = \frac{N_{obs}}{2}\,\ln \sigma_T^2 + \frac{1}{2\sigma_T^2}\sum_{i=1}^{3}\sum_t \left(T^{sim}_i(t;\theta) - T^{obs}_i(t)\right)^2$$

La loi a posteriori est proportionnelle à exp(−E) sur le support des lois a priori. Deux points comptent :

- **σT est fixé**, et non estimé. Par défaut, c'est l'incertitude σ du thermomètre déclarée dans le laboratoire,
  sinon 0,1 °C. Ce choix pèse lourd : un σT trop petit donne des intervalles trop étroits, et un σT trop grand des
  intervalles trop larges. Il doit couvrir l'erreur de mesure, mais aussi l'**erreur de modèle** (hypothèse 1D,
  profondeurs, condition initiale).
- **La pression n'entre pas dans la vraisemblance.** ΔH est un forçage, supposé exact. L'incertitude du capteur de
  pression (`sensors.sigma_pressure`) est enregistrée avec le twin, mais elle ne change pas le résultat. Les
  erreurs d'étalonnage de ΔH ne se propagent donc pas dans les intervalles.

### 4.2 Algorithme DREAM

L'échantillonneur est DREAM, *DiffeRential Evolution Adaptive Metropolis* (Vrugt et al., 2009 ; Vrugt, 2016). Il
reprend l'algorithme de pyheatmy, avec toutes les chaînes mises à jour en parallèle.

1. **Initialisation.** N chaînes (`nb_chains`, au moins 2δ + 1) sont tirées uniformément dans les lois a priori.
2. **Proposition** pour la chaîne j et chaque couche :

   $$\mathrm{d}x = \zeta + (1 + \lambda)\,\gamma \sum_{p=1}^{\delta}\left(x_{a_p} - x_{b_p}\right), \qquad
   \gamma = \frac{2{,}38}{\sqrt{2\, d^*\, \delta}}$$

   - a_p et b_p sont 2δ autres chaînes tirées au hasard ;
   - λ suit U(−c, c) et ζ suit N(0, c*²) ;
   - d* est le nombre de paramètres mis à jour, choisis par crossover avec la probabilité CR ∈ {1/nCR, …, 1} ;
   - une fois sur cinq, γ = 1, pour sauter entre des modes ;
   - les bornes sont gérées par repli périodique (« effet Pac-Man »).
3. **Acceptation de Metropolis** : la proposition est acceptée si ln u < E(x) − E(x′), avec u ~ U(0, 1).
4. **Burn-in**, par blocs de 10 itérations :
   - les probabilités de crossover s'adaptent ;
   - toutes les 50 itérations, une chaîne aberrante est replacée sur la meilleure : son énergie moyenne récente
     dépasse Q3 + max(2·IQR, marge) ;
   - le burn-in s'arrête dès que le **R̂ « split »** (Gelman et al., BDA3), calculé sur la seconde moitié de
     l'historique, est inférieur au seuil de Gelman-Rubin (`gr_threshold`, 1,2 par défaut) pour **tous** les
     paramètres. Sinon il s'arrête à `burn_in_max`, et le résultat est marqué **non convergé**.
5. **Échantillonnage** : `nb_iter` itérations par chaîne, sans adaptation. On en garde une sur `thin` pour les
   paramètres, les énergies et l'acceptation. Des champs (T, q) sont conservés à intervalles réguliers
   (`n_field_samples` au total, ou `fields_per_chain` par chaîne) pour les quantiles et les bandes.

Le **meilleur** jeu de paramètres est celui d'énergie minimale rencontré au cours du calcul. Le modèle stocké pour
les séries « meilleur ajustement » est celui de ce jeu.

## 5. Paramètres et réglages

Valeurs par défaut, bornes et clés de `twin.yaml`, lues dans le code de cette version. Dans l'application, chaque
réglage a sa fiche détaillée : **Aide › Notice des paramètres** (Maj+F1).

<!-- début de la table des paramètres : générée par tools/gen_notice_tables.py, ne pas modifier à la main -->
**Couches**

| Paramètre | Unité | Par défaut | Valeurs admises | Clé de `twin.yaml` | Définition |
|---|---|---|---|---|---|
| Nom | — | — | — | `layers[].name` | Nom de la couche, pour l'affichage seulement. |
| Bas (m) | m | — | — | `layers[].z_low` | Profondeur du bas de la couche, sous le lit de la rivière. |
| −log₁₀ k — perméabilité | k en m² | 12 (modèle direct) · a priori 11 – 15 (inversion) | domaine physique 3 – 20 | `layers[].neg_log10_k, mcmc.priors[].neg_log10_k` | Perméabilité intrinsèque du sédiment, saisie sous la forme −log₁₀ k. |
| n — porosité | – | 0.15 (modèle direct) · a priori 0.01 – 0.25 (inversion) | domaine physique 0.0001 – 0.9999 | `layers[].porosity, mcmc.priors[].porosity` | Porosité : fraction du volume occupée par l'eau. |
| λs — conductivité des grains | W·m⁻¹·K⁻¹ | 3.4 (modèle direct) · a priori 1 – 5 (inversion) | domaine physique 0.001 – 100 | `layers[].lambda_s, mcmc.priors[].lambda_s` | Conductivité thermique des grains solides. |
| ρs·cs — capacité thermique des grains | J·m⁻³·K⁻¹ | 5e+06 (modèle direct) · a priori 1e+06 – 1e+07 (inversion) | domaine physique 10000 – 1e+08 | `layers[].rhos_cs, mcmc.priors[].rhos_cs` | Capacité thermique volumique des grains solides. |
| Bornes a priori (min, max) | — | — | — | `mcmc.priors` | Dans l'inversion, chaque paramètre est tiré uniformément entre son min et son max. |

**Modèle direct**

| Paramètre | Unité | Par défaut | Valeurs admises | Clé de `twin.yaml` | Définition |
|---|---|---|---|---|---|
| Discrétisation | cellules | 100 | 10 – 2000 | `numerics.nb_cells` | Nombre de cellules de la colonne, du lit de la rivière à la profondeur de la colonne. |

**Inversion**

| Paramètre | Unité | Par défaut | Valeurs admises | Clé de `twin.yaml` | Définition |
|---|---|---|---|---|---|
| Incertitude de mesure σT | °C | incertitude du thermomètre, sinon 0,1 °C | ≥ 1e-06 | `sensors.sigma_temperature` | Écart-type de l'erreur de mesure des températures. Il fixe à quel point le modèle doit coller aux données. |
| Nombre de chaînes | chaînes | 10 | 3 – 200 | `mcmc.nb_chains` | Nombre de chaînes de Markov calculées en parallèle. |
| Itérations conservées | itérations | 1000 | 1 – 1e+06 | `mcmc.nb_iter` | Itérations par chaîne conservées après le burn-in : elles forment l'échantillon a posteriori. |
| Burn-in maximal | itérations | 1000 | 2 – 1e+06 | `mcmc.burn_in_max` | Nombre maximal d'itérations par chaîne avant le début de l'échantillonnage. |
| Quantiles | — | 0.05, 0.5, 0.95 | — | `mcmc.quantiles` | Quantiles des champs de température et de flux, calculés sur les champs conservés. |
| Graine aléatoire | — | 0 | ≥ 0 | `seed` | Graine du générateur de nombres aléatoires. |

**Réglages avancés de l'inversion**

| Paramètre | Unité | Par défaut | Valeurs admises | Clé de `twin.yaml` | Définition |
|---|---|---|---|---|---|
| Paires de chaînes par saut (δ) | paires | 3 | 1 – 20 | `mcmc.delta` | Nombre de paires d'autres chaînes dont les différences construisent une proposition. |
| Valeurs de crossover (nCR) | — | 3 | 1 – 20 | `mcmc.n_cr` | Nombre de valeurs de crossover CR = 1/nCR, 2/nCR, …, 1. |
| c — facteur de saut | — | 0.1 | 1e-09 – 0.999 | `mcmc.c` | Amplitude du facteur aléatoire d'un saut : λ est tiré uniformément dans [−c, c]. |
| c* — bruit de saut | — | 1e-06 | ≥ 1e-15 | `mcmc.c_star` | Écart-type du petit bruit gaussien ζ ajouté à un saut. |
| Seuil de Gelman-Rubin | — | 1.2 | ≥ 1.0001 | `mcmc.gr_threshold` | Le burn-in s'arrête quand R-hat passe sous cette valeur pour tous les paramètres. |
| Champs conservés pour les quantiles | champs | 200 | 2 – 5000 | `mcmc.n_field_samples` | Nombre total de champs de température et de flux (T, q) conservés pour calculer les quantiles. |
| Pas de sous-échantillonnage spatial | — | 1 | ≥ 1 | `mcmc.space_step` | Garder une cellule sur k dans les champs conservés. |
| Pas de sous-échantillonnage temporel | — | 1 | ≥ 1 | `mcmc.time_step` | Garder une date sur k dans les champs conservés. |
| Pas d'éclaircissement (thin) | — | 1 | 1 – 1e+06 | `mcmc.thin` | Garder une itération sur k après le burn-in, pour les paramètres, les énergies, l'acceptation et les champs. |
| Champs par chaîne | champs | non réglé : le nombre total de champs s'applique | 1 – 5000 | `mcmc.fields_per_chain` | Nombre de champs (T, q) conservés par chaîne ; il remplace le nombre total de champs conservés pour les quantiles. |
| Champs stockés | — | sensors | none, sensors, full | `mcmc.store_fields` | Ce que l'inversion garde des champs échantillonnés, en plus des quantiles. |
| Par chaîne | — | — | — | — | Ce que donnent les réglages pour chaque chaîne : itérations, champs conservés, mémoire et évaluations du modèle. |
<!-- fin de la table des paramètres -->

**Ordres de grandeur** (valeurs de la littérature, d'après Freeze et Cherry, 1979). Ils servent à choisir les
bornes a priori, et ne remplacent pas une connaissance du site.

| Sédiment | k (m²) | −log₁₀ k |
|---|---|---|
| graviers | 10⁻¹⁰ – 10⁻⁷ | 7 – 10 |
| sables propres | 10⁻¹³ – 10⁻⁹ | 9 – 13 |
| limons, sables limoneux | 10⁻¹⁶ – 10⁻¹² | 12 – 16 |
| argiles | 10⁻¹⁹ – 10⁻¹⁶ | 16 – 19 |

- **Porosité** des alluvions saturées : 0,2 – 0,4.
- **λs** des minéraux courants : 2 – 4 W·m⁻¹·K⁻¹ (quartz ≈ 7,7).
- **ρs·cs** : 1,5·10⁶ – 2,5·10⁶ J·m⁻³·K⁻¹.

Les bornes a priori par défaut de l'application (voir la table) sont plus larges pour ρs·cs. Pour n, elles
s'arrêtent à 0,25, sous la plupart des porosités d'alluvions. Adaptez-les à votre site.

## 6. Sorties et conventions de signe

| Grandeur | Unité | Convention |
|---|---|---|
| z | m | positif vers le bas, depuis le lit |
| q, flux d'eau | m/s | **q > 0 : infiltration** (rivière → nappe) ; q < 0 : exfiltration (nappe → rivière) |
| flux advectif | W/m² | ρw·cw·q·(T − T_ref), avec T_ref = 0 °C, positif vers le bas |
| flux conductif | W/m² | −λm·∂T/∂z, positif vers le bas |
| flux total | W/m² | advectif + conductif |

- Le flux advectif dépend de la température de référence (0 °C). Seules ses **variations**, et ses bilans entre
  deux profondeurs, ont un sens physique ; pas sa valeur absolue.
- Le flux d'eau rapporté « au fond de la rivière » est celui de la première cellule.
- Pour passer en m/jour, multipliez par 86 400.

**Sorties d'une inversion** :

| Sortie | Définition |
|---|---|
| Distributions a posteriori | Histogrammes des paramètres par couche : meilleur, moyenne, écart-type, HDI 90 % (intervalle de plus haute densité). |
| Intervalle de crédibilité des champs | Quantiles (par défaut 5 %, 50 %, 95 %) de T(z, t) et q(z, t) sur les champs conservés. Il ne traduit que l'incertitude des paramètres. |
| Bande prédictive | Températures des champs conservés aux capteurs, plus un bruit N(0, σT²), à graine fixe. C'est là qu'une nouvelle mesure devrait tomber. La **couverture**, part des mesures dans la bande, doit être proche du niveau de la bande. |
| Flux moyen a posteriori | Moyenne de q(0, t) sur la période pour chaque champ conservé ; la valeur centrale et l'intervalle sont la moyenne et les quantiles de ces moyennes. C'est l'incertitude sur le flux moyen, pas la variabilité de q dans le temps. |
| Sens de l'échange | Part du temps en infiltration et en exfiltration, nombre d'inversions de sens. |

## 7. Juger la fiabilité d'un résultat

Un résultat d'inversion n'est interprétable que si les chaînes ont convergé, si l'échantillon est assez grand et
si le modèle reproduit les mesures. Les diagnostics publiés sont ceux de Vehtari et al. (2021), recalculés sans
ArviZ et comparés à lui dans les tests :

- **R̂ rank-normalized** ;
- **ESS bulk et tail** (taille d'échantillon effective) ;
- **MCSE** de la moyenne ;
- **acceptation** et **réinitialisations** par chaîne.

| Diagnostic | Ce qu'il dit | Vert | Orange | Au-delà |
|---|---|---|---|---|
| R̂ maximal | les chaînes ont-elles exploré la même loi ? | ≤ 1,05 | ≤ 1,2 | non convergé |
| ESS minimal (bulk, tail) | combien d'échantillons indépendants valent les chaînes ? | ≥ 400 | ≥ 100 | intervalles peu fiables |
| RMSE / σT aux capteurs | le modèle reproduit-il les mesures, au regard de σT ? | ≤ 1,5 | ≤ 3 | modèle ou σT inadapté |
| écart couverture − niveau | la bande prédictive est-elle bien calibrée ? | ≤ 0,05 | ≤ 0,15 | σT ou modèle à revoir |
| écart-type a posteriori / a priori | les données apprennent-elles quelque chose sur le paramètre ? | ≤ 0,5 | ≤ 0,8 | paramètre non identifiable |
| part de mesures manquantes | les séries sont-elles complètes ? | ≤ 0,05 | ≤ 0,2 | lacunes importantes |

> **Ces seuils ne sont pas encore validés** (`questions/thresholds.py`, `THRESHOLDS_VALIDATED = False`). Les
> réponses de la ligne de commande portent `thresholds_validated: false`, et les seuils se surchargent par la
> variable `MOLONARI_THRESHOLDS`. Leur validation par un hydrogéologue est attendue.

Il y a trois seuils de R̂, avec trois usages différents :

- le **critère d'arrêt du burn-in** (R̂ split < 1,2 par défaut) ne fait que lancer l'échantillonnage ;
- le tableau des paramètres de l'application qualifie un R̂ ≤ 1,01 de bon : c'est la recommandation de Vehtari et
  al. (2021) ;
- le classement de fiabilité ci-dessus met le vert à 1,05.

Un R̂ ≤ 1,01 et un ESS ≥ 400 sont la cible pour publier un intervalle.

**Exemple : Point034** (version 0.9.0-alpha.1, twins de MOLONARI-TWIN) :

- le modèle direct, monocouche avec −log₁₀ k = 12, donne un flux moyen de −4,5·10⁻⁷ m/s et une RMSE globale de
  0,116 °C ;
- l'inversion (σT = 0,04 °C) donne −4,6·10⁻⁹ m/s, avec un intervalle à 90 % de [−1,4·10⁻⁸ ; −1,0·10⁻⁹] m/s, une
  RMSE de 0,039 °C, un burn-in convergé en 3170 itérations, un R̂ maximal de 1,13 et un ESS minimal de 43.

L'ajustement est bon, mais R̂ et ESS sont en orange et en rouge : l'intervalle doit être présenté comme indicatif.
Il faudrait plus de chaînes ou d'itérations pour le consolider.

## 8. Limites et bonnes pratiques

**Hypothèses du modèle**

- **1D vertical** : pas d'écoulement latéral. L'échange latéral (`q_s`) n'est pas pris en charge dans cette version.
  Près des berges, des méandres ou des structures du lit, le flux vertical apparent peut être biaisé.
- **Milieu saturé, équilibre thermique local** entre l'eau et les grains, propriétés constantes dans le temps
  (pas de colmatage ni de remaniement du lit pendant la période).
- **Conditions aux limites mesurées et supposées exactes** : ΔH, T_riv et t₄. Une dérive du capteur de pression
  biaise directement q.
- **Condition initiale linéaire** : les premières heures ne sont pas fiables.

**Identifiabilité**

- Le flux est contraint par l'advection. Quand |q| est faible, la conduction domine, le profil thermique dépend
  peu de k, et la perméabilité est mal contrainte. La loi a posteriori de −log₁₀ k reste alors large ou s'accumule
  contre une borne. On le juge avec le nombre de Péclet thermique, Pe = ρw·cw·|q|·L / λm : pour Pe ≪ 1, attendez
  une incertitude relative forte sur q.
- n, λs et ρs·cs agissent ensemble, par ρm·cm et λm. Ils sont souvent corrélés entre eux et moins bien contraints
  que q. Interprétez le flux, mieux contraint, plutôt que chaque propriété thermique isolément.
- Si une loi a posteriori s'accumule contre une borne a priori, **élargissez la borne**, dans le domaine physique,
  et relancez.

**Bonnes pratiques**

1. **Période.** Retirez l'installation, le retrait et les épisodes de tige touchée. Préférez une période où la
   rivière présente un signal journalier net : c'est lui qui renseigne le mieux.
2. **Profondeurs.** Vérifiez la profondeur du lit et le décalage Δh : une erreur de quelques centimètres change le
   flux estimé.
3. **σT.** Partez de l'incertitude du thermomètre, puis regardez RMSE/σT et la couverture de la bande prédictive.
   Si la couverture est très inférieure au niveau, σT sous-estime l'erreur totale.
4. **Couches.** Commencez par une couche. N'ajoutez des couches que si le site le justifie et si l'ajustement
   s'améliore nettement : chaque couche ajoute quatre paramètres.
5. **Convergence.** Ne publiez qu'avec un burn-in convergé, R̂ ≤ 1,01 et un ESS suffisant. Comparez deux graines :
   les intervalles doivent se recouvrir.
6. **Modèle direct d'abord.** Un modèle direct avec des valeurs plausibles donne l'ordre de grandeur du flux et
   révèle les problèmes de données avant une longue inversion.

## 9. Reproductibilité

- Chaque calcul enregistre sa configuration, sa graine, la version du moteur et l'appareil.
  **Réutiliser cette configuration** relance un calcul à l'identique.
- Un **twin** fige un calcul dans un `twin.yaml`, avec son fichier de mesures. Son identifiant `twin_id` est le
  SHA-256 de la configuration canonique : empreinte des mesures, réglages, graine et version du backend
  `molonari-twin`. Même config, mêmes données et même code donnent les mêmes fichiers, octet pour octet.
- La commande `molonari-twin reproduce` (réservée au niveau opx) recalcule un twin sans cache et compare les
  empreintes de tous les fichiers produits.
- Les nombres pseudo-aléatoires viennent du générateur JAX initialisé par la graine. Deux graines différentes
  donnent deux échantillons différents d'une même loi a posteriori.

## 10. Références

- Anderson, M. P. (2005). Heat as a ground water tracer. *Ground Water*, 43(6), 951–968.
- Freeze, R. A., & Cherry, J. A. (1979). *Groundwater*. Prentice-Hall.
- Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. *Statistical
  Science*, 7(4), 457–472.
- Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., & Rubin, D. B. (2013). *Bayesian Data
  Analysis* (3ᵉ éd.). CRC Press.
- Stallman, R. W. (1965). Steady one-dimensional fluid flow in a semi-infinite porous medium with sinusoidal surface
  temperature. *Journal of Geophysical Research*, 70(12), 2821–2827.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-normalization, folding, and
  localization: an improved R̂ for assessing convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.
- Vrugt, J. A., ter Braak, C. J. F., Diks, C. G. H., Robinson, B. A., Hyman, J. M., & Higdon, D. (2009).
  Accelerating Markov chain Monte Carlo simulation by differential evolution with self-adaptive randomized subspace
  sampling. *International Journal of Nonlinear Sciences and Numerical Simulation*, 10(3), 273–290.
- Vrugt, J. A. (2016). Markov chain Monte Carlo simulation using the DREAM software package: theory, concepts, and
  MATLAB implementation. *Environmental Modelling & Software*, 75, 273–316.
- pyheatmy (dépôt MOLONARI-computo) : `pyheatmy/MCMC_README.md` et `pyheatmy/core.py`, modèle de référence.
