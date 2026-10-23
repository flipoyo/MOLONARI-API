# API /v1 de molonari-twin : référence des questions

<!-- Fichier généré par tools/gen_api_doc.py à partir du catalogue (molonari_twin.api). Ne pas modifier à la main. -->

Version de l'API : **1.0.0-alpha.2**. Schéma complet : `GET /openapi.json` ; catalogue avec les schémas de
paramètres : `GET /v1/questions` (`?twin=<ref>` : les questions de son mode).

## Poser une question

```bash
molonari-twin serve --local --twins-folder examples/twins
curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/questions/exchanged_volume \
     -H 'Content-Type: application/json' -d '{"start": "2016-06-28T00:00:00", "area": 2.0}'
curl -X POST http://127.0.0.1:8765/v1/twins/twin_03/questions/HN-12 -H 'Accept: application/x-npz' -o answer.npz
```

Une question se désigne par son identifiant (`exchanged_volume`) ou par sa référence du plan (`HN-12`). Le
corps de la requête contient ses paramètres ; un paramètre inconnu est refusé. Les questions lisent la
configuration ou les résultats d'un twin : elles répondent tout de suite, sans tâche et sans charger le moteur.
Celles qui n'ont pas besoin de twin (catalogue des paramètres) se posent à `POST /v1/questions/{id}`.
Calculer un twin passe par `POST /v1/twins/{ref}/run`, puis par le suivi de la tâche (`/v1/tasks/…`).

## Réponse

`{data: {nom: {value, unit, interval}}, provenance, warnings}`. `interval` vaut `[bas, haut]`, de la forme de
`value`, ou `null`. La provenance donne la question, ses paramètres, le twin, son `twin_id` et ses résultats.

- **JSON** (par défaut) : un tableau s'écrit `{dtype, shape, data}`, valeurs à plat dans l'ordre C. NaN et
  dates absentes valent `null`, ±∞ s'écrivent `"inf"` et `"-inf"`. Un flottant scalaire non fini s'écrit
  comme un tableau de forme `[]`.
- **npz** (`Accept: application/x-npz`) : chaque tableau va dans l'archive (`data/<nom>/value`, `…/low`,
  `…/high`) ; la réponse elle-même est l'entrée `meta` (JSON). L'archive se lit sans pickle.
- **Dates** : ISO 8601 à la seconde, **sans fuseau**, dans le repère des données du twin. Un fuseau donné en
  paramètre est refusé tant que la convention de dates n'est pas fixée (avec le pipeline brut).
- **Unités** : SI (m/s, m³/m², °C, m, s). Les conversions d'affichage (m/jour, L/m²) sont faites par le client.
- **Tableaux aux capteurs** : capteur d'abord, `(3, T)`.

## Erreurs

Toute erreur attendue a la forme `{code, field, params, message_en}` ; le client traduit à partir de `code`
et de `params`.

| Code | Statut HTTP |
|---|---|
| `E_QUESTION_CHAINS` | 422 |
| `E_QUESTION_LEVEL` | 422 |
| `E_QUESTION_NOT_AVAILABLE` | 409 |
| `E_QUESTION_NO_FIELDS` | 422 |
| `E_QUESTION_NO_POSTERIOR` | 422 |
| `E_QUESTION_NO_TWIN` | 409 |
| `E_QUESTION_OUT_OF_RANGE` | 422 |
| `E_QUESTION_PARAMS` | 422 |
| `E_QUESTION_PERIOD_EMPTY` | 422 |
| `E_QUESTION_SHAPES` | 422 |
| `E_QUESTION_SIGMA` | 422 |
| `E_QUESTION_SIGMA_REQUIRED` | 422 |
| `E_QUESTION_THIN` | 422 |
| `E_QUESTION_TIMES` | 422 |
| `E_QUESTION_TWIN_REQUIRED` | 409 |
| `E_QUESTION_UNKNOWN` | 404 |
| `E_QUESTION_Z_OR_T` | 422 |
| `E_REQUEST_INVALID` | 422 |
| `E_TASK_FINISHED` | 409 |
| `E_TASK_KIND_UNKNOWN` | 422 |
| `E_TASK_NOT_FINISHED` | 409 |
| `E_TASK_NOT_FOUND` | 404 |
| `E_TWIN_FROZEN` | 409 |
| `E_TWIN_NOT_COMPUTED` | 409 |
| `E_TWIN_NOT_FOUND` | 404 |

Les autres codes `E_…` répondent 400.

## Découverte

### `describe` (HN-02)

**What does this twin contain?** — *Que contient ce twin ?*

Layers, sensors, period, mode and main settings, read from the configuration (no computation needed).

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige la configuration du twin.
- Requête : `POST /v1/twins/{ref}/questions/describe`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `name`, `point`, `mode`, `description`, `start`, `end`, `layers`, `layer_bottoms`, `sensor_depths`, `offset`, `nb_cells`, `sigma_temperature`, `seed`, `nb_chains`, `nb_iter`, `burn_in_max`, `thin`, `store_fields`.

## Échanges d'eau

### `water_flux` (HN-10)

**What is the water flux through the riverbed?** — *Quel est le flux d'eau à travers le lit ?*

Mean, median, 5th and 95th percentiles of the flux q at the riverbed (z = 0) over the period, in m/s, positive downwards (infiltration: the river feeds the aquifer).

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/water_flux`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `start` | date ou null | — |  | First date of the period, included (ISO 8601, no time zone). Default: start of the results. |
| `end` | date ou null | — |  | Last date of the period, included (ISO 8601, no time zone). Default: end of the results. |

Sorties : `mean`, `median`, `p5`, `p95`, `points`, `start`, `end`.

### `exchanged_volume` (HN-12)

**What volume of water crossed 1 m² of riverbed?** — *Quel volume d'eau a traversé 1 m² de lit ?*

Infiltrated, exfiltrated and net volumes over the period, integrated by trapezoids with the sign change cut at the zero crossing: m³/m², or m³ when an area is given.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/exchanged_volume`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `start` | date ou null | — |  | First date of the period, included (ISO 8601, no time zone). Default: start of the results. |
| `end` | date ou null | — |  | Last date of the period, included (ISO 8601, no time zone). Default: end of the results. |
| `area` | nombre ou null | — | > 0 | Riverbed area represented by the point (m²). Without it, volumes are per m² (m³/m²). |

Sorties : `infiltration`, `exfiltration`, `net`.

## Températures

### `temperature` (HN-30)

**What is the temperature at a depth and a date?** — *Quelle température à une profondeur et à une date ?*

Simulated temperature in °C, linearly interpolated: one value (z and t), a profile T(z) (t alone) or a series T(t) (z alone).

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/temperature`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `z` | nombre ou null | — | ≥ 0 | Depth (m, positive downwards) between the centres of the first and last computed cells, e.g. 0.004 to 0.396 m for 50 cells over 0.4 m (the error gives the range). Alone: the series T(t). |
| `t` | date ou null | — |  | Date (ISO 8601, no time zone). Alone: the profile T(z). |

Sorties : `temperature`, `depth`, `time`.

### `sensor_intervals` (HN-33)

**What uncertainty surrounds each measurement point?** — *Quelle incertitude autour de chaque point de mesure ?*

Per sensor: the measurement with its instrument interval (± z·σ), the simulation and, in mcmc mode with stored fields, the credible band (model) and the predictive band (a new measurement), with the share of measurements the predictive band covers. Arrays are sensor first: (3, T).

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/sensor_intervals`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `level` | nombre | `0.9` | > 0, < 1 | Level of the intervals, e.g. 0.9 for 90 %. |
| `sigma` | nombre ou null | — | ≥ 0 | Measurement uncertainty of the thermometers (°C). Default: the one recorded with the results. |
| `seed` | entier | `0` | ≥ 0 | Seed of the measurement noise drawn for the predictive band (same seed, same band). |

Sorties : `time`, `measured`, `simulated`, `level`, `z`, `sigma`, `field_time`, `credible`, `predictive`, `coverage`, `coverage_per_sensor`, `points`.

## Fiabilité

### `provenance` (HN-43)

**Where does this result come from?** — *D'où vient ce résultat ?*

Versions, data fingerprint, platform and duration of the computation, the command that reproduces it; the commits and state of each repository are in the provenance of the answer.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/provenance`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `twin_id`, `code_version`, `results_format`, `created_at`, `device`, `duration`, `data_file`, `data_sha256`, `reproduce`.

## Accès bruts aux résultats

### `residuals` (BN-31)

**Residuals at each sensor** — *Résidus par capteur*

Measured, simulated and residual (simulated − measured) temperatures per sensor, with the RMSE; option `intervals`: the credible and predictive bands.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/residuals`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `level` | nombre | `0.9` | > 0, < 1 | Level of the intervals, e.g. 0.9 for 90 %. |
| `sigma` | nombre ou null | — | ≥ 0 | Measurement uncertainty of the thermometers (°C). Default: the one recorded with the results. |
| `seed` | entier | `0` | ≥ 0 | Seed of the measurement noise drawn for the predictive band (same seed, same band). |
| `intervals` | booléen | `False` |  | Add the credible and predictive bands (mcmc mode, stored fields). |

Sorties : `time`, `measured`, `simulated`, `residual`, `rmse`, `field_time`, `credible`, `predictive`, `coverage`, `coverage_per_sensor`, `points`.

### `chain_data` (BN-34)

**Everything about one or more chains** — *Tout sur une ou plusieurs chaînes*

Parameters, energies, acceptance and burn-in resets of the chosen chains, with the computed iteration of each kept sample; read thinning and iteration range as in the computation.

- Modes : mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/chain_data`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `chains` | list[entier] ou null | — |  | Chains to read, numbered from 0. Default: all. |
| `thin` | entier | `1` | ≥ 1 | Keep one kept iteration out of `thin`. |
| `iterations` | [entier ou null, entier ou null] ou null | — |  | (first, last) computed iteration, included; null: no bound. |

Sorties : `iteration`, `chain`, `samples.neg_log10_k`, `samples.porosity`, `samples.lambda_s`, `samples.rhos_cs`, `energy`, `acceptance`, `resets`, `count`.

### `sampling_layout` (BN-36)

**How was this twin sampled?** — *Comment ce twin a-t-il été échantillonné ?*

Chains, iterations per chain, burn-in, thin, total samples, fields per chain and the interval between them, space and time steps, store_fields and stored size; from the configuration, before or after the run.

- Modes : mcmc ; depuis l'API 1.0.0 ; exige la configuration du twin.
- Requête : `POST /v1/twins/{ref}/questions/sampling_layout`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `nb_chains`, `burn_in_max`, `nb_iter`, `thin`, `kept_per_chain`, `samples`, `fields_per_chain`, `fields_total`, `field_interval`, `nb_cells`, `field_cells`, `nb_times`, `field_times`, `space_step`, `time_step`, `store_fields`, `stored_bytes`.
