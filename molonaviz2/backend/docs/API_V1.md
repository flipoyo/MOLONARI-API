# API /v1 de molonari-twin : référence des questions

<!-- Fichier généré par tools/gen_api_doc.py à partir du catalogue (molonari_twin.api). Ne pas modifier à la main. -->

Version de l'API : **1.0.0-alpha.5**. Schéma complet : `GET /openapi.json` ; catalogue avec les schémas de
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
Celles qui n'ont pas besoin de twin (catalogue des paramètres, données d'un point) se posent à `POST /v1/questions/{id}`.
Les questions de données lisent les dossiers d'un point et d'un laboratoire de la machine qui sert l'API, qui
n'écoute que sur 127.0.0.1 : elles refusent par une erreur codée ce qu'elles ne savent pas lire.
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
  paramètre est refusé tant que la convention de dates n'est pas fixée (P2.1f, après le pipeline brut).
- **Unités** : SI (m/s, m³/m², °C, m, s). Les conversions d'affichage (m/jour, L/m²) sont faites par le client.
- **Tableaux aux capteurs** : capteur d'abord, `(3, T)`.

## Erreurs

Toute erreur attendue a la forme `{code, field, params, message_en}` ; le client traduit à partir de `code`
et de `params`.

| Code | Statut HTTP |
|---|---|
| `E_QUESTION_CHAINS` | 422 |
| `E_QUESTION_LEVEL` | 422 |
| `E_QUESTION_MODELS_INCOMPARABLE` | 422 |
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
| `E_QUESTION_VARIABLE` | 422 |
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

Sorties : `name`, `point`, `mode`, `description`, `start`, `end`, `layers`, `layer_bottoms`, `sensor_depths`, `offset`, `nb_cells`, `sigma_temperature`, `nb_chains`, `nb_iter`, `burn_in_max`, `thin`, `store_fields`.

## Données et traitement

### `raw_series` (BN-03)

**Raw measurements of a point** — *Mesures brutes d'un point*

The raw measurements of a point folder (CSV exports of the logger software), between two dates: thermometer temperatures (°C), voltage (V) and temperature of the pressure sensor (°C), and the number of file lines left out (logger events, empty or unreadable values, duplicate dates). It is about no twin. Formats that are not supported are refused with a coded error, never read wrongly.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige aucun twin.
- Requête : `POST /v1/questions/raw_series`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `point_folder` | texte | requis |  | Folder of the sampling point: its info file and its temperature and pressure measurement files, as exported by the logger software (CSV). |
| `start` | date ou null | — |  | First date returned, included (ISO 8601, no time zone). Default: start of the measurements. |
| `end` | date ou null | — |  | Last date returned, included (ISO 8601, no time zone). Default: end of the measurements. |

Sorties : `temperature_time`, `temperature`, `temperature_dropped`, `pressure_time`, `voltage`, `pressure_temperature`, `pressure_dropped`.

### `processing_preview` (BN-04)

**Effect of a cleaning** — *Effet d'un nettoyage*

What a cleaning would do to a point, without saving anything: every date common to the temperature and pressure files, with the measurements (the pressure converted to a differential head, in m, with the calibration of the laboratory), the reason each date is left out (`outside_window`, `manual`, `missing`, `iqr:<variable>` or `zscore:<variable>`; empty when it is kept), the count per reason, the dates found in only one file, and the uncertainties σ of the instruments (Sigma_Meas_P, Sigma_Meas_T). It is about no twin.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige aucun twin.
- Requête : `POST /v1/questions/processing_preview`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `point_folder` | texte | requis |  | Folder of the sampling point: its info file and its temperature and pressure measurement files, as exported by the logger software (CSV). |
| `start` | date ou null | — |  | Start of the studied period, included (ISO 8601, no time zone): earlier dates are left out as `outside_window`. Default: start of the measurements. |
| `end` | date ou null | — |  | End of the studied period, included (ISO 8601, no time zone): later dates are left out as `outside_window`. Default: end of the measurements. |
| `lab_folder` | texte | requis |  | Folder of the laboratory (temperature_sensors, pressure_sensors and shafts): it gives the calibration of the pressure sensor, the depths of the shaft and the sensor uncertainties. |
| `methods` | object ou null | — |  | Outlier detection per variable (temp_bed, t1, t2, t3, t4, pressure): none, iqr (outside [Q1 − 1.5·IQR, Q3 + 1.5·IQR]) or zscore (z-score above 3 in absolute value), computed on the studied period. A variable not listed is not checked. A date is left out as soon as one variable is an outlier. |
| `excluded` | list[date] | `[]` |  | Dates left out by hand (ISO 8601, no time zone), reason `manual`. |

Sorties : `time`, `temperature`, `river_temperature`, `pressure`, `kept`, `reason`, `reasons`, `reason_counts`, `kept_count`, `total_count`, `unmatched_temperature`, `unmatched_pressure`, `sigma_pressure`, `sigma_temperature`.

## Échanges d'eau

### `water_flux` (HN-10)

**What is the water flux through the riverbed?** — *Quel est le flux d'eau à travers le lit ?*

Mean flux q at the riverbed (z = 0) over the period, in m/s, positive downwards (infiltration: the river feeds the aquifer), with the median and the 5th and 95th percentiles of q over time. In mcmc mode `mean` is the posterior mean flux and its interval, at `level` (0.9 by default), is the posterior interval of that mean, drawn from the series q(0, t) stored for every kept sample (`mean_best`: the mean of the best sample); in direct mode `mean` is a single value, without interval (`uncertainty` is none).

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/water_flux`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `start` | date ou null | — |  | First date of the period, included (ISO 8601, no time zone). Default: start of the results. |
| `end` | date ou null | — |  | Last date of the period, included (ISO 8601, no time zone). Default: end of the results. |
| `level` | nombre | `0.9` | > 0, < 1 | Level of the posterior interval of the mean flux, e.g. 0.9 for 90 % (mcmc mode; in direct mode there is no interval). |

Sorties : `mean`, `median`, `p5`, `p95`, `points`, `start`, `end`, `level`, `model`, `uncertainty`, `mean_best`, `samples`.

### `exchange_direction` (HN-11)

**Does the river feed the aquifer?** — *La rivière alimente-t-elle la nappe ?*

Share of the time in infiltration (q ≥ 0, the river feeds the aquifer) and in exfiltration, weighted by the time intervals, dominant regime and number of reversals of q at the riverbed over the period. In mcmc mode with stored water-flux fields (`store_fields: full`), the probability that q ≥ 0 over the posterior samples, as a series and averaged over time.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/exchange_direction`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `start` | date ou null | — |  | First date of the period, included (ISO 8601, no time zone). Default: start of the results. |
| `end` | date ou null | — |  | Last date of the period, included (ISO 8601, no time zone). Default: end of the results. |

Sorties : `infiltration_share`, `exfiltration_share`, `dominant`, `reversals`, `points`, `start`, `end`, `p_infiltration`, `p_infiltration_series`, `field_time`, `samples`.

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

## Chaleur

### `heat_budget` (HN-20)

**Which processes carry the heat?** — *Quels processus transportent la chaleur ?*

Mean advective, conductive and total heat fluxes (W/m², positive downwards) at a depth over the period, their shares (absolute value of the mean) and the dominant process. The lateral exchange is not modelled: it is zero, with a warning. Needs results of format 2.0.0.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/heat_budget`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `start` | date ou null | — |  | First date of the period, included (ISO 8601, no time zone). Default: start of the results. |
| `end` | date ou null | — |  | Last date of the period, included (ISO 8601, no time zone). Default: end of the results. |
| `z` | nombre | `0.0` | ≥ 0 | Depth (m, positive downwards) where the heat fluxes are read. From 0 to the centre of the first cell, the first cell is used (like the flux at the riverbed); deeper, the fluxes are interpolated linearly. |

Sorties : `advective`, `conductive`, `lateral`, `total`, `advective_share`, `conductive_share`, `lateral_share`, `dominant`, `depth`, `points`, `start`, `end`.

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

### `fit_quality` (HN-40)

**Does the model reproduce the measurements?** — *Le modèle reproduit-il les mesures ?*

Per sensor: RMSE, bias (simulated − measured), RMSE/σT, lag-1 autocorrelation of the residuals and, in mcmc mode with stored fields, the share of measurements the predictive band covers; then the total RMSE (root mean square of the three). Arrays are sensor first: (3,). A direct twin is the forward model (`model` forward, `uncertainty` none); an mcmc twin is the inversion model, represented by its best sample, and `median_rmse` and `median_rmse_total` give the RMSE of the posterior median in addition. The direct difference forward − inversion is `TwinAPI.model_difference`, which needs two twins.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/fit_quality`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `level` | nombre | `0.9` | > 0, < 1 | Level of the intervals, e.g. 0.9 for 90 %. |
| `sigma` | nombre ou null | — | ≥ 0 | Measurement uncertainty of the thermometers (°C). Default: the one recorded with the results. |
| `seed` | entier | `0` | ≥ 0 | Seed of the measurement noise drawn for the predictive band (same seed, same band). |

Sorties : `rmse`, `bias`, `rmse_over_sigma`, `lag1_autocorrelation`, `sigma`, `level`, `coverage_per_sensor`, `rmse_total`, `sensor`, `model`, `uncertainty`, `median_rmse`, `median_rmse_total`.

### `reliability` (HN-41)

**Can I trust this result?** — *Puis-je faire confiance à ce résultat ?*

A green, orange or red verdict, the worst of the criteria, each with its level and its reason: convergence (R̂, ESS) in mcmc mode, fit (RMSE/σT), identifiability and predictive coverage in mcmc mode, gaps in the data and provenance. **The thresholds are provisional and not validated by a hydrogeologist** (`thresholds_validated` is false): the response gives those it used, and they can be overridden.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/reliability`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `level` | nombre | `0.9` | > 0, < 1 | Level of the intervals, e.g. 0.9 for 90 %. |
| `sigma` | nombre ou null | — | ≥ 0 | Measurement uncertainty of the thermometers (°C). Default: the one recorded with the results. |
| `seed` | entier | `0` | ≥ 0 | Seed of the measurement noise drawn for the predictive band (same seed, same band). |
| `thresholds` | object ou null | — |  | Override of the reliability thresholds (a partial object), on top of the MOLONARI_THRESHOLDS file. Keys: rhat, ess, fit, coverage, identifiability and gaps, each with `_green` and `_orange` (e.g. `rhat_green`). The defaults are provisional and not validated by a hydrogeologist. |

Sorties : `verdict`, `criteria`, `levels`, `reasons`, `threshold_names`, `thresholds`, `thresholds_validated`, `level`.

### `public_convergence` (HN-45)

**Has the sampling converged?** — *L'échantillonnage a-t-il convergé ?*

The convergence of the algorithm for the public report, without a verdict: largest R̂ and smallest ESS over all parameters, mean acceptance, whether the burn-in converged, chains and kept samples per chain, with the reference thresholds (the defaults of the reliability thresholds, not their overrides) and `thresholds_validated`, false until a hydrogeologist validates them. No colour and no verdict; the developer's own R̂ threshold is private and not given. mcmc mode only.

- Modes : mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/public_convergence`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `rhat_max`, `ess_min`, `acceptance_mean`, `burn_in_converged`, `burn_in_iterations`, `chains`, `samples_per_chain`, `rhat_green`, `rhat_orange`, `ess_green`, `ess_orange`, `thresholds_validated`.

### `provenance` (HN-43)

**Where does this result come from?** — *D'où vient ce résultat ?*

Versions, data fingerprint, platform and duration of the computation, the command that reproduces it; the commits and state of each repository are in the provenance of the answer.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/provenance`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `twin_id`, `code_version`, `results_format`, `created_at`, `device`, `duration`, `data_file`, `data_sha256`, `reproduce`.

### `soil_properties` (HN-42)

**Which soil properties were estimated?** — *Quelles propriétés du sol ont été estimées ?*

Per layer: permeability k (m²), hydraulic conductivity K (m/s, k·ρw·g/μ with water at 20 °C: an approximation), porosity n, solid thermal conductivity λs and heat capacity ρs·cs. In mcmc mode the best value, the median, the highest-density interval (computed in the space of the priors, −log₁₀ k for k) and the identifiability (posterior standard deviation over that of a uniform prior: near 1, the data teach nothing). In direct mode, the values of the configuration.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/soil_properties`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `level` | nombre | `0.9` | > 0, < 1 | Probability mass of the highest-density interval (HDI), e.g. 0.9 for 90 %. |

Sorties : `layers`, `level`, `k`, `k_median`, `k_identifiability`, `K`, `K_median`, `K_identifiability`, `n`, `n_median`, `n_identifiability`, `lambda_s`, `lambda_s_median`, `lambda_s_identifiability`, `rhos_cs`, `rhos_cs_median`, `rhos_cs_identifiability`.

## Construire

### `parameters_catalog` (BN-10)

**Which parameters can be set?** — *Quels paramètres peut-on régler ?*

The settings with their bounds, defaults and units, the values of `store_fields`, the default priors and the physical bounds of the inverted parameters, and the keys of the layers and of the mcmc section of twin.yaml. It is about no twin.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige aucun twin.
- Requête : `POST /v1/questions/parameters_catalog`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `settings.name`, `settings.minimum`, `settings.maximum`, `settings.default`, `settings.unit`, `settings.description`, `store_fields.values`, `store_fields.default`, `store_fields.description`, `priors.parameter`, `priors.label`, `priors.low`, `priors.high`, `priors.physical_low`, `priors.physical_high`, `layer_keys.name`, `layer_keys.description`, `mcmc_keys.name`, `mcmc_keys.description`.

### `check` (BN-11)

**Is this configuration valid, and what will it cost?** — *Cette configuration est-elle valide, et que coûtera-t-elle ?*

Whether the configuration of a twin folder is valid, with its problems (also given as warnings); if so, the twin_id expected for the published code, the state (new, computed, duplicate or modified) and, in mcmc mode, the number of samples, the evaluations and the stored size, which depends on `store_fields`. An invalid configuration is an answer (`valid` is false), not an error.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige le dossier d'un twin, dont la configuration peut être invalide.
- Requête : `POST /v1/twins/{ref}/questions/check`

Aucun paramètre : le corps de la requête est vide (ou `{}`).

Sorties : `valid`, `name`, `mode`, `twin_id`, `twin_id_local`, `state`, `duplicate_of`, `samples`, `max_evaluations`, `stored_bytes`, `store_fields`, `issue_codes`, `issue_fields`, `issue_messages`.

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

### `posterior_samples` (BN-32)

**Posterior samples of the parameters** — *Échantillons a posteriori des paramètres*

The posterior samples as an array (chain, iteration, layer, parameter) in the space of the priors (−log₁₀ k, n, λs, ρs·cs; k = 10^−x in m²), with its axes. Chains, range of computed iterations, burn (kept iterations skipped at the start) and thin choose the samples.

- Modes : mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/posterior_samples`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `chains` | list[entier] ou null | — |  | Chains to read, numbered from 0. Default: all. |
| `burn` | entier | `0` | ≥ 0 | Kept iterations to skip at the start of each chain (on top of the burn-in of the computation). |
| `thin` | entier | `1` | ≥ 1 | Keep one kept iteration out of `thin`. |
| `iter_range` | [entier ou null, entier ou null] ou null | — |  | (first, last) computed iteration, included; null: no bound. |

Sorties : `samples`, `iteration`, `chain`, `layer`, `parameter`, `parameter_unit`, `space`.

### `fields` (BN-30)

**Complete fields** — *Champs complets*

Temperature (°C), water flux (m/s) and advective and conductive heat fluxes (W/m², positive downwards) as arrays (depth, time) of the best fit, with their axes. Choose the variables, period, depth range and time step. They are large: prefer Accept: application/x-npz.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/fields`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `start` | date ou null | — |  | First date of the period, included (ISO 8601, no time zone). Default: start of the results. |
| `end` | date ou null | — |  | Last date of the period, included (ISO 8601, no time zone). Default: end of the results. |
| `variables` | list[texte] ou null | — |  | Fields to return, among temperature, water_flux, heat_flux_advective and heat_flux_conductive. Default: all four. |
| `depth_range` | [nombre ou null, nombre ou null] ou null | — |  | (shallowest, deepest) depth in m, included; null: no bound. |
| `time_step` | entier | `1` | ≥ 1 | Keep one date out of `time_step`. The fields are large: prefer Accept: application/x-npz. |

Sorties : `depth`, `time`, `temperature`, `water_flux`, `heat_flux_advective`, `heat_flux_conductive`.

## Diagnostics

### `convergence` (BN-40)

**Have the chains converged?** — *Les chaînes ont-elles convergé ?*

Per parameter and layer: R̂ (rank-normalized), bulk and tail ESS and MCSE of the mean. Per chain: acceptance, mean energy, bulk ESS and the largest R̂ without that chain. The statistics follow the selection (chains, burn, thin); `rhat_max_all` uses everything, to show the effect of the selection.

- Modes : mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/convergence`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `chains` | list[entier] ou null | — |  | Chains to read, numbered from 0. Default: all. |
| `burn` | entier | `0` | ≥ 0 | Kept iterations to skip at the start of each chain (on top of the burn-in of the computation). |
| `thin` | entier | `1` | ≥ 1 | Keep one kept iteration out of `thin`. |

Sorties : `rhat`, `ess_bulk`, `ess_tail`, `mcse_mean`, `rhat_max`, `rhat_max_all`, `chain`, `acceptance`, `mean_energy`, `chain_ess_bulk`, `rhat_without_chain`, `layer`, `parameter`, `samples_per_chain`, `burn_in_converged`, `burn_in_iterations`.

### `energy_breakdown` (BN-43)

**What is the energy made of?** — *De quoi l'énergie est-elle faite ?*

The terms of the energy (negative Gaussian log-likelihood) of the recorded fit: N·ln σ², the residual term Σr²/(2σ²), the implicit RMSE and the reference N/2, the value the residual term takes when σ is right.

- Modes : direct, mcmc ; depuis l'API 1.0.0 ; exige ses résultats (twin calculé).
- Requête : `POST /v1/twins/{ref}/questions/energy_breakdown`

| Paramètre | Type | Défaut | Bornes | Description |
|---|---|---|---|---|
| `sigma` | nombre ou null | — | > 0 | Measurement uncertainty of the thermometers (°C). Default: the one recorded with the results. |

Sorties : `n_points`, `sigma`, `log_term`, `residual_term`, `energy`, `implicit_rmse`, `reference`, `ratio`.
