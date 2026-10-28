"""Projection publique d'une réponse (P2c.2c, P2c.2d) : la liste blanche de ce qui entre dans le rapport.

La réponse de l'API développeur (`/v1`) reste complète ; le rapport public ne reçoit que la projection. Les listes
sont **blanches** : un champ n'est publié que s'il est listé ici, et une question sans liste n'est pas publiable
(E_REPORT_NO_PROJECTION). Une réponse n'est jamais publiée entière par défaut.

- HN-02 `describe` (SPEC_RAPPORT_PUBLIC, T4, T7) : le mode, la période, les couches, les profondeurs des capteurs et
  les réglages que les échantillons publics révèlent (`nb_chains`, `nb_iter`, `thin`). Ni `name`, `point` ni
  `description` (ils identifient le site, T7 : le site se désigne par `site_label`, public), ni `seed` (réservée à ORX, P2c.2g), ni les réglages privés.
- HN-43 `provenance` : SemVer composé du code, version de build des données (quand elle est enregistrée), versions
  de jax, jaxlib, numpy, plateforme, processeur et appareil de calcul. Ni remotes, ni chemins, ni `data_file`, ni
  `data_sha256`, ni `twin_id`, ni commande `reproduce`, ni commits.
- Les agrégats publiables (flux, échanges, volumes, convergence publique) : leurs grandeurs, sans les séries
  par échantillon (`samples`, `field_time`, `p_infiltration_series`), qui sont des états dérivés (T3).

Livrées avec le générateur (P2c.3) : HN-40 `fit_quality` (RMSE et biais par capteur, RMSE totale ; ni σ, ni RMSE/σ, qui la
révèlent), la différence entre les deux modèles (`model_difference`, nom `compare` dans le rapport), HN-42 `soil_properties`,
BN-32 `posterior_samples` et BN-34 `chain_data`. Les autres questions (HN-20, HN-30, HN-33, BN-30 : des états ou des
mesures ; HN-41 dont le verdict n'est pas public) n'ont pas de projection : elles sont refusées, même déclarées publiques,
tant que la résolution de la grille de requêtes n'est pas décidée (tension T3).
"""

from __future__ import annotations

from dataclasses import replace

from ..errors import UserError
from .answer import Answer, Quantity

# Grandeurs publiées, par question. Une question absente d'ici n'est pas publiable.
PUBLIC_DATA: dict[str, tuple[str, ...]] = {
    "describe": (
        "mode", "start", "end", "layers", "layer_bottoms", "sensor_depths", "offset", "nb_chains", "nb_iter", "thin",
        "site_label", "monitoring_status",  # P2.3 : le libellé public du site et l'état du suivi (carte d'identité)
    ),
    "provenance": ("code_version", "data_release"),
    "water_flux": ("mean", "median", "p5", "p95", "points", "start", "end", "level", "model", "uncertainty", "mean_best"),
    "exchange_direction": (
        "infiltration_share", "exfiltration_share", "dominant", "reversals", "points", "start", "end", "p_infiltration",
    ),
    "exchanged_volume": ("infiltration", "exfiltration", "net"),
    # P2c.3 : agrégats (RMSE), PDF a posteriori (résumé, échantillons, chaînes : publics si `posterior` l'est)
    "fit_quality": ("rmse", "bias", "rmse_total", "sensor", "level", "model", "uncertainty", "median_rmse", "median_rmse_total"),
    "model_difference": ("mean_difference", "rmse", "rmse_total", "sensor", "points"),
    "soil_properties": (
        "layers", "level", *(f"{name}{suffix}" for name in ("k", "K", "n", "lambda_s", "rhos_cs")
                             for suffix in ("", "_median", "_identifiability")),
    ),
    "posterior_samples": ("samples", "iteration", "chain", "layer", "parameter", "parameter_unit", "space"),
    "chain_data": (
        "iteration", "chain", "samples.neg_log10_k", "samples.porosity", "samples.lambda_s", "samples.rhos_cs", "energy",
        "acceptance", "resets", "count",
    ),
    "public_convergence": (
        "rhat_max", "ess_min", "acceptance_mean", "burn_in_converged", "burn_in_iterations", "chains", "samples_per_chain",
        "rhat_green", "rhat_orange", "ess_green", "ess_orange", "thresholds_validated",
    ),
}

# Éléments de l'environnement de calcul publiés par HN-43 (dans provenance["repositories"]["environment"]).
PUBLIC_ENVIRONMENT = ("jax", "jaxlib", "numpy", "platform", "processor", "device")

# Paramètres de la demande qui sont un réglage privé (`inversion_settings`) : retirés de la provenance publique (σ des thermomètres).
PRIVATE_PARAMETERS = ("sigma",)

# Clés de la provenance publique d'une réponse : la question, sa référence, ses paramètres, l'instance du space.
PUBLIC_PROVENANCE = ("question", "ref", "parameters")


def public_answer(question_id: str, answer: Answer, instance: str | None = None) -> Answer:
    """La projection publique de `answer`, réponse à la question `question_id` : une nouvelle réponse (l'originale
    n'est pas modifiée). Données et provenance sont réduites à la liste blanche. `instance` est la référence de
    l'instance du space, ajoutée à la provenance. E_REPORT_NO_PROJECTION si la question n'a pas de liste."""
    if question_id not in PUBLIC_DATA:
        raise UserError("E_REPORT_NO_PROJECTION", question=question_id)
    data = {name: answer.data[name] for name in PUBLIC_DATA[question_id] if name in answer.data}
    if question_id == "provenance":
        data.update(_environment(answer))
    provenance = {key: answer.provenance[key] for key in PUBLIC_PROVENANCE if key in answer.provenance}
    if isinstance(provenance.get("parameters"), dict):
        provenance["parameters"] = {k: v for k, v in provenance["parameters"].items() if k not in PRIVATE_PARAMETERS}
    if instance is not None:
        provenance["instance"] = instance
    return replace(answer, data=data, provenance=provenance)


def _environment(answer: Answer) -> dict[str, Quantity]:
    """Les versions de librairies et la plateforme de HN-43, lues dans la provenance du calcul (jamais ses dépôts)."""
    repositories = answer.provenance.get("repositories") or {}
    environment = repositories.get("environment") or {}
    return {name: Quantity(str(environment[name]), "1") for name in PUBLIC_ENVIRONMENT if environment.get(name) is not None}
