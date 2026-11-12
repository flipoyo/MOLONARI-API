"""Constats du rapport public (F1) : des codes `I_…` structurés, calculés des seules entrées publiées.

`findings(published, chains)` prend la même entrée que `build_figures` (les réponses relues des octets du rapport et
le dictionnaire de `chains.npz`) et rend une liste ordonnée de `{"figure", "code", "params"}`. Les codes sont déclarés
dans `errors.MESSAGES` (modèle anglais) et traduits dans `i18n/fr.json` ; la page `README.md` les rend en français
(`readme.py`). Les nombres des `params` sont des flottants arrondis à 4 chiffres significatifs (mêmes octets d'une
génération à l'autre), les textes des valeurs fermées (`forward`, `inversion`).

**Aucun seuil.** Aucun constat ne dépend d'un seuil et aucun seuil n'est lu ni affiché : ni « bien ou mal contraint »,
ni « accord ou désaccord », ni identifiabilité, ni feu vert de convergence. Un constat dit un signe, un ordre de
grandeur, une part du temps, un rapport d'amplitude, une valeur.
"""

from __future__ import annotations

import math

import numpy as np

from ..questions.answer import Answer
from ..questions.period import time_weights

ROLES = ("forward", "inversion")
DAILY_STEP_MAX = 3 * 3600
"""Pas médian (s) au-delà duquel l'ajustement à 24 h n'est pas fait : ce n'est pas un seuil d'interprétation, c'est la
condition pour que l'ajustement ait un sens (au moins 8 dates par jour)."""
OMEGA = 2 * math.pi / 86400  # rad/s


def rounded(value: float) -> float:
    """Un flottant à 4 chiffres significatifs (`float(f"{x:.4g}")`) : des octets stables."""
    return float(f"{float(value):.4g}")


def daily_fit(seconds: np.ndarray, temperature: np.ndarray) -> dict[str, np.ndarray]:
    """Ajustement par moindres carrés de `T(z, t) ≈ a + b·t + c·cos ωt + d·sin ωt` (ω = 2π / 86 400 s) à chaque profondeur.

    `temperature` est (profondeurs, dates), `seconds` les dates (s). Rend l'amplitude `hypot(c, d)` (°C) et la phase
    `atan2(d, c)` (rad : l'instant du maximum est phase / ω) de chaque profondeur. Le temps est compté depuis la première date."""
    seconds = np.asarray(seconds, float)
    t = seconds - seconds[0]
    design = np.column_stack([np.ones_like(t), t / 86400.0, np.cos(OMEGA * t), np.sin(OMEGA * t)])
    coefficients, *_ = np.linalg.lstsq(design, np.asarray(temperature, float).T, rcond=None)
    c, d = coefficients[2], coefficients[3]
    return {"amplitude": np.hypot(c, d), "phase": np.arctan2(d, c)}


def damping(seconds: np.ndarray, depth: np.ndarray, temperature: np.ndarray) -> dict[str, float]:
    """Amortissement et déphasage de l'onde journalière entre la première et la dernière profondeur : amplitudes (°C),
    rapport bas / haut, retard (h, du maximum du bas sur celui du haut, dans [0, 24[) et profondeur du bas (m)."""
    fit = daily_fit(seconds, temperature)
    top, bottom = float(fit["amplitude"][0]), float(fit["amplitude"][-1])
    lag = ((float(fit["phase"][-1]) - float(fit["phase"][0])) / OMEGA / 3600.0) % 24.0
    return {"amplitude_top": top, "amplitude_bottom": bottom, "ratio": bottom / top if top > 0 else math.nan,
            "lag": lag, "depth_bottom": float(depth[-1])}


def _flux_findings(published: dict[str, Answer]) -> list[dict]:
    found = []
    for role in ROLES:
        answer = published.get(f"{role}-water_flux.json")
        if answer is None:
            continue
        mean = float(answer.data["mean"].value)
        found.append({"figure": "figures/water_flux.svg",
                      "code": "I_FLUX_MEAN_INFILTRATION" if mean >= 0 else "I_FLUX_MEAN_EXFILTRATION",
                      "params": {"model": role, "mean": rounded(mean), "exponent": math.floor(math.log10(abs(mean))) if mean else 0}})
        interval = answer.data["mean"].interval
        if interval is not None:
            low, high = float(interval[0]), float(interval[1])
            found.append({"figure": "figures/water_flux.svg",
                          "code": "I_FLUX_INTERVAL_CONTAINS_ZERO" if low <= 0 <= high else "I_FLUX_INTERVAL_EXCLUDES_ZERO",
                          "params": {"model": role, "low": rounded(low), "high": rounded(high),
                                     "level": rounded(float(answer.data["level"].value) * 100)}})
    return found


def _field_findings(published: dict[str, Answer]) -> tuple[list[dict], list[dict]]:
    """(amortissement de l'onde journalière, part du temps en infiltration au fond) de chaque modèle dont les champs sont publiés."""
    damped, shares = [], []
    for role in ROLES:
        fields = published.get(f"{role}-fields.npz")
        if fields is None:
            continue
        seconds = np.asarray(fields.data["time"].value).astype("datetime64[s]").astype(np.int64).astype(float)
        depth = np.asarray(fields.data["depth"].value, float)
        step = float(np.median(np.diff(seconds))) if len(seconds) > 1 else math.inf
        result = damping(seconds, depth, np.asarray(fields.data["temperature"].value, float)) if step <= DAILY_STEP_MAX else None
        if result is not None and result["amplitude_top"] > 0:
            damped.append({"figure": f"figures/temperature_depths-{role}.svg", "code": "I_DAILY_DAMPING",
                           "params": {"model": role, **{key: rounded(value) for key, value in result.items()}}})
        flux = np.asarray(fields.data["water_flux"].value, float)[0]
        weights = time_weights(np.asarray(fields.data["time"].value))
        finite = np.isfinite(flux)
        if not finite.any():
            continue
        share = float(weights[finite & (flux >= 0)].sum() / weights[finite].sum()) * 100
        shares.append({"figure": f"figures/water_flux_field-{role}.svg", "code": "I_FLUX_INFILTRATION_SHARE",
                       "params": {"model": role, "share": rounded(share), "depth": rounded(depth[0]), "step": rounded(step / 3600)}})
    return damped, shares


def findings(published: dict[str, Answer], chains: dict | None = None) -> list[dict]:
    """Les constats, dans l'ordre des figures : flux (1), amortissement (3), part en infiltration (4), convergence (6),
    ajustement (7), différence entre les modèles (8). Un constat dont un nombre n'est pas fini (R̂ infini d'une chaîne trop
    courte, par exemple) est omis : `report.json` n'écrit pas de NaN ni d'infini ; la figure, elle, écrit la valeur."""
    return [f for f in _findings(published, chains) if all(math.isfinite(v) for v in f["params"].values() if isinstance(v, float))]


def _findings(published: dict[str, Answer], chains: dict | None) -> list[dict]:
    found = _flux_findings(published)
    damped, shares = _field_findings(published)
    found += damped + shares
    convergence = published.get("inversion-public_convergence.json")
    if chains is not None and convergence is not None:
        data = convergence.data
        found.append({"figure": "figures/chains.svg", "code": "I_CONVERGENCE",
                      "params": {"rhat_max": rounded(data["rhat_max"].value), "ess_min": rounded(data["ess_min"].value),
                                 "chains": int(data["chains"].value), "burn_in": int(data["burn_in_iterations"].value)}})
    for role in ROLES:
        fit = published.get(f"{role}-fit_quality.json")
        if fit is not None:
            found.append({"figure": "figures/fit_quality.svg", "code": "I_FIT_RMSE",
                          "params": {"model": role, "rmse": rounded(fit.data["rmse_total"].value)}})
    compare = published.get("compare.json")
    if compare is not None:
        found.append({"figure": "figures/compare.svg", "code": "I_MODEL_DIFFERENCE",
                      "params": {"mean_difference": rounded(np.mean(np.asarray(compare.data["mean_difference"].value, float))),
                                 "rmse": rounded(compare.data["rmse_total"].value)}})
    return found
