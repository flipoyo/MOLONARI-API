"""Statistiques des chaînes MCMC, en numpy et scipy seuls : ni JAX, ni ArviZ.

L'API les utilise pour dire si un résultat a convergé (BN-40, HN-41) sans charger le moteur ni ArviZ. Ce sont les
estimateurs de Vehtari et al. (2021, « Rank-normalization, folding, and localization: an improved R̂ for assessing
convergence of MCMC »), et le HDI le plus étroit sur les échantillons ; les tests les comparent à ArviZ, qui sert
de référence (écart relatif < 1e-6).

Les chaînes sont rangées (N chaînes, n itérations, ...) : les axes suivants (paramètres, couches) sont traités
chacun à leur tour, et le résultat a leur forme.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy import stats
from scipy.fft import next_fast_len

MIN_DRAWS = 4  # en dessous, les statistiques ne sont pas définies (nan)
TAIL_PROBABILITIES = (0.05, 0.95)


def _per_parameter(function: Callable[[np.ndarray], float], chains: np.ndarray) -> np.ndarray:
    """Applique `function` à chaque série (N, n) de `chains` (N, n, ...) ; résultat de forme (...)."""
    chains = np.asarray(chains, np.float64)
    shape = chains.shape[2:]
    out = np.empty(shape)
    for index in np.ndindex(*shape):
        out[index] = function(chains[(slice(None), slice(None), *index)])
    return out


def _usable(chains: np.ndarray, min_chains: int) -> bool:
    return chains.shape[0] >= min_chains and chains.shape[1] >= MIN_DRAWS and not np.isnan(chains).any()


def _split(chains: np.ndarray) -> np.ndarray:
    """Chaque chaîne coupée en deux moitiés de même longueur (N, n) → (2N, n // 2)."""
    half = chains.shape[1] // 2
    return np.vstack((chains[:, :half], chains[:, -half:]))


def _z_scale(chains: np.ndarray) -> np.ndarray:
    """Rangs de toutes les valeurs, ramenés à des quantiles d'une loi normale (c = 3/8, Blom 1958)."""
    ranks = stats.rankdata(chains, method="average").reshape(chains.shape)
    return stats.norm.ppf((ranks - 3 / 8) / (chains.size - 2 * 3 / 8 + 1))


def _rhat(chains: np.ndarray) -> float:
    n = chains.shape[1]
    within = chains.var(axis=1, ddof=1).mean()
    between = n * chains.mean(axis=1).var(ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt((between / within + n - 1) / n))


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """R-hat « split » classique (Gelman et al., BDA3). chains : (N, n, ...) → (...). inf si une série est constante."""
    chains = np.asarray(chains, np.float64)
    n = chains.shape[1] // 2
    if n < 2:
        return np.full(chains.shape[2:], np.nan)
    halves = np.concatenate([chains[:, :n], chains[:, -n:]], axis=0)
    within = halves.var(axis=1, ddof=1).mean(axis=0)
    between_over_n = halves.mean(axis=1).var(axis=0, ddof=1)
    var_hat = (n - 1) / n * within + between_over_n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / within)
    return np.where(within > 0, rhat, np.inf)


def _rank_rhat(chains: np.ndarray) -> float:
    if not _usable(chains, 2):
        return float("nan")
    split = _split(chains)
    bulk = _rhat(_z_scale(split))
    tail = _rhat(_z_scale(np.abs(split - np.median(split))))
    return max(bulk, tail)


def rank_normalized_rhat(chains: np.ndarray) -> np.ndarray:
    """R̂ du plus grand de deux : sur les rangs normalisés (le centre) et sur les rangs de l'écart à la médiane (les
    queues). chains : (N, n, ...) → (...). nan si moins de 2 chaînes ou de 4 itérations."""
    return _per_parameter(_rank_rhat, chains)


def _autocovariance(chains: np.ndarray) -> np.ndarray:
    """Autocovariance de chaque chaîne (N, n) à tous les retards, par FFT."""
    n = chains.shape[1]
    size = next_fast_len(2 * n)
    centred = chains - chains.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centred, n=size, axis=1)
    return np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)[:, :n] / n


def _ess(chains: np.ndarray) -> float:
    """Taille d'échantillon effective (N, n), par la suite positive puis monotone de Geyer."""
    if np.max(chains) - np.min(chains) < np.finfo(float).resolution:
        return float(chains.size)
    n_chain, n_draw = chains.shape
    acov = _autocovariance(chains)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += np.var(chains.mean(axis=1), ddof=1)

    rho = np.zeros(n_draw)
    even = 1.0
    rho[0] = even
    odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho[1] = odd
    t = 1
    while t < n_draw - 3 and even + odd > 0.0:
        even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if even + odd >= 0:
            rho[t + 1] = even
            rho[t + 2] = odd
        t += 2
    last = t - 2
    if even > 0:
        rho[last + 1] = even
    t = 1
    while t <= last - 2:
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]:
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2.0
            rho[t + 2] = rho[t + 1]
        t += 2

    total = chains.size
    tau = -1.0 + 2.0 * np.sum(rho[: last + 1]) + np.sum(rho[last + 1 : last + 2])
    tau = max(tau, 1 / np.log10(total))
    return float("nan") if np.isnan(rho).any() else float(total / tau)


def _ess_bulk(chains: np.ndarray) -> float:
    return _ess(_z_scale(_split(chains))) if _usable(chains, 1) else float("nan")


def _ess_quantile(chains: np.ndarray, probability: float) -> float:
    return _ess(_split(chains <= np.quantile(chains, probability)).astype(float))


def _ess_tail(chains: np.ndarray, probabilities: Sequence[float]) -> float:
    if not _usable(chains, 1):
        return float("nan")
    return min(_ess_quantile(chains, p) for p in sorted(probabilities))


def ess_bulk(chains: np.ndarray) -> np.ndarray:
    """ESS du centre de la loi (rangs normalisés, chaînes coupées en deux). chains : (N, n, ...) → (...)."""
    return _per_parameter(_ess_bulk, chains)


def ess_tail(chains: np.ndarray, probabilities: Sequence[float] = TAIL_PROBABILITIES) -> np.ndarray:
    """ESS des queues : le plus petit de ceux des quantiles `probabilities` (5 % et 95 % par défaut)."""
    return _per_parameter(lambda series: _ess_tail(series, probabilities), chains)


def _mcse_mean(chains: np.ndarray) -> float:
    if not _usable(chains, 1):
        return float("nan")
    return float(np.std(chains, ddof=1) / np.sqrt(_ess(_split(chains))))


def mcse_mean(chains: np.ndarray) -> np.ndarray:
    """Erreur standard de Monte-Carlo de la moyenne : σ / √ESS (même unité que le paramètre)."""
    return _per_parameter(_mcse_mean, chains)


def hdi(samples: np.ndarray, prob: float = 0.9, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Intervalle le plus étroit qui contient la proportion `prob` des échantillons : (bas, haut), sans l'axe `axis`.

    Les bornes sont des échantillons (méthode « nearest » d'ArviZ)."""
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in ]0, 1[")
    values = np.moveaxis(np.asarray(samples, np.float64), axis, 0)
    flat = np.sort(values.reshape(values.shape[0], -1), axis=0)
    n = flat.shape[0]
    count = min(int(np.floor(prob * n)), n - 1)
    width = flat[count:] - flat[: n - count]
    start = np.argmin(width, axis=0)
    columns = np.arange(flat.shape[1])
    low, high = flat[start, columns], flat[start + count, columns]
    return low.reshape(values.shape[1:]), high.reshape(values.shape[1:])
