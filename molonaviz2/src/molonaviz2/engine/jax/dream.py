"""Inversion bayésienne par DREAM (DiffeRential Evolution Adaptive Metropolis), en JAX.

Même algorithme que pyheatmy (core.Column.compute_mcmc, cas multi-chaînes), mais :
- toutes les chaînes sont mises à jour en parallèle (`vmap`) à partir de l'état de
  l'itération précédente (formulation standard de DREAM, Vrugt et al. 2009) ;
- le modèle direct est compilé (`jit`) ;
- l'exécution se fait par blocs d'itérations : entre deux blocs, on rend compte de
  la progression, on teste la convergence et on peut annuler.

Ajout par rapport à pyheatmy, issu du DREAM de référence : pendant le burn-in,
les chaînes aberrantes (énergie moyenne au-delà de Q3 + 2·IQR) sont replacées sur
la meilleure chaîne, ce qui évite qu'une chaîne bloquée loin du mode empêche la
convergence.

Corrections par rapport à pyheatmy (comportement voulu par l'algorithme) :
- les énergies initiales sont calculées sur les températures simulées (et non sur
  un profil nul) ;
- seuls les états acceptés alimentent les quantiles des champs ;
- le saut γ = 1 est tiré avec une probabilité de 0,2 (pyheatmy passait les
  probabilités au paramètre `replace` de `np.random.choice`, soit 0,5).

Les paramètres sont échantillonnés dans l'espace où les lois a priori sont
uniformes : (−log₁₀ k, n, λs, ρs·cs). Les bornes sont gérées par repli périodique
(« effet Pac-Man »), comme pyheatmy.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ...domain.compute import INVERTED_PARAMETERS, InversionConfig
from ...errors import UserError
from ..column import ColumnInputs
from ..results import InversionResult, QuantileResult
from .forward import build_geometry, energy, make_forward, rmse, to_forward_result

N_PARAM = len(INVERTED_PARAMETERS)
GAMMA_ONE_PROBABILITY = 0.2
BURN_IN_BLOCK = 10
OUTLIER_CHECK_EVERY = 50  # itérations (multiple de BURN_IN_BLOCK)
MIN_HISTORY_FOR_RHAT = 8


class Cancelled(Exception):
    """L'utilisateur a annulé le calcul."""


@dataclass
class Progress:
    phase: str  # "compile", "burn_in", "sampling", "finalize"
    iteration: int  # itérations faites dans la phase
    total: int  # itérations prévues dans la phase (maximum pour le burn-in)
    elapsed: float  # secondes depuis le début
    acceptance: float  # taux d'acceptation sur le dernier bloc
    chain_energies: np.ndarray  # énergie courante de chaque chaîne, (N,)
    rhat_max: float  # plus grand R-hat courant (nan si pas encore calculable)


def to_physical(x):
    """(…, L, 4) espace des priors → (…, L, 4) paramètres physiques (k, n, λs, ρs·cs)."""
    return jnp.stack([10.0 ** (-x[..., 0]), x[..., 1], x[..., 2], x[..., 3]], axis=-1)


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """R-hat « split » classique (Gelman et al., BDA3). chains : (N, n, ...) → (...)."""
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


class DreamSampler:
    def __init__(self, inputs: ColumnInputs, config: InversionConfig):
        issues = config.validate(inputs.column_depth)
        if issues:
            raise UserError("\n".join(issue.message for issue in issues))
        self.inputs = inputs
        self.config = config
        self.geom = build_geometry(inputs, [layer.z_low for layer in config.layers], config.nb_cells)
        self.nb_layers = len(config.layers)
        self.bounds = np.array(
            [[[priors[name].low, priors[name].high] for name in INVERTED_PARAMETERS] for priors in config.priors]
        )  # (L, 4, 2)
        self.sigma2 = config.sigma_t**2
        forward = make_forward(self.geom)
        geom = self.geom
        sigma2 = self.sigma2

        def evaluate(x):
            fields = forward(to_physical(x))
            temp = fields["temperature"]
            e = energy(temp, geom, sigma2)
            e = jnp.where(jnp.isfinite(e) & jnp.all(jnp.isfinite(temp)), e, jnp.inf)
            return temp, fields["water_flux"], e

        self._evaluate = jax.vmap(evaluate)
        self._forward = forward
        self._step_fn = self._make_step()
        self._block_cache: dict[tuple[int, bool], Callable] = {}

    # Proposition DREAM pour une chaîne -------------------------------------------------
    def _make_step(self):
        cfg = self.config
        n_chain, n_layer = cfg.nb_chains, self.nb_layers
        delta, n_cr = cfg.delta, cfg.n_cr
        c, c_star = cfg.c, cfg.c_star
        low = jnp.asarray(self.bounds[..., 0])
        width = jnp.asarray(self.bounds[..., 1] - self.bounds[..., 0])
        cr_values = jnp.arange(1, n_cr + 1) / n_cr
        evaluate = self._evaluate

        def propose_one(key, x_all, j, pcr):
            keys = jax.random.split(key, 7)
            id_cr = jax.vmap(lambda k, p: jax.random.choice(k, n_cr, p=p))(jax.random.split(keys[0], n_layer), pcr)
            z = jax.random.uniform(keys[1], (n_layer, N_PARAM))
            mask = z <= cr_values[id_cr][:, None]
            none = ~jnp.any(mask, axis=1)
            forced = jax.nn.one_hot(jnp.argmin(z, axis=1), N_PARAM, dtype=bool)
            mask = jnp.where(none[:, None], forced, mask)
            d_star = jnp.sum(mask, axis=1)
            lam = jax.random.uniform(keys[2], (n_layer, N_PARAM), minval=-c, maxval=c)
            zeta = c_star * jax.random.normal(keys[3], (n_layer, N_PARAM))

            def pairs(k):
                perm = jax.random.permutation(k, n_chain - 1)
                others = perm + (perm >= j)
                return others[:delta], others[delta : 2 * delta]

            a, b = jax.vmap(pairs)(jax.random.split(keys[4], n_layer))  # (L, δ)
            layer_idx = jnp.arange(n_layer)[:, None]
            diff = jnp.sum(x_all[a, layer_idx] - x_all[b, layer_idx], axis=1)  # (L, 4)
            gamma = 2.38 / jnp.sqrt(2.0 * d_star * delta)
            gamma = jnp.where(jax.random.uniform(keys[5], (n_layer,)) < GAMMA_ONE_PROBABILITY, 1.0, gamma)
            dx = jnp.where(mask, zeta + (1.0 + lam) * gamma[:, None] * diff, 0.0)
            proposal = x_all[j] + dx
            proposal = low + jnp.mod(proposal - low, width)
            return proposal, dx, id_cr

        propose_all = jax.vmap(propose_one, in_axes=(0, None, 0, None))

        def step(carry, adapt: bool):
            key, x, e, temp, flux, pcr, jumps, counts = carry
            key, k_prop, k_acc = jax.random.split(key, 3)
            proposal, dx, id_cr = propose_all(jax.random.split(k_prop, n_chain), x, jnp.arange(n_chain), pcr)
            temp_p, flux_p, e_p = evaluate(proposal)
            log_u = jnp.log(jax.random.uniform(k_acc, (n_chain,)))
            accept = log_u < (e - e_p)
            x_new = jnp.where(accept[:, None, None], proposal, x)
            e_new = jnp.where(accept, e_p, e)
            temp_new = jnp.where(accept[:, None, None], temp_p, temp)
            flux_new = jnp.where(accept[:, None, None], flux_p, flux)
            if adapt:
                std = jnp.std(x, axis=0)
                std = jnp.where(std == 0, 1.0, std)
                dx_acc = jnp.where(accept[:, None, None], dx, 0.0)
                contrib = jnp.sum((dx_acc / std) ** 2, axis=2)  # (N, L)
                one_hot = jax.nn.one_hot(id_cr, n_cr)  # (N, L, n_cr)
                jumps = jumps + jnp.sum(one_hot * contrib[..., None], axis=0)
                counts = counts + jnp.sum(one_hot, axis=0)
                ratio = jnp.where(counts > 0, jumps / jnp.where(counts > 0, counts, 1.0), pcr)
                total = jnp.sum(ratio, axis=1, keepdims=True)
                pcr = jnp.where(total > 0, ratio / jnp.where(total > 0, total, 1.0), 1.0 / n_cr)
            carry = (key, x_new, e_new, temp_new, flux_new, pcr, jumps, counts)
            return carry, (x_new, e_new, accept)

        return step

    def _block(self, length: int, adapt: bool):
        cache_key = (length, adapt)
        if cache_key not in self._block_cache:
            step = self._step_fn

            @jax.jit
            def run(carry):
                return jax.lax.scan(lambda c, _: step(c, adapt), carry, None, length=length)

            self._block_cache[cache_key] = run
        return self._block_cache[cache_key]

    # Exécution ------------------------------------------------------------------------
    def run(
        self,
        on_progress: Callable[[Progress], None] | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> InversionResult:
        cfg = self.config
        n_chain = cfg.nb_chains
        start = time.perf_counter()

        def report(phase, iteration, total, accepted, energies, rhat_max):
            if should_cancel is not None and should_cancel():
                raise Cancelled
            if on_progress is not None:
                on_progress(Progress(phase, iteration, total, time.perf_counter() - start, accepted, np.asarray(energies), rhat_max))

        report("compile", 0, 1, float("nan"), np.full(n_chain, np.nan), float("nan"))
        key = jax.random.PRNGKey(cfg.seed)
        key, k_init = jax.random.split(key)
        low, high = self.bounds[..., 0], self.bounds[..., 1]
        x0 = jnp.asarray(low) + jax.random.uniform(k_init, (n_chain, self.nb_layers, N_PARAM)) * jnp.asarray(high - low)
        temp0, flux0, e0 = jax.jit(self._evaluate)(x0)
        pcr = jnp.full((self.nb_layers, cfg.n_cr), 1.0 / cfg.n_cr)
        zeros = jnp.zeros((self.nb_layers, cfg.n_cr))
        carry = (key, x0, e0, temp0, flux0, pcr, zeros, zeros)

        best_x, best_e = np.asarray(x0[np.argmin(e0)]), float(np.min(e0))

        def track_best(xs, es):
            nonlocal best_x, best_e
            es = np.asarray(es)
            flat = int(np.argmin(es))
            i, j = np.unravel_index(flat, es.shape)
            if es[i, j] < best_e:
                best_e, best_x = float(es[i, j]), np.asarray(xs[i, j])

        # Burn-in : adaptation des crossovers jusqu'à convergence (Gelman-Rubin) ou au maximum.
        history: list[np.ndarray] = [np.asarray(x0)]
        energy_history: list[np.ndarray] = [np.asarray(e0)]
        burn_in, converged, rhat_max = 0, False, float("nan")
        while burn_in < cfg.burn_in_max:
            length = min(BURN_IN_BLOCK, cfg.burn_in_max - burn_in)
            carry, (xs, es, acc) = self._block(length, True)(carry)
            burn_in += length
            history.extend(np.asarray(xs))
            energy_history.extend(np.asarray(es))
            track_best(xs, es)
            reset = np.zeros(n_chain, bool)
            if burn_in % OUTLIER_CHECK_EVERY == 0:
                carry, reset = _reset_outlier_chains(
                    carry, np.stack(energy_history[len(energy_history) // 2 :]), margin=float(self.nb_layers * N_PARAM)
                )
            if reset.any():
                # Les chaînes replacées repartent d'un nouvel état : on oublie leur passé pour R-hat.
                history = history[-1:]
                energy_history = energy_history[-1:]
            if len(history) >= MIN_HISTORY_FOR_RHAT:
                recent = np.stack(history[len(history) // 2 :], axis=1)  # (N, n, L, 4)
                rhat = split_rhat(recent)
                rhat_max = float(np.max(rhat))
                if np.all(rhat < cfg.gr_threshold):
                    converged = True
            report("burn_in", burn_in, cfg.burn_in_max, float(np.mean(acc)), carry[2], rhat_max)
            if converged:
                break
        del history

        # Échantillonnage : paramètres à chaque itération, champs (T, q) régulièrement.
        per_chain = max(1, math.ceil(cfg.n_field_samples / n_chain))
        interval = max(1, cfg.nb_iter // per_chain)
        sl_space = slice(None, None, cfg.space_step)
        sl_time = slice(None, None, cfg.time_step)
        samples, energies, accepted = [], [], []
        field_temp, field_flux = [], []
        done = 0
        while done < cfg.nb_iter:
            length = min(interval, cfg.nb_iter - done)
            carry, (xs, es, acc) = self._block(length, False)(carry)
            done += length
            samples.append(np.asarray(xs))
            energies.append(np.asarray(es))
            accepted.append(np.asarray(acc))
            track_best(xs, es)
            if length == interval:
                field_temp.append(np.asarray(carry[3][:, sl_space, sl_time], np.float32))
                field_flux.append(np.asarray(carry[4][:, sl_space, sl_time], np.float32))
            report("sampling", done, cfg.nb_iter, float(np.mean(acc)), carry[2], rhat_max)

        report("finalize", 0, 1, float("nan"), carry[2], rhat_max)
        samples_arr = np.concatenate(samples)  # (I, N, L, 4)
        energies_arr = np.concatenate(energies)
        accepted_arr = np.concatenate(accepted)
        final_rhat = split_rhat(np.moveaxis(samples_arr, 1, 0))

        best_fields = jax.jit(self._forward)(to_physical(jnp.asarray(best_x)))
        best = to_forward_result(best_fields, self.geom, self.inputs)

        temps = np.concatenate(field_temp)  # (S, C', T') en K
        fluxes = np.concatenate(field_flux)
        field_depths = self.geom.z_solve[sl_space]
        field_times = self.inputs.times[sl_time]
        sensor_rows = np.array([np.argmin(np.abs(field_depths - d)) for d in best.sensor_depths])
        observed = self.geom.observed[:, sl_time]
        quantiles = []
        for q, temp_q, flux_q in zip(
            cfg.quantiles, np.quantile(temps, cfg.quantiles, axis=0), np.quantile(fluxes, cfg.quantiles, axis=0)
        ):
            per_sensor = np.sqrt(np.nansum((temp_q[sensor_rows] - observed) ** 2, axis=1) / observed.shape[1])
            total = np.sqrt(np.nansum(per_sensor**2) / len(per_sensor))
            quantiles.append(QuantileResult(float(q), temp_q - 273.15, flux_q, np.append(per_sensor, total)))

        return InversionResult(
            best=best,
            best_parameters=best_x,
            samples=samples_arr,
            energies=energies_arr,
            acceptance=accepted_arr.mean(axis=0),
            burn_in_iterations=burn_in,
            burn_in_converged=converged,
            rhat=final_rhat,
            quantiles=quantiles,
            field_depths=field_depths,
            field_times=field_times,
        )


def _reset_outlier_chains(carry, recent_energies: np.ndarray, margin: float):
    """Replace les chaînes dont l'énergie moyenne récente est aberrante sur la meilleure chaîne.

    recent_energies : (n, N). Règle de Vrugt et al. (2009) sur la log-densité moyenne,
    exprimée ici en énergie (= −log densité) : aberrante si > Q3 + max(2·IQR, margin).
    Le plancher `margin` (de l'ordre de la dimension du problème) évite, quand les
    chaînes bougent peu et que l'IQR est quasi nul, de replacer des chaînes saines
    et de détruire la diversité dont DREAM a besoin.
    """
    mean_energy = np.mean(np.where(np.isfinite(recent_energies), recent_energies, np.inf), axis=0)
    finite = mean_energy[np.isfinite(mean_energy)]
    if finite.size < 4:
        return carry, np.zeros(len(mean_energy), bool)
    q1, q3 = np.percentile(finite, [25, 75])
    outliers = mean_energy > q3 + max(2.0 * (q3 - q1), margin)
    if not outliers.any():
        return carry, outliers
    key, x, e, temp, flux, pcr, jumps, counts = carry
    best = int(np.argmin(np.asarray(e)))
    mask = jnp.asarray(outliers)
    x = jnp.where(mask[:, None, None], x[best], x)
    e = jnp.where(mask, e[best], e)
    temp = jnp.where(mask[:, None, None], temp[best], temp)
    flux = jnp.where(mask[:, None, None], flux[best], flux)
    return (key, x, e, temp, flux, pcr, jumps, counts), outliers


def single_energy(inputs: ColumnInputs, config: InversionConfig, x: np.ndarray) -> float:
    """Énergie d'un jeu de paramètres (L, 4) dans l'espace des priors (utilitaire de test)."""
    geom = build_geometry(inputs, [layer.z_low for layer in config.layers], config.nb_cells)
    fields = make_forward(geom)(to_physical(jnp.asarray(x)))
    return float(energy(fields["temperature"], geom, config.sigma_t**2))


__all__ = ["Cancelled", "DreamSampler", "Progress", "rmse", "single_energy", "split_rhat", "to_physical"]
