"""Les deux DREAM échantillonnent la même loi : celui du twin (JAX, `engine/jax/dream.py`) et celui de
bayesian_inversion 0.4.0 (`samplers.DREAM`, par `run_mcmc`).

Ce que les tests vérifient :
- test 1, une cible gaussienne connue (deux coordonnées gaussiennes, deux libres et uniformes) : chaque DREAM retrouve
  la moyenne et la variance de la cible, et les deux s'accordent entre eux ;
- test 2, une colonne synthétique à une couche, dont les observations sortent du moteur JAX plus un bruit gaussien :
  les médianes et les quantiles à 5 % et 95 % des quatre paramètres, et du flux d'eau à une profondeur, s'accordent.

Pourquoi on n'attend pas l'identité au bit : les deux algorithmes diffèrent par construction. Le DREAM du twin met à
jour toutes les chaînes en parallèle à partir de l'état de l'itération précédente (Vrugt et al. 2009), celui de
bayesian_inversion les met à jour l'une après l'autre ; les générateurs aléatoires (JAX, `random` et `numpy`) et le
remplacement des chaînes aberrantes pendant le burn-in (JAX seulement) diffèrent aussi. On compare donc des lois, à
l'erreur Monte-Carlo près.

Méthode d'erreur, reprise de `tests/test_samplers.py` de bayesian_inversion (bb07a12) : moyennes de lots d'itérations
consécutives (`N_BATCH`), erreur type = écart-type des moyennes de lots / sqrt(`N_BATCH`). Les chaînes d'un DREAM sont
corrélées dans le temps : un lot d'itérations consécutives en tient compte, pas une erreur calculée comme si les
échantillons étaient indépendants. Cette méthode sert au test 1 (gaussienne). Au test 2 (colonne), l'a posteriori est
presque aussi large que l'a priori, et l'erreur par lots y était sous-estimée (dispersion réelle entre graines 0,05 à
0,07 sur la médiane de neg_log10_k, pour 0,025 annoncés). Ce n'est pas que l'autocorrélation dépasse un lot : le temps
d'autocorrélation intégré vaut environ 39 itérations (19 à 79 selon les chaînes), soit 2,6 par lot de 100. Mais un lot
regroupe toutes les chaînes. Une chaîne qui garde longtemps sa propre région (vu côté bayesian_inversion, qui ne
remplace pas les chaînes aberrantes) décale alors tous les lots de la même façon, et leur dispersion ne la voit pas. On
y calcule donc chaque quantile sur chaque chaîne entière, la valeur comparée est celle des chaînes
regroupées, et l'erreur est std(statistiques par chaîne, ddof=1)/sqrt(nombre de chaînes), soit 16 chaînes et 15 degrés de liberté pour l'erreur. La tolérance est la même
partout : `N_STANDARD_ERRORS` erreurs types, jamais élargie pour faire passer un test.

Test 2 : avec un bruit de 0,5 °C, lambda_s, rhos_cs et porosity restent proches de leur loi a priori : la comparaison
contraint surtout neg_log10_k (et le flux, qui en dépend).

Même loi a priori des deux côtés (vérifiée par les tests) : uniforme dans `neg_log10_k`, `porosity`, `lambda_s` et
`rhos_cs` entre les bornes de `config.priors`. Côté bayesian_inversion, `IntrinK` en échelle `log` équivaut à une loi
uniforme en `neg_log10_k = -log10(IntrinK)` ; les trois autres paramètres sont en échelle `linear`.
"""

import dataclasses
import math
import random
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import synthetic_inputs
from molonari_twin.domain.compute import (
    INVERTED_PARAMETERS,
    ForwardConfig,
    InversionConfig,
    LayerSpec,
)
from molonari_twin.engine.api import run_forward
from molonari_twin.engine.jax.dream import DreamSampler
from molonari_twin.engine.reference import _column

pytestmark = pytest.mark.slow

N_BATCH = 20  # lots d'itérations consécutives pour l'erreur Monte-Carlo
N_STANDARD_ERRORS = 4  # la seule tolérance du fichier, en erreurs types ; fixée avant le premier lancement

# Réglages DREAM communs, passés explicitement des deux côtés (les défauts de c_star diffèrent : 1e-6 et 1e-12).
DELTA = 3
N_CR = 3
C = 0.1
C_STAR = 1e-6


# Erreur Monte-Carlo ---------------------------------------------------------------------------------------------
def batches(samples):
    """(itérations, chaînes, K) → (N_BATCH, taille du lot, chaînes, K), les dernières itérations en trop sont écartées."""
    size = samples.shape[0] // N_BATCH
    assert size >= 1, "pas assez d'itérations pour N_BATCH lots"
    return samples[: size * N_BATCH].reshape(N_BATCH, size, *samples.shape[1:])


def batch_mean_and_error(samples):
    """Moyenne et erreur type (par moyennes de lots) de chaque colonne de samples (itérations, chaînes, K)."""
    means = batches(samples).mean(axis=(1, 2))
    return means.mean(axis=0), means.std(axis=0, ddof=1) / math.sqrt(N_BATCH)


def batch_second_moment_and_error(samples, center):
    """Moyenne de (x − center)² et son erreur type : la variance si center est la moyenne."""
    return batch_mean_and_error((samples - center) ** 2)


def chain_quantiles_and_errors(samples, quantiles):
    """Quantiles des chaînes regroupées (itérations, chaînes, K) et erreur type entre chaînes : (Q, K).

    La statistique est calculée sur chaque chaîne entière ; l'erreur est std(par chaîne, ddof=1)/sqrt(chaînes)."""
    pooled = np.quantile(samples.reshape(-1, samples.shape[-1]), quantiles, axis=0)
    per_chain = np.quantile(samples, quantiles, axis=0)  # (Q, chaînes, K)
    return pooled, per_chain.std(axis=1, ddof=1) / math.sqrt(samples.shape[1])


def check_close(label, value, expected, error, failures):
    """Un écart de plus de N_STANDARD_ERRORS erreurs types est consigné ; l'erreur nulle est un test invalide."""
    ok = bool(np.all(np.abs(value - expected) < N_STANDARD_ERRORS * error))
    if not ok:
        failures.append(f"{label} : {np.round(value, 6)} contre {np.round(expected, 6)}, erreur {np.round(error, 6)}")


def check_agree(label, first, first_error, second, second_error, failures):
    """Deux estimations s'accordent si |a − b| < N_STANDARD_ERRORS·sqrt(ea² + eb²)."""
    bound = N_STANDARD_ERRORS * np.sqrt(first_error**2 + second_error**2)
    if not bool(np.all(np.abs(first - second) < bound)):
        failures.append(
            f"{label} : JAX {np.round(first, 6)} contre bayesian_inversion {np.round(second, 6)}, "
            f"écart {np.round(np.abs(first - second), 6)}, tolérance {np.round(bound, 6)}"
        )


# Test 1 : cible gaussienne --------------------------------------------------------------------------------------
GAUSS_MEAN = np.array([13.0, 0.13])  # neg_log10_k et porosity, comme test_engine_dream.py
GAUSS_SD = np.array([0.2, 0.012])
GAUSS_CHAINS = 10
GAUSS_BURN_IN = 1000  # pas, fixes des deux côtés
GAUSS_ITERATIONS = 3000


def gaussian_config(**overrides):
    values = {
        "layers": [LayerSpec("a", 0.4)], "nb_cells": 20, "nb_chains": GAUSS_CHAINS, "nb_iter": GAUSS_ITERATIONS,
        "burn_in_max": GAUSS_BURN_IN, "delta": DELTA, "n_cr": N_CR, "c": C, "c_star": C_STAR,
        # limite basse documentée (SETTING_LIMITS) : le burn-in dure burn_in_max pas, pour qu'il soit le même des deux côtés
        "gr_threshold": 1.0001, "n_field_samples": 10, "seed": 0,
    }
    values.update(overrides)
    return InversionConfig(**values)


def gaussian_energy(coordinates):
    """Énergie de la cible : gaussienne indépendante sur les deux premières coordonnées, nulle ailleurs (a priori seul)."""
    return 0.5 * np.sum(((coordinates[:2] - GAUSS_MEAN) / GAUSS_SD) ** 2)


def jax_gaussian_samples(config):
    sampler = DreamSampler(synthetic_inputs(nb_times=20), config)
    mean, sd = jnp.asarray(GAUSS_MEAN), jnp.asarray(GAUSS_SD)

    def gaussian(x):
        d = (x[:, 0, :2] - mean) / sd
        dummy = jnp.zeros((x.shape[0], 20, 20))
        return dummy, dummy, 0.5 * jnp.sum(d**2, axis=1)

    sampler._evaluate = gaussian
    sampler._step_fn = sampler._make_step()
    result = sampler.run()
    return result


def bayesian_inversion_gaussian_samples(config):
    from bayesian_inversion import DREAM, Prior

    bounds = [(config.priors[0][name].low, config.priors[0][name].high) for name in INVERTED_PARAMETERS]
    priors = [[Prior(b, 0.01 * (b[1] - b[0])) for b in bounds]]
    # la même loi a priori : une échelle `log` ou `symlog` changerait la mesure uniforme des coordonnées
    assert [prior.scale for prior in priors[0]] == ["linear"] * 4
    sigma2_prior = Prior((1.0, 1.0), 0, lambda x: 1)  # sigma2 sans effet : l'énergie ne l'utilise pas
    sampler = DREAM(
        lambda x: np.array(x), lambda output, sigma2: gaussian_energy(output[0]), priors, sigma2_prior, config.nb_chains,
        DELTA, N_CR, C, C_STAR,
    )
    random.seed(0)
    np.random.seed(0)
    sampler.initialize()
    for _ in range(config.burn_in_max):
        sampler.step(adapt=True)
    samples = np.zeros((config.nb_iter, config.nb_chains, 4))
    acceptance = np.zeros(config.nb_chains)
    for i in range(config.nb_iter):
        acceptance += sampler.step()
        samples[i] = sampler.x[:, 0, :]
    return samples, acceptance / config.nb_iter


def test_both_dreams_sample_the_same_gaussian():
    config = gaussian_config()
    result = jax_gaussian_samples(config)
    # burn-in de longueur fixe : gr_threshold ne peut pas être atteint, le burn-in dure burn_in_max pas
    assert result.burn_in_iterations == GAUSS_BURN_IN, "le burn-in JAX devait durer burn_in_max pas"
    assert result.resets.sum() == 0, f"{int(result.resets.sum())} chaînes JAX remplacées pendant le burn-in"
    jax_samples = result.samples[:, :, 0, :]  # (itérations, chaînes, 4)
    bi_samples, bi_acceptance = bayesian_inversion_gaussian_samples(config)
    assert jax_samples.shape == bi_samples.shape == (GAUSS_ITERATIONS, GAUSS_CHAINS, 4)

    low = np.array([config.priors[0][name].low for name in INVERTED_PARAMETERS])
    high = np.array([config.priors[0][name].high for name in INVERTED_PARAMETERS])
    target_mean = np.concatenate([GAUSS_MEAN, (low[2:] + high[2:]) / 2])  # coordonnées libres : uniformes
    target_var = np.concatenate([GAUSS_SD**2, (high[2:] - low[2:]) ** 2 / 12])

    failures = []
    estimates = {}
    for name, samples in (("jax", jax_samples), ("bi", bi_samples)):
        mean, mean_error = batch_mean_and_error(samples)
        var, var_error = batch_second_moment_and_error(samples, mean)
        # variance autour de la moyenne de la cible, pour la comparer à la cible
        target_second, target_second_error = batch_second_moment_and_error(samples, target_mean)
        assert np.all(mean_error > 0) and np.all(var_error > 0)
        check_close(f"{name} : moyenne", mean, target_mean, mean_error, failures)
        check_close(f"{name} : variance", target_second, target_var, target_second_error, failures)
        estimates[name] = (mean, mean_error, var, var_error)
    check_agree("moyennes", *estimates["jax"][:2], *estimates["bi"][:2], failures)
    check_agree("variances", *estimates["jax"][2:], *estimates["bi"][2:], failures)

    print(
        f"\n[gaussienne] burn-in JAX {result.burn_in_iterations} pas, resets JAX {int(result.resets.sum())}, "
        f"acceptation JAX {result.acceptance.mean():.3f}, bayesian_inversion {bi_acceptance.mean():.3f}"
    )
    for name, (mean, mean_error, var, var_error) in estimates.items():
        print(f"[gaussienne] {name:4s} moyennes {mean} ± {mean_error}")
        print(f"[gaussienne] {name:4s} variances {var} ± {var_error}")
    print(f"[gaussienne] cible moyennes {target_mean}, variances {target_var}")
    assert not failures, "\n".join(failures)


# Test 2 : colonne synthétique -----------------------------------------------------------------------------------
COLUMN_TIMES = 60
COLUMN_CELLS = 20
COLUMN_CHAINS = 16
COLUMN_BURN_IN_MAX = 1500
COLUMN_ITERATIONS = 2000
SIGMA_T = 0.5  # bruit des observations et sigma_t de la config : à 0,1 °C l'a posteriori est si étroit que le burn-in dépasse 1000 pas
COLUMN_GR_THRESHOLD = 1.2
COLUMN_STRIDE = 10  # une itération sur 10 pour le flux, recalculé par le modèle direct
EVAL_BATCH = 100  # taille fixe des paquets de _evaluate (une autre forme recompile)
TRUE_LAYER = LayerSpec("a", 0.4, neg_log10_k=12.5, porosity=0.15, lambda_s=3.0, rhos_cs=4e6)
NOISE_SEED = 12345
DREAM_SEED = 0
QUANTILES = (0.05, 0.5, 0.95)


def column_inputs():
    """Observations produites par le moteur JAX pour TRUE_LAYER, plus un bruit gaussien d'écart-type SIGMA_T."""
    inputs = synthetic_inputs(nb_times=COLUMN_TIMES, sigma_t=SIGMA_T)
    clean = run_forward(inputs, ForwardConfig([TRUE_LAYER], nb_cells=COLUMN_CELLS))
    noise = np.random.default_rng(NOISE_SEED).normal(0.0, SIGMA_T, size=inputs.t_sensors.shape)
    return dataclasses.replace(inputs, t_sensors=clean.temperature[clean.sensor_index].T + noise)


def pyheatmy_column(inputs, config):
    from pyheatmy.layers import layersListCreator

    column = _column(inputs, config.nb_cells)
    layers = [(layer.name, layer.z_low, *layer.physical(), 0.0) for layer in config.layers]  # q_s = 0, comme reference.forward
    column.set_layers(layersListCreator(layers))
    return column


def column_state_space(config):
    from bayesian_inversion import Prior, StateSpace

    priors = config.priors[0]
    k = priors["neg_log10_k"]
    ranges = {
        "IntrinK": (10.0 ** -k.high, 10.0 ** -k.low),  # uniforme en neg_log10_k = log en IntrinK
        "n": (priors["porosity"].low, priors["porosity"].high),
        "lambda_s": (priors["lambda_s"].low, priors["lambda_s"].high),
        "rhos_cs": (priors["rhos_cs"].low, priors["rhos_cs"].high),
    }
    block = {name: Prior(r, 0.01 * (r[1] - r[0])) for name, r in ranges.items()}
    block["q_s"] = Prior(0.0, 0.0)
    state_space = StateSpace({config.layers[0].name: block})
    assert [block[name].scale for name in ("IntrinK", "n", "lambda_s", "rhos_cs", "q_s")] == [
        "log", "linear", "linear", "linear", "fixed"
    ], "l'échelle d'un Prior ne donne plus la même loi a priori que JAX : ne pas changer les bornes en silence"
    return state_space


def to_prior_space(physical):
    """(…, 4) paramètres physiques de bayesian_inversion (IntrinK, n, λs, ρs·cs) → espace des priors de JAX."""
    return np.stack([-np.log10(physical[..., 0]), physical[..., 1], physical[..., 2], physical[..., 3]], axis=-1)


def water_flux_at_depth(sampler, states, cell):
    """Flux d'eau moyen dans le temps à la cellule `cell`, pour chaque état (N, 1, 4) en espace des priors.

    Le modèle est celui du DreamSampler (compilé, vectorisé), commun aux deux DREAM, par paquets de taille fixe."""
    evaluate = jax.jit(sampler._evaluate)
    padding = -len(states) % EVAL_BATCH
    padded = np.concatenate([states, np.repeat(states[-1:], padding, axis=0)])
    flux = np.concatenate([np.asarray(evaluate(jnp.asarray(padded[i : i + EVAL_BATCH]))[1]) for i in range(0, len(padded), EVAL_BATCH)])
    return flux[: len(states), cell, :].mean(axis=1)


def test_both_dreams_agree_on_a_synthetic_column():
    from bayesian_inversion import run_mcmc

    inputs = column_inputs()
    config = InversionConfig(
        layers=[TRUE_LAYER], nb_cells=COLUMN_CELLS, nb_chains=COLUMN_CHAINS, nb_iter=COLUMN_ITERATIONS,
        burn_in_max=COLUMN_BURN_IN_MAX, delta=DELTA, n_cr=N_CR, c=C, c_star=C_STAR, gr_threshold=COLUMN_GR_THRESHOLD,
        sigma_t=SIGMA_T, n_field_samples=16, seed=DREAM_SEED,
    )
    assert not config.validate(inputs.column_depth)

    # JAX : burn-in naturel, convergence vérifiée.
    sampler = DreamSampler(inputs, config)  # `_evaluate` n'est pas remplacé : il sert aussi à recalculer les flux
    result = sampler.run()
    assert result.burn_in_converged, (
        f"le burn-in JAX n'a pas convergé en {result.burn_in_iterations} pas (burn_in_max {COLUMN_BURN_IN_MAX}) : "
        "comparaison douteuse"
    )

    # bayesian_inversion : même observations, même modèle, même énergie (sigma2 fixe = sigma_t²).
    column = pyheatmy_column(inputs, config)
    state_space = column_state_space(config)
    observations = column.get_measured_temperatures()
    np.testing.assert_allclose(observations - 273.15, inputs.t_sensors.T, atol=1e-9)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bi = run_mcmc(
            column.simulate, observations, state_space, stored=("flows",), typealgo="no sigma", sigma2=config.sigma_t**2,
            nb_iter=COLUMN_ITERATIONS, nb_chain=COLUMN_CHAINS, nitmaxburning=COLUMN_BURN_IN_MAX, delta=DELTA, n_CR=N_CR,
            c=C, c_star=C_STAR, threshold=COLUMN_GR_THRESHOLD, seed=DREAM_SEED,
        )
    not_converged = [str(w.message) for w in caught if "without converging" in str(w.message)]
    assert not not_converged, f"le burn-in de bayesian_inversion n'a pas convergé : {not_converged}"
    assert bi.nb_burn_in_iter < COLUMN_BURN_IN_MAX, "le burn-in de bayesian_inversion a atteint nitmaxburning"

    # Même énergie : à deux états différents, la différence d'énergie est la même (les constantes ne comptent pas).
    states = [np.array([[12.0, 0.1, 2.0, 3e6]]), np.array([[13.0, 0.2, 4.0, 6e6]])]
    jax_energy = [float(sampler._evaluate(jnp.asarray(s[None]))[2][0]) for s in states]
    bi_energy = []
    for s in states:
        output = column.simulate([{"IntrinK": 10.0 ** -s[0, 0], "n": s[0, 1], "lambda_s": s[0, 2], "rhos_cs": s[0, 3], "q_s": 0.0}])
        residual = output["sensors"] - observations
        bi_energy.append(np.sum(residual**2) / (2 * config.sigma_t**2))
    assert jax_energy[1] - jax_energy[0] == pytest.approx(bi_energy[1] - bi_energy[0], rel=1e-6)

    jax_samples = result.samples[:, :, 0, :]  # (itérations, chaînes, 4), espace des priors
    bi_physical = bi.chains()[1:, :, 0, :4]  # les nb_chain premiers états sont la fin du burn-in
    bi_samples = to_prior_space(bi_physical)
    assert jax_samples.shape == bi_samples.shape == (COLUMN_ITERATIONS, COLUMN_CHAINS, 4)

    # Flux d'eau à la cellule du capteur du milieu, recalculé par le même modèle pour les états des deux DREAM.
    cell = int(sampler.geom.sensor_index[1])
    jax_states = jax_samples[::COLUMN_STRIDE]
    bi_states = bi_samples[::COLUMN_STRIDE]
    jax_flux = water_flux_at_depth(sampler, jax_states.reshape(-1, 1, 4), cell).reshape(*jax_states.shape[:2], 1)
    bi_flux = water_flux_at_depth(sampler, bi_states.reshape(-1, 1, 4), cell).reshape(*bi_states.shape[:2], 1)
    # le modèle du sampler égale pyheatmy sur un état de bayesian_inversion
    one = bi_physical[-1, 0]
    reference = column.simulate([dict(zip(("IntrinK", "n", "lambda_s", "rhos_cs", "q_s"), (*one, 0.0)))])["flows"][cell].mean()
    assert water_flux_at_depth(sampler, bi_samples[-1:, 0:1, :].reshape(1, 1, 4), cell)[0] == pytest.approx(reference, rel=1e-6)

    failures = []
    report = {}
    margins = {}
    for label, jax_values, bi_values in (
        ("paramètres", jax_samples, bi_samples), ("flux d'eau", jax_flux, bi_flux),
    ):
        jax_q, jax_e = chain_quantiles_and_errors(jax_values, QUANTILES)
        bi_q, bi_e = chain_quantiles_and_errors(bi_values, QUANTILES)
        assert np.all(jax_e > 0) and np.all(bi_e > 0)
        for i, q in enumerate(QUANTILES):
            bound = N_STANDARD_ERRORS * np.sqrt(jax_e[i] ** 2 + bi_e[i] ** 2)
            margins[f"{label} q{q:.2f}"] = np.abs(jax_q[i] - bi_q[i]) / bound
            check_agree(f"{label}, quantile {q}", jax_q[i], jax_e[i], bi_q[i], bi_e[i], failures)
        report[label] = (jax_q, jax_e, bi_q, bi_e)

    print(
        f"\n[colonne] burn-in réel : JAX {result.burn_in_iterations} pas (convergé {result.burn_in_converged}), "
        f"bayesian_inversion {bi.nb_burn_in_iter} pas ; resets JAX {int(result.resets.sum())} "
        f"({result.resets.tolist()}) ; R-hat final JAX {np.round(result.rhat[0], 3)}"
    )
    print(f"[colonne] acceptation : JAX {result.acceptance.mean():.3f}, bayesian_inversion {np.mean(bi.acceptance):.3f}")
    names = (*INVERTED_PARAMETERS,)
    for label, (jax_q, jax_e, bi_q, bi_e) in report.items():
        for i, q in enumerate(QUANTILES):
            tag = names if label == "paramètres" else ("flux",)
            for k, name in enumerate(tag):
                print(
                    f"[colonne] {label} {name} q{q:.2f} : JAX {jax_q[i, k]:.6g} ± {jax_e[i, k]:.2g}, "
                    f"bayesian_inversion {bi_q[i, k]:.6g} ± {bi_e[i, k]:.2g}"
                )
    for label, ratio in margins.items():
        print(f"[colonne] marge (écart / tolérance) {label} : {np.round(ratio, 2)}")
    assert not failures, "\n".join(failures)
