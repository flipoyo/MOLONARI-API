"""Inversion DREAM : exactitude de l'échantillonneur, déterminisme, annulation, convergence."""

import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import synthetic_inputs

from molonaviz2.domain.compute import InversionConfig, LayerSpec
from molonaviz2.engine.api import run_forward, run_inversion
from molonaviz2.engine.jax.dream import (
    Cancelled,
    DreamSampler,
    sampling_layout,
    split_rhat,
)
from molonaviz2.errors import UserError


def small_config(**overrides):
    values = {
        "layers": [LayerSpec("a", 0.4)], "nb_cells": 20, "nb_chains": 7, "nb_iter": 30, "burn_in_max": 20, "n_field_samples": 14, "seed": 1
    }
    values.update(overrides)
    return InversionConfig(**values)


def test_split_rhat_matches_arviz():
    rng = np.random.default_rng(0)
    chains = rng.normal(size=(4, 200, 3)) + np.array([0.0, 0.1, 0.5])[None, None, :] * np.arange(4)[:, None, None]
    ours = split_rhat(chains)
    for p in range(3):
        expected = float(az.rhat(chains[:, :, p], method="split"))
        assert ours[p] == pytest.approx(expected, rel=1e-10)


def test_sampler_recovers_a_correlated_gaussian():
    """Cible connue (gaussienne 4D, corrélation 0,9) : vérifie l'algorithme indépendamment de la physique."""
    inputs = synthetic_inputs(nb_times=20)
    sampler = DreamSampler(inputs, small_config(nb_chains=10, nb_iter=3000, burn_in_max=1000, n_field_samples=10, seed=0))
    mean = jnp.array([13.0, 0.13, 3.0, 5.5e6])
    sd = jnp.array([0.2, 0.012, 0.2, 4.5e5])
    cov = (0.9 * jnp.ones((4, 4)) + 0.1 * jnp.eye(4)) * sd[:, None] * sd[None, :]
    precision = jnp.linalg.inv(cov)

    def gaussian(x):
        d = x[:, 0, :] - mean
        dummy = jnp.zeros((x.shape[0], 20, 20))
        return dummy, dummy, 0.5 * jnp.einsum("ni,ij,nj->n", d, precision, d)

    sampler._evaluate = gaussian
    sampler._step_fn = sampler._make_step()
    result = sampler.run()
    post = result.samples[:, :, 0, :].reshape(-1, 4)
    assert result.burn_in_converged
    np.testing.assert_allclose((post.mean(0) - mean) / sd, 0.0, atol=0.15)
    np.testing.assert_allclose(post.std(0) / sd, 1.0, atol=0.1)
    np.testing.assert_allclose(np.corrcoef(post.T)[0, 1:], 0.9, atol=0.05)
    assert np.all(result.rhat < 1.05)
    assert 0.1 < result.acceptance.mean() < 0.4


def test_same_seed_same_result():
    inputs = synthetic_inputs(nb_times=60)
    first = run_inversion(inputs, small_config())
    second = run_inversion(inputs, small_config())
    np.testing.assert_array_equal(first.samples, second.samples)
    other = run_inversion(inputs, small_config(seed=2))
    assert not np.array_equal(first.samples, other.samples)


def test_result_content():
    inputs = synthetic_inputs(nb_times=60)
    config = small_config(quantiles=(0.1, 0.5, 0.9), time_step=2)
    result = run_inversion(inputs, config)
    assert result.samples.shape == (30, 7, 1, 4)
    assert result.energies.shape == (30, 7)
    assert [q.quantile for q in result.quantiles] == [0.1, 0.5, 0.9]
    assert result.quantiles[0].temperature.shape == (20, 30)
    # Les quantiles sont ordonnés en chaque point.
    assert np.all(result.quantiles[0].temperature <= result.quantiles[2].temperature + 1e-9)
    # Paramètres dans les bornes a priori.
    for p, name in enumerate(("neg_log10_k", "porosity", "lambda_s", "rhos_cs")):
        prior = config.priors[0][name]
        assert np.all((result.samples[..., p] >= prior.low) & (result.samples[..., p] <= prior.high))
    # Le meilleur jeu de paramètres est celui de plus basse énergie rencontrée.
    assert result.best.temperature.shape == (20, 60)


def test_progress_phases_and_cancellation():
    inputs = synthetic_inputs(nb_times=40)
    phases = []
    run_inversion(inputs, small_config(), on_progress=lambda p: phases.append(p.phase))
    assert phases[0] == "compile" and phases[-1] == "finalize"
    assert "burn_in" in phases and "sampling" in phases

    calls = []

    def cancel_after_burn_in_starts():
        calls.append(1)
        return len(calls) > 2

    with pytest.raises(Cancelled):
        run_inversion(inputs, small_config(burn_in_max=200), should_cancel=cancel_after_burn_in_starts)


@pytest.mark.parametrize("nb_chains, nb_iter, fields, per_chain, interval", [
    (7, 30, 14, 2, 15),  # cas courant : 2 champs par chaîne
    (7, 30, 15, 3, 10),  # 15 demandés → 3 par chaîne, soit 21
    (7, 30, 3, 1, 30),  # moins de champs que de chaînes : 1 par chaîne au moins
    (7, 5, 100, 5, 1),  # plus de champs par chaîne que d'itérations : 1 par itération
])
def test_sampling_layout_per_chain(nb_chains, nb_iter, fields, per_chain, interval):
    layout = sampling_layout(small_config(nb_chains=nb_chains, nb_iter=nb_iter, n_field_samples=fields, space_step=3, time_step=4), nb_times=301)
    assert (layout.fields_per_chain, layout.interval) == (per_chain, interval)
    assert layout.fields_total == nb_chains * per_chain and layout.parameter_sets == nb_chains * nb_iter
    blocks = layout.blocks()
    assert sum(length for length, _ in blocks) == nb_iter
    assert sum(keep for _, keep in blocks) == per_chain
    assert (layout.field_cells, layout.field_times) == (7, 76)  # ⌈20 / 3⌉, ⌈301 / 4⌉
    assert layout.field_bytes == 2 * 4 * layout.fields_total * 7 * 76


def test_sampling_layout_is_what_the_sampler_stores(monkeypatch):
    """Le nombre de champs (T, q) réellement conservés est celui qu'affiche l'interface."""
    from molonaviz2.engine.jax import dream

    inputs = synthetic_inputs(nb_times=60)
    config = small_config(n_field_samples=15, time_step=2)
    stored = []
    original = np.quantile

    def spy(values, *args, **kwargs):
        if np.ndim(values) == 3:
            stored.append(np.shape(values))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(dream.np, "quantile", spy)
    result = run_inversion(inputs, config)
    layout = sampling_layout(config, nb_times=60)
    assert stored[0] == (layout.fields_total, layout.field_cells, layout.field_times)  # champs de température
    assert result.samples.shape[:2] == (layout.nb_iter, layout.nb_chains)


def test_invalid_configuration_is_rejected():
    inputs = synthetic_inputs(nb_times=40)
    with pytest.raises(UserError, match="2·δ"):
        run_inversion(inputs, small_config(nb_chains=5, delta=3))


def test_older_configurations_take_the_defaults_of_the_new_settings():
    stored = small_config().to_dict()
    for key in ("thin", "fields_per_chain", "store_fields"):
        del stored[key]
    config = InversionConfig.from_dict(stored)
    assert (config.thin, config.fields_per_chain, config.store_fields) == (1, None, "sensors")


@pytest.mark.parametrize("overrides, code", [
    ({"thin": 30}, None),  # thin = nb_iter : une itération gardée par chaîne
    ({"thin": 0}, "E_MCMC_THIN"),
    ({"thin": 31}, "E_MCMC_THIN"),
    ({"thin": 3, "fields_per_chain": 10}, None),  # 10 itérations gardées
    ({"thin": 3, "fields_per_chain": 11}, "E_MCMC_FIELDS_PER_CHAIN"),
    ({"fields_per_chain": 0}, "E_MCMC_FIELDS_PER_CHAIN"),
    ({"store_fields": "none"}, None),
    ({"store_fields": "full"}, None),
    ({"store_fields": "tout"}, "E_MCMC_STORE_FIELDS"),
])
def test_new_settings_are_checked_at_their_bounds(overrides, code):
    codes = [issue.code for issue in small_config(**overrides).validate()]
    assert codes == ([code] if code else [])


@pytest.mark.slow
def test_recovers_known_parameters_on_synthetic_column():
    """Données synthétiques bruitées (σ = 0,3 °C), paramètres connus.

    Critères statistiquement fondés (exiger que 4 intervalles à 90 % contiennent tous la
    vérité échouerait une fois sur trois même avec un échantillonneur parfait) :
    - les vrais paramètres ont une énergie typique du postérieur ;
    - le meilleur ajustement retrouve le niveau de bruit ;
    - la perméabilité, bien contrainte par les données, est retrouvée.
    """
    from molonaviz2.domain.compute import ForwardConfig
    from molonaviz2.engine.jax.dream import single_energy

    nb_times = 400
    true = LayerSpec("a", 0.4, 11.3, 0.15, 2.0, 3.5e6)
    base = synthetic_inputs(nb_times=nb_times, sigma_t=0.3)
    base = type(base)(**{**base.__dict__, "head": 0.08 + 0.02 * np.sin(np.arange(nb_times) / 40)})
    # Deux passes pour que le profil initial soit cohérent avec le modèle.
    first = run_forward(base, ForwardConfig([true], nb_cells=40))
    consistent = type(base)(**{**base.__dict__, "t_sensors": first.simulated_at_sensors()})
    clean = run_forward(consistent, ForwardConfig([true], nb_cells=40)).simulated_at_sensors()
    noisy = clean + np.random.default_rng(1).normal(0, 0.3, clean.shape)
    inputs = type(base)(**{**base.__dict__, "t_sensors": noisy})

    config = InversionConfig(
        layers=[LayerSpec("a", 0.4)], nb_cells=40, sigma_t=0.3, nb_chains=8, nb_iter=1500, burn_in_max=2000, n_field_samples=40, seed=3
    )
    result = run_inversion(inputs, config)
    truth = np.array([[11.3, 0.15, 2.0, 3.5e6]])
    assert single_energy(inputs, config, truth) <= np.quantile(result.energies, 0.99)
    assert result.best.rmse[-1] == pytest.approx(0.3, abs=0.03)
    assert abs(result.best_parameters[0, 0] - 11.3) < 0.2
    median_k = np.median(result.samples[:, :, 0, 0])
    assert abs(median_k - 11.3) < 0.2


def test_single_energy_matches_the_direct_model():
    """L'énergie d'un jeu de paramètres (espace des priors) est celle du modèle direct avec les mêmes paramètres."""
    from molonaviz2.engine.jax.dream import single_energy
    from molonaviz2.engine.jax.forward import build_geometry, energy, make_forward

    inputs = synthetic_inputs(nb_times=120)
    config = InversionConfig([LayerSpec("a", 0.4)], nb_cells=30, sigma_t=0.1)
    x = np.array([[11.5, 0.12, 2.5, 4e6]])
    geom = build_geometry(inputs, [0.4], 30)
    fields = make_forward(geom)(jnp.asarray([LayerSpec("a", 0.4, *x[0]).physical()]))
    expected = float(energy(fields["temperature"], geom, 0.01))
    assert single_energy(inputs, config, x) == pytest.approx(expected, rel=1e-12)
