"""Inversion DREAM : exactitude de l'échantillonneur, déterminisme, annulation, convergence."""

import math

import arviz as az
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import synthetic_inputs
from molonari_twin.domain.compute import InversionConfig, LayerSpec
from molonari_twin.engine.api import run_forward, run_inversion
from molonari_twin.engine.jax.dream import (
    Cancelled,
    DreamSampler,
    sampling_layout,
    split_rhat,
)
from molonari_twin.errors import UserError


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


def blocks_of_0_3_0(nb_chains, nb_iter, n_field_samples):
    """Blocs de 0.3.0 (sans thin ni fields_per_chain), recopiés pour la comparaison."""
    per_chain = max(1, math.ceil(n_field_samples / nb_chains))
    interval = max(1, nb_iter // per_chain)
    full, rest = divmod(nb_iter, interval)
    return [(interval, True)] * full + ([(rest, False)] if rest else [])


@pytest.mark.parametrize("nb_chains, nb_iter, fields", [
    (7, 30, 14), (7, 30, 15), (7, 30, 3), (7, 5, 100), (10, 1000, 200), (8, 300, 200), (12, 997, 37), (3, 1, 2),
])
def test_default_sampling_layout_keeps_the_blocks_of_0_3_0(nb_chains, nb_iter, fields):
    layout = sampling_layout(small_config(nb_chains=nb_chains, nb_iter=nb_iter, n_field_samples=fields))
    assert layout.blocks() == blocks_of_0_3_0(nb_chains, nb_iter, fields)
    assert layout.kept == nb_iter and layout.parameter_sets == nb_chains * nb_iter


def test_thinning_keeps_one_iteration_out_of_thin():
    layout = sampling_layout(small_config(nb_iter=10, thin=3, n_field_samples=7))
    assert (layout.kept, layout.parameter_sets) == (3, 7 * 3)
    assert sum(length for length, _ in layout.blocks()) == 10
    # Chaque champ tombe sur une itération gardée (multiple de thin).
    ends = np.cumsum([length for length, _ in layout.blocks()])
    assert all(end % 3 == 0 for end, (_, keep) in zip(ends, layout.blocks(), strict=True) if keep)


@pytest.mark.parametrize("thin", [1, 2, 7])
def test_fields_per_chain_sets_the_number_of_fields_exactly(thin):
    layout = sampling_layout(small_config(nb_chains=8, nb_iter=300, thin=thin, fields_per_chain=5))
    assert (layout.fields_per_chain, layout.fields_total, layout.fields_requested) == (5, 40, 40)
    assert sum(keep for _, keep in layout.blocks()) == 5
    assert sum(length for length, _ in layout.blocks()) == 300


def test_stored_bytes_by_kind_of_stored_fields():
    layout = sampling_layout(small_config(space_step=3, time_step=4), nb_times=301)  # 14 champs, 7 cellules, 76 dates
    assert layout.stored_bytes("none") == 0
    assert layout.stored_bytes("sensors") == 8 * 14 * 3 * 76
    assert layout.stored_bytes("full") == layout.field_bytes == 2 * 4 * 14 * 7 * 76
    assert sampling_layout(small_config()).stored_bytes("sensors") is None


def test_sampling_layout_is_what_the_sampler_stores(monkeypatch):
    """Le nombre de champs (T, q) réellement conservés est celui qu'affiche l'interface."""
    from molonari_twin.engine.jax import dream

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
    assert result.samples.shape[:2] == (layout.kept, layout.nb_chains)


@pytest.mark.slow
def test_thinning_gives_exactly_a_subset_of_the_unthinned_result():
    inputs = synthetic_inputs(nb_times=60)
    full = run_inversion(inputs, small_config(thin=1))
    thinned = run_inversion(inputs, small_config(thin=3))
    np.testing.assert_array_equal(thinned.samples, full.samples[2::3])
    np.testing.assert_array_equal(thinned.energies, full.energies[2::3])
    assert thinned.thin == 3 and full.thin == 1


def test_iterations_not_multiple_of_thin():
    result = run_inversion(synthetic_inputs(nb_times=60), small_config(nb_iter=10, thin=3, n_field_samples=7))
    assert result.samples.shape[0] == result.energies.shape[0] == 3  # itérations 3, 6 et 9
    assert set(result.field_iteration) <= {3, 6, 9}


@pytest.mark.slow
def test_each_stored_field_is_the_direct_model_of_its_chain_and_iteration():
    from molonari_twin.engine.jax.dream import to_physical

    config = small_config(thin=2, n_field_samples=14, time_step=3)
    sampler = DreamSampler(synthetic_inputs(nb_times=60), config)
    result = sampler.run()
    layout = sampling_layout(config)
    assert result.sensor_temperature.dtype == np.float64
    assert result.sensor_temperature.shape == (layout.fields_total, 3, len(result.field_times))
    np.testing.assert_array_equal(result.field_chain, np.tile(np.arange(7), layout.fields_per_chain))
    for k, (chain, iteration) in enumerate(zip(result.field_chain, result.field_iteration, strict=True)):
        x = result.samples[iteration // result.thin - 1, chain]
        temperature = np.asarray(sampler._forward(to_physical(jnp.asarray(x)))["temperature"])
        expected = temperature[sampler.geom.sensor_index][:, ::3] - 273.15
        np.testing.assert_allclose(result.sensor_temperature[k], expected, rtol=1e-12, atol=0)


@pytest.mark.slow
def test_resets_are_counted_per_chain(monkeypatch):
    from molonari_twin.engine.jax import dream

    inputs = synthetic_inputs(nb_times=60)
    result = run_inversion(inputs, small_config())  # burn-in de 20 itérations : aucun contrôle des chaînes aberrantes
    assert result.resets.shape == (7,) and not result.resets.any()

    def reset_first_chain(carry, recent_energies, margin):
        return carry, np.arange(7) == 0

    monkeypatch.setattr(dream, "_reset_outlier_chains", reset_first_chain)
    result = run_inversion(inputs, small_config(burn_in_max=100, gr_threshold=1.0001))
    assert result.burn_in_iterations == 100
    np.testing.assert_array_equal(result.resets, [2, 0, 0, 0, 0, 0, 0])  # contrôles à 50 et 100


@pytest.mark.parametrize("store_fields", ["none", "sensors", "full"])
def test_stored_fields_by_setting(store_fields):
    config = small_config(store_fields=store_fields, space_step=2, time_step=3)
    result = run_inversion(synthetic_inputs(nb_times=60), config)
    layout = sampling_layout(config, nb_times=60)
    fields, cells, times = layout.fields_total, layout.field_cells, layout.field_times
    assert result.field_chain.shape == result.field_iteration.shape == (fields,)
    if store_fields == "sensors":
        assert result.sensor_temperature.shape == (fields, 3, times)
    else:
        assert result.sensor_temperature is None
    if store_fields == "full":
        assert result.field_temperature.shape == result.field_water_flux.shape == (fields, cells, times)
        assert result.field_temperature.dtype == result.field_water_flux.dtype == np.float32
        median = np.quantile(result.field_temperature, 0.5, axis=0)
        np.testing.assert_allclose(median, result.quantiles[1].temperature, atol=1e-3)
    else:
        assert result.field_temperature is None and result.field_water_flux is None


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
    from molonari_twin.domain.compute import ForwardConfig
    from molonari_twin.engine.jax.dream import single_energy

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
    from molonari_twin.engine.jax.dream import single_energy
    from molonari_twin.engine.jax.forward import build_geometry, energy, make_forward

    inputs = synthetic_inputs(nb_times=120)
    config = InversionConfig([LayerSpec("a", 0.4)], nb_cells=30, sigma_t=0.1)
    x = np.array([[11.5, 0.12, 2.5, 4e6]])
    geom = build_geometry(inputs, [0.4], 30)
    fields = make_forward(geom)(jnp.asarray([LayerSpec("a", 0.4, *x[0]).physical()]))
    expected = float(energy(fields["temperature"], geom, 0.01))
    assert single_energy(inputs, config, x) == pytest.approx(expected, rel=1e-12)
