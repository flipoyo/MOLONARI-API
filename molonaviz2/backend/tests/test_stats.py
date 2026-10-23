"""Statistiques MCMC en numpy (engine/stats.py) : égales à ArviZ, qui sert de référence, et sans JAX ni ArviZ."""

import subprocess
import sys

import arviz as az
import numpy as np
import pytest
import xarray as xr
from molonari_twin.engine import stats

TOLERANCE = 1e-6


def synthetic(kind, chains=4, draws=400, seed=0):
    """Chaînes (N, n, 2) : indépendantes, autocorrélées (AR(1)), ou qui n'ont pas convergé (moyennes décalées)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(chains, draws, 2))
    if kind == "iid":
        return noise
    if kind == "ar1":
        out = np.zeros_like(noise)
        out[:, 0] = noise[:, 0]
        for i in range(1, draws):
            out[:, i] = 0.9 * out[:, i - 1] + noise[:, i]
        return out
    if kind == "drifting":
        return noise + np.linspace(0, 3, draws)[None, :, None] + np.arange(chains)[:, None, None]
    raise ValueError(kind)


def reference(chains):
    return xr.DataArray(chains, dims=["chain", "draw", "p"])


KINDS = ["iid", "ar1", "drifting"]


@pytest.mark.parametrize("kind", KINDS)
def test_rank_normalized_rhat_equals_arviz(kind):
    chains = synthetic(kind)
    np.testing.assert_allclose(stats.rank_normalized_rhat(chains), az.rhat(reference(chains)).values, rtol=TOLERANCE)


@pytest.mark.parametrize("kind", KINDS)
def test_ess_bulk_and_tail_equal_arviz(kind):
    chains = synthetic(kind)
    np.testing.assert_allclose(stats.ess_bulk(chains), az.ess(reference(chains), method="bulk").values, rtol=TOLERANCE)
    # Quantiles 5 % et 95 % de Vehtari et al. : ArviZ prend sinon sa probabilité par défaut (0,94).
    tail = az.ess(reference(chains), method="tail", prob=stats.TAIL_PROBABILITIES).values
    np.testing.assert_allclose(stats.ess_tail(chains), tail, rtol=TOLERANCE)


@pytest.mark.parametrize("kind", KINDS)
def test_mcse_equals_arviz(kind):
    chains = synthetic(kind)
    np.testing.assert_allclose(stats.mcse_mean(chains), az.mcse(reference(chains)).values, rtol=TOLERANCE)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("prob", [0.5, 0.9, 0.95])
def test_hdi_equals_arviz(kind, prob):
    chains = synthetic(kind)
    low, high = stats.hdi(chains.reshape(-1, 2), prob)
    expected = az.hdi(reference(chains), prob=prob).values
    np.testing.assert_allclose(low, expected[:, 0], rtol=TOLERANCE)
    np.testing.assert_allclose(high, expected[:, 1], rtol=TOLERANCE)


def test_split_rhat_is_the_classic_split_rhat():
    chains = synthetic("iid", chains=6, draws=300)
    split = stats.split_rhat(chains)
    assert split.shape == (2,)
    np.testing.assert_allclose(split, az.rhat(reference(chains), method="split").values, rtol=TOLERANCE)


def test_split_rhat_of_a_constant_chain_is_infinite_and_of_a_short_one_is_nan():
    assert np.isinf(stats.split_rhat(np.ones((4, 20, 1)))).all()
    assert np.isnan(stats.split_rhat(np.zeros((4, 3, 1)))).all()


def test_convergence_separates_good_and_bad_chains():
    assert (stats.rank_normalized_rhat(synthetic("iid")) < 1.02).all()
    assert (stats.rank_normalized_rhat(synthetic("drifting")) > 1.2).all()
    assert (stats.ess_bulk(synthetic("ar1")) < 0.2 * 4 * 400).all()


def test_too_short_or_single_chain_statistics_are_nan():
    assert np.isnan(stats.rank_normalized_rhat(synthetic("iid", draws=3))).all()
    assert np.isnan(stats.rank_normalized_rhat(synthetic("iid", chains=1))).all()
    assert np.isnan(stats.ess_bulk(synthetic("iid", draws=3))).all()


def test_scalar_parameters_give_scalars_and_trailing_axes_are_kept():
    chains = synthetic("iid")[..., 0]
    assert stats.ess_bulk(chains).shape == ()
    layered = np.stack([synthetic("iid", seed=s) for s in range(3)], axis=-1)  # (N, n, 2, 3)
    assert stats.mcse_mean(layered).shape == (2, 3)


def test_hdi_of_a_normal_sample_is_plus_or_minus_1_645_sigma():
    sample = np.random.default_rng(1).normal(2.0, 3.0, 200_000)
    low, high = stats.hdi(sample, 0.9)
    assert low == pytest.approx(2.0 - 1.645 * 3.0, abs=0.05)
    assert high == pytest.approx(2.0 + 1.645 * 3.0, abs=0.05)


def test_hdi_is_the_narrowest_interval_of_a_skewed_sample():
    sample = np.random.default_rng(2).exponential(1.0, 100_000)
    low, high = stats.hdi(sample, 0.9)
    assert low == pytest.approx(0.0, abs=1e-3)  # il commence au mode, pas au quantile 5 %
    assert high < np.quantile(sample, 0.95)


def test_hdi_refuses_a_probability_outside_the_open_unit_interval():
    with pytest.raises(ValueError):
        stats.hdi(np.arange(10.0), 1.0)


def test_stats_import_neither_jax_nor_arviz():
    code = "import sys; import molonari_twin.engine.stats; print(sorted(m for m in ('jax', 'arviz') if m in sys.modules))"
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"
