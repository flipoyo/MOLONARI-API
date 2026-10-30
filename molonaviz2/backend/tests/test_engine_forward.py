"""Modèle direct JAX : non-régression contre pyheatmy (référence)."""

import numpy as np
import pytest
from conftest import synthetic_inputs
from molonari_twin.domain.compute import ForwardConfig, LayerSpec
from molonari_twin.engine import api, reference
from molonari_twin.errors import UserError

FIELDS = ["temperature", "water_flux", "advective_flux", "conductive_flux", "rmse"]

CASES = {
    "one_layer": [LayerSpec("a", 0.4, 11.5, 0.12, 2.5, 4e6)],
    "two_layers_interface_between_cells": [LayerSpec("a", 0.2, 11.5, 0.12, 2.5, 4e6), LayerSpec("b", 0.4, 12.5, 0.2, 1.8, 3e6)],
    "two_layers_interface_off_grid": [LayerSpec("a", 0.215, 11.5, 0.12, 2.5, 4e6), LayerSpec("b", 0.4, 12.5, 0.2, 1.8, 3e6)],
    "three_layers": [
        LayerSpec("a", 0.1, 11.0, 0.1, 2.0, 4e6),
        LayerSpec("b", 0.25, 12.0, 0.2, 3.0, 2.5e6),
        LayerSpec("c", 0.4, 13.0, 0.3, 1.5, 3e6),
    ],
}


def max_relative(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30)


@pytest.mark.parametrize("case", CASES)
def test_matches_pyheatmy_exactly_in_float64(case, inputs):
    config = ForwardConfig(CASES[case], nb_cells=100)
    ours = api.run_forward(inputs, config)
    ref = reference.forward(inputs, config)
    np.testing.assert_array_equal(ours.sensor_index, ref.sensor_index)
    np.testing.assert_allclose(ours.depths, ref.depths, rtol=0, atol=1e-12)
    for name in FIELDS:
        assert max_relative(getattr(ours, name), getattr(ref, name)) < 1e-9, name


def test_variable_time_step_and_offset():
    inputs = synthetic_inputs(nb_times=200, offset=0.05)
    jitter = np.zeros(200, "timedelta64[s]")
    jitter[50:] += np.timedelta64(120, "s")  # un pas de temps irrégulier
    inputs = type(inputs)(**{**inputs.__dict__, "times": inputs.times + jitter})
    layers = [LayerSpec("a", 0.25, 11.5, 0.12, 2.5, 4e6), LayerSpec("b", 0.45, 12.5, 0.2, 1.8, 3e6)]
    config = ForwardConfig(layers, nb_cells=60)
    ours, ref = api.run_forward(inputs, config), reference.forward(inputs, config)
    for name in FIELDS:
        assert max_relative(getattr(ours, name), getattr(ref, name)) < 1e-9, name


def test_energy_matches_pyheatmy(inputs):
    from molonari_twin.engine.jax.forward import build_geometry, energy, make_forward

    layers = CASES["one_layer"]
    geom = build_geometry(inputs, [0.4], 100)
    fields = make_forward(geom)(np.array([layer.physical() for layer in layers]))
    temp = np.asarray(fields["temperature"])
    for sigma2 in (1.0, 0.01):
        ours = float(energy(temp, geom, sigma2))
        ref = reference.energy(temp[geom.sensor_index], geom.observed, sigma2)
        assert ours == pytest.approx(ref, rel=1e-12)


def test_result_shapes_and_units(inputs):
    result = api.run_forward(inputs, ForwardConfig(CASES["one_layer"], nb_cells=50))
    assert result.temperature.shape == (50, inputs.nb_times)
    assert result.water_flux.shape == result.temperature.shape
    # Températures en °C, dans la plage des conditions aux limites.
    assert 12.0 < result.temperature.min() and result.temperature.max() < 18.5
    assert result.rmse.shape == (4,)
    np.testing.assert_allclose(result.simulated_at_sensors(), result.temperature[result.sensor_index].T)


def test_invalid_layers_are_rejected_before_computing(inputs):
    with pytest.raises(UserError):
        api.run_forward(inputs, ForwardConfig([LayerSpec("a", 0.3)], nb_cells=50))  # ne descend pas jusqu'au fond
    with pytest.raises(UserError):
        api.run_forward(inputs, ForwardConfig([LayerSpec("a", 0.4, porosity=1.5)], nb_cells=50))


def test_real_point034_from_files_to_model():
    """Chaîne complète sur des mesures réelles : import HOBO, nettoyage, modèle direct JAX = pyheatmy."""
    from conftest import FIXTURES
    from molonari_twin.engine.column import ColumnInputs
    from molonari_twin.io.lab_import import import_lab_directory
    from molonari_twin.io.point_import import read_pressure_file, read_temperature_file
    from molonari_twin.processing.cleanup import CleanupSettings, clean

    lab = import_lab_directory(FIXTURES / "lab").lab
    sensor = next(p for p in lab.pressure_sensors if p.name == "P508")
    shaft = next(s for s in lab.shafts if s.name == "S007")
    raw_t = read_temperature_file(FIXTURES / "point034" / "point034_T_measures.csv").frame
    raw_p = read_pressure_file(FIXTURES / "point034" / "point034_P_measures.csv").frame
    cleaned = clean(raw_t, raw_p, sensor, CleanupSettings()).cleaned.iloc[:400]
    inputs = ColumnInputs.from_cleaned(cleaned, shaft.depths, 0.0, sensor.sigma, 0.03)
    layers = [LayerSpec("l1", 0.3, 11.0, 0.15, 3.4, 5e6), LayerSpec("l2", inputs.column_depth, 12.0, 0.2, 2.0, 4e6)]
    config = ForwardConfig(layers, nb_cells=100)
    ours, ref = api.run_forward(inputs, config), reference.forward(inputs, config)
    for name in FIELDS:
        assert max_relative(getattr(ours, name), getattr(ref, name)) < 1e-9, name


def test_conductive_flux_down_of_a_linear_gradient():
    """T croît vers le bas (dT/dz = 5 K/m), q = 0 : la chaleur remonte, le flux vers le bas vaut −λm·dT/dz."""
    from molonari_twin.engine.jax.forward import conductive_flux
    from molonari_twin.engine.results import ForwardResult

    dz, cells, times = 0.01, 10, 3
    z = (np.arange(cells) + 0.5) * dz
    temperature = np.repeat((288.0 + 5.0 * z)[:, None], times, axis=1)
    lambda_m = np.full(cells, 2.0)
    upward = np.asarray(conductive_flux(temperature, lambda_m, dz))
    zeros = np.zeros((cells, times))
    result = ForwardResult(z, np.arange(times), temperature - 273.15, zeros, zeros, upward, np.array([1, 2, 3]), np.zeros((times, 3)), np.zeros(4))
    np.testing.assert_allclose(result.conductive_flux_down[1:-1], -2.0 * 5.0, rtol=1e-12)
    assert np.all(result.conductive_flux_down < 0)


def test_advective_flux_is_relative_to_the_reference_temperature():
    """q constant et T uniforme : flux advectif = ρw·cw·q·(T − T_ref), T_ref = 0 °C."""
    import jax.numpy as jnp
    from molonari_twin.engine.jax.forward import (
        C_W,
        HEAT_REFERENCE_C,
        RHO_W,
        build_geometry,
        to_forward_result,
    )

    inputs = synthetic_inputs(nb_times=5)
    geom = build_geometry(inputs, [0.4], 10)
    fields = {"temperature": jnp.full((10, 5), 273.15 + 12.0), "water_flux": jnp.full((10, 5), 2e-6), "lambda_m": jnp.full(10, 2.0)}
    result = to_forward_result(fields, geom, inputs)
    assert HEAT_REFERENCE_C == 0.0
    np.testing.assert_allclose(result.advective_flux, RHO_W * C_W * 2e-6 * (12.0 - HEAT_REFERENCE_C), rtol=1e-12)
    np.testing.assert_allclose(result.conductive_flux_down, 0.0, atol=1e-9)


def test_total_heat_flux_adds_both_components_downwards(inputs):
    result = api.run_forward(inputs, ForwardConfig(CASES["one_layer"], nb_cells=30))
    np.testing.assert_array_equal(result.total_heat_flux, result.advective_flux - result.conductive_flux)
    np.testing.assert_array_equal(result.total_heat_flux, result.advective_flux + result.conductive_flux_down)
