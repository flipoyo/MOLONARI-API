"""Adaptateur vers pyheatmy, moteur de référence utilisé pour valider le moteur JAX.

pyheatmy n'est pas embarqué dans l'application distribuée : ce module n'est
importé que par les tests (dépendance de développement).
"""

from __future__ import annotations

import tempfile

import numpy as np

from ..domain.compute import ForwardConfig
from .column import ZERO_CELSIUS, ColumnInputs
from .results import ForwardResult


def _column(inputs: ColumnInputs, nb_cells: int):
    from pyheatmy import Column

    times = [t.astype("datetime64[s]").item() for t in inputs.times]
    dh_measures = [[d, [h, tr + ZERO_CELSIUS]] for d, h, tr in zip(times, inputs.head, inputs.t_river)]
    t_measures = [
        [d, [*(ts + ZERO_CELSIUS), ta + ZERO_CELSIUS]]
        for d, ts, ta in zip(times, inputs.t_sensors, inputs.t_aquifer)
    ]
    return Column(
        river_bed=1.0,  # non utilisé par le modèle direct
        depth_sensors=list(inputs.sensor_depths),
        offset=inputs.offset,
        dH_measures=dh_measures,
        T_measures=t_measures,
        sigma_meas_P=inputs.sigma_pressure,
        sigma_meas_T=inputs.sigma_temperature,
        inter_mode="linear",
        nb_cells=nb_cells,
        rac=tempfile.gettempdir() + "/molonaviz2_pyheatmy",
    )


def forward(inputs: ColumnInputs, config: ForwardConfig) -> ForwardResult:
    from pyheatmy.layers import layersListCreator

    col = _column(inputs, config.nb_cells)
    col.set_layers(layersListCreator([(layer.name, layer.z_low, *layer.physical(), 0.0) for layer in config.layers]))  # q_s = 0 : pas d'échange latéral
    col.compute_solve_transi(verbose=False)
    temperature = np.asarray(col.get_temperatures_solve(), np.float64)
    return ForwardResult(
        depths=np.asarray(col.get_depths_solve(), np.float64),
        times=inputs.times,
        temperature=temperature - ZERO_CELSIUS,
        water_flux=np.asarray(col.get_flows_solve(), np.float64),
        advective_flux=np.asarray(col.get_vertical_advec_flows_solve(), np.float64),
        conductive_flux=np.asarray(col.get_conduc_flows_solve(), np.float64),
        sensor_index=np.asarray(col.get_id_sensors(), np.int64),
        observed=inputs.t_sensors,
        rmse=np.asarray(col.get_RMSE(), np.float64),
    )


def energy(temp_simul, temp_ref, sigma2) -> float:
    """Énergie de pyheatmy (`compute_energy`) avec une loi a priori constante sur σ²."""
    from pyheatmy.core import compute_energy

    return float(compute_energy(temp_simul, temp_ref, sigma2, lambda x: 1.0))
