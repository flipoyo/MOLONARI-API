"""Bilan de chaleur à une profondeur (HN-20) : flux advectif, conductif et total, moyennés sur une période.

Les flux de chaleur viennent du format 2.0.0 des résultats, en W/m², positifs vers le bas. L'échange latéral
n'est pas pris en charge par cette version (LayerSpec.from_dict ignore q_s et t_lateral) : il vaut 0.
"""

from __future__ import annotations

import numpy as np

from ..errors import Issue, UserError
from ..twin.results import TwinResults
from .answer import Answer, Quantity
from .period import as_seconds, period_parameters, within
from .temperature import interpolate


def _at_depth(depths: np.ndarray, values: np.ndarray, z: float) -> np.ndarray:
    """Valeurs (Z, T) à la profondeur z. De z = 0 au centre de la première cellule : la première cellule, comme
    q au fond de la rivière (HN-10) ; au-delà, interpolation linéaire."""
    if 0.0 <= z < depths[0]:
        return values[0]
    return interpolate(depths, values, z, "z", f"{z:g} m", f"{depths[0]:g} m", f"{depths[-1]:g} m")


def heat_budget(results: TwinResults, z: float = 0.0, start=None, end=None) -> Answer:
    """Moyennes sur la période des flux de chaleur advectif, conductif et total (W/m²) à la profondeur z (m), leurs
    parts (en valeur absolue de la moyenne) et le processus dominant. Le flux latéral vaut 0 (non modélisé)."""
    if results.heat_flux_advective is None or results.heat_flux_conductive is None:
        raise UserError("E_QUESTION_NO_FIELDS")
    depths = np.asarray(results.depths, float)
    times = as_seconds(results.times)
    mask = within(times, start, end)
    advective = float(_at_depth(depths, np.asarray(results.heat_flux_advective, float), z)[mask].mean())
    conductive = float(_at_depth(depths, np.asarray(results.heat_flux_conductive, float), z)[mask].mean())
    magnitudes = np.abs([advective, conductive])
    total = magnitudes.sum()
    shares = magnitudes / total if total > 0 else np.zeros(2)
    return Answer(
        data={
            "advective": Quantity(advective, "W/m²"),
            "conductive": Quantity(conductive, "W/m²"),
            "lateral": Quantity(0.0, "W/m²"),
            "total": Quantity(advective + conductive, "W/m²"),
            "advective_share": Quantity(float(shares[0]), "1"),
            "conductive_share": Quantity(float(shares[1]), "1"),
            "lateral_share": Quantity(0.0, "1"),
            "dominant": Quantity("advection" if abs(advective) > abs(conductive) else "conduction", "1"),
            "depth": Quantity(float(z), "m"),
            "points": Quantity(int(mask.sum()), "1"),
            "start": Quantity(times[mask][0], "datetime64[s]"),
            "end": Quantity(times[mask][-1], "datetime64[s]"),
        },
        provenance={"question": "heat_budget", "parameters": {"z": float(z), **period_parameters(start, end)}},
        warnings=list(results.warnings) + [Issue(None, "W_LATERAL_NOT_MODELLED")],
    )

