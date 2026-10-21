"""Volume d'eau échangé à travers la colonne, en séparant infiltration et exfiltration."""

from __future__ import annotations

import numpy as np

from ..errors import UserError
from .answer import Answer, Quantity


def _seconds(times) -> np.ndarray:
    """Dates (datetime64) ou nombres, en secondes depuis la première valeur."""
    values = np.asarray(times)
    if np.issubdtype(values.dtype, np.datetime64):
        return (values - values[0]) / np.timedelta64(1, "s")
    return values.astype(float) - float(values[0])


def exchanged_volume(times, q, area: float | None = None) -> Answer:
    """Volumes échangés, par la méthode des trapèzes sur des pas de temps éventuellement irréguliers.

    q (m/s, positif vers le bas, de forme (..., T)) est intégré séparément quand il est positif
    (infiltration, ≥ 0) et négatif (exfiltration, ≤ 0) : un intervalle où q change de signe est
    coupé au point d'annulation, par interpolation linéaire. Volumes en m³/m², ou en m³ avec `area` (m²).
    """
    t = _seconds(times)
    q = np.asarray(q, dtype=float)
    if t.ndim != 1 or q.shape[-1:] != t.shape or len(t) < 2:
        raise UserError("E_QUESTION_SHAPES", expected=f"(..., {len(t)})", found=str(q.shape))
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise UserError("E_QUESTION_TIMES")
    q0, q1 = q[..., :-1], q[..., 1:]
    same_sign = q0 * q1 >= 0
    # Même signe : le trapèze entier va dans la partie de son signe.
    trapezoid = 0.5 * (q0 + q1) * dt
    # Changement de signe : q s'annule en t0 + q0/(q0 − q1)·dt ; deux triangles, un par partie.
    with np.errstate(divide="ignore", invalid="ignore"):
        before = np.where(same_sign, 0.0, 0.5 * q0 * q0 / (q0 - q1) * dt)  # du signe de q0
        after = np.where(same_sign, 0.0, 0.5 * q1 * q1 / (q1 - q0) * dt)  # du signe de q1
    whole = np.where(same_sign, trapezoid, 0.0)
    pieces = (whole, before, after)
    infiltration = sum(np.where(piece > 0, piece, 0.0) for piece in pieces).sum(axis=-1)
    exfiltration = sum(np.where(piece < 0, piece, 0.0) for piece in pieces).sum(axis=-1)
    scale, unit = (1.0, "m³/m²") if area is None else (float(area), "m³")
    infiltration, exfiltration = infiltration * scale, exfiltration * scale
    return Answer(
        data={
            "infiltration": Quantity(infiltration, unit),
            "exfiltration": Quantity(exfiltration, unit),
            "net": Quantity(infiltration + exfiltration, unit),
        },
        provenance={"question": "exchanged_volume", "parameters": {"area": None if area is None else float(area)}},
    )
