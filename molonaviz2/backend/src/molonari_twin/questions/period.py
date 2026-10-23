"""Choix d'une période dans une série datée."""

from __future__ import annotations

import numpy as np

from ..errors import UserError


def as_seconds(times) -> np.ndarray:
    return np.asarray(times).astype("datetime64[s]")


def within(times, start=None, end=None) -> np.ndarray:
    """Masque des dates comprises entre `start` et `end`, bornes incluses (None : sans borne).

    E_QUESTION_PERIOD_EMPTY si aucune date n'y tombe."""
    times = as_seconds(times)
    mask = np.ones(times.shape, dtype=bool)
    if start is not None:
        mask &= times >= np.datetime64(start, "s")
    if end is not None:
        mask &= times <= np.datetime64(end, "s")
    if not mask.any():
        raise UserError("E_QUESTION_PERIOD_EMPTY", start=_text(start), end=_text(end))
    return mask


def _text(value) -> str:
    return "…" if value is None else str(np.datetime64(value, "s"))


def period_parameters(start=None, end=None) -> dict:
    return {"start": None if start is None else _text(start), "end": None if end is None else _text(end)}
