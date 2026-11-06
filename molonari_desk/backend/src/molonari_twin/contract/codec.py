"""Réponses aux questions : Answer ⇄ JSON (AnswerOut) et Answer ⇄ npz.

JSON (par défaut) : un tableau numpy devient {dtype, shape, data} (ArrayOut), à plat dans l'ordre C.
- NaN et NaT → null ; ±inf → "inf" / "-inf" ;
- un flottant scalaire non fini est écrit comme un tableau de forme [] : null ne veut jamais dire « nombre » ;
- dates : ISO 8601 à la seconde, sans fuseau (convention de dates fixée plus tard, avec le pipeline brut).

npz (`Accept: application/x-npz`) : chaque tableau d'au moins une dimension est rangé dans l'archive, sous
la clé `data/<nom>/value`, `data/<nom>/low` ou `data/<nom>/high`, et remplacé dans la réponse par {"npz": clé} ;
la réponse elle-même est l'entrée `meta` (JSON). L'archive se relit sans pickle.

Le décodage redonne les mêmes valeurs : mêmes types, mêmes formes, NaN compris.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np

from ..errors import Issue
from ..questions.answer import Answer, Quantity
from ..tasks.model import jsonable
from .models import AnswerOut, ArrayOut, NpzRef, QuantityOut, WarningOut

NPZ_MEDIA_TYPE = "application/x-npz"


# Encodage ----------------------------------------------------------------------------------------


def _array_out(values) -> ArrayOut:
    array = np.asarray(values)
    kind = array.dtype.kind
    flat = array.ravel()
    if kind == "M":
        seconds = flat.astype("datetime64[s]")
        text = np.datetime_as_string(seconds, unit="s")
        data = [None if np.isnat(t) else str(s) for t, s in zip(seconds, text, strict=True)]
        return ArrayOut(dtype="datetime64[s]", shape=list(array.shape), data=data)
    if kind == "f":
        return ArrayOut(dtype=str(array.dtype), shape=list(array.shape), data=[_float(x) for x in flat.tolist()])
    if kind in "iub":
        return ArrayOut(dtype=str(array.dtype), shape=list(array.shape), data=flat.tolist())
    if kind == "U":
        return ArrayOut(dtype="str", shape=list(array.shape), data=[str(x) for x in flat])
    raise TypeError(f"unsupported dtype {array.dtype}")


def _float(x: float):
    if math.isnan(x):
        return None
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return x


def _value_out(value, arrays: dict | None, key: str):
    if isinstance(value, np.ndarray) and value.ndim >= 1 and arrays is not None:
        arrays[key] = value
        return NpzRef(npz=key)
    if isinstance(value, np.ndarray | np.datetime64):
        if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind != "M":
            return _value_out(value.item(), arrays, key)
        return _array_out(value)
    if isinstance(value, np.generic):
        return _value_out(value.item(), arrays, key)
    if isinstance(value, float) and not math.isfinite(value):
        return _array_out(np.float64(value))
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"unsupported value of type {type(value).__name__}")


def _warning_out(issue: Issue) -> WarningOut:
    return WarningOut(code=issue.code, field=issue.field, params=jsonable(issue.params), message_en=issue.message)


def answer_out(answer: Answer, arrays: dict | None = None) -> AnswerOut:
    """Answer → AnswerOut. Avec `arrays` (dictionnaire rempli ici), les tableaux vont dans l'archive npz."""
    data = {}
    for name, quantity in answer.data.items():
        interval = None
        if quantity.interval is not None:
            low, high = quantity.interval
            interval = (_value_out(low, arrays, f"data/{name}/low"), _value_out(high, arrays, f"data/{name}/high"))
        data[name] = QuantityOut(value=_value_out(quantity.value, arrays, f"data/{name}/value"), unit=quantity.unit,
                                 interval=interval)
    return AnswerOut(data=data, provenance=jsonable(answer.provenance), warnings=[_warning_out(w) for w in answer.warnings])


def answer_to_json(answer: Answer) -> dict:
    return answer_out(answer).model_dump(mode="json")


def answer_to_npz(answer: Answer) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    meta = answer_out(answer, arrays).model_dump_json()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, meta=np.array(meta), **arrays)
    return buffer.getvalue()


# Décodage ----------------------------------------------------------------------------------------


def _array_in(array: ArrayOut) -> np.ndarray:
    shape = tuple(array.shape)
    if array.dtype == "datetime64[s]":
        return np.array(["NaT" if x is None else x for x in array.data], dtype="datetime64[s]").reshape(shape)
    if array.dtype == "str":
        return np.array(array.data, dtype=str).reshape(shape)
    if np.dtype(array.dtype).kind == "f":
        values = np.array([math.nan if x is None else float(x) for x in array.data], dtype=np.float64)
        return values.astype(array.dtype).reshape(shape)
    return np.array(array.data, dtype=array.dtype).reshape(shape)


def _value_in(value, arrays):
    if isinstance(value, NpzRef):
        return arrays[value.npz]
    if isinstance(value, ArrayOut):
        decoded = _array_in(value)
        if decoded.ndim == 0 and decoded.dtype.kind == "f":
            return float(decoded)  # flottant scalaire non fini
        return decoded[()] if decoded.ndim == 0 else decoded
    return value


def answer_from_out(out: AnswerOut, arrays=None) -> Answer:
    data = {}
    for name, quantity in out.data.items():
        interval = None
        if quantity.interval is not None:
            interval = tuple(_value_in(bound, arrays) for bound in quantity.interval)
        data[name] = Quantity(_value_in(quantity.value, arrays), quantity.unit, interval)
    warnings = [Issue(w.field, w.code, **w.params) for w in out.warnings]
    return Answer(data=data, provenance=out.provenance, warnings=warnings)


def answer_from_json(document: dict | str | bytes) -> Answer:
    if isinstance(document, str | bytes):
        return answer_from_out(AnswerOut.model_validate_json(document))
    return answer_from_out(AnswerOut.model_validate(document))


def answer_from_npz(content: bytes) -> Answer:
    with np.load(io.BytesIO(content), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files if key != "meta"}
        meta = json.loads(str(archive["meta"]))
    return answer_from_out(AnswerOut.model_validate(meta), arrays)
