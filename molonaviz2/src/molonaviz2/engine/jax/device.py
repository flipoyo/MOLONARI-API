"""Configuration de JAX : float64 obligatoire, choix et description de l'appareil.

Le modèle exige le float64. JAX ne le propose pas sur Metal (Mac) et n'a pas de
GPU natif sous Windows : le GPU n'est utilisé qu'avec CUDA (Linux). Sinon le
calcul tourne sur CPU, compilé par XLA.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)


def device_description() -> str:
    device = jax.devices()[0]
    if device.platform == "gpu":
        return f"GPU CUDA · {device.device_kind} · float64"
    return "CPU · float64"
