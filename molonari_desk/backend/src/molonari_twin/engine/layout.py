"""Disposition de l'échantillonnage DREAM : combien d'itérations, de chaînes et de champs donnent des réglages.

Pur calcul sur des entiers, sans JAX : l'interface et l'API (BN-36) le lisent sans charger le moteur, et
DreamSampler (jax/dream.py) parcourt les mêmes blocs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain.compute import InversionConfig


@dataclass(frozen=True)
class SamplingLayout:
    """Ce que donnent les réglages d'échantillonnage, chaîne par chaîne.

    C'est la seule source de ces nombres : DreamSampler parcourt `blocks()`, et
    l'interface affiche les mêmes valeurs."""

    nb_chains: int
    burn_in_max: int  # itérations de burn-in au plus, par chaîne
    nb_iter: int  # itérations calculées après le burn-in, par chaîne
    fields_requested: int  # nombre total de champs (T, q) demandé
    interval: int  # un champ par chaîne toutes les `interval` itérations gardées
    fields_per_chain: int
    nb_cells: int
    field_cells: int  # cellules gardées après le pas spatial
    nb_times: int | None  # dates des mesures (None : inconnu)
    field_times: int | None  # dates gardées après le pas temporel
    thin: int = 1  # on garde les itérations multiples de thin

    @property
    def kept(self) -> int:
        """Itérations gardées par chaîne."""
        return self.nb_iter // self.thin

    @property
    def parameter_sets(self) -> int:
        return self.nb_chains * self.kept

    @property
    def fields_total(self) -> int:
        return self.nb_chains * self.fields_per_chain

    @property
    def max_evaluations(self) -> int:
        return self.nb_chains * (self.burn_in_max + self.nb_iter)

    @property
    def field_bytes(self) -> int | None:
        """Mémoire des champs conservés : T et q en float32."""
        return None if self.field_times is None else 2 * 4 * self.fields_total * self.field_cells * self.field_times

    def stored_bytes(self, store_fields: str) -> int | None:
        """Mémoire des champs stockés dans le résultat : q(0, t) en float64 pour chaque champ (toujours), plus T aux
        capteurs en float64, ou T et q complets en float32."""
        if self.field_times is None:
            return None
        riverbed = 8 * self.fields_total * self.field_times
        if store_fields == "sensors":
            return riverbed + 8 * self.fields_total * 3 * self.field_times
        if store_fields == "full":
            return riverbed + 2 * 4 * self.fields_total * self.field_cells * self.field_times
        return riverbed

    def blocks(self) -> list[tuple[int, bool]]:
        """Blocs d'itérations calculées : (longueur, champ conservé à la fin du bloc).

        Le champ m (1…F) est pris à l'itération calculée m·interval·thin, qui est gardée."""
        step = self.interval * self.thin
        rest = self.nb_iter - self.fields_per_chain * step
        return [(step, True)] * self.fields_per_chain + ([(rest, False)] if rest else [])


def sampling_layout(config: InversionConfig, nb_times: int | None = None) -> SamplingLayout:
    kept = config.nb_iter // config.thin
    if config.fields_per_chain is None:
        per_chain = max(1, math.ceil(config.n_field_samples / config.nb_chains))
        interval = max(1, kept // per_chain)
        fields_per_chain = kept // interval
        requested = config.n_field_samples
    else:
        interval = max(1, kept // config.fields_per_chain)
        fields_per_chain = min(config.fields_per_chain, kept // interval)
        requested = config.nb_chains * config.fields_per_chain
    return SamplingLayout(
        nb_chains=config.nb_chains,
        burn_in_max=config.burn_in_max,
        nb_iter=config.nb_iter,
        fields_requested=requested,
        interval=interval,
        fields_per_chain=fields_per_chain,
        nb_cells=config.nb_cells,
        field_cells=math.ceil(config.nb_cells / config.space_step),
        nb_times=nb_times,
        field_times=None if nb_times is None else math.ceil(nb_times / config.time_step),
        thin=config.thin,
    )
