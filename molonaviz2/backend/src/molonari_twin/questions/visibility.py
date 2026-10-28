"""Visibilité des éléments d'un twin : ce que le développeur accepte de rendre public dans le rapport (P2c.2).

La visibilité porte sur sept éléments (SPEC_RAPPORT_PUBLIC, section 5). Une question déclare ce que sa réponse
**expose** (`Question.exposes`, dans api/catalog.py), et non ce dont elle dépend : le contrôle (api/publication.py)
refuse d'écrire dans le rapport une réponse dont un élément exposé est privé.

`Visibility` est une entrée des fonctions de contrôle : elle ne lit aucun fichier. La clé `visibility` de
twin.yaml (schéma 2.0.0, P2.3a) la fournit : `Visibility.from_config(config)`. Par défaut tout est privé, sauf
la PDF a posteriori (`posterior`) et les réglages que ses échantillons révèlent de toute façon
(`inversion_settings_public`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ..twin.config import VISIBILITY_ELEMENTS, TwinConfig

ELEMENTS = VISIBILITY_ELEMENTS  # les noms, les défauts et le contrôle de la clé `visibility` sont ceux de twin/config.py (P2.3a)

# Éléments publics de fait : la PDF publique les révèle, la visibilité par défaut les déclare publics.
PUBLIC_DE_FAIT = frozenset({"inversion_settings_public"})

# Mode du twin où un élément existe : les autres éléments existent dans les deux. Un élément d'un autre mode n'est pas
# exposé (la PDF d'un twin direct n'existe pas ; les paramètres du sol d'un twin mcmc sont inversés, pas écrits).
ELEMENT_MODES = {
    "inversion_settings": "M",
    "inversion_settings_public": "M",
    "posterior": "M",
    "forward_parameters": "D",
}


def applicable(elements, mode: str) -> list[str]:
    """Parmi `elements`, ceux qui existent dans un twin de ce mode ('D' ou 'M')."""
    return [e for e in elements if ELEMENT_MODES.get(e, mode) == mode]


@dataclass(frozen=True)
class Visibility:
    """Quels éléments sont publics. Défaut : tout privé, sauf la PDF et les réglages qu'elle révèle."""

    inversion_settings: bool = False
    inversion_settings_public: bool = True
    forward_parameters: bool = False
    posterior: bool = True
    states: bool = False
    raw_data: bool = False
    processed_data: bool = False

    @classmethod
    def from_config(cls, config: TwinConfig) -> Visibility:
        """La déclaration de la clé `visibility` du twin (défauts compris)."""
        return cls(**config.visibility)

    def is_public(self, element: str) -> bool:
        if element not in ELEMENTS:
            raise KeyError(element)
        return bool(getattr(self, element))

    def private_elements(self, elements, mode: str) -> list[str]:
        """Parmi `elements`, ceux qui existent dans un twin de ce mode ('D' ou 'M') et qui sont privés."""
        return [e for e in applicable(elements, mode) if not self.is_public(e)]

    def to_dict(self) -> dict[str, bool]:
        """La déclaration appliquée, telle que `report.json` la recopie."""
        return asdict(self)
