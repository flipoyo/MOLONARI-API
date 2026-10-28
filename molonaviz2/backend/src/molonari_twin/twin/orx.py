"""Couche `orx` d'un twin 2.0.0 : le fichier `orx.yaml`, à côté de `twin.yaml` (P0.7d, P2.2c).

La graine du MCMC n'est plus dans la recette (`twin.yaml`, couche `dev`) : elle est dans `orx.yaml`, couche `orx`,
que seul le niveau ORX monte (PLAN_TWIN_v3 §1ter). Ce n'est pas un secret vis-à-vis du dev, qui peut la retrouver par
force brute (accepté, T12) : retirer le fichier de sa couche oriente l'usage (le dev travaille sur le cache), il ne
protège rien. Sans ce fichier, le twin reste lisible ; il ne peut simplement pas être calculé (`run_twin`).

    schema: molonari-orx/1.0.0        # facultatif
    mcmc:
      seed: 42                        # obligatoire pour calculer un twin mcmc
    platform:                         # plateforme de référence de l'instance (facultative)
      platform: macOS-15.5-arm64-arm-64bit
      python: 3.12.4
      device: cpu

La graine entre dans le `twin_id` (c'est une empreinte, elle détermine le résultat) mais pas dans la recette que le
dev lit (HN-02, `metadata.json["config"]`). La plateforme de référence est une étiquette : elle n'entre pas dans le
`twin_id`. Seuls les twins 2.0.0 lisent ce fichier ; un twin 1.x garde sa graine dans `twin.yaml`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..errors import Issue
from .provenance import LIBRARIES

ORX_NAME = "orx.yaml"
ORX_SCHEMA = "molonari-orx/1.0.0"
ORX_KEYS = {"schema", "mcmc", "platform"}
ORX_MCMC_KEYS = {"seed"}
PLATFORM_KEYS = {"python", "platform", "processor", "device", *LIBRARIES}  # les clés de l'environnement de metadata.json


@dataclass(frozen=True)
class OrxLayer:
    path: Path  # chemin de l'orx.yaml
    seed: int | None  # None : le fichier existe mais ne donne pas de graine
    platform: dict[str, str] | None  # plateforme de référence, telle qu'elle est écrite


def orx_path(twin_folder: str | Path) -> Path:
    return Path(twin_folder) / ORX_NAME


def read_orx(twin_folder: str | Path) -> tuple[OrxLayer | None, list[Issue]]:
    """Lit l'`orx.yaml` du dossier d'un twin. (None, []) si le fichier n'existe pas : c'est le cas normal d'un dev, qui
    ne monte pas la couche `orx`. Les problèmes portent le chemin complet (`orx.mcmc.seed`)."""
    path = orx_path(twin_folder)
    if not path.is_file():
        return None, []
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        return None, [Issue("orx", "E_TWIN_INVALID_YAML", error=exc, prefixed=True)]
    issues: list[Issue] = []

    def mapping(value, where: str, allowed: set[str]) -> dict:
        if value is None:
            return {}
        if not isinstance(value, dict):
            issues.append(Issue(where, "E_TWIN_EXPECTED_MAPPING", prefixed=True))
            return {}
        for key in sorted(set(value) - allowed, key=str):
            issues.append(Issue(f"{where}.{key}", "E_TWIN_UNKNOWN_KEY", prefixed=True, keys=", ".join(sorted(allowed))))
        return value

    document = mapping(raw, "orx", ORX_KEYS)
    if "schema" in document and document["schema"] != ORX_SCHEMA:
        issues.append(Issue("orx.schema", "E_TWIN_SCHEMA", prefixed=True, value=document["schema"], expected=ORX_SCHEMA))

    seed = None
    if "mcmc" in document:
        section = mapping(document["mcmc"], "orx.mcmc", ORX_MCMC_KEYS)
        if "seed" in section:
            value = section["seed"]
            if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
                issues.append(Issue("orx.mcmc.seed", "E_TWIN_EXPECTED_INTEGER", prefixed=True))
            elif value < 0:
                issues.append(Issue("orx.mcmc.seed", "E_TWIN_MINIMUM", prefixed=True, v=0))
            else:
                seed = int(value)

    platform = None
    if "platform" in document:
        section = mapping(document["platform"], "orx.platform", PLATFORM_KEYS)
        platform = {}
        for key, value in section.items():
            if key in PLATFORM_KEYS:
                if isinstance(value, bool) or not isinstance(value, str | int | float):
                    issues.append(Issue(f"orx.platform.{key}", "E_TWIN_EXPECTED_TEXT", prefixed=True))
                else:
                    platform[key] = str(value)  # « python: 3.12 » est lu comme un nombre par YAML
    return OrxLayer(path=path, seed=seed, platform=platform), issues
