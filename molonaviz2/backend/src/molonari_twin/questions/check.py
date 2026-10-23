"""Vérification d'une configuration de twin (BN-11) : est-elle valide, quel sera son twin_id, que coûtera-t-elle ?

La question porte sur le dossier d'un twin, même invalide : une configuration invalide est une réponse (valid = false,
avec ses problèmes), pas une erreur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..twin.config import TwinConfigError, load_twin
from .answer import Answer, Quantity


def check_twin(folder: Path) -> Answer:
    """Problèmes (codes, champs, messages ; aussi en avertissements), et si la configuration est valide : `twin_id`
    prévu pour la version publiée du code, `twin_id_local` (celui du code chargé : il diffère pour un build local),
    `state` (new, computed, duplicate ou modified) et, en mode mcmc, le coût de l'échantillonnage."""
    from ..engine.api import release_version
    from ..engine.layout import sampling_layout
    from ..twin.run import twin_status

    try:
        config = load_twin(folder)
    except TwinConfigError as error:
        issues = list(error.issues)
        data = {"valid": Quantity(False, "1")}
        warnings = issues
    else:
        issues, warnings = [], []
        status = twin_status(config)
        data = {
            "valid": Quantity(True, "1"),
            "name": Quantity(config.name, "1"),
            "mode": Quantity(config.mode, "1"),
            "twin_id": Quantity(config.twin_id(release_version()), "1"),
            "twin_id_local": Quantity(status.twin_id, "1"),
            "state": Quantity(status.state, "1"),
            "duplicate_of": Quantity(status.source.name if status.state == "duplicate" else "", "1"),
        }
        if config.mcmc is not None:
            layout = sampling_layout(config.inversion_config(), len(config.load_measurements()))
            data.update({
                "samples": Quantity(layout.parameter_sets, "1"),
                "max_evaluations": Quantity(layout.max_evaluations, "1"),
                "stored_bytes": Quantity(layout.stored_bytes(config.mcmc.store_fields), "B"),
                "store_fields": Quantity(str(config.mcmc.store_fields), "1"),
            })
    data.update({
        "issue_codes": Quantity(np.array([i.code for i in issues], dtype=str), "1"),
        "issue_fields": Quantity(np.array([i.field or "" for i in issues], dtype=str), "1"),
        "issue_messages": Quantity(np.array([i.message for i in issues], dtype=str), "1"),
    })
    return Answer(data=data, provenance={"question": "check", "parameters": {}}, warnings=warnings)
