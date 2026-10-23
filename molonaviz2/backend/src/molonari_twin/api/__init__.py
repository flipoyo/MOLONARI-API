"""Questions posées à un twin : le catalogue et TwinAPI, sans HTTP.

    catalog.py  les questions de l'API (identifiant, référence du plan, paramètres, fonction de calcul)
    service.py  TwinAPI : résout un twin, valide les paramètres, pose la question, renvoie une Answer

Le serveur HTTP (server/) n'est qu'une couche au-dessus ; un notebook peut utiliser TwinAPI directement.
Rien ici ne charge JAX : les questions lisent des résultats, elles ne calculent pas de modèle.
"""

from .catalog import CATALOG, Question
from .service import TwinAPI

__all__ = ["CATALOG", "Question", "TwinAPI"]
