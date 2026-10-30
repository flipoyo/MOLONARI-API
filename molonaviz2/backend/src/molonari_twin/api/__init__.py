"""Questions posées à un twin : le catalogue et TwinAPI, sans HTTP.

    catalog.py  les questions de l'API (identifiant, référence du plan, paramètres, fonction de calcul)
    service.py  TwinAPI : résout un twin, valide les paramètres, pose la question, renvoie une Answer
    publication.py  contrôle de visibilité et projection publique d'une réponse, pour le rapport (P2c.2)
    access.py   niveaux d'accès `user` et `admin` des questions (T11), une couche au-dessus de TwinAPI

Le serveur HTTP (server/) n'est qu'une couche au-dessus ; un notebook peut utiliser TwinAPI directement.
Rien ici ne charge JAX : les questions lisent des résultats, elles ne calculent pas de modèle.
"""

from .access import ADMIN, LEVELS, USER, USER_QUESTIONS, LeveledAPI
from .catalog import CATALOG, DEV, Question
from .publication import check_publishable, exposed_elements, publishable_answer
from .service import TwinAPI

__all__ = ["ADMIN", "CATALOG", "DEV", "LEVELS", "USER", "USER_QUESTIONS", "LeveledAPI", "Question", "TwinAPI", "check_publishable",
           "exposed_elements", "publishable_answer"]
