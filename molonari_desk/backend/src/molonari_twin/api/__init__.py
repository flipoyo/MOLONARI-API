"""Questions posees a un twin : le catalogue et TwinAPI, sans HTTP.

    catalog.py  les questions de l'API (identifiant, reference du plan, parametres, fonction de calcul)
    service.py  TwinAPI : resout un twin, valide les parametres, pose la question, renvoie une Answer
    publication.py  controle de visibilite et projection publique d'une reponse, pour le rapport (P2c.2)
    access.py   niveaux d'acces `user`, `dev` et `opx` des questions, une couche au-dessus de TwinAPI

Le serveur HTTP (server/) n'est qu'une couche au-dessus ; un notebook peut utiliser TwinAPI directement.
Rien ici ne charge JAX : les questions lisent des resultats, elles ne calculent pas de modele.
"""

from .access import DEV, LEVELS, OPX, USER, USER_QUESTIONS, LeveledAPI
from .catalog import CATALOG, Question
from .publication import check_publishable, exposed_elements, publishable_answer
from .service import TwinAPI

__all__ = ["CATALOG", "DEV", "LEVELS", "OPX", "USER", "USER_QUESTIONS", "LeveledAPI", "Question", "TwinAPI", "check_publishable",
           "exposed_elements", "publishable_answer"]
