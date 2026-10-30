"""Contrat de l'API /v1 : modèles des requêtes et des réponses, codec des réponses aux questions, version.

    models.py  modèles pydantic (tâches, erreurs, questions, réponses)
    codec.py   Answer ⇄ JSON et Answer ⇄ npz (tableaux binaires)

Le contrat ne contient aucune logique métier, et n'importe ni FastAPI, ni JAX, ni Qt : le client HTTP de la
phase 4 l'utilisera tel quel. Il reste avec le backend (PLAN_TWIN_v2, phase 6).
"""

API_VERSION = "1.0.0-alpha.8"
