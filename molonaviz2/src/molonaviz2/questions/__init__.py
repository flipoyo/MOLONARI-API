"""Questions : réponses calculées à partir d'un résultat (volumes, intervalles, bande prédictive, chaînes).

Partie du cœur : ni Qt, ni HTTP, ni phrase traduite. Chaque question renvoie une `Answer`.
"""

from .answer import Answer, Quantity

__all__ = ["Answer", "Quantity"]
