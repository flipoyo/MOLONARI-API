"""Exceptions partagées par toutes les couches (domaine, stockage, moteur, interface)."""


class UserError(Exception):
    """Erreur attendue, causée par une donnée ou une action de l'utilisateur.

    Le message est destiné à l'utilisateur : il est affiché tel quel, sans trace technique.
    """
