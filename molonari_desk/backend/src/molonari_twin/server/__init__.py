"""API HTTP du backend (extra `server` : FastAPI, uvicorn).

    app.py      create_app(runner, twins_folder) : endpoints /v1 ; les modèles sont ceux du contrat

L'API ne calcule rien : elle confie chaque calcul à un TaskRunner, dont elle ne connaît que le protocole.
Seule la ligne de commande (`molonari-twin serve`) importe ce paquet, au moment de l'appel.
"""
