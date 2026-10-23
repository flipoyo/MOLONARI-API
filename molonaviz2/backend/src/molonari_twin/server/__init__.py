"""API HTTP du backend (extra `server` : FastAPI, uvicorn).

    schemas.py  modèles pydantic des réponses (ils passeront dans contract/ en fin de phase 3)
    app.py      create_app(runner, twins_folder) : endpoints /v1

L'API ne calcule rien : elle confie chaque calcul à un TaskRunner, dont elle ne connaît que le protocole.
Seule la ligne de commande (`molonari-twin serve`) importe ce paquet, au moment de l'appel.
"""
