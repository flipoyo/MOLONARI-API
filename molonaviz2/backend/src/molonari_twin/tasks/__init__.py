"""Tâches de calcul : l'API crée une tâche, un worker séparé l'exécute (PLAN_TWIN_v2, section 1bis).

    model.py    TaskRecord, TaskEvent, états et erreurs sérialisables
    runner.py   protocole TaskRunner et clé d'idempotence task_key
    store.py    magasin SQLite partagé par le runner et ses workers
    jobs.py     types de tâches : twin.run, diag.sleep
    worker.py   exécution d'une tâche dans un sous-processus
    local.py    LocalTaskRunner (phases 3–4) ; CeleryTaskRunner viendra en phase 5

Ce paquet n'importe ni FastAPI ni JAX au niveau des modules : JAX n'est chargé que dans le worker.
"""
