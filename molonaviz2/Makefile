.PHONY: run twin test test-backend test-frontend test-slow test-all package selftest-package strings

# Application de bureau (frontend) et ligne de commande des twins (backend).
run:
	uv run molonaviz2

twin:
	uv run molonari-twin --help

# Deux suites séparées ; les tests longs (slow) portent sur le calcul, dans le backend.
test: test-backend test-frontend

test-backend:
	cd backend && uv run pytest

test-frontend:
	cd frontend && uv run pytest

test-slow:
	cd backend && uv run pytest -m slow

test-all: test test-slow

# Chaînes de l'interface et du backend sans traduction française (catalogue commun).
strings:
	uv run python tools/extract_strings.py

package:
	uv sync
	uv run pyinstaller --noconfirm --clean --distpath dist --workpath build frontend/packaging/molonaviz2.spec

selftest-package:
	./dist/Molonaviz2/Molonaviz2 --selftest
