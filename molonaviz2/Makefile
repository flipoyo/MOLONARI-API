.PHONY: run test test-slow test-all package selftest-package strings

run:
	uv run molonaviz2

test:
	uv run pytest

test-slow:
	uv run pytest -m slow

test-all: test test-slow

# Chaînes de l'interface sans traduction française.
strings:
	uv run python tools/extract_strings.py

package:
	uv sync --group package
	uv run pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/molonaviz2.spec

selftest-package:
	./dist/Molonaviz2/Molonaviz2 --selftest
