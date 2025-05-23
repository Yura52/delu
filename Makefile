.PHONY: default clean coverage _docs docs doctest spelling format lint pages pre-commit test typecheck

PYTEST_CMD = pytest delu
VIEW_HTML_CMD = open
DOCS_DIR = docs

default:
	echo "Hello, World!"

clean:
	find delu -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -f .coverage
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf dist
	rm -rf $(DOCS_DIR)/api/generated
	uv run make -C $(DOCS_DIR) clean

coverage:
	uv run coverage run -m $(PYTEST_CMD)
	uv run coverage report -m

docs:
	uv run make -C $(DOCS_DIR) html

_docs: docs
	$(VIEW_HTML_CMD) $(DOCS_DIR)/build/html/index.html

pages:
	git checkout main
	make clean
	make docs
	git checkout gh-pages
	rm -r dev
	mv docs/build/html dev
	rm -r docs
	git add -A

doctest:
	uv run xdoctest delu --global-exec "import torch; import torch.nn as nn; import delu"

spelling:
	if [[ $(shell uname -m) != "arm64" ]]; then\
		uv run make -C $(DOCS_DIR) docs SPHINXOPTS="-W -b spelling";\
	fi

lint:
	uv run -m pre_commit_hooks.debug_statement_hook delu/*.py
	uv run -m pre_commit_hooks.debug_statement_hook delu/**/*.py
	uv run ruff format --check
	uv run ruff check .

# The order is important: clean must be first, docs must precede doctest.
pre-commit: clean lint test docs doctest spelling typecheck

test:
	uv run $(PYTEST_CMD) $(ARGV)

typecheck:
	uv run mypy delu
