.PHONY: default clean coverage _docs docs dtest spelling format lint pages pre-commit test typecheck

PYTEST_CMD = pytest zero
VIEW_HTML_CMD = open
DOCS_DIR = docs

default:
	echo "Hello, World!"

clean:
	find zero -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -f .coverage
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf $(DOCS_DIR)/reference/api
	make -C $(DOCS_DIR) clean

coverage:
	coverage run -m $(PYTEST_CMD)
	coverage report -m

docs:
	make -C $(DOCS_DIR) html

_docs: docs
	$(VIEW_HTML_CMD) $(DOCS_DIR)/build/html/index.html

pages:
	git checkout master
	make clean
	make docs
	git checkout gh-pages
	rm -r dev
	mv docs/build/html dev
	rm -r docs
	git add -A

dtest:
	make -C $(DOCS_DIR) doctest

spelling:
	make -C $(DOCS_DIR) docs SPHINXOPTS="-W -b spelling"

lint:
	python -m pre_commit_hooks.debug_statement_hook zero/*.py
	python -m pre_commit_hooks.debug_statement_hook zero/**/*.py
	isort zero --check-only
	black zero --check
	flake8 zero

# the order is important: clean must be first, docs must precede dtest
pre-commit: clean lint test docs dtest spelling typecheck

test:
	PYTHONPATH='.' $(PYTEST_CMD) $(ARGV)

typecheck:
	mypy zero
