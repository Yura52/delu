.PHONY: default clean coverage _docs docs dtest format lint pages pre-commit test typecheck

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
	rm -rf $(DOCS_DIR)/source/reference/api
	make -C $(DOCS_DIR) clean

coverage:
	coverage run -m $(PYTEST_CMD)
	coverage report -m

_docs:
	make -C $(DOCS_DIR) html

docs: _docs
	$(VIEW_HTML_CMD) $(DOCS_DIR)/build/html/index.html

pages:
	git checkout master
	make clean
	make _docs
	git checkout gh-pages
	rm -r dev
	mv docs/build/html dev
	rm -r docs
	git add -A

dtest:
	make -C $(DOCS_DIR) doctest

lint:
	python -m pre_commit_hooks.debug_statement_hook zero/**/*.py
	isort zero --check-only
	black zero --check
	flake8 zero

# the order is important: clean must be first, _docs must precede dtest
pre-commit: clean lint test _docs dtest typecheck

test:
	PYTHONPATH='.' $(PYTEST_CMD) $(ARGV)

typecheck:
	mypy zero
