.PHONY: default clean coverage _docs docs dtest format lint pages pre-commit test typecheck

PYTEST_CMD = pytest zero
TEST_CMD = PYTHONPATH='.' $(PYTEST_CMD)
VIEW_HTML_CMD = open
DOCSRC = docsrc

default:
	echo "Hello, World!"

clean:
	find zero -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -f .coverage
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf dist
	rm -rf $(DOCSRC)/source/reference/api
	cd $(DOCSRC) && make clean

coverage:
	coverage run -m $(PYTEST_CMD)
	coverage report -m

_docs:
	cd $(DOCSRC) && make html

docs: _docs
	$(VIEW_HTML_CMD) $(DOCSRC)/build/html/index.html

dtest:
	cd $(DOCSRC) && make doctest

pages:
	rm -r docs
	cp -r $(DOCSRC)/build/html docs
	touch docs/.nojekyll

lint:
	python -m pre_commit_hooks.debug_statement_hook zero/**/*.py
	isort zero --recursive --check-only
	black zero --check
	flake8 zero

# the order is important
pre-commit: clean lint test dtest typecheck _docs

format:
	isort zero --recursive -y
	black zero

test:
	$(TEST_CMD) $(ARGV)

typecheck:
	mypy zero
