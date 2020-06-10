.PHONY: default clean coverage lint pre-commit pretty test typecheck

PYTEST_CMD = pytest zero
TEST_CMD = PYTHONPATH='.' $(PYTEST_CMD)

default:
	echo "Hello, World!"

clean:
	find zero -type f -name "*.py[co]" -delete -o -type d -name __pycache__ -delete
	rm -f .coverage
	rm -rf .ipynb_checkpoints
	rm -rf .mypy_cache
	rm -rf .pytest_cache

coverage:
	coverage run -m $(PYTEST_CMD)
	coverage report -m

lint:
	python -m pre_commit_hooks.debug_statement_hook zero/**/*.py
	isort --check-only
	black zero --check
	flake8 zero

pre-commit: lint test typecheck

pretty:
	isort -y
	black zero

test: clean
	$(TEST_CMD) $(ARGV)

typecheck:
	mypy zero
