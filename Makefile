.PHONY: all install lint test coverage help

all: help

install:
	poetry install

test:
	poetry run python -m pytest tests/

coverage:
	poetry run python -m pytest --cov \
		--cov-report term \
	tests/
	poetry run python -m pytest --cov=app \
		--cov-report=term-missing \
		--cov-report=html:coverage_html_report \
		tests/
	@echo "===================="
	@echo "Coverage HTML report generated in 'coverage_html_report' directory."

format:
	poetry run isort .
	poetry run black .

lint:
	poetry run flake8 .
	poetry run mypy .

help:
	@echo '===================='
	@echo '-- RUNTIME --'
	@echo 'install      - install dependencies'
	@echo '-- LINTING --'
	@echo 'format       - run code formatters'
	@echo 'lint         - run linters'
	@echo '-- TESTS --'
	@echo 'coverage     - run unit tests and generate coverage report'
	@echo 'test         - run unit tests'
	