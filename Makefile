SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	poetry run mypy gans_zoo tests/**/*.py
	poetry run flake8 .

.PHONY: unit
unit:
	poetry run pytest

.PHONY: package
package:
	poetry check
	poetry run pip check

.PHONY: test
test: lint package unit

.PHONY: install
install:
	poetry check
	poetry config virtualenvs.in-project true
	poetry install

.PHONY: setup
setup:
	pyenv local 3.8.5
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
