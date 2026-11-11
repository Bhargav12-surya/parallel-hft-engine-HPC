.PHONY: build test lint format train benchmark demo package clean
PYTHON ?= python
build:
	$(PYTHON) scripts/build.py --test
test:
	$(PYTHON) -m pytest -q
lint:
	$(PYTHON) -m ruff check hftlab tests scripts
format:
	$(PYTHON) -m ruff format hftlab tests scripts
train:
	$(PYTHON) -m hftlab train --dataset data/generated/demo.csv --output models/reference
benchmark:
	$(PYTHON) -m hftlab compare --dataset data/generated/demo.csv --all --repeats 3
demo:
	$(PYTHON) -m hftlab demo
package:
	$(PYTHON) -m build --no-isolation
clean:
	$(PYTHON) scripts/clean.py
