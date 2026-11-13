# Common tasks. Every target is a thin wrapper around uv; run `make help` for the list.
DATA ?= data/measurements-v1.1.0
OWNER ?= natiixnt

.PHONY: help sync check lint test download reproduce figures

help:
	@echo "make sync        install the results path (numpy, scipy, matplotlib) and dev tools"
	@echo "make check       ruff + pytest"
	@echo "make download     fetch and verify the measurement package into data/ (OWNER=$(OWNER))"
	@echo "make reproduce   recompute every table and compare byte for byte (DATA=$(DATA))"
	@echo "make figures     redraw docs/figures from results/results.json and the package (DATA=$(DATA))"

sync:
	uv sync

lint:
	uv run ruff check .

test:
	uv run pytest

check: lint test

download:
	uv run fcb download --owner $(OWNER)

reproduce:
	uv run fcb reproduce --data $(DATA) --check

figures:
	uv run fcb figures --data $(DATA)
