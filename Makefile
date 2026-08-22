PY := .venv/bin/python
VRAG := .venv/bin/vrag

.PHONY: setup ingest test lint bench eval serve all clean

setup:
	uv venv --python 3.12 .venv
	uv pip install -e ".[ingest,dev]"
	cp -n .env.example .env || true

ingest:
	$(VRAG) ingest --queries 3000 --text-field both

test:
	$(PY) -m pytest tests/ -q

lint:
	.venv/bin/ruff check src tests
	.venv/bin/ruff format --check src tests

bench:
	$(VRAG) bench --n 300 --warmup 30

eval:
	$(VRAG) eval-chunking --max-queries 250
	$(VRAG) eval-guardrails

serve:
	$(VRAG) serve

all: test bench eval

clean:
	rm -rf data/index data/corpus reports .pytest_cache
