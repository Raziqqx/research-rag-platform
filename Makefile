# Reproduce run-A results: make db corpus ingest eval   (or: make all)
PY ?= $(HOME)/rag-data/venv/bin/python

.PHONY: all db corpus ingest eval test lint
all: db corpus ingest eval

db:
	docker compose up -d --wait db

corpus:
	$(PY) scripts/fetch_corpus.py

ingest:
	$(PY) -m rag.ingest

eval:
	$(PY) scripts/eval_retrieval.py --strategy recursive

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check rag scripts tests && $(PY) -m ruff format --check rag scripts tests
