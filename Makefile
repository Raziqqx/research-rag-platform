# One command to regenerate every file in results/:  make reproduce
# (db -> fetch corpus -> ingest 3 chunkers + bge-small -> all evals). ~25 min on a laptop CPU.
PY ?= $(HOME)/rag-data/venv/bin/python
TF_IMAGE ?= hashicorp/terraform:1.9.8
TF_DIR := infra/terraform

.PHONY: all reproduce db corpus ingest ingest-bge eval eval-retrieval test lint calibrate api mcp \
	web-build web-test up down tf-check
all: reproduce
reproduce: db corpus ingest ingest-bge eval-retrieval calibrate eval

db:
	docker compose up -d --wait db

corpus:
	$(PY) scripts/fetch_corpus.py

ingest:
	$(PY) -m rag.ingest

ingest-bge:
	$(PY) -m rag.ingest --strategies recursive --model BAAI/bge-small-en-v1.5

# Full eval: ablation grid (retrieval) + end-to-end answers (faithfulness, citations, abstention)
eval:
	$(PY) scripts/eval_ablation.py
	$(PY) scripts/eval_answers.py

eval-retrieval:
	$(PY) scripts/eval_retrieval.py --strategy recursive

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check rag scripts tests && $(PY) -m ruff format --check rag scripts tests

calibrate:
	$(PY) scripts/calibrate_abstain.py

api:
	$(PY) -m uvicorn rag.api:app --port 8000

mcp:
	$(PY) -m rag.mcp_server

web-build:
	cd web && npm ci && npm run build

web-test:
	cd web && npm test

up:
	docker compose up -d --build --wait

down:
	docker compose down

# terraform fmt + validate in Docker (no local terraform, no cloud credentials needed)
tf-check:
	docker run --rm -v "$(CURDIR)/$(TF_DIR):/tf" -w /tf $(TF_IMAGE) fmt -check -recursive
	docker run --rm -v "$(CURDIR)/$(TF_DIR):/tf" -w /tf $(TF_IMAGE) init -backend=false -input=false
	docker run --rm -v "$(CURDIR)/$(TF_DIR):/tf" -w /tf $(TF_IMAGE) validate
