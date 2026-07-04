# Real-Time Fraud Detection — reproducible pipeline.
# Everything is offline, CPU-only, deterministic (seed=42).

PYTHON ?= python
export PYTHONPATH := src
export MPLBACKEND := Agg

EVENTS ?= 5000000

.PHONY: help setup data run test bench screenshots clean all

help:
	@echo "make setup       - install deps (offline stack assumed present)"
	@echo "make data        - stream EVENTS=$(EVENTS) events -> data/features.parquet"
	@echo "make run         - full pipeline: stream -> train -> eval -> serve"
	@echo "make test        - pytest suite (leakage, rolling, PR-AUC, latency)"
	@echo "make bench       - scaling benchmark (throughput + flat memory)"
	@echo "make screenshots - render 4 PNG dashboards into assets/"
	@echo "make all         - run + screenshots"

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/generate_data.py --rows $(EVENTS)

run:
	$(PYTHON) scripts/run_pipeline.py --events $(EVENTS)

test:
	$(PYTHON) -m pytest tests/ -q

bench:
	$(PYTHON) benchmarks/scaling_bench.py

screenshots:
	$(PYTHON) scripts/make_screenshots.py

all: run screenshots

clean:
	rm -f data/*.parquet data/*.pkl data/*.json data/*.npz
	rm -rf src/**/__pycache__ tests/__pycache__ .pytest_cache
