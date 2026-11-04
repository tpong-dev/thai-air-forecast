# Thai Air Forecast: common commands.
# Windows without make: run the command after each target by hand, with PYTHONPATH=src;.
# Windows: .venv/Scripts/python; Linux/macOS: make PY=.venv/bin/python
PY ?= .venv/Scripts/python
export PYTHONPATH := src$(if $(filter Windows_NT,$(OS)),;,:).

.PHONY: install data backtest train report live api dashboard test lint docker

install:
	$(PY) -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[deep,viz,demo,dev]"

# Download and clean the three sources into data/processed/ (cached in data/raw/).
data:
	$(PY) -m thaiair.pm25
	$(PY) -m thaiair.weather
	$(PY) -m thaiair.cams

# Rolling-origin backtest of every model -> reports/backtest_predictions.parquet + metrics.
backtest:
	$(PY) -m thaiair.evaluate

# Fit the served models on all data -> models/*.joblib + models/manifest.json.
train:
	$(PY) -m thaiair.train

# Tables and figures for README / reports/results.md.
report:
	$(PY) -m thaiair.report

# Today's forecast for all cities from the live APIs.
live:
	$(PY) -m thaiair.live --all --out live_forecast.json

api:
	$(PY) -m uvicorn app.api:app --reload --port 8000

dashboard:
	$(PY) -m streamlit run app/dashboard.py

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check .

docker:
	docker build -t thai-air-forecast .
