PY = venv/bin/python

setup:
	python3 -m venv venv
	$(PY) -m pip install -r requirements-dev.txt
	test -f .env || cp .env.example .env

run:
	SIMULATION_MODE=true $(PY) -m uvicorn app.main:app --reload --port 8000

serve:
	$(PY) -m uvicorn app.main:app --port 8000

test:
	$(PY) -m ruff check app tests scripts
	$(PY) -m pytest -q

demo:
	$(PY) scripts/build_static_demo.py

.PHONY: setup run serve test demo
