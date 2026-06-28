# camdetect - convenience targets
#
#   make setup     create .venv and install dependencies
#   make demo      run with synthetic objects (no cameras/GPU needed)
#   make check     test RTSP connectivity + save snapshots
#   make run       run live (real cameras)
#   make calib     build calibrations from calibrate/points.example.json
#   make clean     remove the virtualenv
#
# Override host/port:  make demo PORT=9000 HOST=0.0.0.0

PY ?= python3
VENV := .venv
BIN := $(VENV)/bin
HOST ?= 0.0.0.0
PORT ?= 8000

.PHONY: setup demo run live check calib clean

$(BIN)/activate:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

setup: $(BIN)/activate ## install deps into .venv
	@echo "Environment ready. Try: make demo"

demo: $(BIN)/activate
	CAMDETECT_MODE=demo $(BIN)/uvicorn backend.app:app --host $(HOST) --port $(PORT)

run live: $(BIN)/activate
	CAMDETECT_MODE=live $(BIN)/uvicorn backend.app:app --host $(HOST) --port $(PORT)

check: $(BIN)/activate
	$(BIN)/python -m backend.check_streams --save

calib: $(BIN)/activate
	$(BIN)/python -m calibrate.calibrate --from-points calibrate/points.example.json

clean:
	rm -rf $(VENV)
