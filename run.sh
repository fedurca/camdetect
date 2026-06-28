#!/usr/bin/env bash
# camdetect launcher.
#
#   ./run.sh             # live mode (real RTSP cameras)
#   ./run.sh demo        # demo mode (synthetic objects, no cameras/GPU needed)
#   ./run.sh check       # test RTSP connectivity and save snapshots
#
# Honors:
#   PORT       (default 8000)
#   HOST       (default 0.0.0.0)
#   CAMDETECT_CONFIG  (default ./config.yaml)
set -euo pipefail

cd "$(dirname "$0")"

VENV=".venv"
if [[ ! -d "$VENV" ]]; then
  echo "Creating virtualenv..."
  python3 -m venv "$VENV"
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  pip install --upgrade pip
  pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
fi

MODE="${1:-live}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

case "$MODE" in
  check)
    exec python -m backend.check_streams --save
    ;;
  demo|live)
    export CAMDETECT_MODE="$MODE"
    echo "Starting camdetect ($MODE) on http://$HOST:$PORT"
    exec uvicorn backend.app:app --host "$HOST" --port "$PORT"
    ;;
  *)
    echo "Usage: ./run.sh [live|demo|check]" >&2
    exit 1
    ;;
esac
