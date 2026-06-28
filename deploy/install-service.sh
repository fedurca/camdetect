#!/usr/bin/env bash
# Install camdetect as a systemd service with autostart on boot and
# auto-recovery (Restart=always) on crash.
#
#   sudo ./deploy/install-service.sh [live|demo] [port]
#   ./deploy/install-service.sh --dry-run [live|demo] [port]   # validate only, no root
#
# Re-run to update. Uninstall with:
#   sudo systemctl disable --now camdetect && sudo rm /etc/systemd/system/camdetect.service
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then
  DRY_RUN=1
  shift
fi

MODE="${1:-live}"
PORT="${2:-8000}"
SERVICE="camdetect"
UNIT="/etc/systemd/system/${SERVICE}.service"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_AS="${SUDO_USER:-$(id -un)}"

render_unit() {  # $1 = destination path
  sed -e "s|__USER__|${RUN_AS}|g" \
      -e "s|__WORKDIR__|${WORKDIR}|g" \
      -e "s|__MODE__|${MODE}|g" \
      -e "s|__PORT__|${PORT}|g" \
      "${SCRIPT_DIR}/camdetect.service" > "$1"
}

echo "camdetect systemd installer"
echo "  user:    ${RUN_AS}"
echo "  workdir: ${WORKDIR}"
echo "  mode:    ${MODE}   port: ${PORT}"
echo

# Validate the rendered unit (best-effort, no root needed).
TMP_UNIT="$(mktemp --suffix=.service)"
render_unit "${TMP_UNIT}"
if command -v systemd-analyze >/dev/null 2>&1; then
  echo "Validating unit with systemd-analyze..."
  systemd-analyze verify "${TMP_UNIT}" && echo "  unit OK"
fi

if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo
  echo "--- rendered ${SERVICE}.service (dry-run, not installed) ---"
  cat "${TMP_UNIT}"
  rm -f "${TMP_UNIT}"
  echo "--- dry-run complete (no changes made) ---"
  exit 0
fi

# Real install requires root + a running systemd.
if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run with sudo: sudo $0 ${MODE} ${PORT}" >&2
  rm -f "${TMP_UNIT}"; exit 1
fi
if [[ ! -d /run/systemd/system ]]; then
  echo "systemd is not the init system here (container?). Cannot install." >&2
  rm -f "${TMP_UNIT}"; exit 1
fi

# Make sure the launcher is executable and the venv exists (run.sh creates it).
chmod +x "${WORKDIR}/run.sh"
if [[ ! -d "${WORKDIR}/.venv" ]]; then
  echo "Creating virtualenv / installing deps (first run)..."
  sudo -u "${RUN_AS}" bash -lc "cd '${WORKDIR}' && python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt"
fi

install -m 0644 "${TMP_UNIT}" "${UNIT}"
rm -f "${TMP_UNIT}"

systemctl daemon-reload
systemctl enable "${SERVICE}"
systemctl restart "${SERVICE}"

echo
echo "Done. camdetect is enabled (autostart on boot) and running."
echo "  status: systemctl status ${SERVICE}"
echo "  logs:   journalctl -u ${SERVICE} -f"
echo "  web:    http://<this-host-ip>:${PORT}"
