#!/usr/bin/env bash
# Install camdetect as a systemd service with autostart on boot and
# auto-recovery (Restart=always) on crash.
#
#   sudo ./deploy/install-service.sh [live|demo] [port]
#
# Re-run to update. Uninstall with:
#   sudo systemctl disable --now camdetect && sudo rm /etc/systemd/system/camdetect.service
set -euo pipefail

MODE="${1:-live}"
PORT="${2:-8000}"
SERVICE="camdetect"
UNIT="/etc/systemd/system/${SERVICE}.service"

# Resolve the repo directory (parent of this script) and the invoking user.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_AS="${SUDO_USER:-$(id -un)}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Please run with sudo: sudo $0 ${MODE} ${PORT}" >&2
  exit 1
fi

echo "Installing ${SERVICE} service:"
echo "  user:    ${RUN_AS}"
echo "  workdir: ${WORKDIR}"
echo "  mode:    ${MODE}   port: ${PORT}"

# Make sure the launcher is executable and the venv exists (run.sh creates it).
chmod +x "${WORKDIR}/run.sh"
if [[ ! -d "${WORKDIR}/.venv" ]]; then
  echo "Creating virtualenv / installing deps (first run)..."
  sudo -u "${RUN_AS}" bash -lc "cd '${WORKDIR}' && python3 -m venv .venv && \
    .venv/bin/pip install --upgrade pip && .venv/bin/pip install -r requirements.txt"
fi

sed -e "s|__USER__|${RUN_AS}|g" \
    -e "s|__WORKDIR__|${WORKDIR}|g" \
    -e "s|__MODE__|${MODE}|g" \
    -e "s|__PORT__|${PORT}|g" \
    "${SCRIPT_DIR}/camdetect.service" > "${UNIT}"

systemctl daemon-reload
systemctl enable "${SERVICE}"
systemctl restart "${SERVICE}"

echo
echo "Done. camdetect is enabled (autostart on boot) and running."
echo "  status: systemctl status ${SERVICE}"
echo "  logs:   journalctl -u ${SERVICE} -f"
echo "  web:    http://<this-host-ip>:${PORT}"
