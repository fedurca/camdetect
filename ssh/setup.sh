#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSH_DIR="${WORKSPACE_DIR}/ssh"
HOST_AGENT_SOCK="/run/host-services/ssh-auth.sock"

install_packages() {
  if ! command -v sshd >/dev/null 2>&1; then
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server openssh-client
  fi
}

start_sshd() {
  if ! pgrep -x sshd >/dev/null 2>&1; then
    sudo service ssh start || sudo /usr/sbin/sshd
  fi
}

configure_client() {
  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"

  if [[ ! -f "${HOME}/.ssh/config" ]]; then
    ln -sf "${SSH_DIR}/config" "${HOME}/.ssh/config"
  fi

  if [[ -S "${HOST_AGENT_SOCK}" ]]; then
    if ! grep -q 'host-services/ssh-auth.sock' "${HOME}/.bashrc" 2>/dev/null; then
      cat >> "${HOME}/.bashrc" <<'EOF'

# Use the host machine's forwarded SSH agent in this workspace.
if [[ -S /run/host-services/ssh-auth.sock ]]; then
  export SSH_AUTH_SOCK=/run/host-services/ssh-auth.sock
fi
EOF
    fi
    export SSH_AUTH_SOCK="${HOST_AGENT_SOCK}"
  fi
}

install_authorized_keys() {
  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"

  if [[ -f "${SSH_DIR}/authorized_keys" ]]; then
    cp "${SSH_DIR}/authorized_keys" "${HOME}/.ssh/authorized_keys"
    chmod 600 "${HOME}/.ssh/authorized_keys"
  fi
}

install_workspace_key() {
  mkdir -p "${HOME}/.ssh"
  chmod 700 "${HOME}/.ssh"

  if [[ ! -f "${SSH_DIR}/id_ed25519" ]]; then
    ssh-keygen -t ed25519 -f "${SSH_DIR}/id_ed25519" -N "" -C "camdetect-workspace@$(hostname)"
    cat "${SSH_DIR}/id_ed25519.pub" >> "${SSH_DIR}/authorized_keys"
  fi

  ln -sf "${SSH_DIR}/id_ed25519" "${HOME}/.ssh/id_ed25519_workspace"
  ln -sf "${SSH_DIR}/id_ed25519.pub" "${HOME}/.ssh/id_ed25519_workspace.pub"
  chmod 600 "${SSH_DIR}/id_ed25519" "${HOME}/.ssh/id_ed25519_workspace"
}

add_key() {
  local key_file="$1"
  if [[ ! -f "${key_file}" ]]; then
    echo "Key file not found: ${key_file}" >&2
    exit 1
  fi
  grep -vF "$(cat "${key_file}")" "${SSH_DIR}/authorized_keys" > "${SSH_DIR}/authorized_keys.tmp" || true
  cat "${key_file}" >> "${SSH_DIR}/authorized_keys.tmp"
  mv "${SSH_DIR}/authorized_keys.tmp" "${SSH_DIR}/authorized_keys"
  install_authorized_keys
  echo "Added key from ${key_file}"
}

print_status() {
  echo
  echo "SSH setup complete."
  echo "  SSH server: $(pgrep -x sshd >/dev/null && echo running || echo stopped) on port 22"
  echo "  Hostname:   $(hostname)"
  echo "  User:       $(whoami)"
  echo "  Addresses:  $(hostname -I 2>/dev/null | xargs echo || true)"
  if [[ -S "${HOST_AGENT_SOCK}" ]]; then
    echo "  Agent:      ${HOST_AGENT_SOCK} (keys: $(SSH_AUTH_SOCK="${HOST_AGENT_SOCK}" ssh-add -l 2>/dev/null | wc -l))"
  else
    echo "  Agent:      not available (host socket missing)"
  fi
}

main() {
  case "${1:-}" in
    --add-key)
      shift
      add_key "$1"
      ;;
    *)
      install_packages
      start_sshd
      install_workspace_key
      configure_client
      install_authorized_keys
      print_status
      ;;
  esac
}

main "$@"
