#!/usr/bin/env bash
# SSH tunnel: forward local 11434 -> remote Ollama 11434
# Usage: ./scripts/start_tunnel.sh
# Password is prompted (placeholder); recommend setting up SSH keys later.
set -e

REMOTE_USER="${REMOTE_SSH_USER:-veridian}"
REMOTE_HOST="${REMOTE_SSH_HOST:-192.168.144.129}"
REMOTE_PORT="${REMOTE_OLLAMA_PORT:-11434}"
LOCAL_PORT="${LOCAL_TUNNEL_PORT:-11434}"

echo "Opening SSH tunnel: localhost:${LOCAL_PORT} -> ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
ssh -N -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" "${REMOTE_USER}@${REMOTE_HOST}"
