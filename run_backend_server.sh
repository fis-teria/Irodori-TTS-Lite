#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

exec "${IRODORI_TTS_LITE_VENV}/bin/python" \
  "${SCRIPT_DIR}/server/tts_backend_server.py" \
  "$@"
