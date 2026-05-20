#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

UPSTREAM_REPO_URL="${IRODORI_TTS_UPSTREAM_REPO_URL:-https://github.com/Aratako/Irodori-TTS.git}"
UPSTREAM_REF="${IRODORI_TTS_UPSTREAM_REF:-main}"
INSTALL_UPSTREAM="${IRODORI_TTS_INSTALL_UPSTREAM:-1}"
INSTALL_LITE="${IRODORI_TTS_INSTALL_LITE:-1}"
RUN_IMPORT_CHECK="${IRODORI_TTS_RUN_IMPORT_CHECK:-1}"
DOWNLOAD_MODELS="${IRODORI_TTS_DOWNLOAD_MODELS:-0}"
TORCH_INDEX_URL="${IRODORI_TTS_TORCH_INDEX_URL:-}"
TORCH_SPEC="${IRODORI_TTS_TORCH_SPEC:-torch}"

usage() {
  cat <<'EOF'
Usage: ./setup.sh [options]

Build an isolated uv environment for Irodori-TTS-Lite. All runtime caches,
upstream source checkout, Hugging Face downloads, torch/triton caches, and the
virtualenv are kept under this irodori_tts_lite directory by env.sh.

Options:
  --skip-upstream       Do not clone/install upstream Aratako/Irodori-TTS.
  --skip-lite           Do not install this irodori-tts-lite package.
  --skip-check          Skip the import check.
  --download-models     Pre-download the default int4 model into the local HF cache.
  --torch-index-url URL Install torch from this package index URL.
  -h, --help            Show this help.

Environment variables:
  IRODORI_TTS_UPSTREAM_REPO_URL   Default: https://github.com/Aratako/Irodori-TTS.git
  IRODORI_TTS_UPSTREAM_REF        Default: main
  IRODORI_TTS_INSTALL_UPSTREAM    Default: 1
  IRODORI_TTS_INSTALL_LITE        Default: 1
  IRODORI_TTS_RUN_IMPORT_CHECK    Default: 1
  IRODORI_TTS_DOWNLOAD_MODELS     Default: 0
  IRODORI_TTS_TORCH_INDEX_URL     Optional torch wheel index URL
  IRODORI_TTS_TORCH_SPEC          Default: torch
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-upstream)
      INSTALL_UPSTREAM=0
      ;;
    --skip-lite)
      INSTALL_LITE=0
      ;;
    --skip-check)
      RUN_IMPORT_CHECK=0
      ;;
    --download-models)
      DOWNLOAD_MODELS=1
      ;;
    --torch-index-url)
      shift
      TORCH_INDEX_URL="${1:-}"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
  shift
done

log() {
  printf '[irodori-tts-lite setup] %s\n' "$*"
}

require_command() {
  local command_name="$1"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "[ERROR] Required command not found: ${command_name}" >&2
    if [[ "${command_name}" == "uv" ]]; then
      echo "[HINT] Install uv first, for example: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    fi
    exit 1
  fi
}

uv_pip_install() {
  uv pip install --python "${IRODORI_TTS_LITE_VENV}/bin/python" "$@"
}

install_torch_from_index() {
  if [[ -z "${TORCH_INDEX_URL}" ]]; then
    return
  fi

  log "Installing ${TORCH_SPEC} from ${TORCH_INDEX_URL}"
  uv pip install \
    --python "${IRODORI_TTS_LITE_VENV}/bin/python" \
    --index-url "${TORCH_INDEX_URL}" \
    "${TORCH_SPEC}"
}

clone_or_update_upstream() {
  local upstream_dir="${IRODORI_TTS_LITE_VENDOR_DIR}/Irodori-TTS"

  if [[ "${INSTALL_UPSTREAM}" != "1" ]]; then
    return
  fi

  mkdir -p "${IRODORI_TTS_LITE_VENDOR_DIR}"
  if [[ -d "${upstream_dir}/.git" ]]; then
    log "Updating upstream Irodori-TTS in ${upstream_dir}"
    git -C "${upstream_dir}" fetch --depth 1 origin "${UPSTREAM_REF}"
    git -C "${upstream_dir}" checkout FETCH_HEAD
  else
    log "Cloning upstream Irodori-TTS into ${upstream_dir}"
    git clone --depth 1 --branch "${UPSTREAM_REF}" "${UPSTREAM_REPO_URL}" "${upstream_dir}"
  fi

  log "Installing upstream Irodori-TTS editable package"
  uv_pip_install -e "${upstream_dir}"
}

install_lite_package() {
  if [[ "${INSTALL_LITE}" != "1" ]]; then
    return
  fi

  log "Installing Irodori-TTS-Lite editable package"
  uv_pip_install -e "${SCRIPT_DIR}[infer]"
}

download_default_models() {
  if [[ "${DOWNLOAD_MODELS}" != "1" ]]; then
    return
  fi

  log "Downloading default Irodori-TTS-Lite model into ${HUGGINGFACE_HUB_CACHE}"
  "${IRODORI_TTS_LITE_VENV}/bin/python" - <<'PY'
import irodori_tts_lite

path = irodori_tts_lite.resolve_checkpoint()
print(f"default_checkpoint={path}")
PY
}

run_import_check() {
  if [[ "${RUN_IMPORT_CHECK}" != "1" ]]; then
    return
  fi

  log "Running import check"
  "${IRODORI_TTS_LITE_VENV}/bin/python" - <<'PY'
import os
import irodori_tts_lite

print(f"irodori_tts_lite={irodori_tts_lite.__file__}")
try:
    import irodori_tts
    print(f"irodori_tts={irodori_tts.__file__}")
except Exception as exc:
    print(f"irodori_tts_import_error={type(exc).__name__}: {exc}")
    raise

for name in (
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "TORCH_HOME",
    "TRITON_CACHE_DIR",
    "UV_CACHE_DIR",
):
    print(f"{name}={os.environ.get(name, '')}")
PY
}

main() {
  require_command uv
  require_command git

  mkdir -p \
    "${IRODORI_TTS_LITE_MODEL_DIR}" \
    "${UV_CACHE_DIR}" \
    "${PIP_CACHE_DIR}" \
    "${HUGGINGFACE_HUB_CACHE}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${NUMBA_CACHE_DIR}" \
    "${MPLCONFIGDIR}" \
    "${PYTHONPYCACHEPREFIX}"

  if [[ ! -x "${IRODORI_TTS_LITE_VENV}/bin/python" ]]; then
    log "Creating uv virtualenv at ${IRODORI_TTS_LITE_VENV}"
    uv venv --python 3.10 "${IRODORI_TTS_LITE_VENV}"
  else
    log "Using existing uv virtualenv at ${IRODORI_TTS_LITE_VENV}"
  fi

  install_torch_from_index
  clone_or_update_upstream
  install_lite_package
  download_default_models
  run_import_check

  log "Done. To use this environment:"
  log "  source ${SCRIPT_DIR}/env.sh"
  log "  ${IRODORI_TTS_LITE_VENV}/bin/python ${SCRIPT_DIR}/example/run_tts.py --text 'こんにちは' --output-wav /tmp/sample.wav --no-ref"
}

main "$@"
