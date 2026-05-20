#!/usr/bin/env bash

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  _irodori_tts_lite_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  _irodori_tts_lite_root="$(pwd)"
fi

export IRODORI_TTS_LITE_ROOT="${_irodori_tts_lite_root}"
export IRODORI_TTS_LITE_VENV="${IRODORI_TTS_LITE_VENV:-${IRODORI_TTS_LITE_ROOT}/.venv}"
export IRODORI_TTS_LITE_VENDOR_DIR="${IRODORI_TTS_LITE_VENDOR_DIR:-${IRODORI_TTS_LITE_ROOT}/vendor}"
export IRODORI_TTS_LITE_MODEL_DIR="${IRODORI_TTS_LITE_MODEL_DIR:-${IRODORI_TTS_LITE_ROOT}/models}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${IRODORI_TTS_LITE_ROOT}/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${IRODORI_TTS_LITE_ROOT}/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${IRODORI_TTS_LITE_ROOT}/.cache/pip}"
export HF_HOME="${HF_HOME:-${IRODORI_TTS_LITE_ROOT}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${IRODORI_TTS_LITE_ROOT}/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${IRODORI_TTS_LITE_ROOT}/.cache/triton}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-${IRODORI_TTS_LITE_ROOT}/.cache/numba}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${IRODORI_TTS_LITE_ROOT}/.cache/matplotlib}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${IRODORI_TTS_LITE_ROOT}/.cache/pycache}"

if [[ -d "${IRODORI_TTS_LITE_VENV}/bin" ]]; then
  export PATH="${IRODORI_TTS_LITE_VENV}/bin:${PATH}"
fi

unset _irodori_tts_lite_root
