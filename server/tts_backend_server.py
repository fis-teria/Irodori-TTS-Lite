#!/usr/bin/env python3
"""Small HTTP backend for Irodori-TTS-Lite.

The server is intentionally independent from ROS 2. Route guidance clients can
POST OSRM/phone route status JSON, and future SLM clients can POST plain text.
Both paths flow into the same synthesis endpoint and audio cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import unquote, urlparse


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"
DEFAULT_AUDIO_DIR = ROOT_DIR / "runtime" / "audio"


def json_dumps(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def bool_value(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = clean_text(value).lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def list_value(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = clean_text(value)
    return [text] if text else []


def format_distance(distance_m: float) -> str:
    if distance_m <= 0.0:
        return ""
    if distance_m >= 1000.0:
        return f"約{distance_m / 1000.0:.1f}キロ先"
    return f"約{round(distance_m)}メートル先"


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except ModuleNotFoundError:
        return load_simple_yaml(path)


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Small fallback parser for this file's simple config.yaml shape."""

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_simple_yaml_scalar(value)
    return root


def parse_simple_yaml_scalar(value: str) -> Any:
    if value in ("''", '""'):
        return ""
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_simple_yaml_scalar(item.strip()) for item in inner.split(",")]
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("null", "none", "~"):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def resolve_path(value: Any, base_dir: Path) -> Path:
    text = clean_text(value)
    path = Path(text) if text else Path()
    return path if path.is_absolute() else base_dir / path


class IrodoriCliEngine:
    """Runs the existing example/run_tts.py path in the isolated uv venv."""

    def __init__(
        self,
        root_dir: Path,
        audio_dir: Path,
        python_executable: Path,
        default_no_ref: bool,
        default_checkpoint: str,
        default_extra_args: list[str],
        timeout_sec: float,
        default_model: str,
        models: dict[str, dict[str, Any]],
        source_models: dict[str, str],
    ) -> None:
        self.root_dir = root_dir
        self.audio_dir = audio_dir
        self.python_executable = python_executable
        self.default_no_ref = default_no_ref
        self.default_checkpoint = default_checkpoint
        self.default_extra_args = default_extra_args
        self.timeout_sec = timeout_sec
        self.default_model = default_model
        self.models = models
        self.source_models = source_models
        self._lock = threading.Lock()

    def synthesize(self, request: dict[str, Any]) -> dict[str, Any]:
        text = clean_text(request.get("text"))
        if not text:
            raise ValueError("text is required")

        request_options = self._request_options(request)
        cache_key = self._cache_key(request_options, text)
        output_path = self.audio_dir / f"{cache_key}.wav"
        if output_path.exists() and output_path.stat().st_size > 0:
            return self._result_payload(text, output_path, cached=True, model=request_options["model"])

        self.audio_dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=f".{cache_key}.", suffix=".wav", dir=str(self.audio_dir)
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)

        command = [
            str(self.python_executable),
            str(self.root_dir / "example" / "run_tts.py"),
            "--text",
            text,
            "--output-wav",
            str(tmp_path),
        ]

        checkpoint = clean_text(request_options.get("checkpoint"))
        if checkpoint:
            command.extend(["--checkpoint", checkpoint])

        seconds = request_options.get("seconds")
        if seconds not in (None, ""):
            command.extend(["--seconds", str(safe_float(seconds))])

        no_ref = bool_value(request_options.get("no_ref"), self.default_no_ref)
        if no_ref:
            command.append("--no-ref")

        if bool_value(request_options.get("no_fused")):
            command.append("--no-fused")
        if bool_value(request_options.get("no_fp16")):
            command.append("--no-fp16")

        command.extend(list_value(request_options.get("extra_args")))

        env = os.environ.copy()
        env.setdefault("IRODORI_TTS_LITE_ROOT", str(self.root_dir))
        env.setdefault("XDG_CACHE_HOME", str(self.root_dir / ".cache"))
        env.setdefault("UV_CACHE_DIR", str(self.root_dir / ".cache" / "uv"))
        env.setdefault("PIP_CACHE_DIR", str(self.root_dir / ".cache" / "pip"))
        env.setdefault("HF_HOME", str(self.root_dir / ".cache" / "huggingface"))
        env.setdefault("HF_HUB_CACHE", str(Path(env["HF_HOME"]) / "hub"))
        env.setdefault("HUGGINGFACE_HUB_CACHE", env["HF_HUB_CACHE"])
        env.setdefault("TORCH_HOME", str(self.root_dir / ".cache" / "torch"))
        env.setdefault("TRITON_CACHE_DIR", str(self.root_dir / ".cache" / "triton"))
        env.setdefault("PYTHONPYCACHEPREFIX", str(self.root_dir / ".cache" / "pycache"))

        started = time.time()
        with self._lock:
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(self.root_dir),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.timeout_sec,
                    check=False,
                )
                if proc.returncode != 0:
                    raise RuntimeError(
                        "Irodori-TTS-Lite failed "
                        f"exit={proc.returncode} stderr={proc.stderr[-1200:]}"
                    )
                tmp_path.replace(output_path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        payload = self._result_payload(
            text,
            output_path,
            cached=False,
            model=request_options["model"],
        )
        payload["elapsed_sec"] = round(time.time() - started, 3)
        return payload

    def preview(self, request: dict[str, Any]) -> dict[str, Any]:
        text = clean_text(request.get("text"))
        if not text:
            raise ValueError("text is required")
        request_options = self._request_options(request)
        return {
            "ok": True,
            "text": text,
            "model": request_options["model"],
            "dry_run": True,
        }

    def _request_options(self, request: dict[str, Any]) -> dict[str, Any]:
        model_name = clean_text(request.get("model"))
        if not model_name:
            source = clean_text(request.get("source"))
            model_name = self.source_models.get(source, "") if source else ""
        if not model_name:
            model_name = self.default_model

        model_config = dict(self.models.get(model_name, {}))
        if model_name and model_name not in self.models:
            raise ValueError(f"unknown model: {model_name}")

        options: dict[str, Any] = {
            "model": model_name,
            "checkpoint": model_config.get("checkpoint", self.default_checkpoint),
            "seconds": model_config.get("seconds"),
            "no_ref": model_config.get("no_ref", self.default_no_ref),
            "no_fused": model_config.get("no_fused", False),
            "no_fp16": model_config.get("no_fp16", False),
            "extra_args": list_value(model_config.get("extra_args")) or self.default_extra_args,
        }
        for key in ("checkpoint", "seconds", "no_ref", "no_fused", "no_fp16", "extra_args"):
            if key in request:
                options[key] = request[key]
        return options

    def _cache_key(self, request: dict[str, Any], text: str) -> str:
        relevant = {
            "text": text,
            "model": clean_text(request.get("model")),
            "checkpoint": clean_text(request.get("checkpoint")),
            "seconds": request.get("seconds"),
            "no_ref": bool_value(request.get("no_ref"), self.default_no_ref),
            "no_fused": bool_value(request.get("no_fused")),
            "no_fp16": bool_value(request.get("no_fp16")),
            "extra_args": list_value(request.get("extra_args")),
        }
        raw = json.dumps(relevant, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def _result_payload(text: str, output_path: Path, cached: bool, model: str) -> dict[str, Any]:
        return {
            "ok": True,
            "text": text,
            "model": model,
            "cached": cached,
            "audio_path": str(output_path),
            "audio_url": f"/audio/{output_path.name}",
        }


class TtsBackend:
    def __init__(self, engine: IrodoriCliEngine) -> None:
        self.engine = engine

    def synthesize_text(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = clean_text(payload.get("text"))
        if payload.get("dry_run"):
            if not text:
                raise ValueError("text is required")
            return self.engine.preview(payload)
        return self.engine.synthesize(payload)

    def synthesize_route_guidance(self, payload: dict[str, Any]) -> dict[str, Any]:
        text = self.route_guidance_text(payload)
        request = dict(payload)
        request["text"] = text
        request.setdefault("source", "route_guidance")
        if payload.get("dry_run"):
            result = self.engine.preview(request)
            result["source"] = "route_guidance"
            return result
        return self.engine.synthesize(request)

    def route_guidance_text(self, payload: dict[str, Any]) -> str:
        explicit_text = clean_text(payload.get("text"))
        if explicit_text:
            return explicit_text

        step = payload.get("step")
        distance_m = safe_float(payload.get("distance_m"), 0.0)
        if isinstance(step, dict):
            if distance_m <= 0.0:
                distance_m = safe_float(step.get("distance_m"), 0.0)
            return self._step_text(step, distance_m)

        phone_status = payload.get("phone_status", payload)
        if isinstance(phone_status, dict):
            selected_route = phone_status.get("selected_route")
            current_index = safe_int(
                payload.get("current_route_index", phone_status.get("current_route_index")),
                0,
            )
            if isinstance(selected_route, dict):
                next_step = self._next_step(selected_route, current_index)
                if next_step is not None:
                    dist = safe_float(payload.get("distance_m"), 0.0)
                    if dist <= 0.0:
                        dist = safe_float(next_step.get("distance_m"), 0.0)
                    return self._step_text(next_step, dist)

        raise ValueError("route guidance text could not be resolved")

    def _next_step(self, selected_route: dict[str, Any], current_index: int) -> dict[str, Any] | None:
        steps = selected_route.get("guidance_steps")
        if not isinstance(steps, list) or not steps:
            return None
        candidates = [item for item in steps if isinstance(item, dict)]
        for item in candidates:
            if safe_int(item.get("route_index"), 0) >= current_index:
                return item
        return candidates[-1] if candidates else None

    def _step_text(self, step: dict[str, Any], distance_m: float) -> str:
        base = clean_text(step.get("text") or "道なりに進んでください")
        maneuver_type = clean_text(step.get("type"))
        if maneuver_type == "arrive" or "到着" in base:
            return "目的地付近です。到着します。"

        distance_phrase = format_distance(distance_m)
        if distance_phrase and not base.startswith(distance_phrase):
            return f"{distance_phrase}、{base}。"
        return f"{base}。"


def make_handler(backend: TtsBackend, audio_dir: Path, engine: IrodoriCliEngine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "IrodoriTtsLiteBackend/0.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[tts-backend] {self.address_string()} - {fmt % args}")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/health"):
                self._send_json(
                    {
                        "ok": True,
                        "service": "irodori_tts_lite_backend",
                        "default_model": engine.default_model,
                        "models": sorted(engine.models.keys()),
                        "endpoints": [
                            "/api/synthesize",
                            "/api/route_guidance",
                            "/api/slm_text",
                        ],
                    }
                )
                return
            if parsed.path.startswith("/audio/"):
                self._send_audio(parsed.path.removeprefix("/audio/"))
                return
            self.send_error(404)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._send_cors_headers()
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            try:
                payload = self._read_json()
                if parsed.path == "/api/synthesize":
                    self._send_json(backend.synthesize_text(payload))
                    return
                if parsed.path == "/api/route_guidance":
                    self._send_json(backend.synthesize_route_guidance(payload))
                    return
                if parsed.path == "/api/slm_text":
                    payload.setdefault("source", "slm")
                    self._send_json(backend.synthesize_text(payload))
                    return
                self.send_error(404)
            except Exception as exc:
                self._send_json(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    status=400,
                )

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON object is required")
            return payload

        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json_dumps(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)

        def _send_cors_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")

        def _send_audio(self, name: str) -> None:
            safe_name = Path(unquote(name)).name
            path = audio_dir / safe_name
            if not path.exists() or path.suffix.lower() != ".wav":
                self.send_error(404)
                return
            body = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(body)

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=os.environ.get("IRODORI_TTS_CONFIG", str(DEFAULT_CONFIG_PATH)))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--audio-dir", default=None)
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=None,
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--no-ref", dest="no_ref", action="store_true", default=None)
    parser.add_argument("--ref", dest="no_ref", action="store_false")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def pick(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = ROOT_DIR / config_path
    config = load_config(config_path)
    server_config = config.get("server", {}) if isinstance(config.get("server"), dict) else {}
    runtime_config = config.get("runtime", {}) if isinstance(config.get("runtime"), dict) else {}
    defaults_config = config.get("defaults", {}) if isinstance(config.get("defaults"), dict) else {}
    models_config = config.get("models", {}) if isinstance(config.get("models"), dict) else {}
    sources_config = config.get("sources", {}) if isinstance(config.get("sources"), dict) else {}

    host = pick(args.host, os.environ.get("IRODORI_TTS_BACKEND_HOST"), server_config.get("host"), default="127.0.0.1")
    port = int(pick(args.port, os.environ.get("IRODORI_TTS_BACKEND_PORT"), server_config.get("port"), default=8766))
    audio_dir = resolve_path(
        pick(args.audio_dir, os.environ.get("IRODORI_TTS_AUDIO_DIR"), server_config.get("audio_dir"), default=str(DEFAULT_AUDIO_DIR)),
        ROOT_DIR,
    ).resolve()
    python_executable = resolve_path(
        pick(
            args.python_executable,
            os.environ.get("IRODORI_TTS_PYTHON"),
            runtime_config.get("python"),
            default=str(ROOT_DIR / ".venv" / "bin" / "python"),
        ),
        ROOT_DIR,
    )
    default_model = clean_text(
        pick(args.model, os.environ.get("IRODORI_TTS_MODEL"), defaults_config.get("model"), default="irodori_lite_int4")
    )
    default_checkpoint = clean_text(
        pick(args.checkpoint, os.environ.get("IRODORI_TTS_CHECKPOINT"), defaults_config.get("checkpoint"), default="")
    )
    default_no_ref = bool_value(
        pick(args.no_ref, os.environ.get("IRODORI_TTS_NO_REF"), defaults_config.get("no_ref"), default=True),
        default=True,
    )
    default_extra_args = list_value(defaults_config.get("extra_args"))
    cli_extra_args = list(args.extra_args or [])
    if cli_extra_args:
        default_extra_args = cli_extra_args
    timeout_sec = safe_float(
        pick(args.timeout_sec, os.environ.get("IRODORI_TTS_TIMEOUT_SEC"), server_config.get("timeout_sec"), default=240),
        default=240.0,
    )

    models: dict[str, dict[str, Any]] = {
        name: dict(value) for name, value in models_config.items() if isinstance(value, dict)
    }
    models.setdefault(
        default_model,
        {
            "checkpoint": default_checkpoint,
            "no_ref": default_no_ref,
            "extra_args": default_extra_args,
        },
    )
    source_models = {
        str(name): clean_text(value.get("model"))
        for name, value in sources_config.items()
        if isinstance(value, dict) and clean_text(value.get("model"))
    }
    engine = IrodoriCliEngine(
        root_dir=ROOT_DIR,
        audio_dir=audio_dir,
        python_executable=python_executable,
        default_no_ref=default_no_ref,
        default_checkpoint=default_checkpoint,
        default_extra_args=default_extra_args,
        timeout_sec=max(1.0, timeout_sec),
        default_model=default_model,
        models=models,
        source_models=source_models,
    )
    backend = TtsBackend(engine)
    httpd = ThreadingHTTPServer((host, port), make_handler(backend, audio_dir, engine))
    print(f"[tts-backend] config={config_path}")
    print(f"[tts-backend] serving on http://{host}:{port}")
    print(f"[tts-backend] audio_dir={audio_dir}")
    print(f"[tts-backend] default_model={default_model}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
