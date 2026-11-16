"""OpenPI JAX in an isolated local process; no robot access in the worker."""

import base64
import hashlib
import io
import json
import os
import select
import subprocess
import threading
import time
from collections import deque
from types import SimpleNamespace

import numpy as np

from .images import model_image
from .policy import model_identity
from .protocol import FEATURES, validate_action

OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"


def jax_contract(path):
    manifest = json.loads((path / "onero_openpi.json").read_text())
    expected = {
        "schema_version": 1,
        "upstream_commit": OPENPI_COMMIT,
        "action_representation": "absolute_joint_positions",
        "units": "radians_and_normalized_grippers",
        "state_features": list(FEATURES),
        "action_features": list(FEATURES),
        "model_type": "pi05",
        "action_dim": 32,
        "action_horizon": 50,
        "max_token_len": 200,
        "num_steps": 10,
        "use_quantiles": True,
        "delta_action_features": [],
    }
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest[key] != value:
            raise ValueError(f"unsupported OpenPI model contract: {key}")
    if type(manifest.get("test_only", False)) is not bool:
        raise ValueError("test_only must be boolean")
    for key in ("norm_stats", "tokenizer"):
        file = path / manifest[key]
        if not file.resolve().is_relative_to(path.resolve()) or not file.is_file():
            raise ValueError(f"invalid local OpenPI asset: {key}")
    if not (path / "params").is_dir():
        raise ValueError("OpenPI requires native JAX params")
    return manifest


def verify_openpi(source, inventory):
    expected = json.loads(inventory.read_text())
    if expected["commit"] != OPENPI_COMMIT:
        raise ValueError("unsupported OpenPI source inventory")
    actual = {
        p.relative_to(source).as_posix()
        for base in ("src/openpi", "packages/openpi-client/src/openpi_client")
        for p in (source / base).rglob("*.py")
    }
    if actual != {p for p in expected["files"] if p.endswith(".py")}:
        raise ValueError("OpenPI source file inventory differs")
    for name, digest in expected["files"].items():
        file = source / name
        if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError(f"OpenPI source hash mismatch: {name}")


class JaxPolicy:
    def __init__(self, settings):
        self.settings = settings
        self.contract = jax_contract(settings.model_path)
        if settings.image_resolution != (168, 224):
            raise ValueError("this JAX checkpoint requires image_resolution: [168,224]")
        if not settings.device.startswith("cuda"):
            raise ValueError("OpenPI JAX backend requires device: cuda")
        runtime = settings.jax_python.parent.parent.parent
        verify_openpi(settings.openpi_source, runtime / "upstream-files.json")
        self.identity = model_identity(settings.model_path) | {
            "backend": "openpi_jax",
            "upstream_commit": OPENPI_COMMIT,
        }
        env = dict(os.environ)
        env.update(
            JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false",
            # Thor's stock Triton path fails to load kernels with this pinned JAX.
            XLA_FLAGS=(
                "--xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer= "
                "--xla_gpu_autotune_level=0"
            ),
            PYTHONNOUSERSITE="1",
            PYTHONPATH=os.pathsep.join(
                [
                    str(runtime),
                    str(settings.openpi_source / "src"),
                    str(settings.openpi_source / "packages/openpi-client/src"),
                ]
            ),
        )
        if settings.device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = settings.device[5:]
        self._buffer = b""
        self._lock = threading.Lock()
        self._queue = deque()
        self._process = subprocess.Popen(
            [
                str(settings.jax_python),
                "-m",
                "onero_jax_worker.worker",
                "--model",
                str(settings.model_path),
                "--task",
                settings.task,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            env=env,
            bufsize=0,
        )
        try:
            reply = self._read()
            if not reply.get("ready") or reply.get("platforms") != ["gpu"]:
                raise RuntimeError("JAX worker did not confirm a CUDA GPU")
            self.identity["runtime"] = reply
        except BaseException:
            self.close()
            raise

    def _read(self):
        deadline = time.monotonic() + self.settings.backend_timeout_s
        while b"\n" not in self._buffer:
            left = deadline - time.monotonic()
            if left <= 0 or not select.select([self._process.stdout], [], [], max(0, left))[0]:
                raise TimeoutError("OpenPI JAX worker response timeout")
            chunk = os.read(self._process.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("OpenPI JAX worker exited before responding")
            self._buffer += chunk
            if len(self._buffer) > 8 * 1024 * 1024:
                raise ValueError("oversized JAX worker response")
        line, self._buffer = self._buffer.split(b"\n", 1)
        value = json.loads(line)
        if "error" in value:
            raise RuntimeError(f"OpenPI JAX: {value['error']}")
        return value

    def predict_chunk(self, raw):
        obs = {"state": np.asarray([raw[k] for k in FEATURES], dtype=np.float32)}
        obs.update(
            {
                k: model_image(raw[k], self.settings.image_resolution)
                for k in ("head", "left", "right")
            }
        )
        stream = io.BytesIO()
        np.savez(stream, **obs)
        request = {
            "op": "infer",
            "observation": base64.b64encode(stream.getvalue()).decode("ascii"),
        }
        with self._lock:
            payload = memoryview((json.dumps(request) + "\n").encode())
            while payload:
                written = self._process.stdin.write(payload)
                if not written:
                    raise RuntimeError("OpenPI JAX worker pipe closed during request")
                payload = payload[written:]
            self._process.stdin.flush()
            actions = np.asarray(self._read()["actions"], dtype=np.float32)
        if actions.shape != (50, 16) or not np.isfinite(actions).all():
            raise ValueError("invalid OpenPI JAX actions")
        for action in actions:
            validate_action(dict(zip(FEATURES, action, strict=True)))
        return actions

    def select(self, raw):
        if not self._queue:
            self._queue.extend(self.predict_chunk(raw))
        return validate_action(dict(zip(FEATURES, self._queue.popleft(), strict=True)))

    def reset(self):
        self._queue.clear()

    def validate_chunk_mode(self):
        if 50 / self.settings.fps > self.settings.processing.max_chunk_age_s:
            raise ValueError("chunk execution horizon exceeds max_chunk_age_s")
        return SimpleNamespace(execution_steps=50)

    def close(self):
        process = self._process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def create_policy(settings):
    if settings.backend == "openpi_jax":
        return JaxPolicy(settings)
    from .policy import LocalPolicy

    return LocalPolicy(settings)
