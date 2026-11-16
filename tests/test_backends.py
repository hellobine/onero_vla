"""Fail closed at checkpoint, model-image and local-worker boundaries."""

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from lerobot_robot_onero.images import model_image
from lerobot_robot_onero.jax_policy import JaxPolicy, jax_contract, verify_openpi
from lerobot_robot_onero.strict_load import load_pi05_strict


class TinyPI(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = torch.nn.Linear(2, 1)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        pytest.fail("upstream PI05 random-weight fallback must never run")

    def _convert_openpi_state_dict(self, state):
        return {f"model.{key}": value for key, value in state.items()}


def test_pi05_loads_exact_saved_tensors(tmp_path):
    values = {"weight": torch.tensor([[3.0, 5.0]]), "bias": torch.tensor([7.0])}
    save_file(values, str(tmp_path / "model.safetensors"))
    policy = load_pi05_strict(TinyPI, SimpleNamespace(), tmp_path)
    for key, value in values.items():
        torch.testing.assert_close(policy.state_dict()[f"model.{key}"], value, rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["missing", "extra", "shape", "nan", "discarded"])
def test_pi05_rejects_incomplete_or_discarded_weights(tmp_path, fault):
    values = {"weight": torch.ones(1, 2), "bias": torch.ones(1)}
    if fault == "missing":
        del values["bias"]
    elif fault == "extra":
        values["extra"] = torch.ones(1)
    elif fault == "shape":
        values["weight"] = torch.ones(1, 3)
    elif fault == "nan":
        values["weight"][0, 0] = float("nan")
    save_file(values, str(tmp_path / "model.safetensors"))

    class Discarding(TinyPI):
        def _convert_openpi_state_dict(self, state):
            return {"model.weight": state["weight"]}

    cls = Discarding if fault == "discarded" else TinyPI
    with pytest.raises((ValueError, RuntimeError)):
        load_pi05_strict(cls, SimpleNamespace(), tmp_path)


def test_area_image_transport_and_rejected_geometry():
    import cv2

    image = np.random.default_rng(1).integers(0, 256, (1200, 1600, 3), dtype=np.uint8)
    actual = model_image(image, (168, 224))
    np.testing.assert_array_equal(
        actual, cv2.resize(image, (224, 168), interpolation=cv2.INTER_AREA)
    )
    assert actual.flags.c_contiguous
    for resolution in ((224, 224), (2400, 3200)):
        with pytest.raises(ValueError):
            model_image(image, resolution)
    with pytest.raises(ValueError):
        model_image(image.astype(np.float32))


def test_openpi_source_inventory_detects_modified_file(tmp_path):
    import hashlib

    from lerobot_robot_onero.jax_policy import OPENPI_COMMIT

    source = tmp_path / "openpi"
    (source / "src/openpi").mkdir(parents=True)
    (source / "src/openpi/__init__.py").write_text("")
    (source / "LICENSE").write_text("license")
    data = {
        "commit": OPENPI_COMMIT,
        "files": {
            p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source.rglob("*")
            if p.is_file()
        },
    }
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(data))
    verify_openpi(source, inventory)
    data["files"]["LICENSE"] = "0" * 64
    broken = tmp_path / "inventory.json"
    broken.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_openpi(source, broken)


def test_jax_worker_eof_and_error_are_fatal():
    backend = object.__new__(JaxPolicy)
    backend.settings = SimpleNamespace(backend_timeout_s=0.01)
    backend._buffer = b'{"error":"bad params"}\n'
    with pytest.raises(RuntimeError, match="bad params"):
        backend._read()
    backend._buffer = b""
    read, write = __import__("os").pipe()
    __import__("os").close(write)
    with io.FileIO(read, "rb") as stream:
        backend._process = SimpleNamespace(stdout=stream)
        with pytest.raises(RuntimeError, match="exited"):
            backend._read()


def test_jax_contract_rejects_torch_directory(tmp_path):
    (tmp_path / "onero_openpi.json").write_text("{}")
    with pytest.raises(ValueError, match="model contract"):
        jax_contract(tmp_path)


def test_jax_request_survives_partial_pipe_writes():
    import base64
    import threading

    from lerobot_robot_onero.protocol import FEATURES

    class PartialPipe:
        def __init__(self):
            self.data = bytearray()

        def write(self, value):
            self.data.extend(value[:19])
            return min(len(value), 19)

        def flush(self):
            pass

    pipe = PartialPipe()
    backend = object.__new__(JaxPolicy)
    backend.settings = SimpleNamespace(image_resolution=(168, 224))
    backend._process = SimpleNamespace(stdin=pipe)
    backend._lock = threading.Lock()
    backend._read = lambda: {"actions": np.zeros((50, 16)).tolist()}
    raw = dict.fromkeys(FEATURES, 0.0)
    raw.update({k: np.zeros((168, 224, 3), dtype=np.uint8) for k in ("head", "left", "right")})
    assert backend.predict_chunk(raw).shape == (50, 16)
    request = json.loads(pipe.data)
    with np.load(io.BytesIO(base64.b64decode(request["observation"])), allow_pickle=False) as obs:
        assert set(obs.files) == {"state", "head", "left", "right"}
        assert obs["state"].shape == (16,)


def test_jax_architecture_profiles_use_identical_worker_and_source_inventory():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in (
        "onero_jax_worker/__init__.py",
        "onero_jax_worker/worker.py",
        "upstream-files.json",
    ):
        assert (root / "jax_runtime" / name).read_bytes() == (
            root / "jax_runtime_5090" / name
        ).read_bytes()
