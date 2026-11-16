"""Customer YAML, with duplicate/unknown keys rejected before device access."""

import dataclasses
import json
import os
from pathlib import Path

import yaml

from .config import OneroRobotConfig, OneroRosCameraConfig, positive
from .smoothing import ProcessingConfig


class StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def keys(value, allowed, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping")
    if set(value) - set(allowed):
        raise ValueError(f"unknown {where} keys: {sorted(set(value) - set(allowed))}")


@dataclasses.dataclass(frozen=True)
class Settings:
    robot: OneroRobotConfig
    model_path: Path
    device: str = "cpu"
    task: str = ""
    fps: float = 30.0
    duration_s: float = 60.0
    rmw: str = "rmw_cyclonedds_cpp"
    zenoh_router: str | None = None
    processing: ProcessingConfig = dataclasses.field(default_factory=ProcessingConfig)
    backend: str = "lerobot_torch"
    image_resolution: tuple[int, int] | None = None
    tokenizer_path: Path | None = None
    compile_model: bool | None = None
    jax_python: Path | None = None
    openpi_source: Path | None = None
    backend_timeout_s: float = 900.0


def load_settings(path: Path) -> Settings:
    path = path.resolve(strict=True)
    data = yaml.load(path.read_text(), Loader=StrictLoader)
    keys(data, {"robot", "model", "runtime", "ros"}, "root")
    robot = dict(data.get("robot", {}))
    keys(robot, {f.name for f in dataclasses.fields(OneroRobotConfig)} - {"read_only"}, "robot")
    if "cameras" in robot:
        keys(robot["cameras"], {"head", "left", "right"}, "cameras")
        cameras = {}
        for name, config in robot["cameras"].items():
            keys(
                config, {f.name for f in dataclasses.fields(OneroRosCameraConfig)}, f"camera {name}"
            )
            cameras[name] = OneroRosCameraConfig(**config)
        robot["cameras"] = cameras
    # Motion is an execution-time choice, never a YAML setting.
    robot_config = OneroRobotConfig(**robot, read_only=True)
    model = data.get("model", {})
    keys(
        model,
        {
            "path",
            "device",
            "task",
            "backend",
            "image_resolution",
            "tokenizer_path",
            "compile_model",
            "jax_python",
            "openpi_source",
            "backend_timeout_s",
        },
        "model",
    )
    if not isinstance(model.get("path"), str) or not model["path"]:
        raise ValueError("model.path must name a local model directory")
    model_path = Path(model["path"]).expanduser()
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    device = model.get("device", "cpu")
    if not isinstance(device, str) or not (
        device in ("cpu", "mps", "cuda") or (device.startswith("cuda:") and device[5:].isdigit())
    ):
        raise ValueError("model.device must be cpu, mps or cuda[:index]")
    if not isinstance(model.get("task", ""), str):
        raise ValueError("model.task must be a string")
    backend = model.get("backend", "lerobot_torch")
    if backend not in ("lerobot_torch", "openpi_jax"):
        raise ValueError("model.backend must be lerobot_torch or openpi_jax")
    resolution = model.get("image_resolution")
    if resolution is not None:
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(type(v) is not int or v <= 0 for v in resolution)
        ):
            raise ValueError("image_resolution must be [height,width] positive integers")
        resolution = tuple(resolution)
    compile_model = model.get("compile_model")
    if compile_model is not None and type(compile_model) is not bool:
        raise ValueError("compile_model must be boolean")
    backend_timeout = positive(model.get("backend_timeout_s", 900), "backend_timeout_s")

    def local_path(name):
        value = model.get(name)
        if value is None:
            return None
        if not isinstance(value, str) or not value:
            raise ValueError(f"model.{name} must be a local path")
        result = Path(value).expanduser()
        # Keep the venv executable's path: resolving its symlink selects system Python.
        return (result if result.is_absolute() else path.parent / result).absolute()

    tokenizer_path, jax_python, openpi_source = (
        local_path(n) for n in ("tokenizer_path", "jax_python", "openpi_source")
    )
    if backend == "openpi_jax" and (jax_python is None or openpi_source is None):
        raise ValueError("openpi_jax requires model.jax_python and model.openpi_source")
    runtime = data.get("runtime", {})
    processing_keys = {f.name for f in dataclasses.fields(ProcessingConfig)}
    keys(runtime, {"fps", "duration_s", *processing_keys}, "runtime")
    processing = ProcessingConfig(**{k: v for k, v in runtime.items() if k in processing_keys})
    fps = positive(runtime.get("fps", 30), "fps")
    if robot_config.command_timeout_s < 1 / fps:
        raise ValueError("command_timeout_s must cover at least one control period")
    duration = positive(runtime.get("duration_s", 60), "duration_s")
    ros = data.get("ros", {})
    keys(ros, {"rmw_implementation", "zenoh_router"}, "ros")
    rmw = ros.get("rmw_implementation", "rmw_cyclonedds_cpp")
    if rmw not in ("rmw_cyclonedds_cpp", "rmw_zenoh_cpp"):
        raise ValueError("unsupported RMW implementation")
    router = ros.get("zenoh_router")
    if rmw == "rmw_zenoh_cpp":
        if not isinstance(router, str) or not router.startswith("tcp/"):
            raise ValueError("Zenoh requires an explicit tcp/address:port endpoint")
        if any(c.isspace() for c in router) or any(c in router for c in (";", '"', "[")):
            raise ValueError("invalid Zenoh endpoint")
    elif router is not None:
        raise ValueError("zenoh_router requires rmw_zenoh_cpp")
    return Settings(
        robot_config,
        model_path.resolve(),
        device,
        model.get("task", ""),
        fps,
        duration,
        rmw,
        router,
        processing,
        backend,
        resolution,
        tokenizer_path,
        compile_model,
        jax_python,
        openpi_source,
        backend_timeout,
    )


def configure_environment(settings):
    os.environ["RMW_IMPLEMENTATION"] = settings.rmw
    if settings.zenoh_router:
        os.environ["ZENOH_CONFIG_OVERRIDE"] = (
            f'mode="client";connect/endpoints={json.dumps([settings.zenoh_router])}'
        )
    else:
        os.environ.pop("ZENOH_CONFIG_OVERRIDE", None)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[name] = "1"
