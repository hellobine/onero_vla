"""Customer entrypoint. Check/preflight are receive-only; motion is explicit."""

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from .protocol import FEATURES
from .release import runtime_identity
from .settings import configure_environment, load_settings


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str))


def readonly_observation(settings):
    from .robot import OneroRobot

    robot = OneroRobot(dataclasses.replace(settings.robot, read_only=True))
    try:
        robot.connect()
        return robot.get_observation()
    finally:
        robot.disconnect()


def preflight(settings, engine, output):
    import numpy as np

    if output.resolve().is_relative_to(settings.model_path):
        raise ValueError("preflight output must be outside the model directory")
    output.mkdir(parents=True, exist_ok=False)
    result = {"ok": False, "motion_attempted": False, "executed_actions": False}
    try:
        raw = readonly_observation(settings)
        np.savez_compressed(output / "observation.npz", **raw)
        start = time.monotonic()
        from .action_runtime import preview_processing

        predicted, processed = preview_processing(engine, raw, settings)
        elapsed = time.monotonic() - start
        np.save(output / "predicted_action.npy", processed[0])
        np.save(output / "raw_actions.npy", predicted)
        np.save(output / "processed_actions.npy", processed)
        result.update(
            ok=True,
            action_shape=[16],
            inference_s=elapsed,
            feature_order=list(FEATURES),
            model=engine.identity,
            processing=dataclasses.asdict(settings.processing),
            preview_steps=len(processed),
            **runtime_identity(),
        )
        return result
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


def _execute(settings, engine, duration, *, motion):
    from .action_runtime import ActionRuntime
    from .robot import OneroRobot

    # Load/validate/prime the model before creating motion publishers.
    if settings.processing.chunked:
        engine.validate_chunk_mode()
        engine.predict_chunk(readonly_observation(settings))
    else:
        engine.select(readonly_observation(settings))
    engine.reset()
    robot = OneroRobot(dataclasses.replace(settings.robot, read_only=not motion))
    runtime = None
    sent = 0
    preview = []
    try:
        robot.connect()
        runtime = ActionRuntime(engine, settings)
        epoch = time.monotonic()
        end = float("inf") if duration is None else epoch + duration
        last_step = -1
        last_action_time = None
        while time.monotonic() < end:
            now = time.monotonic()
            step = int((now - epoch) * settings.fps)
            if step <= last_step:
                time.sleep(max(0, min(end, epoch + (last_step + 1) / settings.fps) - now))
                continue
            observation = robot.get_observation()
            captured = time.monotonic()
            step = int((captured - epoch) * settings.fps)
            action = runtime.tick(observation, step, captured)
            last_step = step
            if time.monotonic() >= end:
                break
            if action is not None:
                if motion:
                    robot.send_action(action)
                    sent += 1
                else:
                    preview.append([action[k] for k in FEATURES])
                last_action_time = time.monotonic()
            elif last_action_time is not None:
                if time.monotonic() - last_action_time > settings.robot.command_timeout_s:
                    raise TimeoutError("action queue starved; dispatch remains disabled")
            elif time.monotonic() - epoch > settings.processing.max_chunk_age_s:
                raise TimeoutError("no valid future action arrived before the startup deadline")
    finally:
        try:
            if runtime is not None:
                runtime.close()
        finally:
            robot.disconnect()
    if not sent and not preview:
        raise RuntimeError("execution window ended without a valid action")
    return {
        "ok": True,
        "sent_actions": sent,
        "preview_actions": len(preview),
        "physical_stop_verified": False,
        "motion_attempted": motion,
        "executed_actions": bool(sent),
        "processing": dataclasses.asdict(settings.processing),
    }, preview


def run(settings, duration=None):
    from .jax_policy import create_policy

    engine = create_policy(settings)
    try:
        if engine.contract.get("test_only", False):
            raise ValueError("test-only checkpoint cannot be used for motion")
        result, _preview = _execute(settings, engine, duration, motion=True)
        return result
    finally:
        if hasattr(engine, "close"):
            engine.close()


def shadow(settings, duration, output):
    import numpy as np

    from .jax_policy import create_policy

    if output.resolve().is_relative_to(settings.model_path):
        raise ValueError("shadow output must be outside the model directory")
    output.mkdir(parents=True, exist_ok=False)
    result = {"ok": False, "motion_attempted": False, "executed_actions": False}
    engine = None
    try:
        engine = create_policy(settings)
        result, actions = _execute(settings, engine, duration, motion=False)
        np.save(output / "processed_actions.npy", np.asarray(actions))
        return result
    except BaseException as exc:
        result["error"] = str(exc)
        raise
    finally:
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="A1/H1 offline LeRobot inference")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("config", "check", "preflight", "shadow", "run"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", required=True, type=Path)
        cmd.add_argument(
            "--inference-mode", choices=("sync", "temporal_smoothing", "kalman_filter")
        )
        cmd.add_argument(
            "--kalman-temporal-smoothing", action=argparse.BooleanOptionalAction, default=None
        )
        if name == "check":
            cmd.add_argument("--no-hardware", action="store_true")
        if name in ("preflight", "shadow"):
            cmd.add_argument("--output", required=True, type=Path)
        if name in ("run", "shadow"):
            cmd.add_argument(
                "--duration",
                type=float,
                help=(
                    "Optional time limit in seconds; run continues until Ctrl+C by default."
                    if name == "run"
                    else "Preview duration in seconds; defaults to runtime.duration_s."
                ),
            )
        if name == "run":
            cmd.add_argument("--enable-motion", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "run" and not args.enable_motion:
            raise ValueError("run requires the operator's explicit --enable-motion")
        settings = load_settings(args.config)
        overrides = {
            k: v
            for k, v in {
                "inference_mode": args.inference_mode,
                "kalman_temporal_smoothing": args.kalman_temporal_smoothing,
            }.items()
            if v is not None
        }
        settings = dataclasses.replace(
            settings, processing=dataclasses.replace(settings.processing, **overrides)
        )
        configure_environment(settings)
        if args.command != "config":
            runtime_identity()
        if args.command == "config":
            emit(dataclasses.asdict(settings) | {"model_path": str(settings.model_path)})
        elif args.command == "check":
            from .policy import model_contract

            if settings.backend == "openpi_jax":
                from .jax_policy import jax_contract

                jax_contract(settings.model_path)
                metadata = {"type": "openpi_jax_pi05"}
            else:
                metadata, _contract = model_contract(settings.model_path)
            result = {"ok": True, "policy_type": metadata["type"], **runtime_identity()}
            if not args.no_hardware:
                observation = readonly_observation(settings)
                result["cameras"] = {k: list(observation[k].shape) for k in settings.robot.cameras}
            result["hardware_checked"] = not args.no_hardware
            result["motion_attempted"] = False
            emit(result)
        elif args.command == "preflight":
            from .jax_policy import create_policy

            with redirect_stdout(sys.stderr):
                engine = create_policy(settings)
                try:
                    result = preflight(settings, engine, args.output)
                finally:
                    if hasattr(engine, "close"):
                        engine.close()
            result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
            (args.output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            emit(result)
        else:
            from .config import positive

            duration = (
                None
                if args.command == "run" and args.duration is None
                else positive(
                    settings.duration_s if args.duration is None else args.duration, "duration"
                )
            )
            with redirect_stdout(sys.stderr):
                result = (
                    shadow(settings, duration, args.output)
                    if args.command == "shadow"
                    else run(settings, duration)
                )
            emit(result)
    except KeyboardInterrupt:
        emit({"ok": False, "stopped_by_operator": True, "physical_stop_verified": False})
        return 130
    except Exception as exc:
        emit({"ok": False, "error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
