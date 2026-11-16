"""Identity of the unmodified official source used by this adapter release."""

import importlib.metadata
import subprocess
from pathlib import Path

VERSION = "0.3.3"
LEROBOT_VERSION = "0.7.0"
LEROBOT_COMMIT = "410c095eacc0011d998a73383eb94abed2715af0"


def runtime_identity():
    import lerobot

    version = importlib.metadata.version("lerobot")
    if version != LEROBOT_VERSION:
        raise ValueError(f"expected LeRobot {LEROBOT_VERSION}, got {version}")
    source = Path(lerobot.__file__).resolve()
    checkout = source.parents[2]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(checkout), *args], text=True, stderr=subprocess.PIPE, timeout=10
        ).strip()

    if git("rev-parse", "--show-toplevel") != str(checkout):
        raise ValueError("LeRobot must be imported from the provided official checkout")
    if git("rev-parse", "HEAD") != LEROBOT_COMMIT:
        raise ValueError("LeRobot source commit does not match this adapter release")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("official LeRobot checkout contains tracked modifications")
    return {
        "lerobot_version": version,
        "lerobot_commit": LEROBOT_COMMIT,
        "lerobot_path": str(source),
        "adapter_version": VERSION,
    }
