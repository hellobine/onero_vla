"""Same-user advisory ownership for cooperating local control processes."""

import fcntl
import hashlib
import os
import stat
import tempfile
from pathlib import Path


class MotionLease:
    def __init__(self, domain, topics):
        self.keys = [hashlib.sha256(f"{domain}:{t}".encode()).hexdigest() for t in sorted(topics)]
        self.fds = []

    def acquire(self):
        if self.fds:
            raise RuntimeError("motion lease already held; finish teardown first")
        root = Path(tempfile.gettempdir()) / f"onero-motion-{os.getuid()}"
        root.mkdir(mode=0o700, exist_ok=True)
        info = root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise PermissionError("motion lock directory must be private and owned by this user")
        try:
            for key in self.keys:
                fd = os.open(root / key, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
                self.fds.append(fd)
                info = os.fstat(fd)
                if (
                    not stat.S_ISREG(info.st_mode)
                    or info.st_uid != os.getuid()
                    or info.st_nlink != 1
                ):
                    raise PermissionError("invalid motion lock file")
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.release()
            raise

    def release(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()
