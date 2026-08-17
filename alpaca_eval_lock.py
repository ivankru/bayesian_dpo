"""Shared .eval.lock helpers for alpaca eval scripts."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import IO, Optional, Union


def _read_lock_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _process_cmdline(pid: int) -> Optional[str]:
    proc = Path(f"/proc/{pid}/cmdline")
    if not proc.is_file():
        return None
    try:
        return proc.read_bytes().replace(b"\0", b" ").decode("utf-8", errors="replace")
    except OSError:
        return None


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return Path(f"/proc/{pid}").exists()


def _pid_holds_alpaca_eval_lock(pid: int) -> bool:
    """True if pid is alive and running alpaca_eval_judge or ifeval_run."""
    if not _is_pid_alive(pid):
        return False
    cmdline = _process_cmdline(pid)
    if cmdline is None:
        return False
    return "alpaca_eval_judge.py" in cmdline or "ifeval_run.py" in cmdline


def clear_stale_eval_lock(
    lock_path: os.PathLike | str,
    out_dir: Optional[os.PathLike | str] = None,
) -> bool:
    """
    Remove .eval.lock if no live alpaca_eval_judge owns it.

    Returns True if a live judge still holds the lock (do not start).
    out_dir is ignored (kept for call-site compatibility).
    """
    _ = out_dir
    path = Path(lock_path)
    if not path.is_file():
        return False
    pid = _read_lock_pid(path)
    if pid is None:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    if _pid_holds_alpaca_eval_lock(pid):
        return True
    try:
        path.unlink()
    except OSError:
        pass
    return False


def _pid_holds_batch_lock(pid: int) -> bool:
    """True if pid is alive and running a batch launcher script."""
    if not _is_pid_alive(pid):
        return False
    cmdline = _process_cmdline(pid)
    if cmdline is None:
        return False
    markers = (
        "run_ifeval_batch.sh",
        "run_alpaca2_batch.sh",
        "ifeval_batch.py",
        "alpaca_eval_batch.py",
    )
    return any(m in cmdline for m in markers)


def clear_stale_batch_lock(lock_path: os.PathLike | str) -> bool:
    """
    Remove batch lock if the owning process is gone.

    Returns True if a live batch still holds the lock.
    """
    path = Path(lock_path)
    if not path.is_file():
        return False
    pid = _read_lock_pid(path)
    if pid is None:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    if _pid_holds_batch_lock(pid):
        return True
    try:
        path.unlink()
    except OSError:
        pass
    return False


def acquire_batch_lock(lock_path: os.PathLike | str) -> bool:
    """Write this process pid to lock file. Returns False if another live batch runs."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and clear_stale_batch_lock(path):
        return False
    path.write_text(str(os.getpid()), encoding="utf-8")
    return True


def release_batch_lock(lock_path: os.PathLike | str) -> None:
    path = Path(lock_path)
    if not path.is_file():
        return
    pid = _read_lock_pid(path)
    if pid == os.getpid():
        try:
            path.unlink()
        except OSError:
            pass


def clear_stale_eval_locks_in_tree(root: os.PathLike | str) -> int:
    """Remove stale .eval.lock files under root. Returns count removed."""
    removed = 0
    root_path = Path(root)
    if not root_path.is_dir():
        return removed
    for lock_path in root_path.rglob(".eval.lock"):
        if not clear_stale_eval_lock(lock_path):
            removed += 1
    return removed


def count_live_eval_locks(root: os.PathLike | str) -> int:
    """Return number of output dirs with a live alpaca_eval_judge lock."""
    root_path = Path(root)
    if not root_path.is_dir():
        return 0
    live = 0
    for lock_path in root_path.rglob(".eval.lock"):
        if clear_stale_eval_lock(lock_path):
            live += 1
    return live


def try_acquire_eval_lock(
    lock_path: os.PathLike | str,
    out_dir: Optional[os.PathLike | str] = None,
) -> Optional[IO]:
    """
    Atomically acquire .eval.lock via flock.

    Returns an open file handle (keep open until release_eval_lock) or None if
    another live judge holds the lock.
    """
    _ = out_dir
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and clear_stale_eval_lock(path):
        return None

    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        if clear_stale_eval_lock(path):
            return None
        fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None

    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    os.fsync(fd)
    return os.fdopen(fd, "w")


def release_eval_lock(
    lock_file: Optional[Union[IO, int]],
    lock_path: Optional[os.PathLike | str] = None,
) -> None:
    if lock_file is None:
        return
    try:
        if isinstance(lock_file, int):
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            os.close(lock_file)
        else:
            fd = lock_file.fileno()
            fcntl.flock(fd, fcntl.LOCK_UN)
            lock_file.close()
    except OSError:
        pass
    if lock_path is not None:
        try:
            Path(lock_path).unlink(missing_ok=True)
        except OSError:
            pass
