from __future__ import annotations

import json
import os
import random
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


TERMINAL_STATUSES = {
    "FAILED_SAFE", "NEEDS_AGENT", "READY_FOR_LEADERBOARD",
    "READY_FOR_HUMAN_REVIEW", "WAITING_FOR_LEADERBOARD",
    "AWAITING_COLAB_EMBEDDING_EQUIVALENCE",
}


@contextmanager
def _state_lock(path: Path, *, attempts: int = 80) -> Iterator[None]:
    """Serialize writers with an OS lock that is released if a process dies."""
    lock = path.with_name(f".{path.name}.lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    handle = lock.open("a+b")
    handle.seek(0, os.SEEK_END)
    if handle.tell() == 0:
        handle.write(b"0"); handle.flush()
    acquired = False
    try:
        for attempt in range(attempts):
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                time.sleep(min(0.01 * (attempt + 1), 0.2) + random.uniform(0, 0.01))
        if not acquired:
            raise TimeoutError(f"state lock remained busy: {lock}")
        yield
    finally:
        try:
            if acquired:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _replace_json(path: Path, value: Mapping[str, Any], *, attempts: int = 40) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
    )
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(0.01 * (attempt + 1), 0.2) + random.uniform(0, 0.01))
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Replace a JSON state atomically under the shared cross-process lock."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(value)
    with _state_lock(destination):
        _replace_json(destination, payload)
    return payload


def update_json(
    path: str | Path,
    updates: Mapping[str, Any],
    *,
    preserve_terminal: bool = True,
) -> dict[str, Any]:
    """Read/merge/write while locked so a stale writer cannot erase progress.

    A RUNNING heartbeat is not allowed to overwrite a terminal status belonging
    to the same started run. Starting a new supervisor uses ``write_json`` and is
    therefore an explicit lifecycle reset.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with _state_lock(destination):
        current: dict[str, Any] = {}
        if destination.exists():
            current = json.loads(destination.read_text(encoding="utf-8"))
        payload = {**current, **dict(updates)}
        if (
            preserve_terminal
            and current.get("status") in TERMINAL_STATUSES
            and updates.get("status") == "RUNNING"
        ):
            payload["status"] = current["status"]
            if "stage" in current:
                payload["stage"] = current["stage"]
        _replace_json(destination, payload)
    return payload
