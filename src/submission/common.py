from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any, *, trailing_newline: bool = True) -> bytes:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return payload + (b"\n" if trailing_newline else b"")


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def atomic_write_json(path: str | Path, value: Any, *, pretty: bool = True) -> None:
    if pretty:
        payload = (
            json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n"
        ).encode("utf-8")
    else:
        payload = canonical_json_bytes(value)
    atomic_write_bytes(path, payload)


def write_submission_stream(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Atomically write a compact UTF-8 JSON array without materializing it twice."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(b"[")
            first = True
            for record in records:
                if not first:
                    handle.write(b",")
                handle.write(canonical_json_bytes(record, trailing_newline=False))
                first = False
            handle.write(b"]\n")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_deterministic_zip(json_path: str | Path, zip_path: str | Path) -> None:
    """Create a byte-stable ZIP containing exactly one root-level JSON file."""
    source = Path(json_path)
    destination = Path(zip_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    info = zipfile.ZipInfo(source.name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            with archive.open(info, mode="w", force_zip64=True) as target:
                with source.open("rb") as source_handle:
                    shutil.copyfileobj(source_handle, target, length=1024 * 1024)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
