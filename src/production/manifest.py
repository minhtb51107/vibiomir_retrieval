from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class StageManifest:
    stage: str
    source_sha256: str
    config_sha256: str
    stage_version: int
    selected_count: int
    row_start: int | None = None
    row_end: int | None = None
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    status: str = "RUNNING"
    completed_count: int = 0
    failure_count: int = 0
    resume_count: int = 0
    output_hashes: dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    completed_at: str | None = None
    run_id: str = ""

    def __post_init__(self) -> None:
        if self.selected_count < 0 or self.completed_count < 0:
            raise ValueError("manifest counts cannot be negative")
        if self.completed_count + self.failure_count > self.selected_count:
            raise ValueError("manifest progress exceeds selected count")
        if not self.run_id:
            signature = {
                "stage": self.stage,
                "source_sha256": self.source_sha256,
                "config_sha256": self.config_sha256,
                "stage_version": self.stage_version,
                "selected_count": self.selected_count,
                "row_start": self.row_start,
                "row_end": self.row_end,
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
            }
            self.run_id = hashlib.sha256(_canonical(signature).encode()).hexdigest()[:24]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ManifestStore:
    def __init__(self, path: str | Path, expected: StageManifest):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            stored = StageManifest(**json.loads(self.path.read_text(encoding="utf-8")))
            if stored.run_id != expected.run_id:
                raise ValueError("stage manifest signature mismatch")
            stored.resume_count += 1
            stored.updated_at = _now()
            self.manifest = stored
            self._write()
        else:
            self.manifest = expected
            self._write()

    def record_progress(self, *, completed_count: int, failure_count: int) -> None:
        if completed_count < self.manifest.completed_count:
            raise ValueError("completed count cannot move backwards")
        if failure_count < self.manifest.failure_count:
            raise ValueError("failure count cannot move backwards")
        if completed_count + failure_count > self.manifest.selected_count:
            raise ValueError("manifest progress exceeds selected count")
        self.manifest.completed_count = completed_count
        self.manifest.failure_count = failure_count
        self.manifest.updated_at = _now()
        self._write()

    def complete(self, output_hashes: dict[str, str]) -> None:
        if self.manifest.completed_count + self.manifest.failure_count != self.manifest.selected_count:
            raise ValueError("cannot complete a stage with unaccounted records")
        self.manifest.status = "COMPLETE"
        self.manifest.output_hashes = dict(sorted(output_hashes.items()))
        self.manifest.updated_at = _now()
        self.manifest.completed_at = self.manifest.updated_at
        self._write()

    def fail(self) -> None:
        self.manifest.status = "FAILED"
        self.manifest.updated_at = _now()
        self._write()

    def _write(self) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.manifest.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
