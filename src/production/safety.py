from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class DiskAudit:
    path: str
    total_bytes: int
    free_bytes: int
    used_bytes: int

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            total_gib=round(self.total_bytes / GIB, 3),
            free_gib=round(self.free_bytes / GIB, 3),
            used_gib=round(self.used_bytes / GIB, 3),
        )
        return value


def audit_disk(path: str | Path) -> DiskAudit:
    target = Path(path).resolve()
    usage = shutil.disk_usage(target)
    return DiskAudit(
        path=str(target),
        total_bytes=int(usage.total),
        free_bytes=int(usage.free),
        used_bytes=int(usage.used),
    )


def evaluate_disk_guard(
    audit: DiskAudit,
    *,
    projected_growth_bytes: int,
    minimum_free_bytes: int,
    headroom_multiplier: float,
    maximum_projected_disk_fraction: float,
) -> dict[str, Any]:
    if projected_growth_bytes < 0:
        raise ValueError("projected growth cannot be negative")
    if minimum_free_bytes < 0 or headroom_multiplier < 1:
        raise ValueError("invalid disk safety threshold")
    required = max(
        minimum_free_bytes,
        int(projected_growth_bytes * headroom_multiplier),
    )
    free_after = audit.free_bytes - projected_growth_bytes
    projected_fraction = (audit.used_bytes + projected_growth_bytes) / max(
        audit.total_bytes, 1
    )
    allowed = (
        audit.free_bytes >= required
        and free_after >= minimum_free_bytes
        and projected_fraction <= maximum_projected_disk_fraction
    )
    return {
        "allowed": allowed,
        "projected_growth_bytes": projected_growth_bytes,
        "required_free_before_start_bytes": required,
        "minimum_free_after_stage_bytes": minimum_free_bytes,
        "free_before_bytes": audit.free_bytes,
        "projected_free_after_bytes": free_after,
        "projected_disk_fraction": round(projected_fraction, 6),
        "maximum_projected_disk_fraction": maximum_projected_disk_fraction,
    }


def validate_execution_gate(
    config: dict[str, Any],
    *,
    stage: str,
    max_records: int | None,
    bounded_confirmation: str | None,
    full: bool,
    full_confirmation: str | None,
) -> None:
    stages = config["production_stages"]
    if stage not in stages:
        raise ValueError(f"unknown production stage: {stage}")
    safety = config["safety"]
    if full:
        if not bool(safety["unrestricted_enabled"]):
            raise ValueError("unrestricted production is disabled by the capacity gate")
        if full_confirmation != safety["full_confirmation"]:
            raise ValueError("full production requires the exact confirmation token")
        if max_records is not None:
            raise ValueError("full production cannot also specify a bounded record count")
        return
    if max_records is None or max_records <= 0:
        raise ValueError("bounded production requires a positive --max-records")
    if bounded_confirmation != safety["bounded_confirmation"]:
        raise ValueError("bounded production requires the exact confirmation token")
    if stage == "8A" and max_records > int(safety["maximum_bounded_crawl_records"]):
        raise ValueError("bounded crawl exceeds the configured Phase 8 ceiling")
    if stage in {"8C", "8D", "8E", "8F"}:
        raise ValueError(f"{stage} remains locked until prior production gates pass")
