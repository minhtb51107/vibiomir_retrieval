"""Shared, low-overhead progress contract for future supervised pipelines."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Sequence


INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True)
class StageProgress:
    stage: str
    current_operation: str
    done: int | None
    total: int | None
    started_at: dt.datetime
    heartbeat_at: dt.datetime
    worker_alive: bool


def _percent(done: int | None, total: int | None) -> float | str:
    if done is None or total is None or total <= 0:
        return INDETERMINATE
    if done < 0 or done > total:
        raise ValueError(f"invalid progress counter: done={done} total={total}")
    return round(100.0 * done / total, 2)


def progress_snapshot(
    progress: StageProgress,
    *,
    ordered_stages: Sequence[str],
    terminal: bool = False,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    """Normalize state without inventing progress or ETA.

    Overall progress is the fraction of ordered stage gates completed plus the
    measurable fraction of the current gate. If the current stage has no exact
    denominator, both the current fraction contribution and ETA are explicitly
    indeterminate. Only a terminal state may publish overall 100%.
    """
    if progress.stage not in ordered_stages:
        raise ValueError(f"unknown stage {progress.stage!r}")
    now = now or dt.datetime.now(dt.timezone.utc)
    stage_percent = _percent(progress.done, progress.total)
    stage_index = ordered_stages.index(progress.stage)
    if terminal:
        overall_percent: float | str = 100.0
    elif stage_percent == INDETERMINATE:
        overall_percent = INDETERMINATE
    else:
        overall_percent = round(100.0 * (stage_index + float(stage_percent) / 100.0) / len(ordered_stages), 2)
        overall_percent = min(overall_percent, 99.99)
    elapsed = max(0.0, (now - progress.started_at).total_seconds())
    eta: float | str = INDETERMINATE
    if progress.done and progress.total and 0 < progress.done < progress.total:
        eta = round(elapsed / progress.done * (progress.total - progress.done), 1)
    return {
        "stage": progress.stage,
        "current_operation": progress.current_operation,
        "done": progress.done,
        "total": progress.total,
        "stage_percent": stage_percent,
        "overall_percent": overall_percent,
        "overall_basis": "ordered_stage_gate_fraction",
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": eta,
        "heartbeat_at": progress.heartbeat_at.isoformat(),
        "worker_alive": progress.worker_alive,
    }


PHASE15A_STAGES = (
    "CONTROL_REPLAY",
    "DOC_TOP2_MEAN",
    "CHUNK_DIVERSITY",
    "VALIDATION",
    "REVIEW_PREPARATION",
    "PACKAGING",
)
