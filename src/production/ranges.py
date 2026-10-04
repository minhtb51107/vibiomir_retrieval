from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class RowRange:
    range_id: str
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start

    def as_dict(self) -> dict[str, int | str]:
        return {**asdict(self), "count": self.count}


def plan_row_ranges(total_rows: int, rows_per_range: int) -> list[RowRange]:
    if total_rows < 0 or rows_per_range <= 0:
        raise ValueError("invalid row range size")
    output = []
    for start in range(0, total_rows, rows_per_range):
        stop = min(start + rows_per_range, total_rows)
        signature = hashlib.sha256(f"{total_rows}:{start}:{stop}".encode()).hexdigest()[:16]
        output.append(RowRange(f"rows-{start:09d}-{stop:09d}-{signature}", start, stop))
    return output
