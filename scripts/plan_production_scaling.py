#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.production.capacity import build_capacity_plan
from src.production.config import load_production_config
from src.production.safety import audit_disk


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 8 capacity and safety plan.")
    parser.add_argument("--config", default="configs/production_pipeline.yaml")
    args = parser.parse_args()
    config = load_production_config(args.config)
    artifacts = Path(config["paths"]["artifacts"])
    c_drive = audit_disk("C:/")
    d_drive = audit_disk(REPOSITORY_ROOT)
    capacity, storage, gate = build_capacity_plan(
        config, c_drive=c_drive, d_drive=d_drive
    )
    _atomic_json(artifacts / "capacity_plan.json", capacity)
    _atomic_json(artifacts / "storage_projection.json", storage)
    _atomic_json(artifacts / "production_gate.json", gate)
    print(json.dumps({"capacity": capacity, "gate": gate}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
