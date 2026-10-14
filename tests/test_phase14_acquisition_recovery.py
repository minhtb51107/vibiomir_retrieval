from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
import yaml

from src.ingestion.checkpoint import CheckpointStore
from src.ingestion.models import CrawlResult, CrawlStatus
from src.source_census.phase14 import (
    PHASE14_NAME,
    acquisition_manifest,
    acquisition_progress,
    crawl_checkpoint_audit,
    load_acquisition_config,
)
from src.source_census.pipeline import load_config as load_legacy_config


def _config(tmp_path: Path) -> Path:
    records = [{"doc_id": value, "url": f"https://medlatec.vn/{value}"} for value in range(1, 3001)]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "experiment": "phase14_medlatec_coverage_3000", "total_incremental_ids": 3000,
        "sources": {"medlatec.vn": {"incremental_records": records}},
    }), encoding="utf-8")
    import hashlib
    digest = hashlib.sha256("\n".join(map(str, range(1, 3001))).encode()).hexdigest()
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"selected_id_sha256": digest}), encoding="utf-8")
    config = {
        "phase": PHASE14_NAME, "run_id": "phase14_medlatec_coverage_3000",
        "focused_scaling": {"phase_key": "phase14_medlatec"},
        "phase14_medlatec": {"manifest": str(manifest), "preflight": str(preflight),
                              "sampling_seed": "vibiomir-phase14-medlatec-coverage-v1",
                              "maximum_new_ids": 3000},
        "acquisition": {"minimum_free_gib": 20, "global_concurrency": 1},
        "models": {"reranker": {"batch_size": 2}},
    }
    path = tmp_path / "phase14.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_phase14_adapter_accepts_its_contract_while_legacy_loader_remains_closed(tmp_path):
    path = _config(tmp_path)
    assert load_acquisition_config(path)["phase"] == PHASE14_NAME
    with pytest.raises(ValueError, match="unexpected source-census configuration"):
        load_legacy_config(path)


def test_phase14_manifest_preserves_exact_selection_and_hash(tmp_path):
    config = load_acquisition_config(_config(tmp_path))
    _, records, digest = acquisition_manifest(config)
    assert len(records) == len({row["doc_id"] for row in records}) == 3000
    assert records[0] == {"doc_id": 1, "original_url": "https://medlatec.vn/1"}
    assert len(digest) == 64


def test_checkpoint_audit_and_resume_preserve_completed_rows(tmp_path):
    records = [{"doc_id": value, "original_url": f"https://medlatec.vn/{value}"} for value in (1, 2, 3)]
    database = tmp_path / "crawl.sqlite"
    with CheckpointStore(database) as store:
        store.save(CrawlResult(1, records[0]["original_url"], records[0]["original_url"],
                               status=CrawlStatus.SUCCESS, attempt_count=1))
        store.save(CrawlResult(2, records[1]["original_url"], records[1]["original_url"],
                               status=CrawlStatus.HTTP_ERROR, attempt_count=1))
    before = crawl_checkpoint_audit(database, records)
    assert before == {"passed": True, "integrity": "ok", "rows": 2, "success": 1,
                      "status_counts": {"SUCCESS": 1, "HTTP_ERROR": 1}, "remaining": 1,
                      "unexpected_ids": 0, "url_mismatches": 0}
    with CheckpointStore(database) as resumed:
        assert resumed.completed_doc_ids() == {1, 2}
    after = crawl_checkpoint_audit(database, records)
    assert after["rows"] == 2
    assert sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True).execute(
        "PRAGMA integrity_check").fetchone()[0] == "ok"


def test_stage_progress_exposes_both_monitor_contracts(tmp_path):
    state = acquisition_progress(3000, 856, rate=59.0, database=tmp_path / "crawl.sqlite")
    assert (state["selected"], state["completed"]) == (3000, 856)
    assert (state["total"], state["done"], state["remaining"]) == (3000, 856, 2144)
    assert state["progress"] == {"stage": "ACQUIRE", "total": 3000, "done": 856, "remaining": 2144}
