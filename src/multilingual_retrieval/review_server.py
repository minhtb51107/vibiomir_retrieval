from __future__ import annotations

import csv
import json
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import yaml

from .evaluation import LABELS, VERIFIED, render_review


class HumanLabelStore:
    def __init__(self, config_path: Path, split: str) -> None:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        root = config_path.resolve().parents[1]
        self.label_path = root / config["output_dir"] / "human_labels.csv"
        self.split = split
        self._lock = threading.Lock()

    def _read(self) -> tuple[list[str], list[dict[str, str]]]:
        with self.label_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames or []
        if not fields or not rows:
            raise ValueError("human label contract is empty")
        return fields, rows

    @staticmethod
    def _atomic_write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader(); writer.writerows(rows)
                stream.flush(); os.fsync(stream.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, path)
                    return
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (2 ** attempt))
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        reviewer = str(payload.get("reviewer") or "").strip()
        judgments = payload.get("judgments")
        if not reviewer:
            raise ValueError("reviewer is required")
        if not isinstance(judgments, dict):
            raise ValueError("judgments must be an object keyed by case_id")
        with self._lock:
            fields, rows = self._read()
            active = {row["case_id"]: row for row in rows if row["split"] == self.split}
            unknown = set(judgments) - active.keys()
            if unknown:
                raise ValueError(f"unknown or non-{self.split} case IDs: {len(unknown)}")
            saved = 0; timestamp = datetime.now(timezone.utc).isoformat()
            for case_id, value in judgments.items():
                if not isinstance(value, dict):
                    raise ValueError(f"invalid judgment payload for {case_id}")
                judgment = str(value.get("judgment") or "").strip()
                if not judgment:
                    continue
                if judgment not in LABELS:
                    raise ValueError(f"invalid judgment {judgment!r} for {case_id}")
                row = active[case_id]
                row["judgment"] = judgment
                row["review_notes"] = str(value.get("review_notes") or "").strip()
                row["verification_status"] = VERIFIED
                row["reviewer"] = reviewer
                row["verified_at"] = timestamp
                saved += 1
            self._atomic_write(self.label_path, fields, rows)
        return {"saved": saved, "split": self.split, "path": str(self.label_path)}

    def status(self) -> dict[str, Any]:
        _, rows = self._read()
        active = [row for row in rows if row["split"] == self.split]
        verified = [row for row in active if row["verification_status"] == VERIFIED]
        return {"split": self.split, "rows": len(active), "verified": len(verified), "remaining": len(active) - len(verified)}


def serve_review(config_path: Path, split: str, host: str, port: int) -> None:
    review = render_review(config_path, split)
    html_path = Path(review["path"])
    store = HumanLabelStore(config_path, split)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: dict[str, Any]) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in {"/", "/index.html"}:
                body = html_path.read_bytes()
                self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            elif self.path == "/api/status":
                self._json(200, store.status())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/save":
                self._json(404, {"error": "not found"}); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                self._json(200, store.save(payload))
            except (ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(json.dumps({"status": "REVIEW_SERVER_READY", "url": f"http://{host}:{port}/", **store.status()}, ensure_ascii=False), flush=True)
    server.serve_forever()
