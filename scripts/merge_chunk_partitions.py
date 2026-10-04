#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream deterministic chunk partitions into one Parquet input."
    )
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    sources = sorted(Path(args.input_directory).glob("part-*.parquet"))
    if not sources:
        raise ValueError("no chunk partitions found")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    identifiers: set[str] = set()
    digest = hashlib.sha256()
    writer = None
    expected_schema = None
    try:
        for source in sources:
            parquet = pq.ParquetFile(source)
            if writer is None:
                expected_schema = parquet.schema_arrow
                writer = pq.ParquetWriter(temporary, expected_schema)
            elif parquet.schema_arrow != expected_schema:
                raise ValueError(f"partition schema mismatch: {source}")
            for batch in parquet.iter_batches(batch_size=2048):
                table = pa.Table.from_batches([batch])
                for chunk_id in table.column("chunk_id").to_pylist():
                    value = str(chunk_id)
                    if value in identifiers:
                        raise ValueError(f"duplicate chunk_id: {value}")
                    identifiers.add(value)
                    digest.update(value.encode("utf-8"))
                    digest.update(b"\0")
                writer.write_table(table)
                rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, destination)
    summary = {
        "input_directory": args.input_directory,
        "partition_count": len(sources),
        "chunk_count": rows,
        "unique_chunk_ids": len(identifiers),
        "chunk_id_order_sha256": digest.hexdigest(),
        "output": str(destination),
        "output_size_bytes": destination.stat().st_size,
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
