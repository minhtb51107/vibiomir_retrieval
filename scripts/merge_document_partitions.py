#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream document partitions into Parquet.")
    parser.add_argument("--input-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    sources = sorted(Path(args.input_directory).glob("part-*.parquet"))
    if not sources:
        raise ValueError("no document partitions found")
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    doc_ids: set[int] = set()
    writer = None
    schema = None
    try:
        for source in sources:
            parquet = pq.ParquetFile(source)
            if writer is None:
                schema = parquet.schema_arrow
                writer = pq.ParquetWriter(temporary, schema)
            elif parquet.schema_arrow != schema:
                raise ValueError(f"partition schema mismatch: {source}")
            for batch in parquet.iter_batches(batch_size=1024):
                table = pa.Table.from_batches([batch])
                for value in table.column("doc_id").to_pylist():
                    doc_id = int(value)
                    if doc_id in doc_ids:
                        raise ValueError(f"duplicate doc_id: {doc_id}")
                    doc_ids.add(doc_id)
                writer.write_table(table)
                rows += len(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, destination)
    summary = {
        "partition_count": len(sources),
        "document_count": rows,
        "unique_doc_ids": len(doc_ids),
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
