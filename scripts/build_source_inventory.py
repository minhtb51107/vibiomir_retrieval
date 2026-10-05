from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.site_discovery.inventory import build_inventory, mark_selection, select_representative_domains


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Phase 10B1 local 97-domain inventory.")
    parser.add_argument("--corpus", default="data/raw/links_corpus.parquet")
    parser.add_argument("--chunks", default="data/chunks/phase4_bge_m3/tokens_512_overlap_64.parquet")
    parser.add_argument("--probe", default="artifacts/source_probe/probe_results.json")
    parser.add_argument("--bing", default="artifacts/phase10b0_search/domain_analysis.json")
    parser.add_argument("--out", default="artifacts/phase10b1_site_discovery/domain_inventory.json")
    args = parser.parse_args()
    rows = build_inventory(
        corpus_path=args.corpus, chunks_path=args.chunks,
        probe_path=args.probe, bing_path=args.bing,
    )
    selected = select_representative_domains(rows)
    mark_selection(rows, selected)
    payload = {
        "total_domains": len(rows),
        "total_corpus_rows": sum(int(row["corpus_rows"]) for row in rows),
        "selected_domains": selected,
        "selected_corpus_share": round(sum(float(row["corpus_share"]) for row in rows if row["selected_stage0"]), 8),
        "domains": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "domains"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
