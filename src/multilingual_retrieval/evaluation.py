from __future__ import annotations

import csv
import hashlib
import html
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq
import yaml


LABELS = {"0", "1", "2", "UNSURE"}
VERIFIED = "HUMAN_VERIFIED"
MISSING_RANK = 1_000_000_000


def _rank(value: Any) -> int:
    return MISSING_RANK if value is None else int(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_han(text: str) -> bool:
    return any("\u3400" <= char <= "\u9fff" for char in str(text))


def _split(case_key: str) -> str:
    return "held_out" if int(hashlib.sha256(case_key.encode()).hexdigest()[:8], 16) % 10 < 3 else "development"


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or int(value.get("version", 0)) != 1:
        raise ValueError("multilingual retrieval config version must be 1")
    return value


def _query_map(path: Path) -> dict[int, str]:
    table = pq.read_table(path, columns=["id", "query"])
    return {int(row["id"]): str(row["query"]) for row in table.to_pylist()}


def _translation_map(path: Path) -> dict[int, str]:
    table = pq.read_table(path, columns=["query_id", "query_en"])
    return {int(row["query_id"]): str(row["query_en"]) for row in table.to_pylist()}


def _phase7_rankings(path: Path, query_ids: set[int]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    columns = [
        "query_id", "query_text", "doc_id", "chunk_id", "chunk_text", "source_url",
        "dense_rank", "sparse_rank", "hybrid_rank", "rerank_rank", "rerank_score",
    ]
    for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=2048):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            if query_id not in query_ids:
                continue
            language = "zh" if has_han(row["chunk_text"]) else "vi"
            grouped[(query_id, language)].append({
                "query_id": query_id,
                "query_text": str(row["query_text"]),
                "target_language": language,
                "doc_id": int(row["doc_id"]),
                "chunk_id": str(row["chunk_id"]),
                "evidence_text": str(row["chunk_text"]),
                "source_url": str(row["source_url"]),
                "score": float(row["rerank_score"]),
                "source_rank": int(row["rerank_rank"]),
                "dense_rank": _rank(row["dense_rank"]),
                "lexical_rank": _rank(row["sparse_rank"]),
                "hybrid_rank": _rank(row["hybrid_rank"]),
                "reranker_rank": _rank(row["rerank_rank"]),
                "retrieval_branch": "phase7_multilingual_reranker",
            })
    return grouped


def _dedupe_documents(
    rows: Iterable[dict[str, Any]], depth: int = 20, rank_field: str = "source_rank"
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in sorted(rows, key=lambda item: (int(item[rank_field]), int(item["doc_id"]), str(item["chunk_id"]))):
        if int(row[rank_field]) >= MISSING_RANK:
            continue
        if int(row["doc_id"]) in seen:
            continue
        seen.add(int(row["doc_id"]))
        output.append({**row, "rank": len(output) + 1})
        if len(output) == depth:
            break
    return output


def _pubmed_rankings(hybrid_path: Path, docs_path: Path, query_ids: set[int]) -> dict[int, list[dict[str, Any]]]:
    wanted_docs: set[int] = set()
    ranks: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for batch in pq.ParquetFile(hybrid_path).iter_batches(
        columns=["query_id", "doc_id", "rank", "rrf_score", "dense_rank", "lexical_rank"], batch_size=4096
    ):
        for row in batch.to_pylist():
            query_id = int(row["query_id"])
            branch_ranks = (_rank(row["rank"]), _rank(row["dense_rank"]), _rank(row["lexical_rank"]))
            if query_id not in query_ids or min(branch_ranks) > 20:
                continue
            doc_id = int(row["doc_id"]); wanted_docs.add(doc_id)
            ranks[query_id].append({
                "doc_id": doc_id,
                "hybrid_rank": _rank(row["rank"]),
                "dense_rank": _rank(row["dense_rank"]),
                "lexical_rank": _rank(row["lexical_rank"]),
                "score": float(row["rrf_score"]),
            })
    documents: dict[int, dict[str, Any]] = {}
    columns = ["doc_id", "original_url", "title", "abstract_text"]
    for batch in pq.ParquetFile(docs_path).iter_batches(columns=columns, batch_size=4096):
        for row in batch.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in wanted_docs:
                documents[doc_id] = row
    if wanted_docs - documents.keys():
        raise ValueError(f"missing canonical PubMed documents: {len(wanted_docs - documents.keys())}")
    output: dict[int, list[dict[str, Any]]] = {}
    for query_id, rows in ranks.items():
        enriched = []
        for row in sorted(rows, key=lambda value: (value["hybrid_rank"], value["doc_id"])):
            doc = documents[row["doc_id"]]
            title = str(doc.get("title") or "").strip(); abstract = str(doc.get("abstract_text") or "").strip()
            enriched.append({
                "query_id": query_id, "target_language": "en", "doc_id": row["doc_id"],
                "chunk_id": "", "evidence_text": (title + "\n\n" + abstract).strip(),
                "source_url": str(doc["original_url"]), "score": row["score"], "source_rank": row["hybrid_rank"],
                "hybrid_rank": row["hybrid_rank"], "dense_rank": row["dense_rank"],
                "lexical_rank": row["lexical_rank"], "retrieval_branch": "phase19_translated_document",
            })
        output[query_id] = enriched
    return output


def _official_map(path: Path, wanted: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    for batch in pq.ParquetFile(path).iter_batches(columns=["id", "url"], batch_size=65536):
        for doc_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            key = int(doc_id)
            if key in wanted:
                found[key] = str(url)
    return found


def _phase29_rankings(path: Path, query_ids: set[int]) -> dict[tuple[int, str], list[dict[str, Any]]]:
    connection = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    try:
        if connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("Phase29 packaging SQLite quick_check failed")
        placeholders = ",".join("?" for _ in query_ids)
        records = connection.execute(
            f"SELECT query_id,record_json FROM records WHERE query_id IN ({placeholders}) ORDER BY query_id",
            tuple(sorted(query_ids)),
        ).fetchall()
        if len(records) != len(query_ids):
            raise ValueError(f"Phase29 record coverage incomplete: {len(records)}/{len(query_ids)}")
        candidates: list[dict[str, Any]] = []
        wanted: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for query_id, raw_record in records:
            record = json.loads(raw_record)
            evidence_by_doc: dict[int, list[str]] = defaultdict(list)
            for chunk in record["relevant_chunks"]:
                evidence_by_doc[int(chunk["doc_id"])].append(str(chunk["chunk_text"]))
            for rank, doc_id in enumerate(record["relevant_docs"], start=1):
                doc_id = int(doc_id)
                if not evidence_by_doc.get(doc_id):
                    continue
                evidence_text = evidence_by_doc[doc_id][0]
                candidate = {
                    "query_id": int(query_id), "doc_id": doc_id, "rank": rank,
                    "evidence_text": evidence_text,
                }
                candidates.append(candidate)
                wanted[(doc_id, evidence_text)].append(candidate)
        cursor = connection.execute("SELECT chunk_id,doc_id,expanded_text FROM expanded")
        while True:
            rows = cursor.fetchmany(128)
            if not rows:
                break
            for chunk_id, doc_id, text in rows:
                for candidate in wanted.get((int(doc_id), str(text)), []):
                    candidate["chunk_id"] = str(chunk_id)
        missing = [row for row in candidates if "chunk_id" not in row]
        if missing:
            raise ValueError(f"Phase29 exact supporting chunks missing: {len(missing)}")
    finally:
        connection.close()

    output: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        language = "zh" if has_han(row["evidence_text"]) else "vi"
        output[(row["query_id"], language)].append({
            **row,
            "target_language": language,
            "source_url": "",
            "score": None,
            "source_rank": row["rank"],
            "phase29_rank": row["rank"],
            "retrieval_branch": "phase29_frozen_output",
        })
    return output


def _verify_phase4_chunks(path: Path, rows: list[dict[str, Any]]) -> None:
    wanted = {str(row["chunk_id"]): (int(row["doc_id"]), str(row["evidence_text"])) for row in rows}
    found: set[str] = set()
    for batch in pq.ParquetFile(path).iter_batches(columns=["chunk_id", "doc_id", "raw_text"], batch_size=4096):
        for chunk_id, doc_id, raw_text in zip(
            batch.column(0).to_pylist(), batch.column(1).to_pylist(), batch.column(2).to_pylist(), strict=True
        ):
            key = str(chunk_id)
            if key not in wanted:
                continue
            expected_doc, expected_text = wanted[key]
            if int(doc_id) != expected_doc or str(raw_text) != expected_text:
                raise ValueError(f"Phase4 chunk provenance mismatch: {key}")
            found.add(key)
    if found != wanted.keys():
        raise ValueError(f"Phase4 canonical chunks missing: {len(wanted.keys() - found)}")


def prepare(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path); root = config_path.resolve().parents[1]
    paths = {key: root / value for key, value in config["paths"].items()}
    output = root / config["output_dir"]; output.mkdir(parents=True, exist_ok=True)
    queries = _query_map(paths["queries"])
    translations = _translation_map(paths["phase19_translations"])

    sanity = json.loads(paths["phase7_sanity"].read_text(encoding="utf-8"))
    phase7_ids = sorted({int(row["query_id"]) for row in sanity["rows"]}, key=lambda q: hashlib.sha256(f"phase7:{q}".encode()).hexdigest())
    if len(phase7_ids) != 40:
        raise ValueError("expected frozen Phase7 40-query sanity set")
    zh_ids = set(phase7_ids[:20]); vi_ids = set(phase7_ids[20:])

    pubmed_evidence = json.loads(paths["phase17_pubmed_evidence"].read_text(encoding="utf-8"))
    available_en = [int(row["query_id"]) for row in pubmed_evidence if int(row["query_id"]) not in set(phase7_ids)]
    en_ids = set(sorted(available_en, key=lambda q: hashlib.sha256(f"phase17-en:{q}".encode()).hexdigest())[:20])
    if len(en_ids) != 20:
        raise ValueError("insufficient distinct frozen English review queries")

    phase7 = _phase7_rankings(paths["phase7_rankings"], vi_ids | zh_ids)
    pubmed = _pubmed_rankings(paths["phase19_hybrid"], paths["pubmed_documents"], en_ids)
    expected_phase29_sha = str(config["contracts"]["phase29_zip_sha256"])
    actual_phase29_sha = sha256_file(paths["phase29_zip"])
    if actual_phase29_sha != expected_phase29_sha:
        raise ValueError(f"frozen Phase29 SHA-256 mismatch: {actual_phase29_sha}")
    phase29 = _phase29_rankings(paths["phase29_packaging"], vi_ids | en_ids | zh_ids)
    raw_rows = [row for values in phase7.values() for row in values]
    raw_rows.extend(row for values in pubmed.values() for row in values)
    raw_rows.extend(row for values in phase29.values() for row in values)
    official = _official_map(paths["official_corpus"], {int(row["doc_id"]) for row in raw_rows})
    historical_non_current_docs = {int(row["doc_id"]) for row in raw_rows} - official.keys()
    historical_url_mismatches = sum(
        1 for row in raw_rows
        if row["source_url"] and int(row["doc_id"]) in official and official[int(row["doc_id"])] != row["source_url"]
    )

    def current_official(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "artifact_source_url": row["source_url"],
                "source_url": official[int(row["doc_id"])],
            }
            for row in rows
            if int(row["doc_id"]) in official
        ]

    retrieval_rows: list[dict[str, Any]] = []
    units: list[tuple[int, str, dict[str, list[dict[str, Any]]]]] = []
    for query_id in sorted(vi_ids):
        rows = current_official(phase7[(query_id, "vi")])
        arms = {
            arm: _dedupe_documents(rows, rank_field=rank_field)
            for arm, rank_field in (("dense", "dense_rank"), ("lexical", "lexical_rank"), ("hybrid", "hybrid_rank"), ("reranker", "reranker_rank"))
        }
        arms["phase29"] = _dedupe_documents(current_official(phase29.get((query_id, "vi"), [])), rank_field="phase29_rank")
        units.append((query_id, "vi", arms))
    for query_id in sorted(en_ids):
        rows = current_official(pubmed.get(query_id, []))
        arms = {
            arm: _dedupe_documents(rows, rank_field=rank_field)
            for arm, rank_field in (("dense", "dense_rank"), ("lexical", "lexical_rank"), ("hybrid", "hybrid_rank"))
        }
        arms["phase29"] = []
        units.append((query_id, "en", arms))
    for query_id in sorted(zh_ids):
        rows = current_official(phase7[(query_id, "zh")])
        arms = {
            arm: _dedupe_documents(rows, rank_field=rank_field)
            for arm, rank_field in (("dense", "dense_rank"), ("lexical", "lexical_rank"), ("hybrid", "hybrid_rank"), ("reranker", "reranker_rank"))
        }
        arms["phase29"] = _dedupe_documents(current_official(phase29.get((query_id, "zh"), [])), rank_field="phase29_rank")
        units.append((query_id, "zh", arms))
    if any(not any(arms.values()) for _, _, arms in units):
        empty = [(query_id, language) for query_id, language, arms in units if not any(arms.values())]
        raise ValueError(f"no current-official candidates for review units: {empty}")

    review_rows: list[dict[str, Any]] = []
    for query_id, language, arms in units:
        case_key = f"{query_id}:{language}"
        pooled: dict[int, dict[str, Any]] = {}
        for arm, rows in arms.items():
            for row in rows:
                retrieval_rows.append({**row, "query_text": queries[query_id], "split": _split(case_key), "arm": arm})
            if rows:
                doc_id = int(rows[0]["doc_id"])
                if doc_id not in pooled:
                    pooled[doc_id] = {**rows[0], "pool_arms": [arm]}
                else:
                    pooled[doc_id]["pool_arms"].append(arm)
        primary = arms.get("reranker", arms.get("hybrid", []))
        for row in primary:
            if len(pooled) >= 5:
                break
            pooled.setdefault(int(row["doc_id"]), {**row, "pool_arms": ["pool_fill"]})
        for presentation_order, row in enumerate(
            sorted(pooled.values(), key=lambda value: hashlib.sha256(f"{case_key}:{value['doc_id']}".encode()).hexdigest()),
            start=1,
        ):
            review_rows.append({
                "case_id": f"{case_key}:{row['doc_id']}", "query_id": query_id,
                "query_text": queries[query_id], "target_language": language,
                "supporting_translation": translations[query_id] if language == "en" else "",
                "split": _split(case_key), "presentation_order": presentation_order, "doc_id": row["doc_id"],
                "chunk_id": row["chunk_id"], "canonical_url": row["source_url"],
                "evidence_text": row["evidence_text"], "pool_arms": "|".join(sorted(row["pool_arms"])),
                "judgment": "", "verification_status": "PENDING_HUMAN",
                "reviewer": "", "verified_at": "", "review_notes": "",
            })

    wanted = {int(row["doc_id"]) for row in retrieval_rows}
    missing = wanted - official.keys()
    url_mismatch = [row for row in retrieval_rows if official.get(int(row["doc_id"])) != row["source_url"]]
    if missing or url_mismatch:
        raise ValueError(f"official provenance failed missing={len(missing)} canonical_url_mismatch={len(url_mismatch)}")
    if any(not str(row["evidence_text"]).strip() for row in review_rows):
        raise ValueError("empty canonical evidence in review queue")
    _verify_phase4_chunks(
        paths["phase4_chunks"],
        [row for row in retrieval_rows if row["retrieval_branch"] == "phase7_multilingual_reranker"],
    )

    retrieval_path = output / "cached_retrieval_top20.parquet"
    review_path = output / "review_queue.parquet"
    pq.write_table(pa.Table.from_pylist(retrieval_rows), retrieval_path, compression="zstd")
    pq.write_table(pa.Table.from_pylist(review_rows), review_path, compression="zstd")
    label_rows = [{key: value for key, value in row.items() if key != "pool_arms"} for row in review_rows]
    template_path = output / "human_labels_template.csv"
    with template_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(label_rows[0]))
        writer.writeheader(); writer.writerows(label_rows)
    label_path = output / "human_labels.csv"
    if label_path.exists():
        with label_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream); existing = list(reader); actual_fields = reader.fieldnames or []
        expected_ids = {row["case_id"] for row in label_rows}
        actual_ids = {row.get("case_id", "") for row in existing}
        contract_changed = (
            len(existing) != len(label_rows)
            or actual_ids != expected_ids
            or actual_fields != list(label_rows[0])
        )
        if contract_changed and any(
            row.get("verification_status") == VERIFIED for row in existing
        ):
            raise ValueError("existing human label file does not match the frozen review manifest")
        if contract_changed:
            with label_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(label_rows[0]))
                writer.writeheader(); writer.writerows(label_rows)
    else:
        with label_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(label_rows[0]))
            writer.writeheader(); writer.writerows(label_rows)

    language_counts = {language: sum(1 for _, value, _ in units if value == language) for language in ("vi", "en", "zh")}
    split_counts = {split: len({(row["query_id"], row["target_language"]) for row in review_rows if row["split"] == split}) for split in ("development", "held_out")}
    source_hashes = {key: sha256_file(path) for key, path in paths.items()}
    manifest = {
        "status": "AWAITING_HUMAN_RELEVANCE_LABELS", "version": 1,
        "queries": len(units), "review_rows": len(review_rows), "retrieval_rows": len(retrieval_rows),
        "languages": language_counts, "splits": split_counts,
        "query_overlap_across_language_tracks": len(units) - len({query_id for query_id, _, _ in units}),
        "official_id_missing": 0, "canonical_url_mismatches": 0,
        "historical_non_current_official_docs_excluded": len(historical_non_current_docs),
        "historical_artifact_url_mismatches": historical_url_mismatches, "empty_evidence": 0,
        "source_hashes": source_hashes,
        "outputs": {"retrieval": str(retrieval_path.relative_to(root)), "review_queue": str(review_path.relative_to(root)), "label_template": str(template_path.relative_to(root)), "labels": str(label_path.relative_to(root))},
        "output_hashes": {"retrieval": sha256_file(retrieval_path), "review_queue": sha256_file(review_path), "label_template": sha256_file(template_path)},
        "metric_gate": "Only rows marked HUMAN_VERIFIED are eligible for relevance metrics.",
        "pooling_contract": "One top document from each frozen retrieval arm, then deterministic fill to at most five documents per query-language unit.",
        "limitations": [
            "Phase7 is a 5,643-chunk mixed-language pilot, not full Chinese corpus coverage.",
            "Phase7 candidates absent from the current official snapshot are excluded before review.",
            "Current official URLs replace historical artifact URLs; mismatch counts remain diagnostic metadata.",
            "Phase19 English branch uses frozen Helsinki VI-to-EN translation and cached hybrid retrieval.",
            "The queue contains at most five documents pooled across frozen retrieval arms; unjudged documents are not assumed irrelevant.",
            "No organizer qrels or Codex judgments are treated as human ground truth.",
        ],
    }
    manifest_path = output / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def evaluate(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path); root = config_path.resolve().parents[1]
    output = root / config["output_dir"]
    retrieval = pq.read_table(output / "cached_retrieval_top20.parquet").to_pylist()
    label_path = output / "human_labels.csv"
    with label_path.open("r", encoding="utf-8-sig", newline="") as stream:
        labels = list(csv.DictReader(stream))
    verified = [row for row in labels if row.get("verification_status") == VERIFIED]
    invalid = [row for row in verified if row.get("judgment") not in LABELS or not row.get("reviewer") or not row.get("verified_at")]
    if invalid:
        raise ValueError(f"invalid human-verified labels: {len(invalid)}")
    label_groups: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        label_groups[(int(row["query_id"]), row["target_language"])].append(row)
    fully_judged = {
        key for key, rows in label_groups.items()
        if rows and all(row.get("verification_status") == VERIFIED for row in rows)
    }
    relevant: dict[tuple[int, str], set[int]] = defaultdict(set)
    for row in verified:
        if row["judgment"] == "2":
            relevant[(int(row["query_id"]), row["target_language"])].add(int(row["doc_id"]))
    rankings: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    for row in sorted(
        retrieval,
        key=lambda value: (int(value["query_id"]), value["target_language"], value["arm"], int(value["rank"])),
    ):
        rankings[(int(row["query_id"]), row["target_language"], row["arm"])].append(int(row["doc_id"]))
    def arm_metrics(keys: list[tuple[int, str]], arm: str) -> dict[str, Any]:
        reciprocal = []; hits = 0; recalls = []
        for query_id, target_language in keys:
            ranked = rankings.get((query_id, target_language, arm), [])[:20]
            gold = relevant[(query_id, target_language)]
            matched = [index + 1 for index, doc_id in enumerate(ranked) if doc_id in gold]
            hits += bool(matched); reciprocal.append(1 / min(matched) if matched else 0.0)
            recalls.append(len(set(ranked) & gold) / len(gold))
        return {
            "supported_units": len(keys),
            "pooled_hit_at_20": (hits / len(keys)) if keys else None,
            "pooled_mrr_at_20": (sum(reciprocal) / len(keys)) if keys else None,
            "pooled_recall_at_20": (sum(recalls) / len(keys)) if keys else None,
        }

    metrics: dict[str, Any] = {}
    for language in ("vi", "en", "zh"):
        keys = [key for key, docs in relevant.items() if key[1] == language and docs and key in fully_judged]
        arms = sorted({key[2] for key in rankings if key[1] == language})
        language_labels = [row for row in labels if row["target_language"] == language]
        metrics[language] = {
            "pool_units": len({(int(row["query_id"]), row["target_language"]) for row in language_labels}),
            "fully_judged_units": sum(key[1] == language for key in fully_judged),
            "supported_units": len(keys),
            "verified_rows": sum(row.get("verification_status") == VERIFIED for row in language_labels),
            "direct_support_label_2": sum(row.get("verification_status") == VERIFIED and row.get("judgment") == "2" for row in language_labels),
            "partial_support_label_1": sum(row.get("verification_status") == VERIFIED and row.get("judgment") == "1" for row in language_labels),
            "arms": {},
        }
        for arm in arms:
            metrics[language]["arms"][arm] = arm_metrics(keys, arm)
    overall_arms: dict[str, Any] = {}
    for arm in sorted({key[2] for key in rankings}):
        keys = [key for key, docs in relevant.items() if docs and key in fully_judged and (key[0], key[1], arm) in rankings]
        overall_arms[arm] = arm_metrics(keys, arm)
    failure_rows: list[dict[str, Any]] = []
    for query_id, language in sorted(fully_judged):
        judged = label_groups[(query_id, language)]
        direct = {int(row["doc_id"]) for row in judged if row["judgment"] == "2"}
        partial = {int(row["doc_id"]) for row in judged if row["judgment"] == "1"}
        ranks_by_arm: dict[str, dict[int, int]] = {}
        for (rank_query, rank_language, arm), docs in rankings.items():
            if (rank_query, rank_language) == (query_id, language):
                ranks_by_arm[arm] = {doc_id: index + 1 for index, doc_id in enumerate(docs[:20])}
        categories: list[str] = []
        if not direct:
            categories.append("NO_DIRECT_SUPPORT_IN_JUDGED_POOL")
        if direct and "dense" in ranks_by_arm and "hybrid" in ranks_by_arm:
            if any(doc in ranks_by_arm["dense"] and doc not in ranks_by_arm["hybrid"] for doc in direct):
                categories.append("DIRECT_SUPPORT_DROPPED_BEFORE_OR_AT_FUSION")
        if direct and "hybrid" in ranks_by_arm and "reranker" in ranks_by_arm:
            if any(doc in ranks_by_arm["hybrid"] and doc not in ranks_by_arm["reranker"] for doc in direct):
                categories.append("DIRECT_SUPPORT_DROPPED_AFTER_RERANKER")
        if partial and not direct:
            categories.append("EVIDENCE_PARTIAL_NOT_DIRECT_ANSWER")
        failure_rows.append({
            "query_id": query_id, "target_language": language,
            "direct_support_docs": sorted(direct), "partial_support_docs": sorted(partial),
            "direct_support_ranks_by_arm": {
                arm: {str(doc): ranks[doc] for doc in direct if doc in ranks}
                for arm, ranks in ranks_by_arm.items()
            },
            "categories": categories,
            "cannot_determine_from_pooled_judging": [
                "CORPUS_MISSING_RELEVANT_EVIDENCE",
                "RELEVANT_EVIDENCE_IN_CORPUS_BUT_OUTSIDE_ALL_CANDIDATE_POOLS",
            ],
        })
    status = "EVALUATED" if all(metrics[language]["supported_units"] > 0 for language in metrics) else "INSUFFICIENT_VERIFIED_LABELS"
    result = {
        "status": status,
        "judging": {
            "total_rows": len(labels), "verified_rows": len(verified),
            "fully_judged_units": len(fully_judged), "total_units": len(label_groups),
        },
        "metrics": metrics,
        "overall": {"arms": overall_arms},
        "metric_boundary": "Recall is pooled Recall@20 over HUMAN_VERIFIED label-2 documents in fully judged units, not exhaustive corpus recall. Unjudged documents are never negatives.",
    }
    (output / "evaluation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "failure_analysis.json").write_text(
        json.dumps({"fully_judged_units": len(fully_judged), "rows": failure_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def render_review(config_path: Path, split: str = "development") -> dict[str, Any]:
    if split not in {"development", "held_out"}:
        raise ValueError("review split must be development or held_out")
    config = _load_config(config_path); root = config_path.resolve().parents[1]
    output = root / config["output_dir"]
    label_path = output / "human_labels.csv"
    with label_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"no review rows for split {split}")
    payload = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    title = f"ViBioMIR multilingual review — {split}"
    html_path = output / f"review_{split}.html"
    html_path.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>"
        "body{font:15px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#f5f7fa;color:#17202a}"
        ".unit{background:white;border:1px solid #d9e1e8;border-radius:10px;padding:16px;margin:18px 0}"
        ".candidate{border-top:1px solid #e7ebef;padding:14px 0}.evidence{white-space:pre-wrap;background:#f8fafc;padding:12px;border-radius:6px}"
        "label{margin-right:14px}input[type=text]{width:100%;padding:7px}button{padding:10px 16px;margin-right:8px}"
        ".meta{color:#52606d;font-size:13px}.held{color:#9c2c2c;font-weight:700}</style></head><body>"
        f"<h1>{html.escape(title)}</h1><p class='held'>Judge source evidence only. Retrieval arm, rank, and score are hidden.</p>"
        "<p><label>Reviewer <input id='reviewer' type='text' style='width:260px'></label> "
        "<button onclick='saveLocal()'>Save draft in browser</button><button onclick='saveVerified()'>Save verified labels</button>"
        "<button onclick='exportCsv()'>Download backup CSV</button></p><div id='status'></div>"
        "<div id='root'></div><script>"
        f"const allRows={payload}; const activeSplit={json.dumps(split)};"
        "const key='vibiomir-multilingual-review-'+activeSplit;"
        "const esc=s=>String(s??'').replace(/[&<>\"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;',\"'\":'&#39;'}[c]));"
        "const groups={}; allRows.filter(r=>r.split===activeSplit).forEach(r=>(groups[r.query_id+':'+r.target_language]??=[]).push(r));"
        "function draw(){const saved=JSON.parse(localStorage.getItem(key)||'{}');document.getElementById('reviewer').value=saved.reviewer||'';"
        "document.getElementById('root').innerHTML=Object.entries(groups).map(([unit,rs])=>`<section class='unit'><h2>${esc(unit)}</h2><h3>Original Vietnamese query</h3><div class='evidence'>${esc(rs[0].query_text)}</div>${rs[0].supporting_translation?`<h3>Supporting English translation <span class='meta'>(mechanical query representation; not evidence)</span></h3><div class='evidence'>${esc(rs[0].supporting_translation)}</div>`:''}`+rs.sort((a,b)=>+a.presentation_order-+b.presentation_order).map(r=>{const v=saved[r.case_id]||{judgment:r.judgment,review_notes:r.review_notes};return `<div class='candidate'><div class='meta'>Candidate ${esc(r.presentation_order)} · official doc ${esc(r.doc_id)} · canonical chunk ${esc(r.chunk_id)} · <a target='_blank' href='${esc(r.canonical_url)}'>official URL</a></div><h3>Original source evidence</h3><div class='evidence'>${esc(r.evidence_text)}</div><p>${[['2','2 — directly relevant'],['1','1 — partially relevant'],['0','0 — not relevant'],['UNSURE','UNSURE']].map(([x,label])=>`<label><input type='radio' name='${esc(r.case_id)}' value='${x}' ${v.judgment===x?'checked':''}>${label}</label>`).join('')}</p><input type='text' data-note='${esc(r.case_id)}' value='${esc(v.review_notes||'')}' placeholder='Reason grounded in the query and exact evidence'></div>`}).join('')+`</section>`).join('');}"
        "function collect(){const out={reviewer:document.getElementById('reviewer').value.trim()};Object.values(groups).flat().forEach(r=>{const pick=document.querySelector(`input[name='${CSS.escape(r.case_id)}']:checked`);const note=document.querySelector(`[data-note='${CSS.escape(r.case_id)}']`);out[r.case_id]={judgment:pick?.value||'',review_notes:note?.value||''};});return out;}"
        "function saveLocal(){localStorage.setItem(key,JSON.stringify(collect()));alert('Saved in this browser.');}"
        "async function saveVerified(){const state=collect();if(!state.reviewer){alert('Reviewer is required.');return;}const judgments=Object.fromEntries(Object.entries(state).filter(([k])=>k!=='reviewer'));try{const response=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({reviewer:state.reviewer,judgments})});const result=await response.json();if(!response.ok)throw new Error(result.error||'save failed');document.getElementById('status').textContent=`Saved ${result.saved} verified labels directly to human_labels.csv.`;localStorage.setItem(key,JSON.stringify(state));}catch(error){document.getElementById('status').textContent=`Direct save unavailable: ${error}. Run the review server or download the backup CSV.`;}}"
        "function exportCsv(){const state=collect();if(!state.reviewer){alert('Reviewer is required.');return;}const now=new Date().toISOString();allRows.forEach(r=>{if(r.split!==activeSplit)return;const v=state[r.case_id]||{};if(v.judgment){r.judgment=v.judgment;r.review_notes=v.review_notes;r.verification_status='HUMAN_VERIFIED';r.reviewer=state.reviewer;r.verified_at=now;}});const cols=Object.keys(allRows[0]);const quote=v=>'\"'+String(v??'').replaceAll('\"','\"\"')+'\"';const csv='\\ufeff'+cols.map(quote).join(',')+'\\r\\n'+allRows.map(r=>cols.map(c=>quote(r[c])).join(',')).join('\\r\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='human_labels.csv';a.click();URL.revokeObjectURL(a.href);}draw();"
        "</script></body></html>",
        encoding="utf-8",
    )
    return {"status": "READY_FOR_HUMAN_REVIEW", "split": split, "rows": len(selected), "path": str(html_path)}
