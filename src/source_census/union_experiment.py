from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import zipfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.source_census.cached_subsets import build_cached_subset_rankings
from src.source_census.pipeline import atomic_json, load_config
from src.submission.common import file_sha256, write_deterministic_zip, write_submission_stream
from src.submission.expansion import SourceWindowExpander
from src.submission.validator import expected_query_ids, normalize_validation_result
from src.validation.pre_submission import audit_score_cache


def _host(url: str) -> str:
    return (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")


def _digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):
            value.update(block)
    return value.hexdigest()


def _runtime_config(spec: dict) -> dict:
    config=load_config(spec["base_config"])
    config["outputs"]={
        **config["outputs"],
        "root":spec["outputs"]["root"],
        "artifacts":spec["outputs"]["artifacts"],
        "submissions":spec["outputs"]["submissions"],
    }
    return config


def _parent_cache_root(parent: dict) -> Path:
    return Path(parent.get("cache_root",Path(parent["root"])/"round1"))


def _contract(spec: dict) -> dict:
    parents={name:load_config(row["config"]) for name,row in spec["parents"].items()}
    first=next(iter(parents.values()))
    fixed={
        "queries":lambda c:c["inputs"]["queries"],
        "pilot_documents":lambda c:c["inputs"]["pilot_documents"],
        "pilot_chunks":lambda c:c["inputs"]["pilot_chunks"],
        "pilot_pool":lambda c:c["inputs"]["pilot_rerank_pool"],
        "pilot_scores":lambda c:c["inputs"]["pilot_rerank_scores"],
        "candidate_policy":lambda c:{key:c["candidate_cache"][key] for key in ("candidate_caps","dense_depth","sparse_depth","rrf_constant","bm25_k1","bm25_b")},
        "embedder":lambda c:c["models"]["embedder"],
        "reranker":lambda c:{**{key:c["models"]["reranker"].get(key) for key in ("name","revision","batch_size","max_length")},"precision":c["models"]["reranker"].get("precision","fp16")},
        "submission":lambda c:c["submission"],
    }
    checks={name:all(getter(config)==getter(first) for config in parents.values()) for name,getter in fixed.items()}
    checks["chosen_m_is_8"]=first["candidate_cache"]["candidate_caps"]==[8]
    checks["expected_unique_sources"]=len({source for row in spec["parents"].values() for source in row["sources"]})==int(spec["expected_source_count"])
    return {"passed":all(checks.values()),"checks":checks,"intended_variable":spec["intended_variable"]}


def _link_or_verify(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        if source.stat().st_size!=destination.stat().st_size or _digest(source)!=_digest(destination):
            raise ValueError(f"existing union cache differs from parent: {destination}")
        return "existing_exact"
    try:
        os.link(source,destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source,destination)
        if _digest(source)!=_digest(destination):
            raise RuntimeError(f"copy integrity failure: {destination}")
        return "copied"


def _prepare_candidate_parts(spec: dict, root: Path) -> dict:
    destination=root/"round1/source_candidate_parts"; destination.mkdir(parents=True,exist_ok=True)
    expected_sources={source for row in spec["parents"].values() for source in row["sources"]}
    found={}; methods=Counter()
    for parent in spec["parents"].values():
        source_root=_parent_cache_root(parent)/"source_candidate_parts"
        parent_sources=set(parent["sources"])
        for marker_path in sorted(source_root.glob("*.json")):
            marker=json.loads(marker_path.read_text(encoding="utf-8")); source=str(marker["source"])
            if source not in parent_sources: continue
            parquet_path=marker_path.with_suffix(".parquet")
            if source in found: raise ValueError(f"duplicate source candidate part: {source}")
            methods[_link_or_verify(parquet_path,destination/parquet_path.name)]+=1
            shutil.copy2(marker_path,destination/marker_path.name)
            found[source]={"rows":pq.ParquetFile(parquet_path).metadata.num_rows,"parent":str(parquet_path)}
    missing=sorted(expected_sources-set(found))
    if missing: raise ValueError(f"missing parent source candidate parts: {missing}")
    return {"sources":found,"reuse_methods":dict(methods),"candidate_rows":sum(row["rows"] for row in found.values())}


def _merge_score_databases(spec: dict, destination: Path) -> dict:
    destination.parent.mkdir(parents=True,exist_ok=True); temporary=destination.with_suffix(".sqlite.tmp")
    temporary.unlink(missing_ok=True)
    connection=sqlite3.connect(temporary)
    connection.execute("CREATE TABLE pairs(query_id INTEGER,chunk_id TEXT,query_text TEXT,chunk_text TEXT,rerank_score REAL,inference_ms REAL,PRIMARY KEY(query_id,chunk_id))")
    source_rows={}; intersections={}; mismatches={}
    for index,(name,parent) in enumerate(spec["parents"].items()):
        path=(_parent_cache_root(parent)/"source_scores.sqlite").resolve(); alias=f"src{index}"
        connection.execute(f"ATTACH DATABASE ? AS {alias}",(str(path),))
        connection.execute("DROP TABLE IF EXISTS temp.parent_keys")
        connection.execute("CREATE TEMP TABLE parent_keys(query_id INTEGER,chunk_id TEXT,PRIMARY KEY(query_id,chunk_id)) WITHOUT ROWID")
        selected=set(parent["sources"]); parts=_parent_cache_root(parent)/"source_candidate_parts"
        for marker_path in sorted(parts.glob("*.json")):
            marker=json.loads(marker_path.read_text(encoding="utf-8"))
            if str(marker["source"]) not in selected: continue
            for batch in pq.ParquetFile(marker_path.with_suffix(".parquet")).iter_batches(columns=["query_id","chunk_id"],batch_size=8192):
                connection.executemany("INSERT OR IGNORE INTO parent_keys VALUES (?,?)",[(int(q),str(c)) for q,c in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist(),strict=True)])
        source_rows[name]=int(connection.execute("SELECT COUNT(*) FROM parent_keys").fetchone()[0])
        before=connection.total_changes
        connection.execute(f"INSERT INTO pairs SELECT s.* FROM {alias}.pairs s JOIN parent_keys k ON k.query_id=s.query_id AND k.chunk_id=s.chunk_id")
        if connection.total_changes-before!=source_rows[name]: raise RuntimeError(f"incomplete score copy: {name}")
        intersections[name]=int(connection.execute(f"SELECT COUNT(*) FROM pairs d JOIN parent_keys k USING(query_id,chunk_id) JOIN {alias}.pairs s USING(query_id,chunk_id)").fetchone()[0])
        mismatches[name]=int(connection.execute(f"SELECT COUNT(*) FROM pairs d JOIN parent_keys k USING(query_id,chunk_id) JOIN {alias}.pairs s USING(query_id,chunk_id) WHERE d.rerank_score!=s.rerank_score OR d.inference_ms!=s.inference_ms OR d.query_text!=s.query_text OR d.chunk_text!=s.chunk_text").fetchone()[0])
        connection.commit(); connection.execute(f"DETACH DATABASE {alias}")
    total=int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0])
    integrity=str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    if total!=sum(source_rows.values()) or integrity!="ok" or any(mismatches.values()):
        raise ValueError("union score cache reconciliation failed")
    os.replace(temporary,destination)
    return {"total":total,"source_rows":source_rows,"exact_intersections":intersections,"value_mismatches":mismatches,"sqlite_integrity":integrity,"new_inference":0}


def _source_accounting(spec: dict) -> dict:
    rows={}; all_docs=set(); all_chunks=set(); duplicate_docs=0; duplicate_chunks=0
    for parent_name,parent in spec["parents"].items():
        root=Path(parent["root"])
        by_source={source:{"document_rows":0,"usable_documents":0,"chunks":0} for source in parent["sources"]}
        for batch in pq.ParquetFile(root/"documents.parquet").iter_batches(columns=["doc_id","original_url","extraction_status","normalized_text"],batch_size=4096):
            for doc_id,url,status,text in zip(*(column.to_pylist() for column in batch.columns),strict=True):
                source=_host(url)
                if source not in by_source: continue
                doc_id=int(doc_id); duplicate_docs+=doc_id in all_docs; all_docs.add(doc_id)
                by_source[source]["document_rows"]+=1
                by_source[source]["usable_documents"]+=status=="SUCCESS" and bool(str(text or "").strip())
        for batch in pq.ParquetFile(root/"chunks.parquet").iter_batches(columns=["chunk_id","source_url"],batch_size=8192):
            for chunk_id,url in zip(*(column.to_pylist() for column in batch.columns),strict=True):
                source=_host(url)
                if source not in by_source: continue
                chunk_id=str(chunk_id); duplicate_chunks+=chunk_id in all_chunks; all_chunks.add(chunk_id)
                by_source[source]["chunks"]+=1
        rows.update(by_source)
    return {"sources":rows,"official_document_rows":len(all_docs),"chunks":len(all_chunks),"duplicate_doc_ids":duplicate_docs,"duplicate_chunk_ids":duplicate_chunks,"passed":duplicate_docs==0 and duplicate_chunks==0}


def _embedding_audit(spec: dict) -> dict:
    parents={}; total=0
    model=None
    for name,parent in spec["parents"].items():
        config=load_config(parent["config"]); current_model=config["models"]["embedder"]
        if model is None: model=current_model
        if current_model!=model: raise ValueError("parent embedding contracts differ")
        root=Path(parent["root"]); chunk_path=root/"chunks.parquet"; storage_rows=pq.ParquetFile(chunk_path).metadata.num_rows; dimension=int(model["dimension"])
        selected_sources=set(parent["sources"]); rows=0
        for batch in pq.ParquetFile(chunk_path).iter_batches(columns=["source_url"],batch_size=8192):
            rows+=sum(_host(url) in selected_sources for url in batch.column(0).to_pylist())
        cache_root=_parent_cache_root(parent)
        vectors=cache_root/"chunk_embeddings.f32"; status=cache_root/"chunk_embeddings.streaming.status.u8"
        if vectors.stat().st_size!=storage_rows*dimension*4: raise ValueError(f"embedding shape mismatch: {name}")
        completion_evidence="transactional_bitmap"
        if status.exists():
            if status.stat().st_size!=storage_rows: raise ValueError(f"embedding bitmap shape mismatch: {name}")
            bitmap=np.memmap(status,dtype=np.uint8,mode="r",shape=(storage_rows,))
            if int((bitmap==1).sum())!=storage_rows or int(((bitmap!=0)&(bitmap!=1)).sum()): raise ValueError(f"embedding bitmap incomplete: {name}")
        else:
            summary=Path(parent.get("embedding_completion_artifact",Path(config["outputs"]["artifacts"])/"embedding_summary.json"))
            evidence=json.loads(summary.read_text(encoding="utf-8"))
            if "chunks" in evidence and int(evidence["chunks"])!=storage_rows: raise ValueError(f"legacy embedding completion evidence mismatch: {name}")
            completion_evidence=f"validated_legacy_artifact:{summary.as_posix()}"
        matrix=np.memmap(vectors,dtype=np.float32,mode="r",shape=(storage_rows,dimension)); nonfinite=0
        for start in range(0,storage_rows,4096): nonfinite+=int((~np.isfinite(matrix[start:start+4096])).sum())
        if nonfinite: raise ValueError(f"non-finite parent embeddings: {name}")
        parents[name]={"selected_rows":rows,"storage_rows":storage_rows,"dimension":dimension,"bytes":vectors.stat().st_size,"completion_evidence":completion_evidence,"completed_storage_rows":storage_rows,"nonfinite_values":nonfinite}
        total+=rows
    return {"passed":True,"model":model,"parents":parents,"exact_embeddings_reused":total,"new_embeddings_computed":0}


def _filter_parquet(paths: list[Path], destination: Path, key: str, selected: set[str|int]) -> dict:
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+".tmp"); temporary.unlink(missing_ok=True)
    writer=None; schema=pa.unify_schemas([pq.ParquetFile(path).schema_arrow for path in paths]); seen=set(); written=0; duplicates=0
    try:
        for path in paths:
            parquet=pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=2048):
                table=pa.Table.from_batches([batch]); values=table[key].to_pylist()
                indices=[]
                for index,value in enumerate(values):
                    normalized=str(value) if key=="chunk_id" else int(value)
                    if normalized not in selected: continue
                    if normalized in seen: duplicates+=1; continue
                    seen.add(normalized); indices.append(index)
                if not indices: continue
                filtered=table.take(pa.array(indices,type=pa.int64()))
                if filtered.schema!=schema: filtered=filtered.cast(schema)
                if writer is None: writer=pq.ParquetWriter(temporary,schema)
                writer.write_table(filtered); written+=filtered.num_rows
    finally:
        if writer is not None: writer.close()
    if writer is None or seen!=selected:
        missing=len(selected-seen); temporary.unlink(missing_ok=True)
        raise ValueError(f"canonical {key} filter incomplete: missing={missing}")
    os.replace(temporary,destination)
    return {"selected":len(selected),"written":written,"duplicate_input_rows":duplicates}


def _query_groups(path: Path):
    current=None; rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
        for row in batch.to_pylist():
            query_id=int(row["query_id"])
            if current is not None and query_id!=current:
                yield rows; rows=[]
            current=query_id; rows.append(row)
    if rows: yield rows


def _prepare_package_canonical(config: dict, spec: dict, rankings: Path, group: str="UNION") -> dict:
    chunk_ids=set(); doc_ids=set()
    for batch in pq.ParquetFile(rankings/group/"reranked_chunks.parquet").iter_batches(columns=["chunk_id","doc_id"],batch_size=8192):
        chunk_ids.update(map(str,batch.column(0).to_pylist())); doc_ids.update(map(int,batch.column(1).to_pylist()))
    chunk_paths=[Path(config["inputs"]["pilot_chunks"]),*(Path(parent["root"])/"chunks.parquet" for parent in spec["parents"].values())]
    document_paths=[Path(config["inputs"]["pilot_documents"]),*(Path(parent["root"])/"documents.parquet" for parent in spec["parents"].values())]
    root=Path(config["outputs"]["root"])/"round1"; root.mkdir(parents=True,exist_ok=True)
    chunk_result=_filter_parquet(chunk_paths,root/"combined_chunks.parquet","chunk_id",chunk_ids)
    document_result=_filter_parquet(document_paths,root/"combined_documents.parquet","doc_id",doc_ids)
    return {"candidate_chunk_ids":len(chunk_ids),"candidate_doc_ids":len(doc_ids),"chunks":chunk_result,"documents":document_result}


def _official_id_audit(submission_json: Path, corpus: Path) -> dict:
    remaining={int(doc) for row in _iter_json_array(submission_json) for doc in row["relevant_docs"]}
    expected=len(remaining)
    for batch in pq.ParquetFile(corpus).iter_batches(columns=["id"],batch_size=131072):
        remaining.difference_update(map(int,batch.column(0).to_pylist()))
        if not remaining: break
    return {"unique_emitted_doc_ids":expected,"invalid_official_doc_ids":len(remaining),"passed":not remaining}


def _submission_signature(row: dict) -> tuple[int,list[int],list[tuple[int,str]]]:
    return (int(row["id"]),list(map(int,row["relevant_docs"])),[(int(chunk["doc_id"]),str(chunk["chunk_text"])) for chunk in row["relevant_chunks"]])


def _compare_submission_signatures(reference: Path, current: Path) -> dict[str,int]:
    document_changes=chunk_changes=queries=0
    for expected,observed in zip_longest(_iter_json_array(reference),_iter_json_array(current)):
        if expected is None or observed is None: raise ValueError("submission query counts differ")
        expected_id,expected_docs,expected_chunks=_submission_signature(expected)
        observed_id,observed_docs,observed_chunks=_submission_signature(observed)
        if expected_id!=observed_id: raise ValueError(f"submission query order differs: {expected_id} != {observed_id}")
        document_changes+=expected_docs!=observed_docs; chunk_changes+=expected_chunks!=observed_chunks; queries+=1
    return {"queries_compared":queries,"queries_top10_docs_changed":document_changes,"queries_top20_chunks_changed":chunk_changes}


def _reference_json_sha256(row: dict) -> str:
    if row.get("submission_json"):
        return _digest(Path(row["submission_json"]))
    path=Path(row["submission_zip"])
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=1 or "/" in names[0] or not names[0].endswith(".json"):
            raise ValueError(f"invalid reference submission ZIP: {path}")
        value=hashlib.sha256()
        with archive.open(names[0]) as handle:
            for block in iter(lambda:handle.read(1024*1024),b""): value.update(block)
        return value.hexdigest()


def _build_canonical_store(chunks_path: Path, documents_path: Path, destination: Path) -> dict:
    """Persist canonical text on disk so packaging never inflates the corpus in RAM."""
    temporary=destination.with_suffix(".sqlite.tmp"); temporary.unlink(missing_ok=True)
    connection=sqlite3.connect(temporary)
    connection.executescript(
        "PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF; "
        "CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY,doc_id INTEGER NOT NULL,raw_text TEXT NOT NULL,start_offset INTEGER NOT NULL,end_offset INTEGER NOT NULL);"
        "CREATE INDEX chunks_doc ON chunks(doc_id);"
        "CREATE TABLE documents(doc_id INTEGER PRIMARY KEY,normalized_text TEXT NOT NULL);"
    )
    chunk_rows=0
    for batch in pq.ParquetFile(chunks_path).iter_batches(
        columns=["chunk_id","doc_id","raw_text","start_offset","end_offset"], batch_size=1024
    ):
        rows=[(str(a),int(b),str(c),int(d),int(e)) for a,b,c,d,e in zip(*(column.to_pylist() for column in batch.columns),strict=True)]
        connection.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)",rows); chunk_rows+=len(rows)
    document_rows=0
    for batch in pq.ParquetFile(documents_path).iter_batches(columns=["doc_id","normalized_text"],batch_size=256):
        rows=[(int(a),str(b)) for a,b in zip(*(column.to_pylist() for column in batch.columns),strict=True)]
        connection.executemany("INSERT INTO documents VALUES (?,?)",rows); document_rows+=len(rows)
    connection.commit(); integrity=str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    if integrity!="ok":
        temporary.unlink(missing_ok=True); raise ValueError(f"canonical store integrity failure: {integrity}")
    os.replace(temporary,destination)
    return {"chunks":chunk_rows,"documents":document_rows,"sqlite_integrity":integrity}


def _audit_ranked_provenance_closure(
    ranked_chunks_path: Path,
    canonical_db: Path,
    official_corpus: Path,
    report_path: Path,
) -> dict:
    """Hard-gate packaging on exact ranked/canonical/official provenance closure.

    The canonical store may contain unused valid rows, but every ranked chunk must
    have exactly one canonical owner, agree on its document ID, belong to the
    current official corpus, and reconstruct exactly from the canonical document.
    """
    ranked: dict[str, int] = {}
    ranked_queries: dict[str, set[int]] = {}
    ranked_owner_conflicts: set[str] = set()
    for batch in pq.ParquetFile(ranked_chunks_path).iter_batches(
        columns=["query_id", "chunk_id", "doc_id"], batch_size=8192
    ):
        for query_id, chunk_id, doc_id in zip(
            *(column.to_pylist() for column in batch.columns), strict=True
        ):
            chunk_id = str(chunk_id); doc_id = int(doc_id)
            previous = ranked.setdefault(chunk_id, doc_id)
            if previous != doc_id:
                ranked_owner_conflicts.add(chunk_id)
            ranked_queries.setdefault(chunk_id, set()).add(int(query_id))

    connection = sqlite3.connect(canonical_db)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        canonical_total = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        canonical_distinct = int(connection.execute("SELECT COUNT(DISTINCT chunk_id) FROM chunks").fetchone()[0])
        canonical: dict[str, tuple[int, str, int, int]] = {}
        keys = list(ranked)
        for start in range(0, len(keys), 800):
            window = keys[start:start + 800]
            marks = ",".join("?" for _ in window)
            for chunk_id, doc_id, raw_text, span_start, span_end in connection.execute(
                f"SELECT chunk_id,doc_id,raw_text,start_offset,end_offset FROM chunks WHERE chunk_id IN ({marks})",
                window,
            ):
                canonical[str(chunk_id)] = (int(doc_id), str(raw_text), int(span_start), int(span_end))
        doc_ids = sorted({doc_id for doc_id in ranked.values()})
        documents = _fetch_map(connection, "documents", "doc_id", doc_ids)
    finally:
        connection.close()

    missing = sorted(set(ranked) - set(canonical))
    mismatched = sorted(
        chunk_id for chunk_id, doc_id in ranked.items()
        if chunk_id in canonical and canonical[chunk_id][0] != doc_id
    )
    reconstruction_failures = []
    for chunk_id, (doc_id, raw_text, start, end) in canonical.items():
        document = documents.get(doc_id)
        text = "" if document is None else str(document[1])
        if not raw_text or not text or start < 0 or end < start or end > len(text) or text[start:end] != raw_text:
            reconstruction_failures.append(chunk_id)

    remaining_official = set(doc_ids)
    for batch in pq.ParquetFile(official_corpus).iter_batches(columns=["id"], batch_size=131072):
        remaining_official.difference_update(map(int, batch.column(0).to_pylist()))
        if not remaining_official:
            break
    affected_chunks = set(missing) | set(mismatched) | set(reconstruction_failures) | ranked_owner_conflicts
    affected_queries = sorted({query for chunk_id in affected_chunks for query in ranked_queries.get(chunk_id, ())})
    report = {
        "ranked_unique_chunk_ids": len(ranked),
        "canonical_unique_chunk_ids": canonical_distinct,
        "missing_canonical_ids": len(missing),
        "unexpected_canonical_ids": canonical_total - len(set(ranked) & set(canonical)),
        "ranked_owner_conflicts": len(ranked_owner_conflicts),
        "canonical_duplicate_owners": canonical_total - canonical_distinct,
        "doc_id_mismatches": len(mismatched),
        "non_official_doc_ids": len(remaining_official),
        "reconstruction_failures": len(reconstruction_failures),
        "affected_queries": len(affected_queries),
        "missing_sample": missing[:20],
        "mismatch_sample": mismatched[:20],
        "non_official_doc_id_sample": sorted(remaining_official)[:20],
        "reconstruction_failure_sample": sorted(reconstruction_failures)[:20],
        "sqlite_integrity": integrity,
    }
    report["passed"] = (
        integrity == "ok"
        and report["missing_canonical_ids"] == 0
        and report["ranked_owner_conflicts"] == 0
        and report["canonical_duplicate_owners"] == 0
        and report["doc_id_mismatches"] == 0
        and report["non_official_doc_ids"] == 0
        and report["reconstruction_failures"] == 0
    )
    atomic_json(report_path, report)
    if not report["passed"]:
        raise ValueError(f"ranked provenance closure failed; diagnostic={report_path}; summary={report}")
    return report


def _fetch_map(connection: sqlite3.Connection, table: str, key: str, values: list[str|int]) -> dict:
    unique=list(dict.fromkeys(values))
    output={}
    for start in range(0,len(unique),800):
        window=unique[start:start+800]; marks=",".join("?" for _ in window)
        for row in connection.execute(f"SELECT * FROM {table} WHERE {key} IN ({marks})",window):
            output[row[0]]=row
    return output


def _submission_records_disk(
    query_ids: list[int], documents_path: Path, chunks_path: Path, canonical_db: Path,
    tokenizer, *, document_depth: int, chunk_depth: int, target_tokens: int,
):
    document_groups=iter(_query_groups(documents_path)); chunk_groups=iter(_query_groups(chunks_path))
    connection=sqlite3.connect(canonical_db)
    try:
        for query_id in query_ids:
            document_rows=next(document_groups,None); chunk_rows=next(chunk_groups,None)
            if not document_rows or int(document_rows[0]["query_id"])!=query_id:
                raise ValueError(f"missing document ranking for query {query_id}")
            if not chunk_rows or int(chunk_rows[0]["query_id"])!=query_id:
                raise ValueError(f"missing chunk ranking for query {query_id}")
            document_rows.sort(key=lambda row:(int(row["rank"]),int(row["doc_id"])))
            chunk_rows.sort(key=lambda row:(int(row["rerank_rank"]),str(row["chunk_id"])))
            docs=[]; seen_docs=set()
            for row in document_rows:
                doc_id=int(row["doc_id"])
                if doc_id not in seen_docs: docs.append(doc_id); seen_docs.add(doc_id)
                if len(docs)==document_depth: break
            canonical=_fetch_map(connection,"chunks","chunk_id",[str(row["chunk_id"]) for row in chunk_rows])
            missing=[str(row["chunk_id"]) for row in chunk_rows if str(row["chunk_id"]) not in canonical]
            if missing:
                raise ValueError(f"ranked chunks are absent from canonical provenance: count={len(missing)} sample={missing[:10]}")
            candidate_doc_ids=[int(canonical[str(row["chunk_id"])][1]) for row in chunk_rows]
            stored_docs=_fetch_map(connection,"documents","doc_id",candidate_doc_ids)
            document_text={int(doc_id):str(row[1]) for doc_id,row in stored_docs.items()}
            expander=SourceWindowExpander(document_text,tokenizer,target_tokens)
            emitted=[]; seen_objects=set()
            for row in chunk_rows:
                chunk_id=str(row["chunk_id"]); item=canonical.get(chunk_id)
                if item is None: raise ValueError(f"missing canonical chunk {chunk_id}")
                _,doc_id,raw_text,start,end=item
                if int(row["doc_id"])!=int(doc_id) or str(row["chunk_text"])!=str(raw_text):
                    raise ValueError(f"canonical chunk provenance mismatch: {chunk_id}")
                span=expander.expand(doc_id=int(doc_id),start=int(start),end=int(end),original_text=str(raw_text))
                key=(int(doc_id),span.text)
                if key in seen_objects: continue
                seen_objects.add(key); emitted.append({"doc_id":key[0],"chunk_text":key[1]})
                if len(emitted)==chunk_depth: break
            yield {"id":query_id,"relevant_docs":docs,"relevant_chunks":emitted}
        if next(document_groups,None) is not None or next(chunk_groups,None) is not None:
            raise ValueError("ranking contains unexpected queries")
    finally:
        connection.close()


def _iter_json_array(path: Path):
    """Incrementally decode a compact top-level JSON array."""
    decoder=json.JSONDecoder(); buffer=""; eof=False; started=False
    with path.open("r",encoding="utf-8") as handle:
        while True:
            if not eof and len(buffer)<1024*1024:
                block=handle.read(1024*1024); eof=not block; buffer+=block
            buffer=buffer.lstrip()
            if not started:
                if not buffer and not eof: continue
                if not buffer.startswith("["): raise ValueError("submission root is not an array")
                buffer=buffer[1:]; started=True; continue
            buffer=buffer.lstrip()
            if buffer.startswith("]"): return
            if buffer.startswith(","): buffer=buffer[1:].lstrip()
            try:
                value,index=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof: raise
                block=handle.read(1024*1024); eof=not block; buffer+=block; continue
            yield value; buffer=buffer[index:]


def _stream_validate(
    json_path: Path,
    zip_path: Path,
    query_ids: list[int],
    canonical_db: Path,
    *,
    document_depth: int = 10,
    chunk_depth: int = 20,
) -> dict:
    connection=sqlite3.connect(canonical_db); seen_queries=[]; document_total=chunk_total=source_spans=0; emitted_docs=set()
    try:
        for index,record in enumerate(_iter_json_array(json_path)):
            if set(record)!={"id","relevant_docs","relevant_chunks"}: raise ValueError(f"incorrect fields at query {index}")
            query_id=int(record["id"]); seen_queries.append(query_id)
            docs=record["relevant_docs"]; chunks=record["relevant_chunks"]
            if len(docs)!=document_depth or len(docs)!=len(set(map(int,docs))): raise ValueError(f"invalid documents at query {query_id}")
            if len(chunks)!=chunk_depth: raise ValueError(f"invalid chunk count at query {query_id}")
            objects=[(int(row["doc_id"]),str(row["chunk_text"])) for row in chunks]
            if len(objects)!=len(set(objects)): raise ValueError(f"duplicate chunks at query {query_id}")
            doc_ids=[*map(int,docs),*(doc_id for doc_id,_ in objects)]; stored=_fetch_map(connection,"documents","doc_id",doc_ids)
            if len(stored)!=len(set(doc_ids)): raise ValueError(f"unknown document at query {query_id}")
            for doc_id,text in objects:
                if not text or text not in str(stored[doc_id][1]): raise ValueError(f"provenance mismatch at query {query_id}")
                source_spans+=1
            emitted_docs.update(doc_ids); document_total+=len(docs); chunk_total+=len(chunks)
    finally:
        connection.close()
    if seen_queries!=query_ids: raise ValueError("query IDs/order differ from official input")
    with zipfile.ZipFile(zip_path) as archive:
        names=archive.namelist()
        if names!=[json_path.name]: raise ValueError(f"ZIP root contents invalid: {names}")
        with archive.open(names[0]) as packed, json_path.open("rb") as raw:
            packed_hash=hashlib.sha256(); raw_hash=hashlib.sha256()
            for block in iter(lambda:packed.read(1024*1024),b""): packed_hash.update(block)
            for block in iter(lambda:raw.read(1024*1024),b""): raw_hash.update(block)
            if packed_hash.digest()!=raw_hash.digest(): raise ValueError("ZIP JSON differs from canonical JSON")
    return {"path":str(zip_path),"valid":True,"query_count":len(seen_queries),"document_result_count":document_total,
            "chunk_result_count":chunk_total,"unique_doc_ids":len(emitted_docs),"document_minimum_per_query":document_depth,
            "document_maximum_per_query":document_depth,"chunk_minimum_per_query":chunk_depth,"chunk_maximum_per_query":chunk_depth,
            "invalid_doc_ids":0,"duplicate_doc_ids":0,"duplicate_chunk_objects":0,"chunk_provenance_mismatches":0,
            "source_span_chunks_verified":source_spans}


def _stream_package(config: dict, ranking_root: Path, group: str, name: str, output: Path, canonical_db: Path, tokenizer, *, verify_determinism: bool=True) -> dict:
    query_ids=expected_query_ids(config["inputs"]["queries"]); ranking=ranking_root/group; output.mkdir(parents=True,exist_ok=True)
    json_path=output/f"{name}.json"; zip_path=output/f"{name}.zip"
    kwargs={"query_ids":query_ids,"documents_path":ranking/"reranked_documents.parquet","chunks_path":ranking/"reranked_chunks.parquet",
            "canonical_db":canonical_db,"tokenizer":tokenizer,"document_depth":int(config["submission"]["documents_per_query"]),
            "chunk_depth":int(config["submission"]["chunks_per_query"]),"target_tokens":int(config["submission"]["chunk_expansion_tokens"])}
    write_submission_stream(json_path,_submission_records_disk(**kwargs)); write_deterministic_zip(json_path,zip_path)
    first=(file_sha256(json_path),file_sha256(zip_path)); validation=normalize_validation_result(_stream_validate(
        json_path,zip_path,query_ids,canonical_db,
        document_depth=int(config["submission"]["documents_per_query"]),
        chunk_depth=int(config["submission"]["chunks_per_query"]),
    ))
    if not validation["passed"]:
        raise ValueError(f"submission validation failed: {validation}")
    second=first
    if verify_determinism:
        write_submission_stream(json_path,_submission_records_disk(**kwargs)); write_deterministic_zip(json_path,zip_path)
        second=(file_sha256(json_path),file_sha256(zip_path))
        if first!=second: raise RuntimeError(f"submission regeneration was not deterministic: {group}")
    return {"name":name,"json_path":str(json_path),"zip_path":str(zip_path),"json_sha256":second[0],"zip_sha256":second[1],
            "validation":validation,"determinism_verified":verify_determinism,"model_inference_during_packaging":0,
            "readiness_status":"STRUCTURALLY_VALIDATED_NOT_SCIENTIFICALLY_READY"}


def _external_control_fixture(spec: dict) -> dict:
    """Validate a frozen control replay that is independent of depth artifacts."""
    row=spec.get("control_fixture")
    if not row:
        return {"required":False,"passed":True}
    original=Path(row["original_submission_json"]); replay=Path(row["replay_submission_json"])
    fixture=json.loads(Path(row["fixture_artifact"]).read_text(encoding="utf-8"))
    expected_sources=sorted(map(str,row["sources"])); observed_sources=sorted(map(str,fixture["sources"]))
    original_sha=_digest(original); replay_sha=_digest(replay)
    passed=(
        fixture.get("status")=="READY_FOR_LEADERBOARD"
        and observed_sources==expected_sources
        and fixture.get("original_sha256")==original_sha
        and fixture.get("replay_sha256")==replay_sha
        and original_sha==replay_sha
    )
    return {
        "required":True,"passed":passed,"sources":observed_sources,
        "original_submission_json":str(original),"replay_submission_json":str(replay),
        "original_sha256":original_sha,"replay_sha256":replay_sha,
        "fixture_artifact":str(row["fixture_artifact"]),
    }


def _experiment_inference_accounting(spec: dict) -> dict:
    row=spec.get("experiment_inference")
    if not row:
        return {"score_cache_reused":0,"newly_inferred_pairs":0,"model_inference_during_packaging":0,"passed":True}
    seed=json.loads(Path(row["seed_summary"]).read_text(encoding="utf-8"))
    total=int(seed["total"]); reused=int(seed["reused"]); missing_before=int(seed["missing"])
    newly_inferred=total-reused
    expected=row.get("expected_new_inference")
    passed=missing_before==newly_inferred and (expected is None or int(expected)==newly_inferred)
    return {"score_cache_reused":reused,"newly_inferred_pairs":newly_inferred,
            "missing_before_inference":missing_before,"model_inference_during_packaging":0,
            "seed_summary":str(row["seed_summary"]),"passed":passed}


def _publish_readiness(package: dict, status: str) -> dict:
    """Replace the packager's provisional lifecycle state after all gates run."""
    result=dict(package)
    result["readiness_status"]=status
    return result


def _group_score_audit(score_db: Path, parts_root: Path, sources: list[str]) -> dict:
    connection=sqlite3.connect(score_db); selected=set(sources); candidate_rows=0
    connection.execute("CREATE TEMP TABLE expected(query_id INTEGER,chunk_id TEXT,PRIMARY KEY(query_id,chunk_id)) WITHOUT ROWID")
    for marker_path in sorted(parts_root.glob("*.json")):
        marker=json.loads(marker_path.read_text(encoding="utf-8"))
        if str(marker["source"]) not in selected: continue
        for batch in pq.ParquetFile(marker_path.with_suffix(".parquet")).iter_batches(columns=["query_id","chunk_id"],batch_size=8192):
            rows=[(int(q),str(c)) for q,c in zip(batch.column(0).to_pylist(),batch.column(1).to_pylist(),strict=True)]
            candidate_rows+=len(rows); connection.executemany("INSERT OR IGNORE INTO expected VALUES (?,?)",rows)
    unique=int(connection.execute("SELECT COUNT(*) FROM expected").fetchone()[0])
    matched,missing,unscored,nonfinite,invalid_ms,distinct,min_score,max_score=connection.execute(
        "SELECT COUNT(p.query_id),SUM(p.query_id IS NULL),SUM(p.query_id IS NOT NULL AND p.rerank_score IS NULL),"
        "SUM(p.rerank_score IS NOT NULL AND (p.rerank_score!=p.rerank_score OR ABS(p.rerank_score)>1e308)),"
        "SUM(p.query_id IS NOT NULL AND (p.inference_ms IS NULL OR p.inference_ms<0)),COUNT(DISTINCT p.rerank_score),MIN(p.rerank_score),MAX(p.rerank_score) "
        "FROM expected e LEFT JOIN pairs p ON p.query_id=e.query_id AND p.chunk_id=e.chunk_id"
    ).fetchone(); connection.close()
    result={"candidate_rows":candidate_rows,"unique_candidate_keys":unique,"matched_score_keys":int(matched or 0),"missing_keys":int(missing or 0),
            "unscored_keys":int(unscored or 0),"nonfinite_scores":int(nonfinite or 0),"invalid_inference_timings":int(invalid_ms or 0),
            "distinct_scores":int(distinct or 0),"min_score":float(min_score),"max_score":float(max_score),"reused_scores":int(matched or 0),"new_inference":0}
    result["passed"]=candidate_rows==unique==result["matched_score_keys"] and not any(result[key] for key in ("missing_keys","unscored_keys","nonfinite_scores","invalid_inference_timings")) and result["distinct_scores"]>1
    return result


def _subset_accounting(accounting: dict, sources: list[str]) -> dict:
    rows={source:accounting["sources"][source] for source in sources}
    return {"sources":rows,"official_document_rows":sum(row["document_rows"] for row in rows.values()),
            "usable_documents":sum(row["usable_documents"] for row in rows.values()),"chunks":sum(row["chunks"] for row in rows.values()),
            "duplicate_doc_ids":0,"duplicate_chunk_ids":0,"passed":True}


def run_probe_set(spec_path: str|Path) -> dict:
    spec=yaml.safe_load(Path(spec_path).read_text(encoding="utf-8")); config=_runtime_config(spec); root=Path(spec["outputs"]["root"]); artifacts=Path(spec["outputs"]["artifacts"])
    (root/"round1").mkdir(parents=True,exist_ok=True); artifacts.mkdir(parents=True,exist_ok=True)
    contract=_contract(spec); accounting=_source_accounting(spec); embeddings=_embedding_audit(spec)
    if not contract["passed"] or not accounting["passed"] or not embeddings["passed"]: raise ValueError("probe preflight contract failed")
    parts=_prepare_candidate_parts(spec,root); scores=_merge_score_databases(spec,root/"round1/source_scores.sqlite")
    all_sources=[source for parent in spec["parents"].values() for source in parent["sources"]]
    atomic_json(artifacts/"technical_triage.json",{"healthy_sources":all_sources,"sources":{source:{"healthy":True} for source in all_sources}})
    groups={**{name:row["sources"] for name,row in spec["probes"].items()},**{name:row["sources"] for name,row in spec["controls"].items()}}
    ranking_root=root/"rankings"; build_cached_subset_rankings(config,groups,ranking_root)
    canonical=_prepare_package_canonical(config,spec,ranking_root,spec["canonical_group"])
    canonical_db=root/"round1/canonical_text.sqlite"; canonical_store=_build_canonical_store(root/"round1/combined_chunks.parquet",root/"round1/combined_documents.parquet",canonical_db)
    from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
    from transformers import AutoTokenizer
    tokenizer=HuggingFaceOffsetTokenizer(AutoTokenizer.from_pretrained(config["models"]["embedder"]["name"],revision=config["models"]["embedder"]["revision"],local_files_only=True))
    controls={}; control_root=root/"controls"
    for group,row in spec["controls"].items():
        package=_stream_package(config,ranking_root,group,f"{spec['experiment']}_{group}",control_root,canonical_db,tokenizer,verify_determinism=False)
        reference=_reference_json_sha256(row); package["semantic_replay"]={"passed":reference==package["json_sha256"],"comparison":"byte-identical canonical JSON","reference_sha256":reference,"replay_sha256":package["json_sha256"]}
        controls[group]=package; atomic_json(artifacts/f"submission_{group.lower()}.json",package)
    shared_cache=audit_score_cache(root/"round1/source_scores.sqlite",root/"round1/source_candidate_parts")
    baseline=Path(spec["comparison_submission_json"]); outputs={}; any_failure=False
    for group,row in spec["probes"].items():
        package=_stream_package(config,ranking_root,group,row["submission_name"],Path(spec["outputs"]["submissions"]),canonical_db,tokenizer)
        changes=_compare_submission_signatures(baseline,Path(package["json_path"]))
        scoped_scores=_group_score_audit(root/"round1/source_scores.sqlite",root/"round1/source_candidate_parts",row["sources"])
        official=_official_id_audit(Path(package["json_path"]),Path(config["inputs"]["corpus"])); subset=_subset_accounting(accounting,row["sources"])
        failures=[]
        if not shared_cache["passed"]: failures.append("shared_score_cache")
        if not scoped_scores["passed"]: failures.append("scoped_score_cache")
        if not all(control["semantic_replay"]["passed"] for control in controls.values()): failures.append("full_control_replay")
        if not official["passed"] or not package["validation"]["valid"] or not package["determinism_verified"]: failures.append("package_validation")
        audit={"status":"READY_FOR_LEADERBOARD" if not failures else "NEEDS_AGENT","passed":not failures,"failures":failures,"experiment_contract":contract,
               "source_membership":row["sources"],"corpus_accounting":subset,"embedding_reconciliation":{"reused":subset["chunks"],"computed":0,"parent_audit":embeddings},
               "candidate_score_reconciliation":scoped_scores,"shared_score_cache":shared_cache,"control_replays":controls,"ranking_change_vs_full":changes,
               "official_id_audit":official,"structural_validation":package["validation"],"determinism_verified":package["determinism_verified"],"submission":{"path":package["zip_path"],"sha256":package["zip_sha256"]}}
        atomic_json(row["audit"],audit); outputs[group]={"status":audit["status"],"sources":row["sources"],"corpus":subset,"embeddings_reused":subset["chunks"],"embeddings_new":0,
            "candidate_rows":scoped_scores["candidate_rows"],"reranker_scores_reused":scoped_scores["reused_scores"],"reranker_scores_new":0,"ranking_change_vs_full":changes,"submission":package,"audit":row["audit"]}
        any_failure|=bool(failures)
    report={"experiment":spec["experiment"],"status":"NEEDS_AGENT" if any_failure else "READY_FOR_LEADERBOARD","new_acquisition":0,"new_embeddings":0,"new_reranker_inference":0,
            "full_source_accounting":accounting,"candidate_cache":parts,"score_cache":scores,"controls":controls,"canonical":{**canonical,"disk_backed_store":canonical_store},"probes":outputs}
    atomic_json(spec["outputs"]["report"],report)
    if any_failure: raise RuntimeError("one or more probe scientific gates failed")
    return report


def run_union(spec_path: str|Path) -> dict:
    spec=yaml.safe_load(Path(spec_path).read_text(encoding="utf-8")); config=_runtime_config(spec); root=Path(spec["outputs"]["root"]); artifacts=Path(spec["outputs"]["artifacts"])
    (root/"round1").mkdir(parents=True,exist_ok=True); artifacts.mkdir(parents=True,exist_ok=True)
    contract=_contract(spec)
    if not contract["passed"]: raise ValueError(f"parent experiment contracts differ: {contract}")
    accounting=_source_accounting(spec)
    if not accounting["passed"]: raise ValueError(f"source union identity failure: {accounting}")
    embeddings=_embedding_audit(spec)
    parts=_prepare_candidate_parts(spec,root)
    scores=_merge_score_databases(spec,root/"round1/source_scores.sqlite")
    sources=[source for parent in spec["parents"].values() for source in parent["sources"]]
    atomic_json(artifacts/"technical_triage.json",{"healthy_sources":sources,"sources":{source:{"healthy":True} for source in sources}})
    groups={"UNION":sources,**{name:row["sources"] for name,row in spec["controls"].items()}}
    ranking_root=root/"rankings"; rankings=build_cached_subset_rankings(config,groups,ranking_root)
    canonical=_prepare_package_canonical(config,spec,ranking_root)
    canonical_db=root/"round1/canonical_text.sqlite"
    canonical_store=_build_canonical_store(root/"round1/combined_chunks.parquet",root/"round1/combined_documents.parquet",canonical_db)
    from src.indexing.tokenizer_validation import HuggingFaceOffsetTokenizer
    from transformers import AutoTokenizer
    tokenizer=HuggingFaceOffsetTokenizer(AutoTokenizer.from_pretrained(config["models"]["embedder"]["name"],revision=config["models"]["embedder"]["revision"],local_files_only=True))
    union=_stream_package(config,ranking_root,"UNION",spec["submission_name"],Path(spec["outputs"]["submissions"]),canonical_db,tokenizer)
    atomic_json(artifacts/"submission_union.json",union)
    control_root=root/"controls"; controls={}
    for group,control in spec["controls"].items():
        controls[group]=_stream_package(config,ranking_root,group,f"{spec['experiment']}_{group}",control_root,canonical_db,tokenizer,verify_determinism=False)
        reference_sha256=_reference_json_sha256(control)
        controls[group]["semantic_replay"]={"passed":reference_sha256==controls[group]["json_sha256"],"comparison":"byte-identical canonical JSON","reference_sha256":reference_sha256,"replay_sha256":controls[group]["json_sha256"]}
        atomic_json(artifacts/f"submission_{group.lower()}.json",controls[group])
    cache=audit_score_cache(root/"round1/source_scores.sqlite",root/"round1/source_candidate_parts")
    official=_official_id_audit(Path(union["json_path"]),Path(config["inputs"]["corpus"]))
    changes=_compare_submission_signatures(Path(spec["comparison_submission_json"]),Path(union["json_path"]))
    external_control=_external_control_fixture(spec)
    inference=_experiment_inference_accounting(spec)
    failures=[]
    for name,value in (("contract",contract),("source_accounting",accounting),("embeddings",embeddings),("score_cache",cache),("official_ids",official)):
        if not value.get("passed",False): failures.append(name)
    expected_pairs=int(spec["expected_candidate_pairs"])
    if parts["candidate_rows"]!=expected_pairs or scores["total"]!=expected_pairs or scores["new_inference"]!=0: failures.append("candidate_or_score_accounting")
    if not all(row["semantic_replay"]["passed"] for row in controls.values()): failures.append("parent_control_replay")
    if not external_control["passed"]: failures.append("external_control_replay")
    if not inference["passed"]: failures.append("experiment_inference_accounting")
    if spec.get("require_comparison_byte_identical") and (
        changes["queries_top10_docs_changed"] or changes["queries_top20_chunks_changed"]
        or _digest(Path(spec["comparison_submission_json"]))!=union["json_sha256"]
    ): failures.append("comparison_not_byte_identical")
    if not union["validation"]["valid"] or not union["determinism_verified"]: failures.append("structural_or_determinism")
    audit={"status":"READY_FOR_LEADERBOARD" if not failures else "NEEDS_AGENT","passed":not failures,"failures":failures,"experiment_contract":contract,"source_union":accounting,"embedding_reconciliation":embeddings,"candidate_reuse":parts,"score_cache_integrity":cache,"score_reuse":{"exact_parent_scores_reused":scores["total"],"new_inference_required":0,**scores},"experiment_inference_accounting":inference,"parent_control_replays":controls,"external_control_replay":external_control,"package_canonical":{**canonical,"disk_backed_store":canonical_store},"official_id_audit":official,"structural_validation":union["validation"],"determinism_verified":union["determinism_verified"],"ranking_change_vs_comparison":changes,"submission":{"path":union["zip_path"],"sha256":union["zip_sha256"]}}
    atomic_json(spec["outputs"]["audit"],audit)
    # Packaging is intentionally structural-only until every scientific gate has
    # run.  Publish the resulting lifecycle state in the final report rather
    # than leaking the packager's provisional status into a READY experiment.
    union=_publish_readiness(union,audit["status"])
    atomic_json(artifacts/"submission_union.json",union)
    report={"experiment":spec["experiment"],"scientific_question":spec["scientific_question"],"status":audit["status"],"new_acquisition":0,"new_embeddings":0,"new_reranker_inference":inference["newly_inferred_pairs"],"model_inference_during_packaging":0,"source_union":accounting,"candidate_pairs":parts["candidate_rows"],"reranker_scores_reused":scores["total"],"experiment_inference_accounting":inference,"ranking_change_vs_comparison":changes,"submission":union,"audit":spec["outputs"]["audit"]}
    atomic_json(spec["outputs"]["report"],report)
    if failures: raise RuntimeError(f"union scientific gate failed: {failures}")
    return report
