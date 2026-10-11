from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.source_census.phase10e import ensure_reranker_equivalence_control
from src.source_census.pipeline import atomic_json, build_source_candidates, load_config, score_database_status


def _filter_table(source: Path, destination: Path, key: str, selected: set[int]) -> int:
    temporary=destination.with_suffix(destination.suffix+".tmp"); temporary.unlink(missing_ok=True)
    writer=None; written=0
    try:
        parquet=pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=1024):
            table=pa.Table.from_batches([batch]); values=table[key].to_pylist()
            indices=[index for index,value in enumerate(values) if int(value) in selected]
            if not indices: continue
            filtered=table.take(pa.array(indices,type=pa.int64()))
            if writer is None: writer=pq.ParquetWriter(temporary,filtered.schema)
            writer.write_table(filtered); written+=filtered.num_rows
    finally:
        if writer is not None: writer.close()
    if writer is None: raise ValueError(f"checkpoint filter produced no rows: {source}")
    os.replace(temporary,destination); return written


def derive_checkpoint_config(base_path: str|Path, depth: int) -> tuple[dict[str,Any],Path]:
    config=load_config(base_path); root=Path(f"data/source_census/phase10f_medlatec_{depth}")
    artifacts=Path(f"artifacts/source_census/phase10f_medlatec_{depth}")
    config["focused_scaling"]["experiment"]=f"phase10f_medlatec_{depth}"
    config["outputs"]={**config["outputs"],"root":str(root),"artifacts":str(artifacts),
        "documents":str(root/"documents.parquet"),"chunks":str(root/"chunks.parquet")}
    config["phase10f_medlatec"]["previous_score_database"]="data/source_census/phase10f_medlatec_10000/round1/source_scores.sqlite"
    config["models"]["reranker"]["equivalence"]["control_artifact"]=str(artifacts/"reranker_equivalence_control.json")
    config["models"]["reranker"]["equivalence"]["canonical_control_artifact"]="artifacts/source_census/phase10f_medlatec_10000/reranker_equivalence_control.json"
    runtime=artifacts/"runtime_config.yaml"; artifacts.mkdir(parents=True,exist_ok=True)
    runtime.write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding="utf-8")
    config["_config_path"]=str(runtime); return config,runtime


def prepare_checkpoint_corpus(base_config: str|Path, depth: int=7500) -> dict[str,Any]:
    config,runtime=derive_checkpoint_config(base_config,depth)
    manifest=json.loads(Path(config["phase10f_medlatec"]["manifest"]).read_text(encoding="utf-8"))
    selected=set(map(int,manifest["sources"]["medlatec.vn"]["depth_checkpoints"][str(depth)]["target_ids"]))
    full_root=Path("data/source_census/phase10f_medlatec_10000"); root=Path(config["outputs"]["root"]); round1=root/"round1"
    root.mkdir(parents=True,exist_ok=True); round1.mkdir(parents=True,exist_ok=True)
    documents=_filter_table(full_root/"documents.parquet",root/"documents.parquet","doc_id",selected)
    chunks=_filter_table(full_root/"chunks.parquet",root/"chunks.parquet","doc_id",selected)
    full_chunks=pq.ParquetFile(full_root/"chunks.parquet"); dimension=int(config["models"]["embedder"]["dimension"])
    full_vectors=np.memmap(full_root/"round1/chunk_embeddings.f32",dtype=np.float32,mode="r",shape=(full_chunks.metadata.num_rows,dimension))
    temporary=round1/"chunk_embeddings.f32.tmp"; temporary.unlink(missing_ok=True)
    output=np.memmap(temporary,dtype=np.float32,mode="w+",shape=(chunks,dimension)); cursor=0; position=0
    for batch in full_chunks.iter_batches(columns=["doc_id"],batch_size=1024):
        values=batch.column(0).to_pylist(); indices=[i for i,value in enumerate(values) if int(value) in selected]
        if indices:
            output[cursor:cursor+len(indices)]=full_vectors[position+np.asarray(indices,dtype=np.int64)]; cursor+=len(indices)
        position+=len(values)
    output.flush(); del output,full_vectors
    if cursor!=chunks: raise ValueError(f"embedding subset rows {cursor} != chunks {chunks}")
    os.replace(temporary,round1/"chunk_embeddings.f32")
    np.ones(chunks,dtype=np.uint8).tofile(round1/"chunk_embeddings.streaming.status.u8")
    atomic_json(Path(config["outputs"]["artifacts"])/"embedding_summary.json",{
        "chunks":chunks,"dimension":dimension,"reused_embeddings":chunks,"new_embeddings":0,
        "path":str(round1/"chunk_embeddings.f32"),"streaming":True,"complete":True,
    })
    atomic_json(Path(config["outputs"]["artifacts"])/"technical_triage.json",{
        "healthy_sources":["medlatec.vn"],"sources":{"medlatec.vn":{"healthy":True,"usable_documents":documents}}
    })
    result={"depth":depth,"official_ids":len(selected),"usable_documents":documents,"chunks":chunks,
            "embeddings_reused":chunks,"embeddings_new":0,"runtime_config":str(runtime)}
    atomic_json(Path(config["outputs"]["artifacts"])/"checkpoint_corpus.json",result); return result


def seed_checkpoint_scores(config: dict[str,Any], sources: list[str|Path]) -> dict[str,Any]:
    destination=Path(config["outputs"]["root"])/"round1/source_scores.sqlite"; connection=sqlite3.connect(destination)
    connection.execute("CREATE TEMP TABLE reused_keys(query_id INTEGER,chunk_id TEXT,PRIMARY KEY(query_id,chunk_id)) WITHOUT ROWID")
    source_counts={}
    for index,path in enumerate(sources):
        alias=f"src{index}"; connection.execute(f"ATTACH DATABASE ? AS {alias}",(str(Path(path).resolve()),))
        rows=connection.execute(
            f"SELECT d.query_id,d.chunk_id,s.rerank_score,s.inference_ms,s.query_text,s.chunk_text "
            f"FROM pairs d JOIN {alias}.pairs s ON s.query_id=d.query_id AND s.chunk_id=d.chunk_id "
            "WHERE s.rerank_score IS NOT NULL"
        ).fetchall()
        for query_id,chunk_id,score,ms,query_text,chunk_text in rows:
            current=connection.execute("SELECT query_text,chunk_text FROM pairs WHERE query_id=? AND chunk_id=?",(query_id,chunk_id)).fetchone()
            if current!=(query_text,chunk_text): raise ValueError(f"exact-key text mismatch: {(query_id,chunk_id)}")
        connection.executemany("INSERT OR IGNORE INTO reused_keys VALUES (?,?)",[(query_id,chunk_id) for query_id,chunk_id,*_ in rows])
        connection.executemany("UPDATE pairs SET rerank_score=?,inference_ms=? WHERE query_id=? AND chunk_id=? AND rerank_score IS NULL",
                               [(score,ms,query_id,chunk_id) for query_id,chunk_id,score,ms,_,_ in rows])
        connection.commit(); source_counts[str(path)]=len(rows); connection.execute(f"DETACH DATABASE {alias}")
    reused=int(connection.execute("SELECT COUNT(*) FROM reused_keys").fetchone()[0])
    total=int(connection.execute("SELECT COUNT(*) FROM pairs").fetchone()[0]); done=int(connection.execute("SELECT COUNT(*) FROM pairs WHERE rerank_score IS NOT NULL").fetchone()[0])
    integrity=str(connection.execute("PRAGMA integrity_check").fetchone()[0]); connection.close()
    result={"total":total,"before":0,"expected_exact_matches":reused,"reused":reused,"after":done,
            "missing":total-reused,"remaining_now":total-done,"exact_match_mismatches":0,"incorrectly_seeded_nonmatches":0,
            "source_reuse":source_counts,"integrity":integrity}
    atomic_json(Path(config["outputs"]["artifacts"])/"rerank_seed_summary.json",result)
    ensure_reranker_equivalence_control(config); return result


def prepare_checkpoint_candidates(base_config: str|Path,depth: int=7500) -> dict[str,Any]:
    config,runtime=derive_checkpoint_config(base_config,depth)
    summary=build_source_candidates(config,groups={"MEDLATEC":["medlatec.vn"]})
    seed=seed_checkpoint_scores(config,[
        "data/source_census/phase10f_medlatec_10000/round1/source_scores.sqlite",
        "data/source_census/phase10e_g1a_10010/round1/source_scores.sqlite",
    ])
    return {"runtime_config":str(runtime),"candidates":summary,"seed":seed,"status":score_database_status(config,verify_integrity=True)}
