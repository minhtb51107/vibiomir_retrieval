import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing.streaming_embeddings import prepare_streaming_cache, recovery_paths, validate_profile


def _config(tmp_path: Path) -> dict:
    old_chunks=tmp_path/"old.parquet"; chunks=tmp_path/"chunks.parquet"; old_vectors=tmp_path/"old.f32"
    pq.write_table(pa.table({"chunk_id":["a","b"]}),old_chunks)
    pq.write_table(pa.table({"chunk_id":["b","c","a"],"normalized_text":["B","C","A"]}),chunks)
    vectors=np.memmap(old_vectors,dtype=np.float32,mode="w+",shape=(2,2)); vectors[:]=[[1,2],[3,4]]; vectors.flush(); del vectors
    return {"focused_scaling":{"phase_key":"phase10e_g1a","experiment":"test","group":"G1A"},"phase10e_g1a":{"targets":{"example.org":3},"previous_chunks":str(old_chunks),"previous_embeddings":str(old_vectors)},"outputs":{"root":str(tmp_path/"output"),"chunks":str(chunks)},"models":{"embedder":{"name":"BAAI/bge-m3","revision":"fixed","dimension":2,"batch_size":4,"max_length":512}}}


def test_prepare_streaming_cache_reuses_only_exact_keys(tmp_path):
    config=_config(tmp_path); checkpoint=prepare_streaming_cache(config); paths=recovery_paths(config)
    values=np.memmap(paths["partial"],dtype=np.float32,mode="r",shape=(3,2)); status=np.memmap(paths["status"],dtype=np.uint8,mode="r",shape=(3,))
    assert checkpoint["reused_rows"]==2
    assert checkpoint["missing_rows"]==1
    assert status.tolist()==[1,0,1]
    assert values.tolist()==[[3,4],[0,0],[1,2]]
    assert prepare_streaming_cache(config)==checkpoint


def test_memory_profile_gate_requires_stable_bounded_streaming():
    measurements=[{"phase":"model_loaded","private_mib":3000.0}]
    stable={"completed":True,"rows":2048,"measurements":measurements,"checkpoints":[{"processed_rows":i*128,"rss_mib":500.0,"private_mib":3200.0+i*.25} for i in range(1,17)]}
    assert validate_profile(stable,105050)["passed"]
    growing={**stable,"checkpoints":[{"processed_rows":i*128,"rss_mib":500.0,"private_mib":3200.0+i*10} for i in range(1,17)]}
    assert not validate_profile(growing,105050)["passed"]
