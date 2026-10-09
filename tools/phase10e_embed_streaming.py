#!/usr/bin/env python3
from __future__ import annotations

import argparse,gc,json,os,sys,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.indexing.embedder import normalize_text
from src.indexing.streaming_embeddings import _atomic_json,prepare_streaming_cache,process_memory,recovery_paths,validate_profile
from src.source_census.pipeline import atomic_json,load_config


def gpu_memory(torch):
    return {"gpu_allocated_mib":round(torch.cuda.memory_allocated()/2**20,2),"gpu_reserved_mib":round(torch.cuda.memory_reserved()/2**20,2)}


def load_model(config,measurements):
    import torch
    from sentence_transformers import SentenceTransformer
    model_cfg=config["models"]["embedder"]
    measurements.append({"phase":"before_model_load",**process_memory(),**gpu_memory(torch)})
    started=time.perf_counter(); model=SentenceTransformer(model_cfg["name"],revision=model_cfg["revision"],device="cpu")
    model.max_seq_length=int(model_cfg["max_length"]); model.half(); model.to("cuda")
    measurements.append({"phase":"model_loaded","seconds":time.perf_counter()-started,**process_memory(),**gpu_memory(torch)})
    return model,torch


def profile(config,chunks_path,limit,output):
    output.mkdir(parents=True,exist_ok=True); dimension=int(config["models"]["embedder"]["dimension"]); batch_rows=128
    rows=min(limit,int(pq.ParquetFile(chunks_path).metadata.num_rows)); vectors=np.memmap(output/"profile.f32",dtype=np.float32,mode="w+",shape=(rows,dimension))
    measurements=[{"phase":"process_start",**process_memory()}]; model,torch=load_model(config,measurements)
    processed=0; checkpoints=[]; started=time.perf_counter(); tokenization_measured=False
    for batch in pq.ParquetFile(chunks_path).iter_batches(batch_size=batch_rows,columns=["normalized_text"]):
        texts=[normalize_text(str(value)) for value in batch.column(0).to_pylist()][:max(0,rows-processed)]
        if not texts: break
        measurements.append({"phase":"chunk_text_batch_loaded","processed_rows":processed,**process_memory(),**gpu_memory(torch)}) if processed==0 else None
        if not tokenization_measured:
            tokens=model.tokenizer(texts[:int(config["models"]["embedder"]["batch_size"])],padding=True,truncation=True,max_length=int(config["models"]["embedder"]["max_length"]),return_tensors="pt")
            measurements.append({"phase":"tokenization","processed_rows":processed,**process_memory(),**gpu_memory(torch)}); del tokens; tokenization_measured=True
        encoded=model.encode(texts,batch_size=int(config["models"]["embedder"]["batch_size"]),show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True)
        end=processed+len(texts); vectors[processed:end]=np.asarray(encoded,dtype=np.float32); vectors.flush(); processed=end
        point={"processed_rows":processed,"elapsed_seconds":time.perf_counter()-started,**process_memory(),**gpu_memory(torch)}; checkpoints.append(point)
        if len(checkpoints)==1: measurements.append({"phase":"first_batch_written",**point})
        if processed>=rows: break
    vectors.flush(); del vectors,model; torch.cuda.empty_cache(); gc.collect()
    result={"completed":processed==rows,"rows":rows,"dimension":dimension,"batch_rows":batch_rows,"inference_batch_size":int(config["models"]["embedder"]["batch_size"]),"num_workers":0,"prefetching":False,"accumulates_outputs_in_python":False,"measurements":measurements,"checkpoints":checkpoints,"runtime_seconds":time.perf_counter()-started}
    result["memory_gate"]=validate_profile(result,rows); _atomic_json(output/"profile.json",result); print(json.dumps(result,indent=2)); return 0 if result["memory_gate"]["passed"] else 2


def encode(config,run_state,max_new_rows=0):
    import torch
    paths=recovery_paths(config); checkpoint=prepare_streaming_cache(config); rows=int(checkpoint["signature"]["rows"]); dimension=int(checkpoint["signature"]["dimension"])
    profile_path=Path(config["embedding_recovery"]["profile_artifact"]); profile_data=json.loads(profile_path.read_text(encoding="utf-8")); gate=validate_profile(profile_data,int(checkpoint["missing_rows"]))
    if not gate["passed"]: raise RuntimeError(f"embedding memory profile gate failed: {gate}")
    equivalence_result=json.loads(Path(config["embedding_recovery"]["equivalence_artifact"]).read_text(encoding="utf-8"))
    if not equivalence_result.get("passed"): raise RuntimeError(f"embedding equivalence gate failed: {equivalence_result}")
    # Load CUDA before mapping the full output. On the 5.69 GiB host, mapping
    # the 537 MiB recovery file first reproducibly caused c10.dll to access-
    # violate during model load, while the same model loaded safely in the
    # bounded profile with only a tiny output mapping.
    measurements=[{"phase":"process_start",**process_memory()}]; model,torch=load_model(config,measurements)
    vectors=np.memmap(paths["partial"],dtype=np.float32,mode="r+",shape=(rows,dimension)); status=np.memmap(paths["status"],dtype=np.uint8,mode="r+",shape=(rows,))
    initial=int(status.sum()); processed_new=0; position=0; started=time.perf_counter(); checkpoint_rows=[]
    for batch in pq.ParquetFile(config["outputs"]["chunks"]).iter_batches(batch_size=128,columns=["normalized_text"]):
        texts=batch.column(0).to_pylist(); indices=[position+i for i in range(len(texts)) if not status[position+i]]
        if max_new_rows: indices=indices[:max(0,max_new_rows-processed_new)]
        if indices:
            pending=[normalize_text(str(texts[index-position])) for index in indices]
            encoded=model.encode(pending,batch_size=int(config["models"]["embedder"]["batch_size"]),show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True)
            vectors[indices]=np.asarray(encoded,dtype=np.float32); vectors.flush(); status[indices]=1; status.flush(); processed_new+=len(indices)
        position+=len(texts)
        if indices:
            completed=initial+processed_new; point={"processed_rows":processed_new,"completed_rows":completed,**process_memory(),**gpu_memory(torch)}; checkpoint_rows.append(point)
            checkpoint.update({"embedded_rows":completed-int(checkpoint["reused_rows"]),"completed_rows":completed,"last_scanned_row":position,"last_memory":point}); _atomic_json(paths["checkpoint"],checkpoint)
            if run_state: atomic_json(run_state,{"stage":"EMBEDDINGS","status":"RUNNING","total":int(checkpoint["missing_rows"]),"done":checkpoint["embedded_rows"],"reused_embeddings":checkpoint["reused_rows"],"rss_mib":point["rss_mib"],"private_mib":point["private_mib"],"page_faults":point["page_faults"],"gpu_allocated_mib":point["gpu_allocated_mib"],"gpu_reserved_mib":point["gpu_reserved_mib"],"last_successful_checkpoint":checkpoint["embedded_rows"]})
        del texts,indices; gc.collect() if position%4096<128 else None
        if max_new_rows and processed_new>=max_new_rows: break
    completed_total=int(status.sum())
    if completed_total!=rows:
        if not max_new_rows: raise RuntimeError("streaming embedding status is incomplete")
        del vectors,status,model; torch.cuda.empty_cache(); gc.collect()
        checkpoint.update({"bounded_gate_rows":processed_new,"bounded_gate_complete":True,"last_bounded_gate_memory":checkpoint_rows[-1] if checkpoint_rows else None}); _atomic_json(paths["checkpoint"],checkpoint)
        print(json.dumps({"bounded_gate_complete":True,"new_rows":processed_new,"completed_rows":completed_total,"remaining_rows":rows-completed_total,"memory":checkpoint_rows},indent=2)); return 0
    vectors.flush(); del vectors,status,model; torch.cuda.empty_cache(); gc.collect()
    if paths["final"].exists(): raise FileExistsError(f"refusing to overwrite unverified embedding artifact: {paths['final']}")
    os.replace(paths["partial"],paths["final"])
    checkpoint.update({"complete":True,"completed_rows":rows,"runtime_seconds":time.perf_counter()-started,"memory_gate":gate,"measurements":measurements,"checkpoint_memory":checkpoint_rows}); _atomic_json(paths["checkpoint"],checkpoint)
    summary={"chunks":rows,"dimension":dimension,"reused_embeddings":checkpoint["reused_rows"],"new_embeddings":checkpoint["missing_rows"],"runtime_seconds":checkpoint["runtime_seconds"],"path":str(paths["final"]),"streaming":True,"transactional_checkpoint":str(paths["checkpoint"]),"memory_gate":gate}
    atomic_json(Path(config["outputs"]["artifacts"])/"embedding_summary.json",summary); print(json.dumps(summary,indent=2)); return 0


def equivalence(config,candidate_path,reference_path,rows,output):
    dimension=int(config["models"]["embedder"]["dimension"])
    candidate=np.memmap(candidate_path,dtype=np.float32,mode="r",shape=(rows,dimension))
    reference=np.memmap(reference_path,dtype=np.float32,mode="r").reshape(-1,dimension)
    values=np.asarray(candidate); expected=np.asarray(reference[:rows]); differences=np.abs(values-expected); cosine=np.sum(values*expected,axis=1)
    result={"contract":{"model":config["models"]["embedder"]["name"],"revision":config["models"]["embedder"]["revision"],"dimension":dimension,"batch_size":config["models"]["embedder"]["batch_size"],"max_length":config["models"]["embedder"]["max_length"],"precision":"FP16 CUDA","normalization":True},"rows":rows,"exact_rows":int(np.all(values==expected,axis=1).sum()),"max_absolute_difference":float(differences.max()),"mean_absolute_difference":float(differences.mean()),"minimum_cosine":float(cosine.min()),"absolute_tolerance":float(config["embedding_recovery"]["measured_absolute_tolerance"]),"minimum_cosine_required":float(config["embedding_recovery"]["measured_minimum_cosine"])}
    result["passed"]=result["max_absolute_difference"]<=result["absolute_tolerance"] and result["minimum_cosine"]>=result["minimum_cosine_required"]
    _atomic_json(output,result); print(json.dumps(result,indent=2)); return 0 if result["passed"] else 2


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=["prepare","profile","equivalence","encode"]); parser.add_argument("--config",required=True); parser.add_argument("--run-state"); parser.add_argument("--chunks"); parser.add_argument("--limit",type=int,default=2048); parser.add_argument("--output"); parser.add_argument("--candidate"); parser.add_argument("--reference"); parser.add_argument("--max-new-rows",type=int,default=0)
    args=parser.parse_args(); config=load_config(args.config)
    if args.mode=="prepare": print(json.dumps(prepare_streaming_cache(config),indent=2)); return 0
    if args.mode=="profile": return profile(config,Path(args.chunks),args.limit,Path(args.output))
    if args.mode=="equivalence": return equivalence(config,Path(args.candidate),Path(args.reference),args.limit,Path(args.output))
    return encode(config,Path(args.run_state) if args.run_state else None,args.max_new_rows)
if __name__=="__main__": raise SystemExit(main())
