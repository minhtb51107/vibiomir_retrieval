#!/usr/bin/env python3
"""Build and maintain the human-facing submission lifecycle registry."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STATUSES=("READY_TO_SUBMIT","SUBMITTED_VALID","VALID_CONTROL","SUPERSEDED_VALID","INVALID_DO_NOT_SUBMIT","UNKNOWN_REVIEW_REQUIRED")
LOCAL_RETAINED="RETAINED_LOCAL"
LOCAL_PURGED="INTENTIONALLY_PURGED"
FOLDERS={
    "READY_TO_SUBMIT":"00_READY_TO_UPLOAD","SUBMITTED_VALID":"10_SUBMITTED_VALID",
    "VALID_CONTROL":"20_VALID_CONTROLS","SUPERSEDED_VALID":"30_SUPERSEDED_VALID",
    "INVALID_DO_NOT_SUBMIT":"90_INVALID_DO_NOT_SUBMIT","UNKNOWN_REVIEW_REQUIRED":"99_REVIEW_REQUIRED",
}
READY=set()
INVALID={"phase10d_DEPTH1000_G1A.zip","phase10d_DEPTH1000_G5A.zip","phase10d_DEPTH1000_G6B.zip","smoke_medlatec_do_not_submit.zip"}
SUPERSEDED={
    "phase9_B_pure_top3_mean.zip","phase9_C_max3_best_chunk.zip","phase9_D_controlled_best_chunk.zip",
    *(f"phase10a_E{i}_{suffix}.zip" for i,suffix in [(1,"docs20"),(2,"docs50"),(3,"docs_max"),(4,"chunks20"),(5,"chunks_more"),(6,"chunk_expand_768"),(7,"chunk_expand_1024"),(8,"hybrid"),(9,"sparse")]),
    "phase10d_R3_G1B.zip","phase10d_R3_G5B.zip","phase10d_R3_G6A.zip",
}
CONTROLS={"phase10d_R3_G1A_control_replay.zip","phase10e_control_G5A.zip"}

ALIASES={
    "phase10b3_B0_fixed_baseline":"B0 fixed baseline","phase10b3_S1_cnkang":"S1 cnkang",
    "phase10b3_S2_familydoctor":"S2 familydoctor","phase10b3_S3_ahospital":"S3 a-hospital",
    "phase10b3_S4_suckhoecongdong":"S4 suckhoecongdongonline","phase10b3_S5_120ask_shortqa":"S5 120ask short-QA",
    "phase10b3_S6_ask39_shortqa":"S6 ask39 short-QA","phase10c1_C1_S4_5k":"C1 S4-5K",
    "phase10c1_C2_S2_5k":"C2 S2-5K",
}


def digest(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): value.update(block)
    return value.hexdigest()


def classify(path: Path, leaderboard: dict[str,dict[str,str]]) -> str:
    name=path.name; stem=path.stem
    if name in INVALID: return "INVALID_DO_NOT_SUBMIT"
    if name in CONTROLS: return "VALID_CONTROL"
    key=ALIASES.get(stem,stem)
    if key in leaderboard: return "SUBMITTED_VALID"
    if name in READY: return "READY_TO_SUBMIT"
    if name in SUPERSEDED: return "SUPERSEDED_VALID"
    return "UNKNOWN_REVIEW_REQUIRED"


def phase_for(stem: str) -> str:
    if stem.startswith("phase10e_"): return "Phase 10E"
    if stem.startswith("phase10d_DEPTH"): return "Phase 10D depth1000"
    if stem.startswith("phase10d_R3"): return "Phase 10D Round 3"
    if stem.startswith("phase10d_R2"): return "Phase 10D Round 2"
    if stem.startswith("phase10d_R1"): return "Phase 10D Round 1"
    if stem.startswith("phase10c1_"): return "Phase 10C1"
    if stem.startswith("phase10b3_"): return "Phase 10B3"
    if stem.startswith("phase10a_"): return "Phase 10A"
    if stem.startswith("phase9_"): return "Phase 9"
    return "CONTROL"


def report_for(stem: str) -> tuple[str,str]:
    if stem.startswith("phase10e_G1A_10010"): return "artifacts/source_census/phase10e_g1a_10010_report.json","artifacts/validation/phase10e_g1a_10010_pre_submit_audit.json"
    if stem.startswith("phase10e_G5A_11200"): return "artifacts/source_census/phase10e_g5a_11200_report.json","artifacts/validation/phase10e_g5a_11200_pre_submit_audit.json"
    if stem.startswith("phase10d_DEPTH1000_FIXED"): return "artifacts/source_census/depth1000_fixed_report.json","artifacts/validation/phase10d_depth1000_fixed_pre_submit_audit.json"
    if stem.startswith("phase10d_DEPTH1000_"): return "artifacts/source_census/depth1000_report.json",""
    if stem.startswith("phase10d_R3"): return "artifacts/source_census/round3_report.json",""
    if stem.startswith("phase10d_R2"): return "artifacts/source_census/round2_report.json",""
    if stem.startswith("phase10d_R1"): return "artifacts/source_census/round1_report.json",""
    if stem.startswith("phase10b3"): return "artifacts/phase10b3_source_probe/source_probe_summary.json",""
    if stem.startswith("phase10a"): return "artifacts/phase10a_calibration/manifest.json",""
    if stem.startswith("phase9"): return "artifacts/phase9_submission/submission_manifest.json",""
    return "",""


def discover() -> list[Path]:
    paths=list((ROOT/"submissions").rglob("*.zip"))
    for path in (
        ROOT/"data/source_census/control_replay_g1a/phase10d_R3_G1A_control_replay.zip",
        ROOT/"data/source_census/phase10e_g5a_11200/control_depth1000_g5a/phase10e_control_G5A.zip",
        ROOT/"data/source_census/subset_smoke/output/smoke_medlatec_do_not_submit.zip",
    ):
        if path.exists(): paths.append(path)
    # One row per current path. Test-generated ZIPs are not competition artifacts.
    return sorted(set(paths))


def intentionally_purged(row: dict[str,str]) -> bool:
    return row.get("local_present","").lower()=="false" and row.get("local_retention","")==LOCAL_PURGED


def check_rows(rows: list[dict[str,str]]) -> list[str]:
    """Validate retained artifacts strictly while accepting recorded local purges."""
    failures=[]
    for row in rows:
        target=ROOT/row["relative_path"]
        if intentionally_purged(row):
            if target.exists():
                failures.append(f"{row['relative_path']} (marked purged but present)")
            continue
        if not target.exists():
            failures.append(f"{row['relative_path']} (missing)")
            continue
        if digest(target)!=row["sha256"] or target.stat().st_size!=int(row["size_bytes"]):
            failures.append(f"{row['relative_path']} (integrity mismatch)")
    return failures


def rebuild(output: Path) -> list[dict[str,str]]:
    with (ROOT/"docs/leaderboard_history.csv").open(encoding="utf-8",newline="") as handle:
        leaderboard={row["submission"]:row for row in csv.DictReader(handle)}
    prior=[]
    if output.exists():
        with output.open(encoding="utf-8",newline="") as handle:
            prior=list(csv.DictReader(handle))
    rows=[]
    for path in discover():
        stem=path.stem; key=ALIASES.get(stem,stem); metrics=leaderboard.get(key,{})
        status=classify(path,leaderboard); report,audit=report_for(stem)
        group=""
        for token in ("G1A","G1B","G5A","G5B","G6A","G6B","group_A","group_B","group_C","group_1","group_2","group_3","group_4","group_5","group_6","group_7"):
            if token in stem: group=token; break
        depth="10010" if "10010" in stem else ("11200" if "11200" in stem else ("1000" if "DEPTH1000" in stem else ("5000" if "_5k" in stem.lower() else "")))
        note=""
        if status=="INVALID_DO_NOT_SUBMIT": note="Scientifically invalid or explicit smoke artifact; never upload."
        elif status=="READY_TO_SUBMIT": note="Mandatory validation passed; organizer result pending."
        elif status=="SUPERSEDED_VALID": note="Structurally valid historical variant superseded by later experiments."
        elif status=="VALID_CONTROL": note="Local replay/control artifact; not an upload candidate."
        rows.append({
            "filename":path.name,"relative_path":path.relative_to(ROOT).as_posix(),"phase":phase_for(stem),
            "experiment_id":stem,"group":group,"depth":depth,"status":status,
            "leaderboard_submitted":"true" if metrics else ("false" if status=="READY_TO_SUBMIT" else ""),
            "final_score":metrics.get("final_score",""),"docs_f2":metrics.get("docs_f2",""),"chunks_f2":metrics.get("chunks_f2",""),
            "sha256":digest(path),"size_bytes":str(path.stat().st_size),
            "generated_timestamp":datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
            "report_path":report,"audit_path":audit,"notes":note,
            "local_present":"true","local_retention":LOCAL_RETAINED,"purged_at":"","purge_reason":"",
        })
    discovered={row["relative_path"] for row in rows}
    rows.extend(row for row in prior if intentionally_purged(row) and row["relative_path"] not in discovered)
    output.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]) if rows else []
    with output.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return rows


def organize() -> list[dict[str,object]]:
    """Move submission ZIPs into lifecycle folders, retaining byte evidence."""
    with (ROOT/"docs/leaderboard_history.csv").open(encoding="utf-8",newline="") as handle:
        leaderboard={row["submission"]:row for row in csv.DictReader(handle)}
    submission_root=(ROOT/"submissions").resolve(); records=[]
    for source in sorted((ROOT/"submissions").rglob("*.zip")):
        status=classify(source,leaderboard); destination=(ROOT/"submissions"/FOLDERS[status]/source.name).resolve()
        if submission_root not in destination.parents:
            raise ValueError(f"unsafe submission destination: {destination}")
        if source.resolve() == destination:
            continue
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite: {destination}")
        before={"path":source.relative_to(ROOT).as_posix(),"size_bytes":source.stat().st_size,"sha256":digest(source)}
        destination.parent.mkdir(parents=True,exist_ok=True); os.replace(source,destination)
        after={"path":destination.relative_to(ROOT).as_posix(),"size_bytes":destination.stat().st_size,"sha256":digest(destination)}
        if before["size_bytes"]!=after["size_bytes"] or before["sha256"]!=after["sha256"]:
            raise RuntimeError(f"byte-integrity failure moving {source.name}")
        records.append({"status":status,"before":before,"after":after})
    report=ROOT/"artifacts/submission_registry_cleanup.json"; report.parent.mkdir(parents=True,exist_ok=True)
    prior=[]
    if report.exists():
        prior=json.loads(report.read_text(encoding="utf-8")).get("moved",[])
    temporary=report.with_suffix(".json.tmp"); temporary.write_text(json.dumps({"moved":[*prior,*records]},indent=2)+"\n",encoding="utf-8"); os.replace(temporary,report)
    return records


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["organize","rebuild","check"]); parser.add_argument("--manifest",default="submissions/MANIFEST.csv"); args=parser.parse_args()
    path=ROOT/args.manifest
    if args.command=="organize":
        moved=organize(); print(f"submission ZIPs organized: {len(moved)}"); return 0
    rows=rebuild(path) if args.command=="rebuild" else list(csv.DictReader(path.open(encoding="utf-8",newline="")))
    failures=check_rows(rows)
    if failures: raise SystemExit("registry integrity failure: "+", ".join(failures))
    print(f"submission registry OK: {len(rows)} artifacts")
    return 0


if __name__=="__main__": raise SystemExit(main())
