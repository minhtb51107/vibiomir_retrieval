# Phase 10D Tomorrow Playbook

> Superseded on 2026-10-08 by the controlled depth-1000 experiment. Do not
> execute the earlier adaptive source-splitting or depth-300 plan below. All
> nine sources in G1A, G5A, and G6B remain active until organizer-confirmed
> depth-scaling evidence is available.

## Fixed evidence boundary

- Round 3 uses the unchanged pilot control, unchanged Round-1 source samples,
  candidate cap `m=8`, and the completed 582,000-score reranker cache.
- The six Round-3 ZIPs differ only by source membership.
- Do not infer individual-source value from the Round-2 group scores.
- Do not crawl deeper until organizer evidence identifies a sufficiently strong
  source-level hypothesis.

## Tomorrow's sequence

1. Manually submit the six Round-3 ZIPs: G1A, G1B, G5A, G5B, G6A, G6B.
2. Wait for all six organizer results before making a selection.
3. Compare the sibling pairs G1A/G1B, G5A/G5B, and G6A/G6B across FINAL,
   document F2/precision/recall, and chunk F2/precision/recall. Do not discard a
   side based on a tiny rounded difference.
4. If two trios clearly dominate, use at most four remaining submissions to
   test one source alone and the other two sources together within each leading
   trio. Do not pre-select those sources before seeing all six results.
5. Begin deeper acquisition only after source-level organizer evidence is
   strong enough to justify its time and disk cost.

## Cached arbitrary-subset command

Run from the repository root. Use a new name and new work directory for every
experiment so existing artifacts are never mutated:

```powershell
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe tools\source_census_cached_subset.py make-subset `
  --sources "source_A,source_B" `
  --name "phase10d_R4_descriptive_name" `
  --work-dir "data\source_census\adaptive\phase10d_R4_descriptive_name" `
  --output-dir "submissions"
```

The command performs zero model inference. It streams the fixed pilot and
cached source candidates, reads the existing SQLite reranker scores, applies
the unchanged ranking/submission policy, generates the ZIP twice, validates
all 1,200 queries and provenance, and records SHA-256 in the work directory's
`submission_manifest.json`.

Smoke test completed with `medlatec.vn` under
`data/source_census/subset_smoke/`; it passed strict validation and deterministic
regeneration. Its output is explicitly named `smoke_medlatec_do_not_submit`.

## Prepared deep-acquisition manifests — do not run yet

| Target | Additional official IDs | Sources below target population | Rough retained disk | Source-serial time |
|---|---:|---|---:|---:|
| 300/source | 3,000 | familydoctor.cn; qy.familydoctor.com.cn; v.familydoctor.com.cn | 0.126 GiB | 0.99 h |
| 1,000/source | 13,500 | familydoctor.cn; qy.familydoctor.com.cn; v.familydoctor.com.cn | 0.566 GiB | 4.45 h |

The projections linearly extrapolate each source's Round-1 worker directory
size and wall time. They are operational estimates, not guaranteed concurrent
wall time, content yield, or relevance. The manifests contain only official
IDs, exclude the existing Round-1 IDs, and make no network requests:

- `artifacts/source_census/round3_depth300_manifest.json`
- `artifacts/source_census/round3_depth1000_manifest.json`
- `artifacts/source_census/round3_depth_manifest_summary.json`
