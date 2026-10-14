# Colab A100 reranker proof of concept

This bounded proof of concept checks whether an NVIDIA A100 Colab runtime can
reproduce the repository's existing FP16 reranker scores. It does not modify a
score cache or a frozen submission, and it is not approval to run production
work remotely.

## Frozen scorer contract

- Model: `BAAI/bge-reranker-v2-m3`
- Model/tokenizer revision: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`
- License: Apache-2.0, as declared by the official model card
- Inputs: `(query_text, chunk_text)`
- Precision/device: FP16 CUDA
- Production equivalence batch: 2
- Tokenization: padded batches, `longest_first`, maximum length 512
- Mode: evaluation/inference
- Score: `logits.reshape(-1).float().cpu().numpy()`
- Existing measured absolute tolerance: `0.005859375`

The 200-pair export includes 120 deterministic language/length-stratified rows
and 80 rows forming query-local close-score pairs. The retained Phase 12 data
contains Chinese short and medium passages but no Chinese passage of at least
1,200 characters, so the declared strata reflect that observed distribution;
no synthetic long-Chinese row is created. Every input text has an exact SHA256.

## Windows export

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe tools\colab_reranker_poc.py export
```

This creates the untracked transfer file
`artifacts/colab_reranker_poc/reranker_poc_200.zip` plus a small export report.
The source score database is opened read-only.

## Colab A100 run

1. Create an A100 GPU runtime.
2. Upload `reranker_poc_200.zip` and
   `notebooks/colab_reranker_poc.py` into the Colab session (or copy them
   manually from Google Drive).
3. Run:

```python
!pip -q install transformers==4.48.3 psutil
!python colab_reranker_poc.py --package reranker_poc_200.zip --output colab_scores.json
```

The exact pinned model revision is downloaded from Hugging Face. Download
`colab_scores.json` back to Windows. Optional larger-batch telemetry can be
enabled with `--benchmark-larger-batches`; those scores are explicitly
discarded and never used for equivalence.

## Windows dry-run verification

```powershell
.\.venv\Scripts\python.exe tools\colab_reranker_poc.py verify `
  --results C:\path\to\colab_scores.json
```

Verification is read-only by default. It rejects package/hash/contract/key
mismatches, non-finite values, absolute drift beyond the measured tolerance,
or any query-local ordering change. It never writes to a production cache.
After acceptance only, `--write-staging <new-path.sqlite>` may create a new,
previously nonexistent staging database for inspection; it refuses overwrite.

## Acceptance and future workflow

Current status is `AWAITING_COLAB_RESULT`: local tooling and the transfer
package are validated, but A100 numerical equivalence and performance are not
yet measured. If verification fails, remote integration stops.

If it passes, a future separately approved workflow should export bounded,
non-overlapping work packages with content hashes and immutable manifests;
score batch 2 in package order; import into new staging databases; reconcile
every exact key/hash; and transactionally merge only missing production keys
after a full equivalence gate. Completed package IDs and hashes would be
checkpointed so restart never repeats accepted work.
