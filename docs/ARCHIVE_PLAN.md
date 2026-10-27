# Safe archive plan — 2026-10-27

**STOP BEFORE DELETION. No cleanup is authorized.**
An external archive destination has not been confirmed; no complete independent
backup has been created or verified. A Git push or snapshot on D: is not a
backup of private data. C: and D: share the same physical 512-GB NVMe SSD.

## Exact inventory and interpretation

The bounded inventory ran 50.66 s at reported peak working set 27.07 MiB, with
58,455 entries, zero scan errors and 8 reparse links intentionally not followed.
Sizes are logical file lengths, **not measured allocated/reclaimable disk space**.
Audit/run output directories were excluded to avoid a self-growing inventory.
The snapshot precedes final docs/guard changes; the finalization delta and local
snapshot manifests identify the later work. Refresh source hashes at backup time.

Private files under `artifacts/final_freeze_20261027/`:

- `inventory.sqlite`: exact path, bytes, timestamp, Git state, category/reason.
- `file_inventory.jsonl`: same file-level inventory with absolute paths.
- `folder_inventory.jsonl`: direct and recursive bytes/counts for each folder.
- `inventory_summary.json`: totals, runtime, exclusions and error counts.
- `cleanup_candidates_verified.json`: corrected nonoverlapping C-only targets,
  approval=false. The original unapproved `cleanup_candidates.json` is preserved
  but superseded: its `test_tmp` parent included two protected reparse links.
- `critical_hashes.json`: recomputed BEST/corpus/query checksums.
- `worktree_before_finalization.patch`, index patch and Git-status snapshot:
  preserve pre-existing local changes; private, potentially sensitive.
- `installed_versions_private.json`: package versions without importing models.
- `scheduler/`: original task XML/private definition and disable result.
- Versioned `git_history_*.bundle` and `local_working_source_*.tar`, when created:
  local recovery aids only. Their separate manifests record hashes/roundtrip.

| Classification | Meaning | Files | Logical bytes | GiB |
|---|---|---:|---:|---:|
| A | Code/config/docs and full local Git database | 7,108 | 3,468,369,625 | 3.230 |
| B | Scientific/reproducibility data, scores, indexes, outputs | 12,013 | 167,487,815,625 | 155.985 |
| C | Regenerable environment/test/bytecode caches, conditionally | 37,458 | 5,355,100,385 | 4.987 |
| D | Unknown, sensitive, links and potentially live checkpoint remnants | 1,876 | 5,665,489,458 | 5.276 |
| Total | Before finalization/audit-output delta | 58,455 | 181,976,775,093 | 169.479 |

| Folder | Logical bytes | GiB |
|---|---:|---:|
| data | 150,770,002,971 | 140.416 |
| submissions | 14,847,010,993 | 13.827 |
| .venv | 5,348,986,112 | 4.982 |
| .cache | 4,790,526,323 | 4.462 |
| .git | 3,463,985,021 | 3.226 |
| artifacts | 2,745,744,288 | 2.557 |

Git states at the inventory snapshot: 840 tracked, 5,779 untracked, 45,251 ignored,
6,585 Git-internal files. The 523 category-A non-Git source/config/doc files
occupy 4,384,604 bytes: 393 tracked and 130 untracked. These counts are not the
post-finalization Git status. Untracked/ignored does not mean disposable.

## Backup layout and capacity

Confirm an external physical disk or authorized trusted storage first.
Uncompressed baseline 169.48 GiB plus 20% headroom and 20 GiB reserve is about 223.37 GiB.
Plan for **at least 225 GiB free**, preferably a 500-GB-or-larger external drive.
Neither local partition currently has sufficient space for the full mirror.
Compression savings are UNKNOWN and must not be budgeted as guaranteed.

```text
ViBioMIR_final_20261027/
  project_mirror/           exact approved workspace paths, no flattening
  git/                     versioned bundles + full .git/reflog/object preservation
  manifests/               inventory, full source SHA manifest, verification reports
  recovery/                worktree/index patches, private working-source tar
  environment/             versions, revisions, precision/batch/equivalence contracts
  protected_releases/      Phase27 BEST + JSON/audit and preserved Phase29 control
```

The mirror preserves A+B+D; a full first backup should also retain C unless
the user explicitly chooses otherwise after environment/recovery checks.
Encrypt sensitive configuration, source narratives and recovery snapshots.
Do not publish raw source pages, model weights or private data without rights.

**Git bundle limits:** bundles preserve reachable refs, not untracked files,
unstaged work, reflogs or unreachable objects. The local Git directory is 3.23 GiB;
Git reported about 116 MB reachable storage and 10 temporary-object warnings.
Those objects may be valuable or incomplete; **do not run gc/prune/rewrite**.
Preserve the full Git directory separately, along with patches/source snapshots.

## Verification procedure — after destination approval only

1. Confirm destination path, physical independence, permissions/encryption and
   capacity. Keep all project writers stopped. Record the final commit and
   scheduled-task/freeze state. Do not silently reactivate workers.
2. Review the 8 reparse links and any external/shared model references. Preserve
   relative internal model links and their blobs with a link-aware tool, or
   approve an explicit alternate representation. Never traverse/delete shared
   external targets by assumption. The simple helper below deliberately refuses
   reparse links until this is resolved; this is a blocker, not a reason to skip
   them silently.
3. Produce a **fresh full SHA-256 manifest** after finalization and before copy.
   Hash every included regular file with bounded reads and detect source changes
   during hashing. Include final audit/recovery files and all approved A/B/D
   paths; account separately for link metadata. This full-content hashing has
   NOT been performed across the 169 GiB in the current task.
4. Copy to the approved mirror without changing source files. Do not copy a live
   SQLite main database while omitting its WAL; either retain the quiescent
   main/WAL set or use SQLite's backup API into the destination. Keep original
   checkpoint files untouched. No copy is currently authorized by this plan.
5. Re-hash every copied file and compare path, size and SHA against the manifest.
   Require zero missing/mismatched files. Check completeness against the full
   inventory plus finalization delta, review destination extras, verify links,
   count bytes and verify the final source files did not change during copy.
6. Test restore of the source tar and Git bundle, canonical/score/checkpoint
   SQLite integrity and critical release JSON/ZIP CRC. Phase27 BEST must match
   `b58a1c2983e51586237ba55dc98b85b267e7436385f6f8e14998f81709639d8e`.
   Do not load a model to test an archival restore.
7. Save a signed-off `VERIFIED.json` describing destination, scope, manifest
   SHA, zero mismatches, restore checks and critical hashes. A verification tool's
   exit 0 alone is not proof that the archive contains every essential file.
8. Present exact cleanup targets and verified bytes to the user. **Wait for
   explicit approval.** Only then may a separate cleanup task act.

Available helper, to be used only after the destination/link decisions:

```powershell
# Paths below are placeholders, not an approved destination.
python tools/final_freeze_archive.py hash-source --confirmed-destination --manifest "X:\approved_archive\manifests\source_sha256.jsonl"
python tools/final_freeze_archive.py verify --manifest "X:\approved_archive\manifests\source_sha256.jsonl" --mirror "X:\approved_archive\project_mirror"
```

The helper never copies/deletes/moves project data. It rejects path escapes,
empty/duplicate manifests, missing files and hash mismatches. Its verifier checks
manifest-listed regular files only; completeness, link handling and restore
approval remain mandatory separate checks. A failed/partial hash manifest must
not be accepted as a complete backup or overwritten without inspection.

## Explicit cleanup candidates — proposed, not approved

`cleanup_candidates_verified.json` lists 30 nonoverlapping C-only targets with absolute
paths and sizes. Their total logical size is **5,355,100,385 bytes (4.987GiB)**.
Most is `.venv` (5,348,986,112 bytes); the remainder is test/bytecode cache data.
Preserve environment records, source code and numerical contracts before even
this conditional cleanup. Do not remove the interpreter while verification uses it.
Actual allocated space released is UNKNOWN until measured.

Not cleanup candidates by default: official/raw corpus, acquired bodies,
canonical documents/chunks, score databases, vectors/indexes, BEST/controls,
scientific reports, manifests, full Git objects, source tar, credentials or
unknown files. In particular preserve the 537 MB embedding `.tmp`, 199 MB package
`.tmp`, SQLite WAL/SHM and 8 links until their lifecycle is established.
The 4.46 GiB model cache is not automatically dispensable or project-exclusive.

Moving large B/D data to cold storage might later recover substantially more
local space, but is **not authorized or included in the 4.987 GiB estimate**.
It requires a verified full backup and a new exact-target decision. No broad
workspace/home/drive deletion command is provided.

## Recovery / rollback

- Before any cleanup, recovery is simply to keep the original files untouched.
- If cleanup is later approved, restore each exact path from the verified mirror,
  validate hashes and restore SQLite companions/links. Never fabricate scores
  or regenerate missing official content as if identical.
- Restore code with the source tar, Git bundle and working/index patches in a
  separate directory first. Do not overwrite the existing dirty worktree blindly.
- Keep `PROJECT_FREEZE.json` enabled and the exported scheduled task disabled.
  Re-enabling a task or unfreezing research needs new explicit user approval.
- Frozen numerical contracts may not hold on a different machine; archives do
  not authorize cross-runtime score reuse or new model inference.

**User decisions required:** archive destination/capacity and encryption/link
handling first; exact cleanup scope only after the backup/restore proof passes.
