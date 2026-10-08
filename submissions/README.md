# Competition Submission Registry

## CURRENT FILES TO UPLOAD

Only ZIPs in `00_READY_TO_UPLOAD/` are current upload candidates.

| Experiment | ZIP | SHA-256 | Why ready | Organizer result |
|---|---|---|---|---|
| Phase 10E G5A 11.2K | `00_READY_TO_UPLOAD/phase10e_G5A_11200.zip` | `6881d21914aa44e307254f352354439077426f043c0e6f8c4d4f2afea12fb635` | Mandatory structural and scientific audits passed, including exact-key cache reuse and corrected-depth1000 control replay. | Pending |
| Phase 10D corrected G1A depth1000 | `00_READY_TO_UPLOAD/phase10d_DEPTH1000_FIXED_G1A.zip` | `2b9034fe452c66bc3e8539317dd28d501c388aa14c0dcf5ddfbc301d25d6ecfb` | Corrected cache, strict validation, determinism, and mandatory scientific audit passed. | Pending |
| Phase 10D corrected G6B depth1000 | `00_READY_TO_UPLOAD/phase10d_DEPTH1000_FIXED_G6B.zip` | `01555bd9368ba74c353c6c806ae69e799a09dea797484d5eebc4af7649d5b377` | Corrected cache, strict validation, determinism, and mandatory scientific audit passed. | Pending |

The authoritative inventory is [`MANIFEST.csv`](MANIFEST.csv). It records current paths, hashes, sizes, validation evidence, and known organizer status. Historical reports retain their original generation-time paths; the manifest records the post-audit lifecycle location.

## Folder meanings

- `00_READY_TO_UPLOAD/`: passed all required gates and has no recorded organizer result.
- `10_SUBMITTED_VALID/`: organizer-submitted, scientifically valid historical packages.
- `20_VALID_CONTROLS/`: local replay/control packages; not upload candidates.
- `30_SUPERSEDED_VALID/`: structurally valid historical variants superseded by later experiments.
- `90_INVALID_DO_NOT_SUBMIT/`: invalid experiments or explicit smoke artifacts.
- `99_REVIEW_REQUIRED/`: insufficient evidence to classify safely.

**NEVER upload files from `90_INVALID_DO_NOT_SUBMIT/`.**

## Lifecycle convention

Future packages begin as generated/staging artifacts. Run:

```powershell
.venv\Scripts\python.exe tools\submission_registry.py rebuild
.venv\Scripts\python.exe tools\submission_registry.py check
```

An unrecognized package is classified `UNKNOWN_REVIEW_REQUIRED`; it cannot become ready merely from its filename. Promotion to `READY_TO_SUBMIT` requires an explicit registry classification backed by a `READY_FOR_LEADERBOARD` audit containing the exact canonical ZIP path and SHA-256. After an organizer result, record metrics in `docs/leaderboard_history.csv`, then rebuild the registry so the package moves into submitted history during the next reviewed cleanup.

ZIP files are ignored by Git. `README.md` and `MANIFEST.csv` are tracked; model/data artifacts remain outside version control.
