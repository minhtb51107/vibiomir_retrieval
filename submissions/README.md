# Competition Submission Registry

## CURRENT FILES TO UPLOAD

Only ZIPs in `00_READY_TO_UPLOAD/` are current upload candidates.

| Experiment | ZIP | SHA-256 | Status |
|---|---|---|---|
| _None_ | — | — | The G1A+G6B union has an organizer result; the three-group union is not ready yet. |

Recently submitted valid packages are retained under `10_SUBMITTED_VALID/`:

- `phase10e_G5A_11200.zip` — organizer FINAL `0.0085`
- `phase10d_DEPTH1000_FIXED_G1A.zip` — organizer FINAL `0.0062`
- `phase10e_G1A_10010.zip` — organizer FINAL `0.0234`
- `phase10e_G6B_15000.zip` — organizer FINAL `0.0140`
- `phase10e_G1A_10010_G6B_15000_UNION.zip` — organizer FINAL `0.0332`
- `phase10d_DEPTH1000_FIXED_G6B.zip` — organizer FINAL `0.0045`

The authoritative inventory is [`MANIFEST.csv`](MANIFEST.csv). It records current or historical paths, hashes, sizes, validation evidence, known organizer status, and local-retention state. `INTENTIONALLY_PURGED` means the local ZIP was removed through an audited storage cleanup while its scientific status and byte identity remain recorded; it does not make the experiment invalid. Historical reports retain their original generation-time paths; the manifest records the post-audit lifecycle location.

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
