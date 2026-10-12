# Competition Submission Registry

## CURRENT FILES TO UPLOAD

Only ZIPs in `00_READY_TO_UPLOAD/` are current upload candidates.

| Experiment | ZIP | SHA-256 | Status |
|---|---|---|---|
| Phase 11B NEW-corpus control — Docs@10 / Chunks@20 | `phase11b_NEW_CORPUS_CONTROL.zip` | `d094358409475d28a71dc8591a75d71c0412d8b9dc83d69bd93de1e4b494b35a` | Ready; organizer result pending. |
| Phase 11B Docs@20 / Chunks@20 | `phase11b_DOCS20_CHUNKS20.zip` | `9a4a9a3adc9d0f1319b924b34ea7f11abaab29b8d8516e734e3176b8131d12c9` | Ready; organizer result pending. |
| Phase 11B Docs@50 / Chunks@20 | `phase11b_DOCS50_CHUNKS20.zip` | `069480b6ec2b88ab1a00ee17ec22da4c0e9409debff35b958eddc2984453bfee` | Ready; organizer result pending. |
| FULL minus Vinmec | `phase10f_FULL_MINUS_VINMEC.zip` | `91635fac70d8ff3e8eda7fb43509637073ff06f5e899c3ba05b1e3eb5c622247` | Ready; organizer result pending. |
| FULL minus Tiemchunglongchau | `phase10f_FULL_MINUS_TIEMCHUNGLONGCHAU.zip` | `7ad8c68c4be22c54d49fe48788d370af4a292dda10793977830ce77adc4fde57` | Ready; organizer result pending. |
| FULL minus Hellobacsi | `phase10f_FULL_MINUS_HELLOBACSI.zip` | `3f399709057cf71f8d3965132e73fdbde896d194146bb8b74e2b07fde6852553` | Ready; organizer result pending. |

Recently submitted valid packages are retained under `10_SUBMITTED_VALID/`:

- `phase11a_OUTPUT20_CHUNKS40.zip` — organizer FINAL `0.0470` (historical pre-migration diagnostic)
- `phase11a_CANDIDATE_M16.zip` — organizer FINAL `0.0450` (historical pre-migration diagnostic)
- `FULL_MEDLATEC_7500.zip` — organizer FINAL `0.0398`
- `FULL_MEDLATEC_10000.zip` — organizer FINAL `0.0426`
- `phase10e_G5A_11200.zip` — organizer FINAL `0.0085`
- `phase10d_DEPTH1000_FIXED_G1A.zip` — organizer FINAL `0.0062`
- `phase10e_G1A_10010.zip` — organizer FINAL `0.0234`
- `phase10e_G6B_15000.zip` — organizer FINAL `0.0140`
- `phase10e_G1A_10010_G6B_15000_UNION.zip` — organizer FINAL `0.0332`
- `phase10e_G1A_10010_G6B_15000_G5A_11200_UNION.zip` — organizer FINAL `0.0383`
- `phase10d_DEPTH1000_FIXED_G6B.zip` — organizer FINAL `0.0045`
- `phase10e_FULL_MINUS_MEDLATEC.zip` — organizer FINAL `0.0288`
- `phase10e_FULL_MINUS_SUCKHOEDOISONG.zip` — organizer FINAL `0.0351`
- `phase10e_FULL_PLUS_R3_RESURRECTION.zip` — organizer FINAL `0.0383`

The authoritative inventory is [`MANIFEST.csv`](MANIFEST.csv). It records current or historical paths, hashes, sizes, validation evidence, known organizer status, and local-retention state. `INTENTIONALLY_PURGED` means the local ZIP was removed through an audited storage cleanup while its scientific status and byte identity remain recorded; it does not make the experiment invalid. Historical reports retain their original generation-time paths; the manifest records the post-audit lifecycle location.

## Folder meanings

- `00_READY_TO_UPLOAD/`: passed all required gates and has no recorded organizer result.
- `10_SUBMITTED_VALID/`: organizer-submitted, scientifically valid historical packages.
- `20_VALID_CONTROLS/`: local replay/control packages; not upload candidates.
- `30_SUPERSEDED_VALID/`: structurally valid historical variants superseded by later experiments.
- `90_INVALID_DO_NOT_SUBMIT/`: invalid experiments or explicit smoke artifacts.
- `99_REVIEW_REQUIRED/`: insufficient evidence to classify safely.

**NEVER upload files from `90_INVALID_DO_NOT_SUBMIT/`.**
The Medlatec source-cache intermediate is quarantined there because it is not
a full-union leaderboard experiment and failed the frozen-control/cardinality
gate; only `FULL_MEDLATEC_7500.zip` and `FULL_MEDLATEC_10000.zip` are the
scientifically valid depth packages.

## Lifecycle convention

Future packages begin as generated/staging artifacts. Run:

```powershell
.venv\Scripts\python.exe tools\submission_registry.py rebuild
.venv\Scripts\python.exe tools\submission_registry.py check
```

An unrecognized package is classified `UNKNOWN_REVIEW_REQUIRED`; it cannot become ready merely from its filename. Promotion to `READY_TO_SUBMIT` requires an explicit registry classification backed by a `READY_FOR_LEADERBOARD` audit containing the exact canonical ZIP path and SHA-256. After an organizer result, record metrics in `docs/leaderboard_history.csv`, then rebuild the registry so the package moves into submitted history during the next reviewed cleanup.

ZIP files are ignored by Git. `README.md` and `MANIFEST.csv` are tracked; model/data artifacts remain outside version control.
