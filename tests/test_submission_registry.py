from pathlib import Path

from tools.submission_registry import FOLDERS, LOCAL_PURGED, STATUSES, check_rows, classify, report_for


def test_every_lifecycle_status_has_a_human_folder():
    assert set(FOLDERS)==set(STATUSES)
    assert FOLDERS["READY_TO_SUBMIT"]=="00_READY_TO_UPLOAD"
    assert FOLDERS["INVALID_DO_NOT_SUBMIT"]=="90_INVALID_DO_NOT_SUBMIT"


def test_invalid_package_cannot_be_ready():
    path=Path("phase10d_DEPTH1000_G5A.zip")
    assert classify(path,{"phase10d_DEPTH1000_G5A":{"final_score":"0.0002"}})=="INVALID_DO_NOT_SUBMIT"


def test_submitted_evidence_beats_unknown():
    path=Path("phase10d_R1_group_A.zip")
    assert classify(path,{"phase10d_R1_group_A":{"final_score":"0.0016"}})=="SUBMITTED_VALID"


def test_submitted_evidence_retires_former_ready_package():
    path=Path("phase10e_G5A_11200.zip")
    assert classify(path,{"phase10e_G5A_11200":{"final_score":"0.0085"}})=="SUBMITTED_VALID"


def test_g1a_focused_submission_keeps_scientific_evidence_paths():
    assert report_for("phase10e_G1A_10010")==(
        "artifacts/source_census/phase10e_g1a_10010_report.json",
        "artifacts/validation/phase10e_g1a_10010_pre_submit_audit.json",
    )


def test_unknown_package_is_not_promoted_by_filename_guessing():
    assert classify(Path("phase10e_G5A_11200.zip"),{})=="UNKNOWN_REVIEW_REQUIRED"
    assert classify(Path("phase10e_future.zip"),{})=="UNKNOWN_REVIEW_REQUIRED"


def test_intentionally_purged_artifact_is_not_reported_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.submission_registry.ROOT",tmp_path)
    row={
        "relative_path":"submissions/obsolete.zip","sha256":"preserved-evidence",
        "size_bytes":"123","local_present":"false","local_retention":LOCAL_PURGED,
    }
    assert check_rows([row])==[]


def test_retained_artifact_is_still_required(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.submission_registry.ROOT",tmp_path)
    row={
        "relative_path":"submissions/milestone.zip","sha256":"missing",
        "size_bytes":"123","local_present":"true","local_retention":"RETAINED_LOCAL",
    }
    assert check_rows([row])==["submissions/milestone.zip (missing)"]
