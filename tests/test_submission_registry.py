from pathlib import Path

from tools.submission_registry import FOLDERS, STATUSES, classify


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


def test_ready_requires_explicit_allowlist_not_filename_guessing():
    assert classify(Path("phase10e_G5A_11200.zip"),{})=="READY_TO_SUBMIT"
    assert classify(Path("phase10e_future.zip"),{})=="UNKNOWN_REVIEW_REQUIRED"
