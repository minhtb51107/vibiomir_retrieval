import subprocess
import sys

import pytest

from src.ingestion.cli import build_parser, validate_safety_args


def test_cli_refuses_more_than_pilot_limit_without_full() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/crawl_corpus.py", "--limit", "101"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "refusing 101 URLs without --full" in completed.stderr


def test_cli_refuses_unbounded_full_without_second_confirmation() -> None:
    parser = build_parser()
    with pytest.raises(ValueError, match="--confirm-full-crawl"):
        validate_safety_args(parser.parse_args(["--full"]))


def test_bounded_production_batch_does_not_need_full_confirmation() -> None:
    parser = build_parser()
    args = parser.parse_args(["--full", "--max-new-records", "1000"])
    validate_safety_args(args)


def test_production_mode_refuses_raw_body_blobs() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--full", "--max-new-records", "1000", "--store-raw-body"]
    )
    with pytest.raises(ValueError, match="pilot-only"):
        validate_safety_args(args)
