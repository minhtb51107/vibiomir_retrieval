import subprocess
import sys


def test_cli_refuses_more_than_pilot_limit_without_full() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/crawl_corpus.py", "--limit", "101"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "refusing 101 URLs without --full" in completed.stderr
