import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from src.runtime import state_file
from src.runtime.state_file import update_json, write_json
from tools.phase13a_top3_recovery import open_checkpoint


def test_unique_temporary_names_and_transient_permission_retry(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    real_replace = state_file.os.replace
    names = []
    failures = {"remaining": 2}

    def flaky(source, destination):
        names.append(str(source))
        if failures["remaining"]:
            failures["remaining"] -= 1
            raise PermissionError(5, "transient sharing lock")
        return real_replace(source, destination)

    monkeypatch.setattr(state_file.os, "replace", flaky)
    write_json(path, {"status": "RUNNING"})
    assert json.loads(path.read_text())["status"] == "RUNNING"
    assert len(set(names)) == 1  # one unique temp survives retries; no shared .tmp name
    assert ".tmp." in names[0]


def test_competing_state_writers_merge_without_lost_fields(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"status": "RUNNING", "done": 0})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: update_json(path, {f"field_{index}": index}), range(30)))
    result = json.loads(path.read_text())
    assert all(result[f"field_{index}"] == index for index in range(30))


def test_running_heartbeat_cannot_overwrite_terminal_status(tmp_path):
    path = tmp_path / "state.json"
    write_json(path, {"status": "READY_FOR_LEADERBOARD", "stage": "READY_FOR_LEADERBOARD", "done": 3})
    update_json(path, {"status": "RUNNING", "stage": "PACKAGE", "heartbeat_at": "later"})
    result = json.loads(path.read_text())
    assert result["status"] == "READY_FOR_LEADERBOARD"
    assert result["stage"] == "READY_FOR_LEADERBOARD"
    assert result["heartbeat_at"] == "later"


def test_top3_checkpoint_enforces_query_rank_and_document_uniqueness(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.sqlite"
    monkeypatch.setattr("tools.phase13a_top3_recovery.RUN_DB", path)
    connection = open_checkpoint()
    with connection:
        connection.execute("INSERT INTO rankings VALUES (1,1,10,0.5)")
        connection.execute("INSERT INTO completed VALUES (1,0)")
    with __import__("pytest").raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO rankings VALUES (1,1,11,0.4)")
    with __import__("pytest").raises(sqlite3.IntegrityError):
        connection.execute("INSERT INTO rankings VALUES (1,2,10,0.4)")
    assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0] == 1
    connection.close()
