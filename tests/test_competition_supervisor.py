import json

import pytest

from tools.competition_supervisor import recover_stale_run


def test_dead_run_lock_is_marked_stale_without_losing_checkpoint(tmp_path, monkeypatch):
    run=tmp_path/"run"; run.mkdir()
    (run/"run.lock").write_text("pid=123\n",encoding="utf-8")
    (run/"state.json").write_text(json.dumps({"status":"RUNNING","stage":"RERANK","last_successful_checkpoint":13432}),encoding="utf-8")
    monkeypatch.setattr("tools.competition_supervisor.process_alive",lambda pid:False)
    recovered=recover_stale_run(run,"phase10e_test")
    assert recovered["status"]=="STALE_RUNNING"
    assert recovered["last_successful_checkpoint"]==13432
    assert recovered["checkpoint_preserved"] is True
    assert not (run/"run.lock").exists()


def test_live_run_lock_is_not_removed(tmp_path, monkeypatch):
    run=tmp_path/"run"; run.mkdir()
    lock=run/"run.lock"; lock.write_text("pid=456\n",encoding="utf-8")
    monkeypatch.setattr("tools.competition_supervisor.process_alive",lambda pid:True)
    with pytest.raises(FileExistsError,match="live run owner"):
        recover_stale_run(run,"phase10e_test")
    assert lock.exists()
