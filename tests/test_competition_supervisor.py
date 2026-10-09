import json

import pytest

from tools.competition_supervisor import recover_stale_run
from tools.phase10e_worker import RESUME_STAGES
from tools.phase10e_embedding_continuation import memory_gate_status


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


def test_running_state_without_lock_is_reconciled_as_stale(tmp_path):
    run=tmp_path/"run"; run.mkdir()
    (run/"state.json").write_text(json.dumps({"status":"RUNNING","stage":"EMBEDDINGS","completed":37434}),encoding="utf-8")
    recovered=recover_stale_run(run,"phase10e_test")
    assert recovered["status"]=="STALE_RUNNING"
    assert recovered["interrupted_pid"] is None
    assert recovered["completed"]==37434
    assert recovered["checkpoint_preserved"] is True


def test_phase10e_can_resume_at_atomic_assemble_boundary():
    assert RESUME_STAGES==("full","assemble","embeddings","source_candidates","rerank")


def test_embedding_startup_gate_uses_measured_profile_requirements():
    profile={"measurements":[{"phase":"model_loaded","rss_mib":717.41}],"memory_gate":{"observed_peak_rss_mib":833.41,"projected_peak_private_mib":3931.96}}
    result=memory_gate_status(profile,{"available_physical_mib":950,"available_commit_mib":4000})
    assert result["required_physical_mib"]==pytest.approx(949.41)
    assert result["physical_safety_headroom_mib"]==pytest.approx(116.0)
    assert result["passed"] is True
    assert memory_gate_status(profile,{"available_physical_mib":900,"available_commit_mib":4000})["passed"] is False
    assert memory_gate_status(profile,{"available_physical_mib":950,"available_commit_mib":3900})["passed"] is False
