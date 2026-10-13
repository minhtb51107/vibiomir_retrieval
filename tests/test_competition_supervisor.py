import json

import pytest

from tools.competition_supervisor import SUCCESS_TERMINAL_STATES, reconcile_worker_exit_state, recover_stale_run
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
    assert recovered["termination_reason"]=="Cause of process termination unknown."


def test_dead_supervisor_does_not_start_competitor_while_worker_is_alive(tmp_path,monkeypatch):
    run=tmp_path/"run";run.mkdir()
    (run/"run.lock").write_text("pid=123\n",encoding="utf-8")
    (run/"state.json").write_text(json.dumps({"status":"RUNNING","worker_pid":456}),encoding="utf-8")
    monkeypatch.setattr("tools.competition_supervisor.process_alive",lambda pid:pid==456)
    with pytest.raises(FileExistsError,match="live worker"):
        recover_stale_run(run,"phase12_test")
    assert (run/"run.lock").exists()


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


def test_human_review_is_a_supported_success_terminal_state():
    assert "READY_FOR_HUMAN_REVIEW" in SUCCESS_TERMINAL_STATES


def test_fast_worker_terminal_update_is_not_overwritten_by_stale_supervisor_state(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"status": "READY_FOR_HUMAN_REVIEW", "stage": "AWAITING_HUMAN_REVIEW", "done": 1}),
        encoding="utf-8",
    )
    reconciled = reconcile_worker_exit_state(
        state_path, {"status": "RUNNING", "stage": "STARTING", "done": 0}, worker_pid=321,
    )
    assert reconciled["status"] == "READY_FOR_HUMAN_REVIEW"
    assert reconciled["stage"] == "AWAITING_HUMAN_REVIEW"
    assert reconciled["done"] == 1
    assert reconciled["worker_alive"] is False
