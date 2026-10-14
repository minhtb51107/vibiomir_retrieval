import datetime as dt

import pytest

from src.runtime.progress_contract import INDETERMINATE, StageProgress, progress_snapshot


STAGES = ("CONTROL_REPLAY", "DOC_TOP2_MEAN", "VALIDATION", "PACKAGING")
START = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _progress(stage="VALIDATION", done=2, total=4):
    return StageProgress(stage, "validate provenance", done, total, START, START, True)


def test_reports_exact_stage_and_overall_gate_percentages():
    result = progress_snapshot(_progress(), ordered_stages=STAGES, now=START + dt.timedelta(seconds=20))
    assert result["stage_percent"] == 50.0
    assert result["overall_percent"] == 62.5
    assert result["done"] == 2 and result["total"] == 4
    assert result["eta_seconds"] == 20.0


def test_indeterminate_counter_never_invents_percent_or_eta():
    result = progress_snapshot(_progress(done=None, total=None), ordered_stages=STAGES, now=START)
    assert result["stage_percent"] == INDETERMINATE
    assert result["overall_percent"] == INDETERMINATE
    assert result["eta_seconds"] == INDETERMINATE


def test_nonterminal_last_stage_never_reports_overall_100():
    result = progress_snapshot(_progress(stage="PACKAGING", done=10, total=10), ordered_stages=STAGES, terminal=False, now=START)
    assert result["stage_percent"] == 100.0
    assert result["overall_percent"] == 99.99


def test_only_terminal_state_reports_overall_100():
    result = progress_snapshot(_progress(stage="PACKAGING", done=10, total=10), ordered_stages=STAGES, terminal=True, now=START)
    assert result["overall_percent"] == 100.0


@pytest.mark.parametrize("done,total", [(-1, 10), (11, 10), (1, 0)])
def test_invalid_or_unmeasurable_counters_are_not_presented_as_valid(done, total):
    if total == 0:
        assert progress_snapshot(_progress(done=done, total=total), ordered_stages=STAGES, now=START)["stage_percent"] == INDETERMINATE
    else:
        with pytest.raises(ValueError):
            progress_snapshot(_progress(done=done, total=total), ordered_stages=STAGES, now=START)
