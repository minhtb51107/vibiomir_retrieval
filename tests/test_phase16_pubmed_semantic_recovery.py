import importlib.util
import json
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "tools/phase16_pubmed_semantic_recovery.py"
    spec = importlib.util.spec_from_file_location("semantic_recovery", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_language_signal_is_explicit_and_deterministic():
    mod = load_module()
    assert mod.language_signal("Trẻ bị viêm amidan") == "vi"
    assert mod.language_signal("What treats asthma?") == "en"
    assert mod.language_signal("儿童哮喘") == "zh"


def test_evidence_spans_must_be_verbatim():
    mod = load_module(); payload={"query":"viêm gan", "candidate_A":{"title":"Hepatitis"}, "candidate_B":{"supporting_chunk":"Điều trị viêm gan"}}
    assert mod.validate_spans({"abstain":False,"evidence_spans":[{"source":"A","exact_quote":"Hepatitis"},{"source":"B","exact_quote":"Điều trị viêm gan"}]}, payload)
    assert not mod.validate_spans({"abstain":False,"evidence_spans":[{"source":"A","exact_quote":"invented"},{"source":"B","exact_quote":"Điều trị viêm gan"}]}, payload)


def test_prompt_swap_preserves_membership_and_changes_order():
    mod=load_module(); row={"query_text":"q","pubmed_title":"p","pubmed_abstract":"a","pubmed_chunk_text":"pc","best_url":"u","best_chunk_text":"bc"}
    normal=json.loads(mod.prompt_for(row,False)); swapped=json.loads(mod.prompt_for(row,True))
    assert normal["candidate_A"] == swapped["candidate_B"]
    assert normal["candidate_B"] == swapped["candidate_A"]


def test_calls_fail_closed_without_explicit_approval():
    mod=load_module()
    try: mod.judge(40,20,"",0.05,0.10,True)
    except SystemExit as exc: assert "inference calls disabled" in str(exc)
    else: raise AssertionError("judge did not fail closed")


def test_provider_is_pinned_without_fallback():
    mod = load_module()
    assert mod.HF_ENDPOINT == "https://router.huggingface.co/v1"
    assert mod.ROUTED_MODEL == "Qwen/Qwen3-14B:deepinfra"
    assert mod.PROVIDER == "deepinfra"
    assert mod._provider_is_live({"inferenceProviderMapping": [{"provider": "deepinfra", "status": "live"}]}, "deepinfra")


def test_online_preflight_checks_revision_provider_and_credit(monkeypatch):
    mod = load_module()
    replies = iter([
        {"name": "test-account"},
        {"sha": mod.MODEL_REVISION, "inferenceProviderMapping": {"deepinfra": {"status": "live"}}},
    ])
    monkeypatch.setattr(mod, "_request_json", lambda *args, **kwargs: next(replies))
    result = mod.online_preflight("not-a-real-token", 0.10, 0.02, True)
    assert result["authenticated"] is True
    assert result["provider_pinned"] == "deepinfra"
    assert result["fallback_allowed"] is False


def test_online_preflight_rejects_insufficient_confirmed_credit():
    mod = load_module()
    try:
        mod.online_preflight("not-a-real-token", 0.01, 0.02, True)
    except SystemExit as exc:
        assert "credit gate failed" in str(exc)
    else:
        raise AssertionError("insufficient credit was accepted")
