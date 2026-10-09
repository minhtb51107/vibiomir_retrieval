from src.reranking.equivalence import validate_score_equivalence


CONTRACT={
    "model":"BAAI/bge-reranker-v2-m3","revision":"rev","tokenizer_revision":"rev",
    "batch_size":2,"max_length":512,"precision":"float16","eval_mode":True,
    "padding":True,"truncation":"longest_first",
    "score_extraction":"logits.reshape(-1).float().cpu().numpy()","input_order":"query, chunk",
}


def test_measured_fp16_jitter_with_stable_order_passes():
    result=validate_score_equivalence([-2.80078125,-3.1171875],[-2.802734375,-3.111328125],["a","b"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT,expected_keys=["a","b"],expected_text_hashes=[("q1","c1"),("q2","c2")],observed_text_hashes=[("q1","c1"),("q2","c2")])
    assert result["passed"]


def test_mutated_control_keys_fail():
    result=validate_score_equivalence([1.0,0.0],[1.0,0.0],["current-a","current-b"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT,expected_keys=["expected-a","expected-b"])
    assert not result["passed"] and not result["sample_keys_match_control"]


def test_different_experiment_with_its_own_deterministic_control_passes():
    result=validate_score_equivalence([1.0,0.0],[1.0,0.0],["g1a-a","g1a-b"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT,expected_keys=["g1a-a","g1a-b"])
    assert result["passed"]


def test_mutated_control_text_fails():
    result=validate_score_equivalence([1.0],[1.0],["a"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT,expected_keys=["a"],expected_text_hashes=[("query","chunk")],observed_text_hashes=[("query","mutated")])
    assert not result["passed"] and not result["sample_texts_match_control"]


def test_material_score_difference_fails():
    result=validate_score_equivalence([1.0,0.0],[1.02,0.0],["a","b"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT)
    assert not result["passed"]


def test_changed_ordering_on_close_scores_fails():
    result=validate_score_equivalence([1.0,.999],[.998,1.001],["a","b"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT)
    assert not result["passed"] and not result["ordering_unchanged"]


def test_wrong_contract_fails_before_tolerance_can_accept():
    wrong={**CONTRACT,"revision":"wrong"}
    result=validate_score_equivalence([1.0],[1.0],["a"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=wrong)
    assert not result["passed"] and "revision" in result["contract_mismatches"]
