from src.reranking.equivalence import validate_score_equivalence


CONTRACT={
    "model":"BAAI/bge-reranker-v2-m3","revision":"rev","tokenizer_revision":"rev",
    "batch_size":2,"max_length":512,"precision":"float16","eval_mode":True,
    "padding":True,"truncation":"longest_first",
    "score_extraction":"logits.reshape(-1).float().cpu().numpy()","input_order":"query, chunk",
}


def test_measured_fp16_jitter_with_stable_order_passes():
    result=validate_score_equivalence([-2.80078125,-3.1171875],[-2.802734375,-3.111328125],["a","b"],measured_absolute_tolerance=.005859375,expected_contract=CONTRACT,actual_contract=CONTRACT)
    assert result["passed"]


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
