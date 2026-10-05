from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.acquisition_benchmark.metrics import content_signals, early_stop_decision, language_label, tier_for
from src.acquisition_benchmark.sampling import annotate_phase8_reuse, deterministic_domain_samples, records_manifest


def test_domain_sampling_is_deterministic_and_stratified(tmp_path):
    corpus = tmp_path / "corpus.parquet"
    pq.write_table(pa.table({
        "id": list(range(1, 13)),
        "url": [f"https://a.test/{value}" for value in range(1, 7)]
        + [f"https://b.test/{value}" for value in range(7, 13)],
    }), corpus)
    first, populations = deterministic_domain_samples(
        corpus, domains={"a.test", "b.test"}, sample_size=3, seed="fixed"
    )
    second, _ = deterministic_domain_samples(
        corpus, domains={"a.test", "b.test"}, sample_size=3, seed="fixed"
    )
    assert first == second
    assert populations == {"a.test": 6, "b.test": 6}
    assert all(len(rows) == 3 for rows in first.values())
    assert all(rows == sorted(rows, key=lambda row: (row["rank"], row["doc_id"])) for rows in first.values())


def test_phase8_reuse_requires_exact_url():
    samples = {"a.test": [{"doc_id": 1, "original_url": "https://a.test/1"}]}
    counts = annotate_phase8_reuse(samples, {"records": [{"doc_id": 1, "original_url": "https://a.test/1"}]})
    assert counts == {"a.test": 1}
    assert samples["a.test"][0]["phase8_reusable"] is True
    with pytest.raises(ValueError, match="URL mismatch"):
        annotate_phase8_reuse(
            {"a.test": [{"doc_id": 1, "original_url": "https://a.test/different"}]},
            {"records": [{"doc_id": 1, "original_url": "https://a.test/1"}]},
        )


def test_bounded_manifest_preserves_doc_ids_and_limit():
    rows = [
        {"doc_id": value, "original_url": f"https://a/{value}", "rank": value}
        for value in range(5)
    ]
    manifest = records_manifest(rows, 2)
    assert manifest["records"] == [
        {"doc_id": 0, "original_url": "https://a/0"},
        {"doc_id": 1, "original_url": "https://a/1"},
    ]


def test_predeclared_early_stop_rules():
    assert early_stop_decision(access_rate=0.81, usable_rate=1.0, unsafe_antibot=False)[0]
    assert early_stop_decision(access_rate=0.0, usable_rate=0.19, unsafe_antibot=False)[0]
    assert early_stop_decision(access_rate=0.0, usable_rate=1.0, unsafe_antibot=True)[0]
    assert early_stop_decision(access_rate=0.0, usable_rate=0.20, unsafe_antibot=False) == (
        False, "checkpoint passed"
    )


def test_language_and_medical_signals_are_lightweight():
    chinese = content_signals("医生治疗患者疾病")
    vietnamese = content_signals("Bác sĩ điều trị bệnh nhân và sức khỏe")
    assert chinese["medical_terms"] >= 3
    assert vietnamese["medical_terms"] >= 3
    assert language_label(chinese["han"], chinese["latin"], chinese["vietnamese_specific"]) == "mostly Chinese"
    assert language_label(vietnamese["han"], vietnamese["latin"], vietnamese["vietnamese_specific"]) == "mostly Vietnamese"


def test_tier_rules_match_predeclaration():
    assert tier_for(usable_rate=0.9, access_rate=0.0, projected_50k_fits=True, throughput_acceptable=True) == "A"
    assert tier_for(usable_rate=0.6, access_rate=0.0, projected_50k_fits=True, throughput_acceptable=False) == "B"
    assert tier_for(usable_rate=0.4, access_rate=0.0, projected_50k_fits=True, throughput_acceptable=True) == "C"
    assert tier_for(usable_rate=1.0, access_rate=0.8, projected_50k_fits=True, throughput_acceptable=True) == "BLOCKED"
