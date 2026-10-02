import pyarrow as pa
import pyarrow.parquet as pq

from src.probing.models import ProbeConfig, SamplingRule
from src.probing.probe import _result_from_dict
from src.probing.sampler import select_samples


def _config(path: str) -> ProbeConfig:
    return ProbeConfig(
        input_path=path,
        output_dir="unused",
        seed="fixed-seed",
        maximum_live_urls=4,
        absolute_safety_ceiling=200,
        batch_size=2,
        global_concurrency=1,
        per_domain_concurrency=1,
        domain_delay_seconds=0,
        connect_timeout_seconds=1,
        read_timeout_seconds=1,
        max_response_bytes=1000,
        user_agent="test",
        sampling_rules=(
            SamplingRule("domain_a", "domain", 2, "a.test", "vi"),
            SamplingRule("queries", "query_string", 1),
            SamplingRule("suffix", "suffix", 1, ".vov"),
        ),
    )


def test_seeded_balanced_sampling_is_reproducible(tmp_path) -> None:
    path = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "url": [
                    "https://a.test/one?q=1",
                    "https://a.test/two",
                    "https://a.test/three",
                    "https://b.test/story.vov",
                    "http://c.test/no-extension",
                ],
            }
        ),
        path,
    )
    first, first_counts = select_samples(_config(str(path)))
    second, second_counts = select_samples(_config(str(path)))
    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]
    assert first_counts == second_counts == {"domain_a": 2, "queries": 1, "suffix": 1}
    assert len(first) <= 4
    assert any("queries" in item.sampling_categories for item in first)
    assert any("suffix" in item.sampling_categories for item in first)


def test_legacy_http_error_result_gets_explicit_error_metadata() -> None:
    result = _result_from_dict(
        {
            "doc_id": 7,
            "original_url": "https://example.test/blocked",
            "fetch_url": "https://example.test/blocked",
            "domain": "example.test",
            "sampling_categories": ["test"],
            "inferred_language": None,
            "robots_allowed": True,
            "robots_status": "ALLOWED",
            "status": "HTTP_ERROR",
            "http_status": 403,
        }
    )

    assert result.error_type == "HTTPStatusError"
    assert result.error_message == "HTTP 403"
