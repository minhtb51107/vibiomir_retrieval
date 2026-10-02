import pytest

from src.ingestion.checkpoint import CheckpointStore, InvalidStatusTransition
from src.ingestion.models import CrawlResult, CrawlStatus


def _result(doc_id: int, status: CrawlStatus) -> CrawlResult:
    return CrawlResult(
        doc_id=doc_id,
        original_url=f"https://example.test/{doc_id}#fragment",
        fetch_url=f"https://example.test/{doc_id}",
        status=status,
        attempt_count=1,
    )


def test_checkpoint_resume_and_duplicate_doc_id_protection(tmp_path) -> None:
    path = tmp_path / "crawl.sqlite"
    with CheckpointStore(path) as store:
        store.save(_result(10, CrawlStatus.TIMEOUT))
        store.save(_result(10, CrawlStatus.TIMEOUT))
        store.save(_result(11, CrawlStatus.SUCCESS))
        assert store.count() == 2
        assert store.completed_doc_ids() == {10, 11}
        assert store.completed_doc_ids(retry_failures=True) == {11}

    with CheckpointStore(path) as resumed:
        assert resumed.completed_doc_ids() == {10, 11}


def test_status_transitions_protect_success(tmp_path) -> None:
    with CheckpointStore(tmp_path / "crawl.sqlite") as store:
        store.save(_result(1, CrawlStatus.TIMEOUT))
        store.save(_result(1, CrawlStatus.SUCCESS))
        with pytest.raises(InvalidStatusTransition):
            store.save(_result(1, CrawlStatus.NETWORK_ERROR))
        assert store.status_counts() == {"SUCCESS": 1}
