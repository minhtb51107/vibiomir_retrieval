import pyarrow as pa
import pyarrow.parquet as pq

from src.ingestion.corpus_loader import iter_corpus_records


def test_loader_preserves_original_and_prepares_fetch_url(tmp_path) -> None:
    path = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table(
            {
                "id": [1, 2],
                "url": [
                    "https://a.test/page?q=one#part",
                    "https://b.test/other?q=two",
                ],
            }
        ),
        path,
    )
    records = list(iter_corpus_records(path, limit=2))
    assert records[0].doc_id == 1
    assert records[0].original_url == "https://a.test/page?q=one#part"
    assert records[0].fetch_url == "https://a.test/page?q=one"
    assert records[1].fetch_url.endswith("?q=two")


def test_loader_start_after_id(tmp_path) -> None:
    path = tmp_path / "corpus.parquet"
    pq.write_table(
        pa.table({"id": [1, 2, 3], "url": [f"https://a.test/{i}" for i in range(3)]}),
        path,
    )
    records = list(iter_corpus_records(path, limit=None, start_after_id=1))
    assert [record.doc_id for record in records] == [2, 3]
