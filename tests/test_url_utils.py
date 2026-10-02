import pytest

from src.ingestion.url_utils import domain_from_url, prepare_fetch_url


def test_fragment_is_stripped_but_query_is_preserved() -> None:
    original = "https://Example.COM/article?id=42&lang=vi#section-3"
    assert prepare_fetch_url(original) == "https://Example.COM/article?id=42&lang=vi"
    assert domain_from_url(original) == "example.com"
    assert domain_from_url("https://www.example.com/path") == "example.com"


def test_invalid_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        prepare_fetch_url("ftp://example.com/file")
