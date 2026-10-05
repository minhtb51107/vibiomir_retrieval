from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq


ZH_DOMAINS = {
    "120ask.com", "a-hospital.com", "ask.39.net", "baby.39.net",
    "bingli.iiyi.com", "care.39.net", "cnkang.com", "familydoctor.com.cn",
    "fitness.39.net", "food.39.net", "heart.39.net", "jbk.39.net",
    "nk.39.net", "qihuangzhishu.com", "test.pmphai.com", "woman.39.net",
    "wujue.com", "youlai.cn", "zhongyibaodian.net", "zydcd.com",
    "zysjonline.com",
}
VI_DOT_COM_DOMAINS = {"hellobacsi.com", "vinmec.com"}


def canonical_domain(value: str) -> str:
    try:
        text = str(value).strip()
        host = (urlsplit(text if "://" in text else "//" + text).hostname or "").casefold().rstrip(".")
    except ValueError:
        return ""
    return host[4:] if host.startswith("www.") else host


def likely_language(domain: str) -> tuple[str, str]:
    if (
        domain in ZH_DOMAINS or domain.endswith(".cn") or domain.endswith(".39.net")
        or domain.endswith(".iiyi.com") or domain == "jb39.com" or domain == "baidu.com"
    ):
        return "zh", "known Chinese-associated source or .cn suffix"
    if domain in VI_DOT_COM_DOMAINS or domain.endswith(".vn") or domain.endswith(".com.vn"):
        return "vi", "Vietnamese country-code domain (source association; not content measurement)"
    return "unknown", "hostname alone is insufficient"


def corpus_domain_counts(corpus_path: str | Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    parquet = pq.ParquetFile(corpus_path)
    for batch in parquet.iter_batches(batch_size=131072, columns=["url"]):
        for url in batch.column(0).to_pylist():
            counts[canonical_domain(str(url)) or "__NO_DOMAIN__"] += 1
    return counts


def pilot_domain_counts(chunks_path: str | Path) -> Counter[str]:
    path = Path(chunks_path)
    if not path.exists():
        return Counter()
    table = pq.read_table(path, columns=["doc_id", "source_url"])
    docs: dict[int, str] = {}
    for row in table.to_pylist():
        docs.setdefault(int(row["doc_id"]), canonical_domain(str(row["source_url"])))
    return Counter(docs.values())


def phase2_evidence(probe_path: str | Path) -> dict[str, dict[str, object]]:
    path = Path(probe_path)
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[canonical_domain(str(row.get("original_url", "")))].append(row)
    output = {}
    for domain, values in grouped.items():
        statuses = Counter(str(row.get("status", "UNKNOWN")) for row in values)
        classes = Counter(str(row.get("template_classification", "UNKNOWN")) for row in values)
        languages = Counter(str(row.get("language_signal", "unknown")) for row in values)
        output[domain] = {
            "sample_count": len(values),
            "statuses": dict(sorted(statuses.items())),
            "template_classes": dict(sorted(classes.items())),
            "content_language_signals": dict(sorted(languages.items())),
        }
    return output


def bing_domains(path: str | Path) -> set[str]:
    source = Path(path)
    if not source.exists():
        return set()
    payload = json.loads(source.read_text(encoding="utf-8"))
    return {canonical_domain(str(row["domain"])) for row in payload.get("mapped", [])}


def build_inventory(
    *, corpus_path: str | Path, chunks_path: str | Path,
    probe_path: str | Path, bing_path: str | Path,
) -> list[dict[str, object]]:
    counts = corpus_domain_counts(corpus_path)
    total = sum(counts.values())
    pilot = pilot_domain_counts(chunks_path)
    evidence = phase2_evidence(probe_path)
    bing = bing_domains(bing_path)
    rows = []
    for domain, count in counts.most_common():
        language, basis = likely_language(domain)
        rows.append({
            "domain": domain,
            "corpus_rows": count,
            "corpus_share": round(count / total, 8),
            "likely_language": language,
            "language_basis": basis,
            "phase2_accessibility": evidence.get(domain),
            "pilot_document_count": pilot.get(domain, 0),
            "present_in_pilot": domain in pilot,
            "bing_b0_discovered": domain in bing,
            "selected_stage0": False,
            "selection_reason": None,
            "native_search": "UNKNOWN",
            "sitemap": "UNKNOWN",
            "category_index": "UNKNOWN",
            "mapping_works": "UNTESTED",
            "search_quality_tested": False,
            "recommended_acquisition": "UNKNOWN",
        })
    return rows


def select_representative_domains(rows: list[dict[str, object]], count: int = 19) -> list[str]:
    """Deterministically balance corpus share with source-language diversity."""
    chosen: list[str] = []

    def add(domain: str) -> None:
        if domain not in chosen and any(row["domain"] == domain for row in rows):
            chosen.append(domain)

    # Major Chinese-associated sources (including known blocked/access cases).
    for domain in (
        "cnkang.com", "120ask.com", "familydoctor.com.cn", "ask.39.net",
        "zysjonline.com", "a-hospital.com", "zhongyibaodian.net", "zydcd.com",
    ):
        add(domain)
    # Major and medium Vietnamese-associated sources, including Phase 2 failures.
    for domain in (
        "suckhoecongdongonline.vn", "suckhoedoisong.vn",
        "nhathuoclongchau.com.vn", "thanhnien.vn", "laodong.vn", "vinmec.com",
        "medlatec.vn", "vietnamnet.vn",
    ):
        add(domain)
    # Medium Chinese source and an unknown/English-associated candidate if available.
    add("youlai.cn")
    unknown = [row for row in rows if row["likely_language"] == "unknown"]
    if unknown:
        add(str(unknown[0]["domain"]))
    for row in rows:
        if len(chosen) >= count:
            break
        add(str(row["domain"]))
    return chosen[:count]


def mark_selection(rows: list[dict[str, object]], selected: list[str]) -> None:
    selected_set = set(selected)
    ranks = {domain: index + 1 for index, domain in enumerate(selected)}
    for row in rows:
        domain = str(row["domain"])
        if domain not in selected_set:
            continue
        row["selected_stage0"] = True
        row["selection_reason"] = (
            f"representative rank {ranks[domain]} balancing corpus share, language/source diversity, "
            "and known Phase 2 accessibility behavior"
        )


def weighted_share(rows: list[dict[str, object]], domains: set[str]) -> float:
    return sum(float(row["corpus_share"]) for row in rows if row["domain"] in domains)
