from __future__ import annotations

import asyncio
import csv
import json
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx

from src.ingestion.models import DomainPolicy, RobotsConfig
from src.ingestion.robots import RobotsPolicy
from src.ingestion.scheduler import DomainController

from .extraction import analyze_html, decode_document, declared_http_encoding
from .models import ExtractionMetrics, ProbeConfig, ProbeResult, SampleRecord


def _integer_header(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def _crawler_limits(probe_config: ProbeConfig, crawler_config: Any) -> Any:
    overrides = {
        domain: DomainPolicy(
            concurrency=probe_config.per_domain_concurrency,
            delay_seconds=max(probe_config.domain_delay_seconds, policy.delay_seconds),
        )
        for domain, policy in crawler_config.domain_overrides.items()
    }
    return replace(
        crawler_config,
        global_concurrency=probe_config.global_concurrency,
        default_domain_policy=DomainPolicy(
            concurrency=probe_config.per_domain_concurrency,
            delay_seconds=probe_config.domain_delay_seconds,
        ),
        domain_overrides=overrides,
        robots=RobotsConfig(enabled=True, allow_on_fetch_error=True),
    )


class SourceProbe:
    def __init__(self, probe_config: ProbeConfig, crawler_config: Any):
        self.config = probe_config
        self.crawler_config = _crawler_limits(probe_config, crawler_config)

    async def run(self, records: list[SampleRecord]) -> list[ProbeResult]:
        if len(records) > self.config.maximum_live_urls:
            raise ValueError(
                f"refusing {len(records)} probes; configured maximum is "
                f"{self.config.maximum_live_urls}"
            )
        if len(records) > self.config.absolute_safety_ceiling:
            raise ValueError("absolute live probe ceiling of 200 exceeded")

        timeout = httpx.Timeout(
            connect=self.config.connect_timeout_seconds,
            read=self.config.read_timeout_seconds,
            write=self.config.connect_timeout_seconds,
            pool=self.config.connect_timeout_seconds,
        )
        limits = httpx.Limits(
            max_connections=self.config.global_concurrency,
            max_keepalive_connections=self.config.global_concurrency,
        )
        async with httpx.AsyncClient(
            headers={"User-Agent": self.config.user_agent},
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            max_redirects=10,
        ) as client:
            robots = RobotsPolicy(
                client, self.crawler_config.robots, self.config.user_agent
            )
            controller = DomainController(self.crawler_config)
            queue: asyncio.Queue[SampleRecord | None] = asyncio.Queue()
            for record in records:
                queue.put_nowait(record)
            for _ in range(self.config.global_concurrency):
                queue.put_nowait(None)
            results: list[ProbeResult] = []
            result_lock = asyncio.Lock()

            async def worker() -> None:
                while True:
                    record = await queue.get()
                    try:
                        if record is None:
                            return
                        result = await self._probe_one(record, client, robots, controller)
                        async with result_lock:
                            results.append(result)
                        print(
                            f"doc_id={result.doc_id} domain={result.domain} "
                            f"status={result.status} http={result.http_status} "
                            f"bytes={result.downloaded_bytes}"
                        )
                    finally:
                        queue.task_done()

            workers = [
                asyncio.create_task(worker())
                for _ in range(self.config.global_concurrency)
            ]
            await queue.join()
            await asyncio.gather(*workers)
        return sorted(results, key=lambda item: (item.domain, item.doc_id))

    async def _probe_one(
        self,
        record: SampleRecord,
        client: httpx.AsyncClient,
        robots: RobotsPolicy,
        controller: DomainController,
    ) -> ProbeResult:
        result = ProbeResult(
            doc_id=record.doc_id,
            original_url=record.original_url,
            fetch_url=record.fetch_url,
            domain=record.domain,
            sampling_categories=record.sampling_categories,
            inferred_language=record.inferred_language,
        )
        started = time.perf_counter()
        async with controller.slot(record.domain):
            try:
                allowed = await robots.can_fetch(record.fetch_url)
                result.robots_allowed = allowed
                cache_entry = robots.cache_summary().get(record.domain, {})
                result.robots_status = cache_entry.get("source_status")  # type: ignore[assignment]
                if not allowed:
                    result.status = "ROBOTS_BLOCKED"
                    result.error_type = "RobotsDisallowed"
                    result.elapsed_ms = round((time.perf_counter() - started) * 1000)
                    return result

                async with client.stream("GET", record.fetch_url) as response:
                    result.final_url = str(response.url)
                    result.http_status = response.status_code
                    result.redirect_count = len(response.history)
                    content_type_header = response.headers.get("content-type")
                    result.content_type = (
                        content_type_header.split(";", 1)[0].strip().lower()
                        if content_type_header
                        else None
                    )
                    result.declared_http_encoding = declared_http_encoding(
                        content_type_header
                    )
                    result.client_encoding = response.encoding
                    declared_size = _integer_header(
                        response.headers.get("content-length")
                    )
                    body_parts: list[bytes] = []
                    downloaded = 0
                    async for chunk in response.aiter_bytes():
                        remaining = self.config.max_response_bytes - downloaded
                        if remaining <= 0:
                            result.body_truncated = True
                            break
                        piece = chunk[:remaining]
                        body_parts.append(piece)
                        downloaded += len(piece)
                        if len(piece) < len(chunk):
                            result.body_truncated = True
                            break
                    body = b"".join(body_parts)
                    result.downloaded_bytes = downloaded
                    result.response_size_bytes = declared_size or downloaded
                    result.status = (
                        "SUCCESS"
                        if 200 <= response.status_code < 300
                        else "HTTP_ERROR"
                    )
                    if result.status == "HTTP_ERROR":
                        result.error_type = "HTTPStatusError"
                        result.error_message = f"HTTP {response.status_code}"

                    is_html = result.content_type in {
                        "text/html",
                        "application/xhtml+xml",
                    }
                    if is_html and body:
                        decoded = decode_document(body, content_type_header)
                        result.declared_meta_encoding = decoded.declared_meta_encoding
                        result.detected_encoding = decoded.detected_encoding
                        result.selected_encoding = decoded.selected_encoding
                        result.decoding_replacement_count = decoded.replacement_count
                        result.mojibake_marker_count = decoded.mojibake_marker_count
                        result.unicode_nfc_changed = decoded.unicode_nfc_changed
                        analysis = analyze_html(decoded.text)
                        result.title = analysis.title
                        result.basic_text_length = analysis.basic_text_length
                        result.language_signal = analysis.language_signal
                        result.language_signal_basis = analysis.language_signal_basis
                        result.template_classification = analysis.template_classification
                        result.template_evidence = analysis.template_evidence
                        result.structure = analysis.structure
                        result.extraction = analysis.extraction_metrics
                        result.anti_bot_signals = analysis.anti_bot_signals
                        if response.status_code in {401, 403, 429}:
                            signal = f"HTTP_{response.status_code}"
                            result.anti_bot_signals = sorted(
                                set([*result.anti_bot_signals, signal])
                            )
                            result.template_classification = "ACCESS_RESTRICTED"
                            result.template_evidence = [
                                *result.template_evidence,
                                f"HTTP access response {response.status_code}",
                            ]
                    elif result.status == "SUCCESS" and not body:
                        result.anti_bot_signals.append("EMPTY_BODY")
            except httpx.TimeoutException as exc:
                result.status = "TIMEOUT"
                result.error_type = type(exc).__name__
                result.error_message = str(exc)[:1000]
            except httpx.TooManyRedirects as exc:
                result.status = "REDIRECT_LOOP"
                result.error_type = type(exc).__name__
                result.error_message = str(exc)[:1000]
            except httpx.RequestError as exc:
                result.status = "NETWORK_ERROR"
                result.error_type = type(exc).__name__
                result.error_message = str(exc)[:1000]
        result.elapsed_ms = round((time.perf_counter() - started) * 1000)
        return result


def summarize_results(results: list[ProbeResult]) -> dict[str, Any]:
    html_results = [
        item
        for item in results
        if item.status == "SUCCESS"
        and item.content_type in {"text/html", "application/xhtml+xml"}
    ]
    approach_values: dict[str, list[Any]] = defaultdict(list)
    for result in html_results:
        for metric in result.extraction:
            approach_values[metric.approach].append(metric)
    extraction_summary: dict[str, Any] = {}
    for approach, metrics in sorted(approach_values.items()):
        chars = [item.character_count for item in metrics]
        extraction_summary[approach] = {
            "pages": len(metrics),
            "mean_character_count": round(statistics.mean(chars), 1) if chars else 0,
            "median_character_count": round(statistics.median(chars), 1) if chars else 0,
            "empty_count": sum(item.empty for item in metrics),
            "empty_rate": round(sum(item.empty for item in metrics) / len(metrics), 4)
            if metrics
            else 0,
            "title_preserved_count": sum(item.title_preserved for item in metrics),
            "title_preserved_rate": round(
                sum(item.title_preserved for item in metrics) / len(metrics), 4
            )
            if metrics
            else 0,
            "boilerplate_marker_count": sum(
                item.boilerplate_marker_count for item in metrics
            ),
        }

    domains: dict[str, Any] = {}
    for domain in sorted({item.domain for item in results}):
        group = [item for item in results if item.domain == domain]
        domains[domain] = {
            "sampled": len(group),
            "statuses": dict(Counter(item.status for item in group)),
            "http_statuses": dict(
                Counter(str(item.http_status) for item in group if item.http_status is not None)
            ),
            "content_types": dict(Counter(item.content_type or "missing" for item in group)),
            "language_signals": dict(Counter(item.language_signal for item in group)),
            "template_classifications": dict(
                Counter(item.template_classification for item in group)
            ),
            "redirected": sum(item.redirect_count > 0 for item in group),
            "robots_blocked": sum(item.status == "ROBOTS_BLOCKED" for item in group),
            "anti_bot_signals": dict(
                Counter(signal for item in group for signal in item.anti_bot_signals)
            ),
        }

    return {
        "total_probed": len(results),
        "status_counts": dict(Counter(item.status for item in results)),
        "content_type_counts": dict(
            Counter(item.content_type or "missing" for item in results)
        ),
        "http_status_counts": dict(
            Counter(str(item.http_status) for item in results if item.http_status is not None)
        ),
        "redirected_count": sum(item.redirect_count > 0 for item in results),
        "robots_blocked_count": sum(item.status == "ROBOTS_BLOCKED" for item in results),
        "html_success_count": len(html_results),
        "declared_http_encoding_counts": dict(
            Counter(item.declared_http_encoding or "missing" for item in html_results)
        ),
        "declared_meta_encoding_counts": dict(
            Counter(item.declared_meta_encoding or "missing" for item in html_results)
        ),
        "detected_encoding_counts": dict(
            Counter(item.detected_encoding or "missing" for item in html_results)
        ),
        "client_encoding_counts": dict(
            Counter(item.client_encoding or "missing" for item in html_results)
        ),
        "selected_encoding_counts": dict(
            Counter(item.selected_encoding or "missing" for item in html_results)
        ),
        "language_signal_counts": dict(
            Counter(item.language_signal for item in html_results)
        ),
        "template_classification_counts": dict(
            Counter(item.template_classification for item in html_results)
        ),
        "decoding_replacement_total": sum(
            item.decoding_replacement_count for item in html_results
        ),
        "mojibake_marker_total": sum(
            item.mojibake_marker_count for item in html_results
        ),
        "unicode_nfc_changed_pages": sum(
            item.unicode_nfc_changed for item in html_results
        ),
        "anti_bot_signal_counts": dict(
            Counter(signal for item in results for signal in item.anti_bot_signals)
        ),
        "extraction_comparison": extraction_summary,
        "domains": domains,
    }


def _result_from_dict(data: dict[str, Any]) -> ProbeResult:
    copied = dict(data)
    if not copied.get("client_encoding") and copied.get("declared_http_encoding"):
        # httpx selects a valid declared HTTP charset when one is present.
        copied["client_encoding"] = copied["declared_http_encoding"]
    if (
        copied.get("status") == "HTTP_ERROR"
        and copied.get("http_status") is not None
        and not copied.get("error_type")
    ):
        copied["error_type"] = "HTTPStatusError"
        copied["error_message"] = f"HTTP {copied['http_status']}"
    copied["extraction"] = [
        ExtractionMetrics(**item) for item in copied.get("extraction", [])
    ]
    return ProbeResult(**copied)


def write_probe_results(
    output_dir: str | Path,
    results: list[ProbeResult],
    *,
    merge_existing: bool = False,
) -> tuple[Path, Path, Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "probe_results.json"
    csv_path = directory / "probe_results.csv"
    summary_path = directory / "summary.json"
    if merge_existing and json_path.exists():
        existing = [
            _result_from_dict(item)
            for item in json.loads(json_path.read_text(encoding="utf-8"))
        ]
        merged = {item.doc_id: item for item in existing}
        merged.update({item.doc_id: item for item in results})
        results = sorted(merged.values(), key=lambda item: (item.domain, item.doc_id))
    json_path.write_text(
        json.dumps([item.as_dict() for item in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = summarize_results(results)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "doc_id",
            "domain",
            "sampling_categories",
            "original_url",
            "fetch_url",
            "final_url",
            "status",
            "http_status",
            "content_type",
            "declared_http_encoding",
            "declared_meta_encoding",
            "detected_encoding",
            "client_encoding",
            "selected_encoding",
            "response_size_bytes",
            "downloaded_bytes",
            "body_truncated",
            "redirect_count",
            "elapsed_ms",
            "title",
            "basic_text_length",
            "language_signal",
            "template_classification",
            "robots_allowed",
            "robots_status",
            "anti_bot_signals",
            "error_type",
            "error_message",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            data = result.as_dict()
            row = {field: data.get(field) for field in fields}
            row["sampling_categories"] = ";".join(result.sampling_categories)
            row["anti_bot_signals"] = ";".join(result.anti_bot_signals)
            writer.writerow(row)
    return json_path, csv_path, summary_path
