from __future__ import annotations

import re
import time
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from .cache import CachedResponse, ProbeCache
from .inventory import canonical_domain

QUERY_NAMES = {"s", "q", "query", "keyword", "keywords", "key", "wd", "word", "search", "searchword"}
SEARCH_MARKERS = ("search", "tim-kiem", "timkiem", "tìm-kiếm", "so-search", "sousuo", "搜索")
CATEGORY_MARKERS = ("category", "chuyen-muc", "chuyên-mục", "danh-muc", "tag/", "topic", "频道", "栏目")
ARTICLE_EXCLUDES = ("/search", "tim-kiem", "timkiem", "/category", "/tag/", "javascript:", "mailto:")
CHALLENGE_MARKERS = ("captcha", "cloudflare", "attention required", "verify you are human", "access denied")


@dataclass(frozen=True)
class SearchMechanism:
    template_url: str
    query_parameter: str
    evidence: str


class ConservativeFetcher:
    def __init__(
        self, cache: ProbeCache, *, user_agent: str, timeout: float,
        delay_seconds: float, max_new_requests: int, max_bytes: int = 2_000_000,
    ) -> None:
        self.cache = cache
        self.delay_seconds = delay_seconds
        self.max_new_requests = max_new_requests
        self.max_bytes = max_bytes
        self.new_requests = 0
        self.client = httpx.Client(
            headers={"User-Agent": user_agent}, timeout=timeout,
            follow_redirects=True, max_redirects=5,
        )

    def get(self, url: str) -> tuple[CachedResponse | None, str | None, bool]:
        cached = self.cache.get(url)
        if cached:
            return cached, None, True
        if self.new_requests >= self.max_new_requests:
            return None, "request_cap_reached", False
        if self.new_requests:
            time.sleep(self.delay_seconds)
        self.new_requests += 1
        try:
            with self.client.stream("GET", url) as response:
                data = bytearray()
                truncated = False
                for block in response.iter_bytes():
                    remaining = self.max_bytes - len(data)
                    if remaining <= 0:
                        truncated = True
                        break
                    data.extend(block[:remaining])
                    if len(block) > remaining:
                        truncated = True
                        break
                result = CachedResponse(
                    request_url=url, final_url=str(response.url),
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                    body=bytes(data), truncated=truncated,
                )
                self.cache.put(result)
                return result, None, False
        except (httpx.HTTPError, OSError) as exc:
            return None, type(exc).__name__ + ": " + str(exc)[:300], False

    def close(self) -> None:
        self.client.close()


def decode_body(response: CachedResponse) -> str:
    match = re.search(r"charset\s*=\s*['\"]?([\w.-]+)", response.content_type, re.I)
    encodings = [match.group(1) if match else None, "utf-8", "gb18030"]
    for encoding in encodings:
        if not encoding:
            continue
        try:
            return response.body.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return response.body.decode("utf-8", errors="replace")


def robots_policy(text: str, base_url: str, user_agent: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.set_url(urljoin(base_url, "/robots.txt"))
    parser.parse(text.splitlines())
    return parser


def sitemap_urls(robots_text: str) -> list[str]:
    values = []
    for line in robots_text.splitlines():
        if line.casefold().startswith("sitemap:"):
            value = line.split(":", 1)[1].strip()
            if value and value not in values:
                values.append(value)
    return values


def detect_search_mechanisms(html: str, final_url: str) -> list[SearchMechanism]:
    soup = BeautifulSoup(html, "html.parser")
    mechanisms: list[SearchMechanism] = []
    for form in soup.find_all("form"):
        if str(form.get("method", "get")).casefold() not in {"", "get"}:
            continue
        action = urljoin(final_url, str(form.get("action") or final_url))
        for field in form.find_all(["input", "textarea"]):
            name = str(field.get("name") or "").strip()
            field_type = str(field.get("type") or "text").casefold()
            if name.casefold() in QUERY_NAMES or (
                field_type in {"search"} and name
            ):
                mechanisms.append(SearchMechanism(action, name, "public GET search form on home page"))
                break
    for anchor in soup.find_all("a", href=True):
        href = urljoin(final_url, str(anchor["href"]))
        label = (anchor.get_text(" ", strip=True) + " " + href).casefold()
        if any(marker in label for marker in SEARCH_MARKERS):
            split = urlsplit(href)
            pairs = parse_qsl(split.query, keep_blank_values=True)
            query_name = next((key for key, _ in pairs if key.casefold() in QUERY_NAMES), None)
            if query_name:
                mechanisms.append(SearchMechanism(href, query_name, "public search link on home page"))
    unique = {}
    for mechanism in mechanisms:
        key = (mechanism.template_url, mechanism.query_parameter)
        unique[key] = mechanism
    return list(unique.values())


def build_search_url(mechanism: SearchMechanism, query: str) -> str:
    split = urlsplit(mechanism.template_url)
    pairs = [(key, value) for key, value in parse_qsl(split.query, keep_blank_values=True)
             if key != mechanism.query_parameter]
    pairs.append((mechanism.query_parameter, query))
    return urlunsplit((split.scheme, split.netloc, split.path, urlencode(pairs), ""))


def page_flags(html: str) -> dict[str, object]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    visible_prefix = soup.get_text(" ", strip=True)[:5000]
    challenge_text = (title + " " + visible_prefix).casefold()
    category_links = 0
    for anchor in soup.find_all("a", href=True):
        value = (str(anchor["href"]) + " " + anchor.get_text(" ", strip=True)).casefold()
        if any(marker in value for marker in CATEGORY_MARKERS):
            category_links += 1
    return {
        "title": title,
        "challenge": any(marker in challenge_text for marker in CHALLENGE_MARKERS),
        "category_link_count": category_links,
        "wordpress": bool(soup.find("meta", attrs={"name": re.compile("generator", re.I), "content": re.compile("wordpress", re.I)})),
    }


def extract_result_urls(html: str, final_url: str, official_domain: str, limit: int = 30) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    ranked: list[tuple[int, int, str]] = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        url = urljoin(final_url, str(anchor["href"])).split("#", 1)[0]
        if canonical_domain(url) != official_domain:
            continue
        split = urlsplit(url)
        folded = url.casefold()
        if split.path in {"", "/"} or any(marker in folded for marker in ARTICLE_EXCLUDES):
            continue
        if url in seen:
            continue
        seen.add(url)
        text = anchor.get_text(" ", strip=True)
        classes = " ".join(anchor.get("class", []))
        parent_classes = " ".join(anchor.parent.get("class", [])) if anchor.parent else ""
        class_text = (classes + " " + parent_classes).casefold()
        score = 0
        if any(marker in class_text for marker in ("result", "story", "heading", "title", "cms-link", "news")):
            score += 4
        if re.search(r"\.(?:s?html?|ldo|vov2?|vnp|tpo)(?:$|[?])", folded):
            score += 3
        if re.search(r"\d{5,}", split.path) or split.path.startswith("/w/"):
            score += 2
        if len(split.path) >= 45:
            score += 1
        if len(text) >= 20:
            score += 1
        ranked.append((-score, len(ranked), url))
    ranked.sort()
    return [url for _, _, url in ranked[:limit]]


def extract_sitemap_locs(xml: str, limit: int = 30) -> list[str]:
    values = re.findall(r"<loc>\s*([^<]+?)\s*</loc>", xml, flags=re.I)
    output = []
    for value in values:
        value = value.replace("&amp;", "&").strip()
        if value not in output:
            output.append(value)
        if len(output) >= limit:
            break
    return output


def classify_discovery_url(url: str) -> str:
    split = urlsplit(url)
    path = split.path.casefold()
    if any(marker in path for marker in ("/category", "/tag/", "/search", "tim-kiem", "sitemap")):
        return "CATEGORY_OR_INDEX"
    if path.startswith("/w/"):
        return "CONTENT_LIKE"
    if re.search(r"\d{5,}", path) or re.search(r"\.(?:s?html?|ldo|vov2?|vnp|tpo)$", path):
        return "CONTENT_LIKE"
    if len(path) >= 45 and path.count("/") >= 1:
        return "CONTENT_LIKE"
    return "CATEGORY_OR_INDEX"
