"""Source-grounded Vietnamese-to-English anchors for the PubMed lexical arm."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import yaml


def normalize_vi(value: str) -> str:
    value = unicodedata.normalize("NFD", str(value).casefold())
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def load_lexicon(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("policy") != "local_source_explicit_bilingual_pairs_only":
        raise ValueError("unsupported terminology provenance policy")
    for entry in data.get("entries", []):
        required = {"concept_id", "kind", "vietnamese", "english", "source_doc_id", "source_url", "source_excerpt"}
        missing = required - set(entry)
        if missing or not entry["vietnamese"] or not entry["english"]:
            raise ValueError(f"invalid terminology entry {entry.get('concept_id')}: missing={sorted(missing)}")
    return data


def _contains_phrase(normalized_query: str, normalized_phrase: str) -> bool:
    return f" {normalized_phrase} " in f" {normalized_query} "


def grounded_anchors(query: str, lexicon: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_vi(query)
    anchors: list[dict[str, Any]] = []
    occupied: list[set[str]] = []
    entries = sorted(
        lexicon.get("entries", []),
        key=lambda entry: -max(len(normalize_vi(alias)) for alias in entry["vietnamese"]),
    )
    for entry in entries:
        matches = [alias for alias in entry["vietnamese"] if _contains_phrase(normalized, normalize_vi(alias))]
        if not matches:
            continue
        tokens = set(normalize_vi(max(matches, key=len)).split())
        if any(tokens <= old for old in occupied):
            continue
        occupied.append(tokens)
        anchors.append({
            "concept_id": str(entry["concept_id"]), "kind": str(entry["kind"]),
            "original_phrase": max(matches, key=len), "english": list(map(str, entry["english"])),
            "provenance": {
                "type": "local_explicit_bilingual_source", "doc_id": int(entry["source_doc_id"]),
                "url": str(entry["source_url"]), "excerpt": str(entry["source_excerpt"]),
            },
        })
    matched_literals: list[set[str]] = []
    for literal in sorted(lexicon.get("literal_medical_anchors", []), key=lambda value: (-len(value), value)):
        if _contains_phrase(normalized, normalize_vi(literal)):
            literal_tokens = set(normalize_vi(literal).split())
            if any(literal_tokens <= old for old in matched_literals):
                continue
            matched_literals.append(literal_tokens)
            english = literal.upper() if literal in {"hiv", "aids", "hpv", "pcos"} else literal
            if any(english.casefold() in {term.casefold() for term in row["english"]} for row in anchors):
                continue
            anchors.append({
                "concept_id": f"literal:{normalize_vi(literal)}", "kind": "query_literal",
                "original_phrase": literal, "english": [english],
                "provenance": {"type": "original_query_literal", "query_text": query},
            })
    return anchors


def anchored_fts_query(anchors: list[dict[str, Any]]) -> str:
    terms = []
    for anchor in anchors:
        for term in anchor["english"]:
            clean = str(term).replace('"', '""').strip()
            if clean and clean.casefold() not in {old.casefold() for old in terms}:
                terms.append(clean)
    return " OR ".join(f'"{term}"' for term in terms)


def repaired_search(connection, query: str, lexicon: dict[str, Any], dense: list[dict], limit: int = 100) -> dict:
    from src.retrieval.pubmed_lexical import search

    anchors = grounded_anchors(query, lexicon)
    expression = anchored_fts_query(anchors)
    if not anchors:
        return {
            "anchors": [], "lexical_query": "", "lexical": [], "candidates": dense[:limit],
            "fallback": True, "fallback_reason": "no_source_grounded_anchor",
            "required_anchors_preserved": True,
        }
    lexical = search(connection, expression, limit)
    if not lexical:
        return {
            "anchors": anchors, "lexical_query": expression, "lexical": [], "candidates": dense[:limit],
            "fallback": True, "fallback_reason": "grounded_query_returned_no_documents",
            "required_anchors_preserved": True,
        }
    return {
        "anchors": anchors, "lexical_query": expression, "lexical": lexical, "candidates": lexical,
        "fallback": False, "fallback_reason": None, "required_anchors_preserved": True,
    }
