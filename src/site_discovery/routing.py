from __future__ import annotations


def route_query_variants(language: str, *, translation_available: bool) -> tuple[str, ...]:
    if language == "vi":
        return ("Q0", "Q1")
    if language == "en":
        return ("Q2", "Q1") if translation_available else ("Q1_CONTROL_ONLY",)
    if language == "zh":
        return ("Q3", "Q1") if translation_available else ("Q1_CONTROL_ONLY",)
    return ("Q1_CONTROL_ONLY",)


def dedupe_candidates(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    seen = set()
    output = []
    for row in rows:
        key = (row.get("domain"), row.get("url"))
        if key not in seen:
            seen.add(key)
            output.append(row)
    return output
