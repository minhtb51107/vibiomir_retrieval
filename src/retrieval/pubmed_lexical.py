"""Deterministic disk-backed PubMed lexical retrieval primitives."""
from __future__ import annotations

import hashlib
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Iterable


ENGLISH_STOPWORDS = frozenset("""
a an and are as at be been being by can could did do does doing for from had has have
having he her hers him his how i if in into is it its may might more most no not of on
or our ours she should than that the their theirs them then there these they this those
to under up was we were what when where which who why will with would you your study
studies result results conclusion conclusions method methods background objective
patients patient group groups using use used based among between associated effect
effects analysis data significant compared clinical medical disease treatment
""".split())

TOKEN_RE = re.compile(r"[a-z][a-z0-9-]{2,}")


def english_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(str(text).casefold()) if token not in ENGLISH_STOPWORDS]


def configure(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")  # bounded 32 MiB page cache
    connection.execute("PRAGMA foreign_keys=ON")


def initialize_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    configure(connection)
    connection.executescript("""
    CREATE TABLE IF NOT EXISTS documents(
      doc_id INTEGER PRIMARY KEY,
      ordinal INTEGER NOT NULL UNIQUE,
      official_url TEXT NOT NULL,
      pmid INTEGER NOT NULL,
      title TEXT NOT NULL,
      abstract TEXT NOT NULL,
      content_hash TEXT NOT NULL
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      title, abstract, content='documents', content_rowid='doc_id',
      tokenize='unicode61 remove_diacritics 2'
    );
    CREATE TABLE IF NOT EXISTS build_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    connection.commit()
    return connection


def add_documents(connection: sqlite3.Connection, rows: list[dict]) -> int:
    """Atomically add a batch; exact duplicate resumes are harmless."""
    inserted = 0
    with connection:
        for row in rows:
            digest = hashlib.sha256(
                (str(row["doc_id"]) + "\0" + str(row["title"]) + "\0" + str(row["abstract"])).encode("utf-8")
            ).hexdigest()
            cursor = connection.execute(
                "INSERT OR IGNORE INTO documents(doc_id,ordinal,official_url,pmid,title,abstract,content_hash) VALUES(?,?,?,?,?,?,?)",
                (int(row["doc_id"]), int(row["ordinal"]), str(row["official_url"]), int(row["pmid"]),
                 str(row["title"]), str(row["abstract"]), digest),
            )
            if cursor.rowcount:
                connection.execute(
                    "INSERT INTO documents_fts(rowid,title,abstract) VALUES(?,?,?)",
                    (int(row["doc_id"]), str(row["title"]), str(row["abstract"])),
                )
                inserted += 1
        if rows:
            connection.execute(
                "INSERT INTO build_state(key,value) VALUES('completed_ordinal',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(max(int(row["ordinal"]) for row in rows) + 1),),
            )
    return inserted


def completed_ordinal(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT value FROM build_state WHERE key='completed_ordinal'").fetchone()
    return int(row[0]) if row else 0


def search(connection: sqlite3.Connection, lexical_query: str, limit: int = 100) -> list[dict]:
    if not lexical_query.strip():
        return []
    rows = connection.execute(
        "SELECT d.doc_id,d.official_url,d.pmid,d.title,bm25(documents_fts,4.0,1.0) AS score "
        "FROM documents_fts JOIN documents d ON d.doc_id=documents_fts.rowid "
        "WHERE documents_fts MATCH ? ORDER BY score ASC,d.doc_id ASC LIMIT ?",
        (lexical_query, int(limit)),
    ).fetchall()
    return [
        {"doc_id": int(row[0]), "official_url": row[1], "pmid": int(row[2]),
         "title": row[3], "bm25_score": float(row[4]), "rank": rank}
        for rank, row in enumerate(rows, 1)
    ]


def bridge_terms(
    seed_documents: Iterable[dict], document_frequency: dict[str, int], document_count: int,
    *, maximum_terms: int = 12, maximum_df_ratio: float = 0.05,
) -> list[dict]:
    """Select transparent English terms from frozen dense seeds using TF-IDF."""
    term_frequency = Counter()
    title_frequency = Counter()
    for row in seed_documents:
        term_frequency.update(english_tokens(str(row.get("title") or "") + " " + str(row.get("abstract") or "")))
        title_frequency.update(set(english_tokens(str(row.get("title") or ""))))
    scored = []
    for term, frequency in term_frequency.items():
        df = int(document_frequency.get(term, 0))
        if df <= 0 or df / max(document_count, 1) > maximum_df_ratio:
            continue
        score = (1.0 + math.log(frequency)) * math.log((document_count + 1) / (df + 1))
        score *= 1.25 if title_frequency[term] else 1.0
        scored.append({"term": term, "tf": frequency, "df": df, "score": score})
    return sorted(scored, key=lambda row: (-row["score"], row["term"]))[:maximum_terms]


def lexical_query(terms: Iterable[dict]) -> str:
    safe = [str(row["term"]).replace('"', '""') for row in terms]
    return " OR ".join(f'"{term}"' for term in safe)


def reciprocal_rank_union(dense: list[dict], lexical: list[dict], constant: int = 60) -> list[dict]:
    """Fuse ranks without pretending cosine and BM25 scores share a scale."""
    union: dict[int, dict] = {}
    for arm, rows in (("dense", dense), ("lexical", lexical)):
        for row in rows:
            doc_id = int(row["doc_id"])
            item = union.setdefault(doc_id, {"doc_id": doc_id, "dense_rank": None, "dense_score": None,
                                             "lexical_rank": None, "lexical_score": None, "candidate_arms": []})
            item["candidate_arms"].append(arm)
            item[f"{arm}_rank"] = int(row["rank"])
            item[f"{arm}_score"] = float(row["dense_score"] if arm == "dense" else row["bm25_score"])
    for item in union.values():
        item["candidate_arms"] = sorted(set(item["candidate_arms"]))
        item["rrf_score"] = sum(1.0 / (constant + int(item[f"{arm}_rank"])) for arm in item["candidate_arms"])
    ordered = sorted(union.values(), key=lambda row: (-row["rrf_score"], row["doc_id"]))
    for rank, row in enumerate(ordered, 1):
        row["hybrid_rank"] = rank
    return ordered
