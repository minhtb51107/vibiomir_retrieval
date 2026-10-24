"""Historical document-consensus-v1, isolated from old run paths/deadlines.

Same policy and pure selection semantics as the Phase25 cached experiment.
Multiple lexically independent windows are not semantic relevance labels.
"""
from collections import defaultdict
import itertools
import json
from pathlib import Path

POLICY = {"version": "document-consensus-v1", "protect_head": 15, "max_replacements": 2,
          "minimum_evidence_windows": 2, "maximum_5gram_overlap": 0.5,
          "minimum_window_words": 24, "victim_maximum_chunk_count": 1,
          "chunks": "unchanged", "documents": 20}


def fingerprints(text):
    words = text.casefold().split()
    return set(zip(*(words[i:] for i in range(5)))) if len(words) >= 24 else set()


def independent(windows):
    grams = [fingerprints(text) for _, text in windows]
    return any(a and b and len(a & b) / min(len(a), len(b)) <= .5
               for a, b in itertools.combinations(grams, 2))


def select_documents(record):
    docs = list(record['relevant_docs']); windows = defaultdict(list)
    for rank, chunk in enumerate(record['relevant_chunks'], 1):
        windows[int(chunk['doc_id'])].append((rank, chunk['chunk_text']))
    candidates = [d for d, spans in windows.items() if d not in docs and independent(spans)]
    candidates.sort(key=lambda d: (-len(windows[d]), windows[d][0][0], d))
    victims = [i for i in range(19, 14, -1) if len(windows[docs[i]]) <= 1]
    changes = []
    for candidate, index in zip(candidates[:2], victims):
        changes.append(dict(position=index+1, removed=docs[index], inserted=candidate,
            inserted_evidence_ranks=[rank for rank, _ in windows[candidate]],
            removed_evidence_ranks=[rank for rank, _ in windows[docs[index]]]))
        docs[index] = candidate
    if docs[:15] != record['relevant_docs'][:15] or len(docs) != 20 or len(set(docs)) != 20:
        raise ValueError('document policy invariant')
    return docs, changes


def records(path):
    decoder=json.JSONDecoder();buffer='';opened=False;eof=False
    with Path(path).open(encoding='utf-8') as stream:
        while True:
            buffer=buffer.lstrip()
            if not buffer:
                extra=stream.read(65536)
                if not extra:raise ValueError('truncated submission')
                buffer+=extra;continue
            if not opened:
                if buffer[0]!='[':raise ValueError('submission root not array')
                opened=True;buffer=buffer[1:];continue
            if buffer.startswith(']'):
                if buffer[1:].strip() or stream.read().strip():raise ValueError('trailing JSON')
                return
            try:value,end=decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                extra=stream.read(65536)
                if not extra:raise ValueError('truncated submission')
                buffer+=extra
                if len(buffer)>16*1024*1024:raise ValueError('query exceeds bounded JSON buffer')
                continue
            yield value;buffer=buffer[end:].lstrip()
            while not buffer:
                extra=stream.read(65536)
                if not extra:raise ValueError('truncated array')
                buffer+=extra;buffer=buffer.lstrip()
            if buffer.startswith(','):buffer=buffer[1:]
            elif not buffer.startswith(']'):raise ValueError('missing JSON separator')


def verify_sources(record, canonical):
    if set(record)!={'id','relevant_docs','relevant_chunks'} or type(record['id']) is not int:
        raise ValueError('submission schema')
    docs=record['relevant_docs'];chunks=record['relevant_chunks']
    if len(docs)!=20 or len(set(docs))!=20 or any(type(d) is not int for d in docs) or len(chunks)!=40:
        raise ValueError('submission depth/document identities')
    owners=set(docs);seen=set()
    for chunk in chunks:
        if set(chunk)!={'doc_id','chunk_text'} or type(chunk['doc_id']) is not int or not isinstance(chunk['chunk_text'],str) or not chunk['chunk_text']:
            raise ValueError('chunk schema')
        pair=(chunk['doc_id'],chunk['chunk_text'])
        if pair in seen:raise ValueError('duplicate exact owner/window')
        seen.add(pair);owners.add(chunk['doc_id'])
    source={}
    for doc in owners:
        row=canonical.execute('SELECT normalized_text FROM documents WHERE doc_id=?',(doc,)).fetchone()
        if not row:raise ValueError('canonical owner missing')
        source[doc]=row[0]
    if any(text not in source[doc] for doc,text in seen):raise ValueError('nonverbatim source span')
    return owners
