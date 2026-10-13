# M2A decision

- PubMed FTS5 index: **PASS**, 239,896 documents, 0.63 GiB.
- Operational VI→EN pseudo-relevance expansions: 1,200/1,200; degenerate: 0.
- Additional lexical-only query/document pairs: 97,811; dense/lexical overlap: 22,189.
- Bounded evidence pairs: 435,558; existing scores: 239,954; missing scores: 195,604.
- No relevance labels establish that lexical-only documents are useful. Pseudo-relevance terms may drift semantically; inspect `sample_query_traces.json` before M2B.

M2B is resource-feasible only as resumable candidate-only reranking at batch=2 using the exact missing-pair count above. It is not automatically authorized.
