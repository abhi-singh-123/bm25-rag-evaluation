# BM25 vs Vector Search: Retrieval Evaluation

Replication package for the IEEE Software article **"The Retrieval Layer You Already Have: Why most RAG pipelines reach for a vector database before they need one"** by Kumar Abhishek (Akamai Technologies, Inc.).

## What This Is

A reproducible evaluation comparing BM25 and sentence-transformers (all-MiniLM-L6-v2) retrieval across two independent publicly licensed corpora of Kubernetes/SRE technical documentation.

**Key finding:** The results depend on vocabulary alignment between document authors and query authors. On Corpus 1 (terse diagnostic runbooks), vector search leads by 15 percentage points on P@1. On Corpus 2 (accessible practitioner documentation), BM25 wins by 6 percentage points. In both cases BM25 is 27 to 183x faster to index with zero external dependencies.

## Corpora

| Corpus | Source | License | Docs | Queries |
|--------|--------|---------|------|---------|
| Prometheus Operator Runbooks | [github.com/prometheus-operator/runbooks](https://github.com/prometheus-operator/runbooks) | Apache 2.0 | 30 | 38 |
| Kubernetes Official Documentation | [kubernetes.io/docs/tasks/debug](https://kubernetes.io/docs/tasks/debug/) | CC BY 4.0 | 30 | 35 |

Content fetched verbatim from source repositories. Corpus 1 raw files at `content/runbooks/` in the GitHub repo. Corpus 2 raw files at `content/en/docs/tasks/debug/` in [github.com/kubernetes/website](https://github.com/kubernetes/website).

## Results

### Corpus 1: Prometheus Operator Runbooks (Apache 2.0)

| Metric | BM25 | sentence-transformers (all-MiniLM-L6-v2) |
|--------|------|------------------------------------------|
| P@1 | 0.711 | 0.868 |
| P@3 | 0.325 | 0.368 |
| P@5 | 0.211 | 0.226 |
| MRR | 0.812 | 0.921 |
| Query latency p50 | 0.034 ms | 4.950 ms |
| Query latency p95 | 0.041 ms | 26.669 ms |
| Index build time | 1.222 ms | 222.605 ms |

### Corpus 2: Kubernetes Official Documentation (CC BY 4.0)

| Metric | BM25 | sentence-transformers (all-MiniLM-L6-v2) |
|--------|------|------------------------------------------|
| P@1 | 0.886 | 0.829 |
| P@3 | 0.333 | 0.343 |
| P@5 | 0.211 | 0.206 |
| MRR | 0.918 | 0.894 |
| Query latency p50 | 0.031 ms | 4.812 ms |
| Query latency p95 | 0.041 ms | 5.367 ms |
| Index build time | 2.593 ms | 71.684 ms |

Results from author's machine (Apple MacBook, Python 3.9). Quality metrics (P@1, P@3, P@5, MRR) are deterministic. Latency metrics vary by machine load.

## Reproduction

```bash
pip install -r requirements.txt
python eval_unified.py
```

Outputs `eval_unified_results.json` and `eval_unified_results.md` with all metrics for both corpora.

**Requirements:** Python 3.9+, internet connection for first model download (~90 MB).

## Files

| File | Description |
|------|-------------|
| `eval_unified.py` | Unified evaluation script — both corpora, both retrievers, single run |
| `requirements.txt` | Python dependencies |
| `eval_unified_results.json` | Raw results from author's machine |
| `eval_unified_results.md` | Formatted results tables |
| `README.md` | This file |

## Citation

> Abhishek, K. 2026. The Retrieval Layer You Already Have: Why most RAG pipelines reach for a vector database before they need one. *IEEE Software* (under review).

## Author

Kumar Abhishek — Engineering Manager, Quality Engineering, Akamai Technologies, Inc.
ORCID: [0009-0005-7443-0235](https://orcid.org/0009-0005-7443-0235)
LinkedIn: [linkedin.com/in/kr0abhishek](https://linkedin.com/in/kr0abhishek)

## License

Evaluation code: MIT License.
Corpus 1 content fetched verbatim from Apache 2.0 licensed sources (github.com/prometheus-operator/runbooks).
Corpus 2 content fetched verbatim from CC BY 4.0 licensed sources (github.com/kubernetes/website).
