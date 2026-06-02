# Retrieval Evaluation Results

## Corpus 1: Prometheus Operator Runbooks
**Source:** https://runbooks.prometheus-operator.dev/runbooks/ (Apache 2.0)
**Docs:** 30  **Queries:** 38

| Metric | BM25 | Vector (all-MiniLM-L6-v2) |
|--------|------|--------------------------|
| P@1 | 0.7105 | 0.8684 |
| P@3 | 0.3246 | 0.3684 |
| P@5 | 0.2105 | 0.2263 |
| MRR | 0.8123 | 0.9211 |
| Query latency p50 (ms) | 0.034 | 4.95 |
| Query latency p95 (ms) | 0.041 | 26.669 |
| Index build (ms) | 1.222 | 222.605 |

## Corpus 2: Kubernetes Troubleshooting Documentation
**Source:** https://kubernetes.io/docs/tasks/debug/ (CC BY 4.0)
**Docs:** 30  **Queries:** 35

| Metric | BM25 | Vector (all-MiniLM-L6-v2) |
|--------|------|--------------------------|
| P@1 | 0.8857 | 0.8286 |
| P@3 | 0.3333 | 0.3429 |
| P@5 | 0.2114 | 0.2057 |
| MRR | 0.9181 | 0.894 |
| Query latency p50 (ms) | 0.031 | 4.812 |
| Query latency p95 (ms) | 0.041 | 5.367 |
| Index build (ms) | 2.593 | 71.684 |
