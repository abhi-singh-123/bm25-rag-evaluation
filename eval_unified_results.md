# Retrieval Evaluation Results

## Corpus 1: Prometheus Operator Runbooks
**Source:** https://github.com/prometheus-operator/runbooks (Apache 2.0)
**Docs:** 30  **Queries:** 35

| Metric | BM25 | Vector (all-MiniLM-L6-v2) |
|--------|------|--------------------------|
| P@1 | 0.8 | 0.8286 |
| P@3 | 0.3143 | 0.3429 |
| P@5 | 0.2 | 0.2057 |
| MRR | 0.8652 | 0.9 |
| Query latency p50 (ms) | 0.03 | 5.161 |
| Query latency p95 (ms) | 0.041 | 34.268 |
| Index build (ms) | 0.646 | 153.053 |

## Corpus 2: Kubernetes Troubleshooting Documentation
**Source:** https://kubernetes.io/docs/tasks/debug/ (Apache 2.0)
**Docs:** 30  **Queries:** 36

| Metric | BM25 | Vector (all-MiniLM-L6-v2) |
|--------|------|--------------------------|
| P@1 | 0.8889 | 0.8611 |
| P@3 | 0.3333 | 0.3333 |
| P@5 | 0.2111 | 0.2167 |
| MRR | 0.9211 | 0.912 |
| Query latency p50 (ms) | 0.03 | 4.55 |
| Query latency p95 (ms) | 0.042 | 8.553 |
| Index build (ms) | 0.533 | 53.504 |
