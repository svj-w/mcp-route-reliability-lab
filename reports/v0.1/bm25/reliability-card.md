# MCP Route Reliability Card

> This report is an evaluation artifact, not a safety certification.

## Scope and provenance

| Field | Value |
|---|---|
| Dataset | v0.1 (`sha256:e1fca3ad3a871d3028c0e4989ad3c78bed515724f4f0f0f065dc8f173661c6eb`) |
| Tool corpus | v0.1 (`sha256:a4cb1dc1d0ccfa9bb0b559dae7729c455c45afa15abe04219446e8b65a94eb50`) |
| Router | bm25 1.0.0 |
| Model | rank-bm25:0.2.2 |
| Score metadata | `{"score_type":"similarity","higher_is_better":true,"comparable_across_queries":false}` |
| Feature version | 1.0 |
| Source commit | `working-tree` |
| Calibration / frozen test N | 30 / 30 |

## Benchmark composition

| Case type | Count |
|---|---:|
| clarify | 25 |
| clear | 25 |
| hard_negative | 25 |
| no_match | 25 |

## Environment

| Component | Version |
|---|---|
| platform | Windows |
| python | 3.11 |

## Method comparison on frozen test

| Method | Brier | ECE | Error AUROC | AURC | Coverage | Selective risk | Accepted N |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.0758 | 0.1001 | 0.9444 | 0.5144 | 0.0000 | 0.0000 | 0 |

## Risk-controlled policies

Configured selection: **empirical**. Acceptance uses `confidence >= threshold`.

| Policy | Available | Threshold | Calibration accepted N | Empirical risk | One-sided 95% risk UCB | Reason |
|---|---|---:|---:|---:|---:|---|
| empirical | no | not_available | 0 | not_available | not_available | exploratory_v0_1 |
| conservative | no | not_available | 0 | not_available | not_available | exploratory_v0_1 |

## Grouped bootstrap intervals

Intervals resample complete `base_query_id` groups; variants are never independent units.

| Metric | Estimate | 95% lower | 95% upper | Replicates | Seed | Groups |
|---|---:|---:|---:|---:|---:|---:|
| routing_error | 0.8000 | 0.6667 | 0.9333 | 1000 | 20260921 | 30 |

## Paired robustness

| Condition | Accuracy shift | Top-1 flip | Decision flip | Confidence shift |
|---|---:|---:|---:|---:|
| No perturbation variants in this dataset | not_available | not_available | not_available | not_available |

## Error diagnostics

Exported error rows: **24**. The CSV includes query text, gold set, Top-K, decision, confidence, reason hint, language, and case type.

## Limitations

- V0.1 contains 100 author-created, class-balanced queries and is explicitly non-production.
- Synthetic class balance may differ from production traffic.
- Thresholds apply only to the bound router, model, tool corpus, feature version, score semantics, and a comparable query distribution.
- Absence of an available conservative policy is reported directly; the target risk is never relaxed silently.
- This report must not be interpreted as a safety certification.

## Reproduction commands

```console
mrrl benchmark --config configs/bm25.yaml --tools TOOLS --queries QUERIES --run-dir RUN
```
