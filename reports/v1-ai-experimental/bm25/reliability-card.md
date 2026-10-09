# MCP Route Reliability Card

> This report is an evaluation artifact, not a safety certification.

## Scope and provenance

| Field | Value |
|---|---|
| Dataset | ai-experimental (`sha256:3e5f030f545dad8386ae98e688d81d16640fe03c7b5838291f9baa5741b7b478`) |
| Tool corpus | ai-experimental (`sha256:8983296fa7cc6343c94a11fe15307e87f3bf5c6f03bcd036e971b9e9e18de069`) |
| Router | bm25 1.0.0 |
| Model | rank-bm25:0.2.2 |
| Score metadata | `{"score_type":"similarity","higher_is_better":true,"comparable_across_queries":false}` |
| Feature version | 1.0 |
| Source commit | `6319120606b19f34802d0c9cad4dbf65866921cd` |
| Calibration / frozen test N | 100 / 190 |

## Benchmark composition

| Case type | Count |
|---|---:|
| clarify | 65 |
| clear | 100 |
| hard_negative | 63 |
| multi_valid | 12 |
| no_match | 60 |

## Environment

| Component | Version |
|---|---|
| platform | Windows |
| python | 3.11 |

## Method comparison on frozen test

| Method | Brier | ECE | Error AUROC | AURC | Coverage | Selective risk | Accepted N |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic | 0.1120 | 0.0711 | 0.8722 | 0.6364 | 0.0000 | 0.0000 | 0 |

## Risk-controlled policies

Configured selection: **empirical**. Acceptance uses `confidence >= threshold`.

| Policy | Available | Threshold | Calibration accepted N | Empirical risk | One-sided 95% risk UCB | Reason |
|---|---|---:|---:|---:|---:|---|
| empirical | no | not_available | 0 | not_available | not_available | no_qualifying_threshold |
| conservative | no | not_available | 0 | not_available | not_available | no_qualifying_threshold |

## Grouped bootstrap intervals

Intervals resample complete `base_query_id` groups; variants are never independent units.

| Metric | Estimate | 95% lower | 95% upper | Replicates | Seed | Groups |
|---|---:|---:|---:|---:|---:|---:|
| routing_error | 0.8263 | 0.7617 | 0.8859 | 1000 | 20260924 | 100 |

## Paired robustness

Shifts are variant minus paired base. Decision flip is `not_available` when the selected policy is unavailable.

| Condition | Pairs | Accuracy shift | Top-1 flip | Decision flip | Confidence shift |
|---|---:|---:|---:|---:|---:|
| overall | 90 | 0.0000 | 0.3667 | not_available | 0.0611 |
| paraphrase | 30 | -0.1000 | 0.1667 | not_available | 0.0130 |
| cross_language | 30 | 0.1000 | 0.9333 | not_available | 0.1729 |
| noise | 30 | 0.0000 | 0.0000 | not_available | -0.0025 |

## Error diagnostics

Exported error rows: **157**. The CSV includes query text, gold set, Top-K, decision, confidence, reason hint, language, and case type.

## Limitations

- AI-annotated synthetic experiment with post-hoc revisions and generator-provisional labels; not independent human review or a safety guarantee.
- Synthetic class balance may differ from production traffic.
- Thresholds apply only to the bound router, model, tool corpus, feature version, score semantics, and a comparable query distribution.
- Absence of an available conservative policy is reported directly; the target risk is never relaxed silently.
- This report must not be interpreted as a safety certification.

## Reproduction commands

Run from the repository root. Use a new `--run-dir` for each reproduction.

```console
mrrl benchmark --config configs/v1-ai-bm25.yaml --tools data/v1-ai-experimental/tools.json --queries data/v1-ai-experimental/queries.jsonl --run-dir runs/reproduce-ai-experimental-bm25
```
