# MCP Route Reliability Card

> This report is an evaluation artifact, not a safety certification.

## Scope and provenance

| Field | Value |
|---|---|
| Dataset | v1.0 (`sha256:dataset`) |
| Tool corpus | v1.0 (`sha256:tools`) |
| Router | embedding 1.0.0 |
| Model | sentence-transformers:test-model |
| Score metadata | `{"score_type":"similarity","higher_is_better":true,"comparable_across_queries":false}` |
| Feature version | 1.0 |
| Source commit | `abc123` |
| Calibration / frozen test N | 100 / 100 |

## Benchmark composition

| Case type | Count |
|---|---:|
| clarify | 60 |
| clear | 100 |
| hard_negative | 80 |
| no_match | 60 |

## Environment

| Component | Version |
|---|---|
| platform | test-os |
| python | 3.11 |

## Method comparison on frozen test

| Method | Brier | ECE | Error AUROC | AURC | Coverage | Selective risk | Accepted N |
|---|---:|---:|---:|---:|---:|---:|---:|
| max_score | 0.2200 | 0.1800 | 0.7100 | 0.1300 | 0.7000 | 0.0857 | 70 |
| logistic | 0.1400 | 0.0900 | 0.8300 | 0.0700 | 0.4200 | 0.0476 | 42 |

## Risk-controlled policies

Configured selection: **empirical**. Acceptance uses `confidence >= threshold`.

| Policy | Available | Threshold | Calibration accepted N | Empirical risk | One-sided 95% risk UCB | Reason |
|---|---|---:|---:|---:|---:|---|
| empirical | yes | 0.8200 | 42 | 0.0476 | 0.1130 | — |
| conservative | no | not_available | 0 | not_available | not_available | no_qualifying_threshold |

## Grouped bootstrap intervals

Intervals resample complete `base_query_id` groups; variants are never independent units.

| Metric | Estimate | 95% lower | 95% upper | Replicates | Seed | Groups |
|---|---:|---:|---:|---:|---:|---:|
| selective_risk | 0.0476 | 0.0200 | 0.0910 | 1000 | 20260921 | 100 |

## Paired robustness

Shifts are variant minus paired base. Decision flip is `not_available` when the selected policy is unavailable.

| Condition | Pairs | Accuracy shift | Top-1 flip | Decision flip | Confidence shift |
|---|---:|---:|---:|---:|---:|
| paraphrase | 50 | -0.0200 | 0.0800 | 0.0500 | -0.0300 |
| cross_language | 50 | -0.0700 | 0.1800 | 0.1200 | -0.0900 |

## Error diagnostics

Exported error rows: **1**. The CSV includes query text, gold set, Top-K, decision, confidence, reason hint, language, and case type.

## Limitations

- Synthetic class balance may differ from production traffic.
- The threshold is invalid after router, model, corpus, or score-semantics changes.
- Thresholds apply only to the bound router, model, tool corpus, feature version, score semantics, and a comparable query distribution.
- Absence of an available conservative policy is reported directly; the target risk is never relaxed silently.
- This report must not be interpreted as a safety certification.

## Reproduction commands

Run from the repository root. Use a new `--run-dir` for each reproduction.

```console
mrrl evaluate --config configs/v1.yaml
mrrl report --run runs/example
```
