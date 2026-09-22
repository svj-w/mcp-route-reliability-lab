# Two-minute demo script

## 0:00–0:20 — Problem

“An agent router should not merely pick the most similar tool. When a request is
ambiguous or the Tool Corpus has no matching capability, confidently choosing a tool
is itself an error. This project measures ranking and abstention together.”

Show the README result table and risk-coverage figure.

## 0:20–0:45 — Reproducible data contract

```powershell
mrrl validate --tools data/tools.json --queries data/queries.jsonl --profile v0.1
```

Point out the machine-readable totals: 20 tools, 100 queries, balanced Chinese and
English, and a frozen 40/30/30 split. Open `data/sources.csv` and mention the pinned
upstream commit and per-tool license/transformation record.

## 0:45–1:15 — Run the offline fixture

```powershell
mrrl benchmark `
  --config configs/fixture-bm25.yaml `
  --tools tests/fixtures/tools.jsonl `
  --queries tests/fixtures/queries.jsonl `
  --run-dir runs/demo `
  --json
```

Explain that routing receives `InferenceQuery`, which excludes gold labels. The
calibrator fits only on Calibration, configuration freezes, and Test is evaluated
once.

## 1:15–1:40 — Inspect the evidence

Open:

- `runs/demo/reliability-card.md`;
- `runs/demo/metrics.json`;
- `runs/demo/errors.csv`; and
- `runs/demo/manifest.json`.

“The manifest binds versions and hashes every artifact. The error export includes
the query, gold set, Top-K, confidence, independent reason hint, language, and case
type.”

## 1:40–2:00 — Fail closed and state the boundary

Open a V0.1 `policy.json` and show:

```json
{"available": false, "threshold": null, "unavailable_reason": "exploratory_v0_1"}
```

“The benchmark does not relax the target risk or market a small-sample threshold as
safe. V0.1 is a measured portfolio milestone. V1.0 requires a larger dataset and a
second human’s blinded annotation before any release claim.”
