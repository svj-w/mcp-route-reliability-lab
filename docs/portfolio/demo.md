# Five-minute offline demo (after installation)

This demo shows how the evaluator works; it does **not** reproduce the committed V1 AI experimental scores. The small fixture is deliberately different from the 40-tool V1 dataset. Once Python 3.11 or 3.12 and the [base/report dependencies](../../README.md#installation) are installed, the BM25 path below needs no model download, API key, or paid service.

Run the commands from the repository root in PowerShell.

## 1. Validate the fixture

```powershell
mrrl validate --tools tests/fixtures/tools.jsonl `
  --queries tests/fixtures/queries.jsonl --profile fixture
```

Expected: `dataset is valid`. This checks the fixture's tool IDs, query labels, Tool Corpus hash, and split profile before any routing begins.

## 2. Run one complete evaluation

```powershell
mrrl benchmark --config configs/fixture-bm25.yaml `
  --tools tests/fixtures/tools.jsonl `
  --queries tests/fixtures/queries.jsonl `
  --run-dir runs/portfolio-fixture
```

Expected: `benchmark complete: runs/portfolio-fixture` (the displayed path may be absolute). The run fits confidence on Calibration, freezes its configuration, evaluates Test once, and writes provenance-bound artifacts. `runs/` is ignored by Git; no frozen dataset or committed report is modified.

Inspect four outputs:

| File | What to show |
|---|---|
| `runs/portfolio-fixture/reliability-card.md` | The readable summary, limitations, and policy status. |
| `runs/portfolio-fixture/metrics.json` | Ranking, calibration, selective-risk metrics, and `lifecycle.test_evaluations: 1`. |
| `runs/portfolio-fixture/errors.csv` | One failure row: compare `gold_tool_ids` with `top_k_tool_ids` and the abstention decision. |
| `runs/portfolio-fixture/manifest.json` | The router, model, Tool Corpus/query hashes, and artifact checksums. |

The fixture is a mechanism demo, not a performance claim. Its small Test set and policy behavior must not be described as the V1 result.

## 3. Show the frozen V1 AI experimental evidence

Open the committed [BM25 reliability card](../../reports/v1-ai-experimental/bm25/reliability-card.md) and [embedding reliability card](../../reports/v1-ai-experimental/embedding/reliability-card.md). Both reports evaluate the same frozen 40-tool corpus and Test set of 100 bases plus 90 paired variants. Compare their [BM25 metrics](../../reports/v1-ai-experimental/bm25/metrics.json) and [embedding metrics](../../reports/v1-ai-experimental/embedding/metrics.json), then inspect the [BM25 policy](../../reports/v1-ai-experimental/bm25/policy.json) and [embedding policy](../../reports/v1-ai-experimental/embedding/policy.json): both have `available: false` at the configured 5% target risk.

For a concrete routing error, find `v01_q_015` in the [embedding error export](../../reports/v1-ai-experimental/embedding/errors.csv): the request asks for a commit history list (`git.log` is gold), while `git.show` ranks ahead of it. This is a failure to explain, not a success to hide.

The V1 dataset is AI-annotated and not independently human-reviewed. Its 190 Test records are not 190 independent intents, and the unavailable policies are not safe deployment thresholds. The [Dataset Card](../../data/v1-ai-experimental/DATASET_CARD.md) and [annotation report](../../data/v1-ai-experimental/ANNOTATION_REPORT.md) document those limits.

## If a step fails

- `mrrl` not found or an import fails: activate a Python 3.11/3.12 environment and follow [Installation](../../README.md#installation), including `.[dev,report]` for this demo. Do not substitute a different baseline package silently.
- `runs/portfolio-fixture` already exists: choose a new unused `--run-dir`, such as `runs/portfolio-fixture-2`; the command intentionally refuses to overwrite a previous run.
- The optional multilingual embedding reproduction needs `.[embedding,report]` and the pinned model revision. Its first uncached run may download model files, so it is **not** part of this offline five-minute path. The committed V1 embedding report remains inspectable offline.
