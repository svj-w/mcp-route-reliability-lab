# MCP Route Reliability Lab V1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CPU-reproducible, router-agnostic MCP tool-routing reliability toolkit that imports or produces Top-K predictions, calibrates correctness, selects accept/abstain policies with uncertainty estimates, diagnoses abstention reasons, and publishes an auditable bilingual benchmark.

**Architecture:** A typed Python SDK owns all domain logic. Built-in routers and external prediction importers both emit the same `RouterPrediction`; feature extraction feeds serializable calibrators, reason diagnostics, and a version-bound `PolicyArtifact`; evaluation and reporting consume immutable run artifacts. Typer commands are thin wrappers, and public CI runs only deterministic CPU fixtures.

**Tech Stack:** Python 3.11, uv, Pydantic 2, Typer, NumPy, pandas, PyArrow, scikit-learn, rank-bm25, sentence-transformers, Matplotlib, pytest, Ruff, mypy, GitHub Actions.

**Spec:** `docs/superpowers/specs/MCP-Route-Reliability-Lab-V1.1.md`

## Global Constraints

- Target product version is V1.0; specification revision is 1.1.
- Python floor is exactly 3.11; default workflows must run on CPU, with GPU used only as an optional accelerator.
- Core installation must not require Streamlit, Docker, a cloud LLM SDK, GPU, network access at test time, or an API key.
- Original code uses Apache-2.0; original queries and annotation documentation use CC BY 4.0; upstream Tool Schemas retain their source licenses.
- V1.0 does not execute Tools, generate Tool arguments, evaluate multi-step trajectories, run FastAPI, manage MCP lifecycles, ship a dashboard, add an LLM Router, perturb Tool Schemas, use conformal prediction, or provide a formal safety guarantee.
- Inference code must never receive `relevant_tool_ids`, `case_type`, `should_abstain`, or `abstain_reason`.
- Development, Calibration, and Test each contain 100 base queries; variants stay with their `base_query_id`; Test is read once after configuration freeze.
- Every fitted policy is bound to Router ID/version, model version when applicable, Tool Corpus hash, Feature Version, and Score Metadata; mismatches fail closed.
- Public CI uses only small fixtures and mocks; full embeddings and the frozen benchmark run locally and publish manifests.
- Plan for 2–3 hours per day and a total cash budget no greater than ¥1,000; prefer local/open-source components, record any API or second-annotator expense, and stop before exceeding the cap.
- Commit after every task using only the paths named by that task; never stage unrelated Desktop files.

## Review Focus

- A prediction file with the wrong Tool Corpus or query-set hash must fail before any metric is computed; Task 5 pins this behavior.
- Labels present on `QueryCase` must be impossible to pass into feature extraction or runtime policy application; Tasks 1 and 7 pin this boundary.
- Empty, singleton, tied, negative, or non-finite Top-K scores must produce deterministic validation or feature behavior without division-by-zero; Tasks 3, 5, and 7 pin these cases.
- A 5% target risk with fewer than 30 accepted Calibration samples, or with no threshold whose one-sided bound qualifies, must return an explicit unavailable conservative policy; Task 10 pins this behavior.
- Base-query variants must never cross splits or count as independent Bootstrap units; Tasks 6 and 12 pin this behavior.

---

## Planned File Structure

```text
pyproject.toml                         # package metadata, dependencies, tool configuration
uv.lock                                # reproducible dependency lock
LICENSE                                # Apache-2.0 code license
DATA_LICENSE                           # CC BY 4.0 for original dataset content
README.md                              # install, quick start, results, limitations
src/mcp_route_reliability/
├── __init__.py                        # public package version
├── models.py                          # all versioned Pydantic contracts
├── hashing.py                         # canonical JSON and SHA-256 helpers
├── validation.py                      # corpus/query/split validation
├── datasets.py                        # JSON/JSONL loaders and annotation agreement
├── routers/
│   ├── base.py                        # Router protocol
│   ├── bm25.py                        # deterministic lexical baseline
│   └── embedding.py                   # lazy, injectable multilingual baseline
├── importers/
│   └── jsonl.py                       # external prediction importer
├── confidence/
│   ├── features.py                    # inference-only feature extraction
│   ├── calibrators.py                 # serializable Isotonic/Logistic artifacts
│   ├── reasons.py                     # independent reason-hint diagnostic
│   └── policies.py                    # threshold search and PolicyArtifact creation
├── evaluation/
│   ├── ranking.py                     # Top-1, Recall@K, MRR, multi-valid hit
│   ├── calibration.py                 # Brier, ECE, reliability bins, error AUROC
│   ├── selective.py                   # risk, coverage, AURC, Wilson bound
│   ├── robustness.py                  # paired shifts and decision flips
│   └── bootstrap.py                   # grouped confidence intervals
├── artifacts.py                       # run directory read/write and manifests
├── reporting.py                       # Markdown card, CSV, metrics, PNG figures
├── runtime.py                         # fail-closed `apply_policy`
└── cli.py                             # thin Typer commands
data/
├── tools.json                         # 40–60 normalized public Tool Schemas
├── queries.jsonl                      # 300 base + 90 variant queries
├── sources.csv                        # provenance and licenses
├── annotations/                       # independent labels and adjudication
└── DATASET_CARD.md                    # scope, composition, limitations
configs/
├── bm25.yaml                          # lexical baseline run config
└── embedding.yaml                     # multilingual embedding run config
tests/
├── fixtures/                          # 10-Tool/20-query CPU-only tiny corpus
├── unit/                              # focused module tests
├── integration/                       # end-to-end and CLI tests
└── golden/                            # frozen expected metrics/cards
.github/workflows/ci.yml               # lint, typecheck, tests, tiny benchmark
docs/
├── annotation-guide.md                # operational label rules
└── reliability-card-example.md        # published example output
```


## Pre-Task Learning Checkpoint: Understand the three-tool agent loop

This checkpoint is intentionally outside the package's core code. Use throwaway code that is not committed; commit only the learning note.

**File:**

- Create: `docs/learning-notes.md`

- [ ] Build a local calculator, weather stub, and time tool with explicit JSON Schemas. Make one low-cost model call or use an already available model endpoint to produce a tool name and arguments; keep credentials in environment variables and never write them to disk.
- [ ] Execute the chosen function, return the tool result to the model, and capture one success plus one wrong-tool example. If no API is available, replay a saved, hand-authored model response and label it as a protocol simulation.
- [ ] Write `docs/learning-notes.md` explaining the two boundaries `Query → Tool Name + Arguments` and `Tool Selection ≠ Tool Execution`, plus JSON/HTTP/function-calling observations and the failure example.
- [ ] Confirm `git grep -n -I -E "api[_-]?key|secret|token" -- docs/learning-notes.md` finds no credential, then commit only the note with `git commit -m "docs: record three-tool agent learning"`.

## Task 1: Establish the package and versioned domain contracts

**Files:**

- Create: `pyproject.toml`
- Create: `src/mcp_route_reliability/__init__.py`
- Create: `src/mcp_route_reliability/models.py`
- Create: `tests/unit/test_models.py`
- Create: `LICENSE`
- Create: `DATA_LICENSE`

**Interfaces:**

```python
class ToolRecord(StrictModel): ...
class InferenceQuery(StrictModel): ...
class QueryCase(StrictModel):
    def to_inference_query(self) -> InferenceQuery: ...
class ScoreMetadata(StrictModel): ...
class RankedCandidate(StrictModel): ...
class RouterPrediction(StrictModel): ...
class ConfidenceFeatures(StrictModel): ...
class ReliabilityDecision(StrictModel): ...
class PolicyArtifact(StrictModel): ...
```

- [ ] Create `pyproject.toml` for Python 3.11–3.12 with runtime dependencies `numpy`, `pandas`, `pyarrow`, `pydantic>=2`, `rank-bm25`, `scikit-learn`, and `typer`; add optional groups `embedding` (`sentence-transformers`), `report` (`matplotlib`), and `dev` (`mypy`, `pytest`, `ruff`). Register the `mrrl` console script at `mcp_route_reliability.cli:app` and configure Ruff, mypy strict mode, and pytest.
- [ ] Write `tests/unit/test_models.py` first. Assert that `QueryCase.to_inference_query()` excludes every gold-label field, non-finite numeric candidate scores raise `ValidationError`, duplicate candidate tool IDs raise `ValidationError`, distance metadata with `higher_is_better=True` raises `ValidationError`, rank-only candidates require `score=None`, and empty candidates require a nonempty router error.
- [ ] Run `python -m pytest tests/unit/test_models.py -q`; verify collection fails because the package and models do not exist.
- [ ] Implement frozen Pydantic models with `extra="forbid"`; require schema `1.0` for `ToolRecord`/`QueryCase`, schema `1.1` for `RouterPrediction`, and explicit versions for features, decisions, policies, and annotation records. Include Tool source/license/domain fields, Query difficulty, and prediction latency/error fields. Use literal types for split, case type, decision, abstention reason, reason hint, score type (`similarity`, `logit`, `distance`, `rank_only`), and score direction. Make `RouterPrediction` validate unique tool IDs and contiguous ranks starting at 1.
- [ ] Add Apache-2.0 text to `LICENSE` and a concise CC BY 4.0 notice to `DATA_LICENSE`, explicitly limiting the latter to original tool descriptions, queries, and annotations while retaining upstream schema licenses.
- [ ] Run `python -m pytest tests/unit/test_models.py -q`; verify all model-contract tests pass.
- [ ] Run `python -m ruff check src tests` and `python -m mypy src`; resolve every reported issue.
- [ ] Commit exactly these files with `git commit -m "build: establish package contracts"`.

## Task 2: Canonical hashing, loading, and dataset validation

**Files:**

- Create: `src/mcp_route_reliability/hashing.py`
- Create: `src/mcp_route_reliability/validation.py`
- Create: `src/mcp_route_reliability/datasets.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/tools.jsonl`
- Create: `tests/fixtures/queries.jsonl`
- Create: `tests/unit/test_hashing.py`
- Create: `tests/unit/test_validation.py`

**Interfaces:**

```python
def canonical_sha256(value: object) -> str: ...
def load_tools(path: Path) -> list[ToolRecord]: ...
def load_queries(path: Path) -> list[QueryCase]: ...
def validate_dataset(tools: Sequence[ToolRecord], queries: Sequence[QueryCase], profile: ValidationProfile) -> DatasetSummary: ...
```

- [ ] Add fixture factories in `tests/conftest.py` that deterministically generate 10 synthetic tools and 20 query records with split counts 8 development, 6 calibration, and 6 test; write equivalent checked-in JSONL fixtures without copied third-party schemas.
- [ ] Write hashing tests proving dictionary key order does not affect the SHA-256 value and list order does. Write validation tests for duplicate IDs/text, unknown relevant tool IDs, non-empty labels on `no_match`, inconsistent abstain fields, and two records sharing a `base_query_id` across splits.
- [ ] Run `python -m pytest tests/unit/test_hashing.py tests/unit/test_validation.py -q`; verify imports fail.
- [ ] Implement canonical UTF-8 JSON serialization with sorted keys and compact separators. Implement line-numbered JSONL loading that reports the path, line, and Pydantic validation error.
- [ ] Implement `DatasetSummary` counts by split, language, and case type. Reject duplicate tool/query IDs, normalized duplicate query text, unknown relevant tools, invalid abstention semantics, and cross-split base-query groups. Define a fixture profile for 8/6/6, a V0.1 profile for 40/30/30, and a production profile enforcing 100/100/100 base queries, 90 variants, 50/50 language balance, and case totals 100/60/20/60/60.
- [ ] Run the focused tests, then `python -m pytest -q`, Ruff, and mypy; verify all pass.
- [ ] Commit exactly these files with `git commit -m "feat: validate versioned datasets"`.

## Task 3: Router protocol and deterministic BM25 baseline

**Files:**

- Create: `src/mcp_route_reliability/routers/__init__.py`
- Create: `src/mcp_route_reliability/routers/base.py`
- Create: `src/mcp_route_reliability/routers/bm25.py`
- Create: `tests/unit/test_bm25_router.py`

**Interfaces:**

```python
class Router(Protocol):
    def rank(self, query: InferenceQuery, tools: Sequence[ToolRecord], top_k: int, query_set_hash: str) -> RouterPrediction: ...

def tokenize(text: str) -> list[str]: ...
class BM25Router: ...
```

- [ ] Write tests for mixed Chinese/English tokenization, an empty corpus, invalid `top_k`, unique contiguous ranks, negative BM25 scores, and deterministic `tool_id` ordering when scores tie.
- [ ] Run `python -m pytest tests/unit/test_bm25_router.py -q`; verify the router imports fail.
- [ ] Implement a tokenizer using lower-cased Unicode word tokens plus CJK unigrams and adjacent bigrams. Index tool name, description, and parameter names; compute BM25 scores and sort by descending score then ascending `tool_id`.
- [ ] Populate router version, corpus hash, query-set hash, and `ScoreMetadata(score_type="similarity", higher_is_better=True, comparable_across_queries=False)` in every prediction.
- [ ] Run the focused tests and the full quality suite; verify deterministic output on two repeated runs.
- [ ] Commit exactly these files with `git commit -m "feat: add deterministic BM25 router"`.

## Task 4: Injectable multilingual embedding baseline

**Files:**

- Create: `src/mcp_route_reliability/routers/embedding.py`
- Create: `configs/embedding.yaml`
- Create: `tests/unit/test_embedding_router.py`

**Interfaces:**

```python
class EmbeddingProvider(Protocol):
    @property
    def version(self) -> str: ...
    def encode(self, texts: Sequence[str]) -> NDArray[np.float64]: ...

class SentenceTransformerProvider: ...
class EmbeddingRouter: ...
```

- [ ] Write tests using a fixed fake provider; cover cosine ordering, zero vectors, tied similarities, dimension mismatch, and prove tests make no network or model-download call.
- [ ] Run `python -m pytest tests/unit/test_embedding_router.py -q`; verify imports fail.
- [ ] Implement normalized cosine similarity with zero-vector similarity fixed at `0.0`, validation of two-dimensional finite embeddings, and the same stable tie-break as BM25.
- [ ] Implement `SentenceTransformerProvider` with a lazy optional import, explicit CPU device, and version string containing package/model/revision details. Set the default config to `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and `top_k: 5`.
- [ ] Run focused tests and the full quality suite. Do not download the default model in CI.
- [ ] Commit exactly these files with `git commit -m "feat: add multilingual embedding router"`.
## Task 5: Import external predictions with provenance gates

**Files:**

- Create: `src/mcp_route_reliability/importers/__init__.py`
- Create: `src/mcp_route_reliability/importers/jsonl.py`
- Create: `tests/fixtures/predictions.jsonl`
- Create: `tests/unit/test_prediction_importer.py`

**Interfaces:**

```python
class PredictionImportError(ValueError): ...
def import_predictions(
    path: Path,
    *,
    expected_tool_corpus_hash: str,
    expected_query_set_hash: str,
    expected_query_ids: Collection[str],
    expected_tool_ids: Collection[str],
) -> list[RouterPrediction]: ...
```

- [ ] Write tests first for corpus-hash mismatch, query-set-hash mismatch, missing/duplicate/extra query IDs, an unknown candidate Tool ID, mixed router versions, inconsistent score metadata, non-finite scores, empty candidates without an error, an error record with empty candidates, singleton candidates, tied scores, negative scores, and rank-only input without numeric scores. For the two hash mismatches, patch a metric function with a spy and assert it is never called.
- [ ] Run `python -m pytest tests/unit/test_prediction_importer.py -q`; verify imports fail.
- [ ] Implement streaming JSONL parsing into `RouterPrediction`. Validate file-level invariants before returning any record: exact query-ID set, one prediction per query, a single router/model version and score contract, matching corpus/query hashes, candidate Tool IDs contained in the expected corpus, at least one candidate unless a nonempty router error is recorded, finite numeric values, and ordering consistent with `higher_is_better` or rank-only semantics.
- [ ] Preserve raw numeric scores and metadata without converting distances to similarities. For `rank_only`, require every candidate `score=None`; never synthesize a numeric score or probability, and mark calibration/Risk-Coverage unavailable while retaining ranking metrics.
- [ ] Run focused tests and the full quality suite; verify failures name the offending file, query ID, and invariant.
- [ ] Commit exactly these files with `git commit -m "feat: import version-bound predictions"`.

## Task 6: Group-safe splitting and human annotation agreement

**Files:**

- Modify: `src/mcp_route_reliability/datasets.py`
- Modify: `src/mcp_route_reliability/models.py`
- Create: `tests/unit/test_splits.py`
- Create: `tests/unit/test_annotation_agreement.py`
- Create: `docs/annotation-guide.md`

**Interfaces:**

```python
def assign_grouped_splits(cases: Sequence[QueryCase], seed: int, targets: Mapping[Split, int]) -> dict[str, Split]: ...
def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float: ...
def mean_set_jaccard(first: Sequence[frozenset[str]], second: Sequence[frozenset[str]]) -> float: ...
def build_disagreement_records(primary: Sequence[Annotation], secondary: Sequence[Annotation], adjudicated: Sequence[Annotation]) -> list[DisagreementRecord]: ...
```

- [ ] Write tests showing all variants of a `base_query_id` remain together, identical seed/input yields identical assignments, impossible target counts fail clearly, Kappa returns `1.0` for perfect non-degenerate agreement and `not_available:degenerate_labels` when expected-agreement denominator is zero, Jaccard defines two empty sets as `1.0`, and raw conflicting labels survive adjudication export.
- [ ] Run the two focused test files; verify the new interfaces are missing.
- [ ] Implement deterministic group-aware stratification over case type, language, difficulty, and Tool domain, assigning only base-query groups and checking exact requested base counts.
- [ ] Implement independent-annotation records, Cohen's Kappa for route/abstain labels, per-sample and mean set Jaccard for relevant-tool sets, and disagreement rows containing both raw annotations plus the adjudicated result.
- [ ] Write the annotation guide with operational definitions, positive/negative examples for all five case types, multi-valid handling, abstention-reason precedence, independence requirements, and the 100-sample second-annotator protocol.
- [ ] Run focused tests and the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: add grouped splits and agreement analysis"`.

## Task 7: Extract leakage-safe confidence features

**Files:**

- Create: `src/mcp_route_reliability/confidence/__init__.py`
- Create: `src/mcp_route_reliability/confidence/features.py`
- Create: `tests/unit/test_confidence_features.py`

**Interfaces:**

```python
class FeatureNormalizer:
    @classmethod
    def fit(cls, calibration_predictions: Sequence[RouterPrediction]) -> FeatureNormalizer: ...
    def standardize_top1(self, raw_score: float) -> float: ...
    def standardize_dispersion(self, raw_dispersion: float) -> float: ...

def extract_confidence_features(
    query: InferenceQuery,
    prediction: RouterPrediction,
    tools_by_id: Mapping[str, ToolRecord],
    normalizer: FeatureNormalizer,
    peer_prediction: RouterPrediction | None = None,
) -> ConfidenceFeatures: ...
```

- [ ] Write tests proving a `QueryCase` cannot be passed without an explicit `to_inference_query()` conversion and that feature output cannot contain label field names. Cover similarity and distance direction, singleton Top-K, tied/negative scores, all-equal scores, rank-only scores, non-finite rejection, empty input rejection, missing descriptions, Chinese lexical coverage, and router agreement with and without a peer.
- [ ] Run `python -m pytest tests/unit/test_confidence_features.py -q`; verify imports fail.
- [ ] Implement Calibration-only mean/standard-deviation fitting for both signed Top-1 and raw Top-K dispersion, using `+score` when higher is better and `-score` otherwise; use standardized value `0.0` when the corresponding Calibration standard deviation is zero. Persist both transforms and the feature version in the Manifest.
- [ ] Compute signed top-1/top-2 margin, stable softmax entropy, raw and Calibration-standardized score dispersion, query length/language, selected-tool description length/missing flag, lexical coverage, and Top-1 peer agreement. Define singleton raw margin, entropy, and dispersion as `0.0`; rank-only inputs expose rank/metadata features but set score-derived optional fields to `None`.
- [ ] Do not compute score ratios and do not accept relevant tools, case type, abstention labels, or reasons in any feature API.
- [ ] Run focused tests, full tests, Ruff, and mypy.
- [ ] Commit exactly these files with `git commit -m "feat: extract leakage-safe confidence features"`.

## Task 8: Compute ranking and error-label metrics

**Files:**

- Create: `src/mcp_route_reliability/evaluation/__init__.py`
- Create: `src/mcp_route_reliability/evaluation/ranking.py`
- Create: `tests/unit/test_ranking_metrics.py`

**Interfaces:**

```python
def is_routing_error(case: QueryCase, prediction: RouterPrediction) -> bool: ...
def ranking_metrics(cases: Sequence[QueryCase], predictions: Sequence[RouterPrediction], ks: Sequence[int] = (1, 3, 5)) -> RankingMetrics: ...
def per_case_errors(cases: Sequence[QueryCase], predictions: Sequence[RouterPrediction]) -> list[CaseError]: ...
```

- [ ] Write hand-calculated tests covering clear, hard-negative, multi-valid, clarify, and no-match cases; include a missing prediction, unknown candidate tool, multi-valid hit at rank 2, and deterministic metric keys for K values larger than available candidates.
- [ ] Run `python -m pytest tests/unit/test_ranking_metrics.py -q`; verify imports fail.
- [ ] Define a router error record or empty candidate list as a routing error; otherwise define routing error as accepting an incorrect Top-1 on routeable cases or accepting any tool on gold-abstain cases; keep policy decisions separate so raw ranking metrics report Top-1 accuracy, Recall@K, MRR, and multi-valid hit rate without pretending the router already abstained.
- [ ] Join cases and predictions by query ID with exact-set validation, calculate micro metrics and slices by language/case type, and emit per-case correctness records for later calibration and error analysis.
- [ ] Run focused tests and the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: add ranking evaluation"`.
## Task 9: Fit serializable calibrators and calibration metrics

**Files:**

- Create: `src/mcp_route_reliability/confidence/calibrators.py`
- Create: `src/mcp_route_reliability/evaluation/calibration.py`
- Create: `src/mcp_route_reliability/evaluation/selective.py`
- Create: `tests/unit/test_calibrators.py`
- Create: `tests/unit/test_calibration_metrics.py`
- Create: `tests/unit/test_selective_metrics.py`

**Interfaces:**

```python
class ProbabilityCalibrator(Protocol):
    def predict_correctness(self, rows: Sequence[ConfidenceFeatures]) -> NDArray[np.float64]: ...
    def to_payload(self) -> CalibratorPayload: ...

class IsotonicCalibrator: ...
class LogisticConfidenceModel: ...
def brier_score(y_correct: NDArray[np.int_], p_correct: NDArray[np.float64]) -> float: ...
def equal_frequency_ece(y_correct: NDArray[np.int_], p_correct: NDArray[np.float64], bins: int = 5) -> ECEResult: ...
def error_auroc(y_error: NDArray[np.int_], p_error: NDArray[np.float64]) -> float | None: ...
def risk_coverage_curve(y_error: NDArray[np.int_], p_correct: NDArray[np.float64]) -> RiskCoverageResult: ...
```

- [ ] Write calibrator tests for isotonic monotonicity, L2 logistic fitting with exactly 4–6 named features, missing optional score features, deterministic serialization round-trips, and a single-class calibration target and too few unique Isotonic scores, both producing explicit unavailable artifacts. Write metric tests from small arrays with hand-computed Brier score, five equal-frequency bins including boundaries/counts, error-positive AUROC, risk/coverage points, and trapezoidal AURC.
- [ ] Run the three focused test files; verify imports fail.
- [ ] Implement isotonic calibration over one configured scalar feature and logistic calibration over an explicit ordered 4–6-feature list selected and frozen using Development only. Fit every numeric parameter on Calibration only; median-impute optional values from Calibration and persist medians/scaler/coefficient/intercept values as JSON numbers. Return a serialized `not_available` result for rank-only input, a single-class target, fewer than two unique Isotonic scores, or another documented fit precondition failure; record valid sample and unique-score counts.
- [ ] Implement Brier as mean squared error on correctness probability. Implement exactly five near-equal-count ECE bins using stable confidence ordering; serialize each bin's closed interval, count, mean confidence, accuracy, and absolute gap. Return `None` with a reason when AUROC has one class.
- [ ] Implement risk/coverage by descending correctness confidence, including `(coverage=0, risk=0)` and every accepted prefix; integrate AURC over coverage with the trapezoidal rule. For rank-only input, serialize calibration and Risk-Coverage as `not_available` rather than treating rank as confidence.
- [ ] Run focused tests and the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: calibrate confidence and selective metrics"`.

## Task 10: Select empirical and conservative risk policies

**Files:**

- Create: `src/mcp_route_reliability/confidence/policies.py`
- Create: `tests/unit/test_policy_selection.py`

**Interfaces:**

```python
def wilson_upper_bound(errors: int, accepted: int, confidence: float = 0.95) -> float: ...
def select_policy(
    calibration_rows: Sequence[CalibrationRow],
    *,
    method: Literal["max_score", "margin", "isotonic", "logistic"],
    target_risk: float = 0.05,
    minimum_accepted: int = 30,
) -> PolicySelectionResult: ...
def build_policy_artifact(selection: PolicySelectionResult, provenance: PolicyProvenance) -> PolicyArtifact: ...
```

- [ ] Write tests for the one-sided 95% Wilson upper bound against fixed reference values; threshold ties; exactly 29 versus 30 accepted samples; empirical risk exactly 5%; no empirical candidate; no conservative candidate; and deterministic preference for maximum coverage followed by the highest threshold.
- [ ] Write an artifact test asserting router ID/version, model version, corpus hash, calibration dataset hash, feature version, full score metadata, threshold, risk settings, sample sizes, source commit, calibrator payload, and availability status are serialized.
- [ ] Run `python -m pytest tests/unit/test_policy_selection.py -q`; verify imports fail.
- [ ] Enumerate thresholds from unique Calibration scores plus an above-maximum sentinel. For each threshold calculate accepted count, coverage, empirical risk, and the one-sided Wilson upper bound with `z = NormalDist().inv_cdf(0.95)`, `center = (p + z²/(2n))/(1 + z²/n)`, and `radius = z*sqrt(p(1-p)/n + z²/(4n²))/(1 + z²/n)`. Select the maximum-coverage qualifying threshold, breaking ties by higher threshold.
- [ ] Produce separate empirical (`risk <= 0.05`) and conservative (`Wilson upper <= 0.05`) selections. If accepted count is below 30 or none qualifies, return `available=false` with a machine-readable reason; never silently relax the target.
- [ ] Support raw signed max-score and margin methods plus calibrated isotonic/logistic correctness probabilities while recording each method's exact feature contract.
- [ ] Run focused tests and the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: select risk-controlled policies"`.

## Task 11: Add independent reason hints and fail-closed runtime application

**Files:**

- Create: `src/mcp_route_reliability/confidence/reasons.py`
- Create: `src/mcp_route_reliability/runtime.py`
- Create: `tests/unit/test_reason_hints.py`
- Create: `tests/unit/test_runtime_policy.py`

**Interfaces:**

```python
def infer_reason_hint(query: InferenceQuery, prediction: RouterPrediction, features: ConfidenceFeatures) -> ReasonHint: ...
def evaluate_reason_hints(gold: Sequence[AbstainReason], predicted: Sequence[ReasonHint]) -> ReasonMetrics: ...
def apply_policy(
    policy: PolicyArtifact,
    query: InferenceQuery,
    prediction: RouterPrediction,
    features: ConfidenceFeatures,
) -> ReliabilityDecision: ...
```

- [ ] Write reason tests for `no_match`, `ambiguous`, `insufficient_information`, and fallback `uncertain`, and assert changing the hint cannot change the accept/abstain result. Add hand-computed tests for per-class precision/recall/F1/support, macro F1, confusion matrix, and `uncertain` rate.
- [ ] Write runtime tests for router-ID, router-version, model-version, corpus-hash, feature-version, and score-metadata mismatches; unavailable policy; a prediction with `error` or no candidates; missing calibrator feature; non-finite probability; exact-threshold acceptance; and a calibration-dataset hash that differs from the runtime query set but remains valid provenance.
- [ ] Run both focused test files; verify imports fail.
- [ ] Implement a versioned `ReasonHintConfig` fitted on Calibration only. Apply this precedence: `insufficient_information` when query token count and lexical coverage are both below configured bounds; `no_match` when signed standardized Top-1 is below its bound; `ambiguous` when margin is below or entropy is above its bound; otherwise `uncertain`. Use only inference text, predictions, and confidence features, persist every bound, and compute the hint only after the main decision.
- [ ] Implement policy payload reconstruction and score calculation. Before scoring, compare Router ID/version, model version, corpus hash, feature version, and complete score metadata and raise a typed `PolicyBindingError` on any mismatch. Do not compare the calibration dataset hash to runtime query IDs.
- [ ] Define acceptance as `confidence >= threshold`; validate confidence is finite and in `[0, 1]` for calibrated policies. Return selected tool ID only on accept and include reason hint only on abstain.
- [ ] Run focused tests and the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: apply policies with fail-closed bindings"`.

## Task 12: Quantify paired robustness and grouped uncertainty

**Files:**

- Create: `src/mcp_route_reliability/evaluation/robustness.py`
- Create: `src/mcp_route_reliability/evaluation/bootstrap.py`
- Create: `tests/unit/test_robustness.py`
- Create: `tests/unit/test_grouped_bootstrap.py`

**Interfaces:**

```python
def paired_robustness(base: Sequence[EvaluatedCase], variants: Sequence[EvaluatedCase]) -> RobustnessResult: ...
def grouped_bootstrap(
    rows: Sequence[MetricRow],
    metric: Callable[[Sequence[MetricRow]], float],
    *,
    group_key: str = "base_query_id",
    replicates: int = 1000,
    seed: int = 20260921,
) -> BootstrapInterval: ...
```

- [ ] Write paired tests for paraphrase, cross-language, and noise variants, measuring Top-1 flip, accept/abstain flip, confidence shift, accuracy shift, and missing/duplicate pair rejection.
- [ ] Write bootstrap tests proving every sampled replicate contains complete base-query groups, variants never become independent units, fewer than 1000 replicates is rejected for production mode, a fixed seed is reproducible, and percentile bounds match a saved small example.
- [ ] Run both focused test files; verify imports fail.
- [ ] Join every variant to exactly one base record by `base_query_id`, aggregate overall and by perturbation type, and preserve paired deltas for CSV export.
- [ ] Resample unique base-query IDs with replacement, include every row from each sampled group, calculate at least 1000 replicates, and return estimate plus 2.5/97.5 percentile bounds and seed/replicate metadata.
- [ ] Run focused tests and the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: add paired robustness and grouped bootstrap"`.
## Task 13: Write complete run artifacts and the Reliability Card

**Files:**

- Create: `src/mcp_route_reliability/artifacts.py`
- Create: `src/mcp_route_reliability/reporting.py`
- Create: `tests/unit/test_artifacts.py`
- Create: `tests/integration/test_reporting.py`
- Create: `tests/golden/reliability-card.md`
- Create: `docs/reliability-card-example.md`

**Interfaces:**

```python
class RunWriter:
    def __enter__(self) -> RunWriter: ...
    def write_manifest(self, manifest: RunManifest) -> None: ...
    def finalize(self) -> Path: ...

def build_reliability_card(bundle: EvaluationBundle) -> str: ...
def render_figures(bundle: EvaluationBundle, output_dir: Path) -> list[Path]: ...
```

- [ ] Write artifact tests asserting an incomplete temporary run is never exposed as final and a completed run contains `manifest.json`, `predictions.jsonl`, `features.parquet`, `metrics.json`, `errors.csv`, `policy.json`, `policy-empirical.json`, `policy-conservative.json`, `reliability-card.md`, and PNG figures.
- [ ] Write a golden report test covering dataset/router hashes, environment and source commit, method comparison, Brier/ECE/AUROC, risk/coverage/AURC, empirical and conservative policy availability, grouped confidence intervals, robustness slices, limitations, and exact commands used.
- [ ] Run the focused tests; verify imports fail.
- [ ] Implement a temporary run directory that writes UTF-8/JSON deterministically, validates required files and manifest checksums, then atomically renames to the final timestamped run directory. Write `policy.json` as the exact configured empirical or conservative selection—even when unavailable—and never fall back silently.
- [ ] Implement errors CSV with query text, gold set, Top-K, decision, confidence, reason hint, and slice fields; Parquet feature output must exclude raw gold-label columns but may include a separately joined evaluation target named `is_correct` only in the evaluation artifact.
- [ ] Render reliability, risk/coverage, method-comparison, and robustness PNGs using a noninteractive Matplotlib backend. Generate the Markdown card and copy a sanitized example to `docs/reliability-card-example.md`.
- [ ] Run `python -m pytest tests/unit/test_artifacts.py tests/integration/test_reporting.py -q`, update the golden file only after inspecting the semantic diff, then run the full quality suite.
- [ ] Commit exactly these files with `git commit -m "feat: publish reproducible reliability artifacts"`.

## Task 14: Wire the CLI and a CPU-only end-to-end workflow

**Files:**

- Create: `src/mcp_route_reliability/cli.py`
- Create: `configs/bm25.yaml`
- Create: `tests/integration/test_cli.py`
- Create: `tests/integration/test_end_to_end.py`
- Create: `uv.lock`

**Interfaces:**

```text
mrrl validate --tools PATH --queries PATH --profile fixture|v0.1|production
mrrl route --config PATH --tools PATH --queries PATH --output PATH
mrrl import-predictions --input PATH --tools PATH --queries PATH --output PATH
mrrl evaluate --predictions PATH --tools PATH --queries PATH --run-dir PATH
mrrl fit-policy --run-dir PATH --method max-score|margin|isotonic|logistic
mrrl apply-policy --policy PATH --prediction PATH --query PATH
mrrl report --run-dir PATH
mrrl benchmark --config PATH --tools PATH --queries PATH --run-dir PATH
```

- [ ] Write Typer CLI tests for command help, missing files, invalid config, validation failure, nonzero exit codes, and JSON error output. Write an end-to-end test that runs the 10-tool/20-query BM25 fixture through validation, routing, feature extraction, policy fitting, evaluation, reporting, and runtime application.
- [ ] In the end-to-end test, assert Calibration alone fits normalizers/calibrators/thresholds, Test is evaluated once after a recorded `configuration_frozen=true` transition, and every expected artifact and binding hash is present.
- [ ] Run the two integration test files; verify the CLI entry point is missing.
- [ ] Implement thin commands that call the tested library APIs, read explicit paths/configuration, write no hidden global state, and emit concise human output plus stable machine-readable JSON with `--json`.
- [ ] Add BM25 defaults including seed, Top-K, split names, target risk `0.05`, production minimum accepted `30`, bootstrap replicates `1000`, and report settings. Integration tests use an explicit fixture-only minimum of `3` and must label its policy non-production. Generate and check in `uv.lock` under Python 3.11.
- [ ] Run `python -m pytest tests/integration -q`, `python -m pytest -q`, Ruff, and mypy; then run the documented fixture command twice and compare artifact hashes excluding timestamp/run-ID fields.
- [ ] Commit exactly these files with `git commit -m "feat: add end-to-end reliability CLI"`.

## Task 15: Publish the four-week V0.1 portfolio milestone

**Files:**

- Create: `data/tools.json`
- Create: `data/queries.jsonl`
- Create: `data/sources.csv`
- Create: `data/annotations/primary.jsonl`
- Create: `data/DATASET_CARD.md`
- Create: `README.md`
- Modify: `configs/bm25.yaml`
- Modify: `configs/embedding.yaml`

**Acceptance contract:**

```text
20 normalized tools
100 base queries: 50 Chinese + 50 English
Four V0.1 types only: Clear, Hard Negative, Clarify, No-match
Development / Calibration / Test = 40 / 30 / 30 base queries
All source URLs, licenses, access dates, and transformations recorded
BM25 plus multilingual embedding baseline cards generated
```

- [ ] Select 20 public Tool Schemas across at least five confusable domains. Record source URL, upstream license, access date, original identifier, transformation notes, and redistribution status before copying any schema; exclude sources without clear redistribution permission.
- [ ] Normalize schema names/descriptions/parameters into `ToolRecord`, preserving the original license per record. Run corpus validation and manually inspect near-duplicate descriptions so hard negatives are meaningful rather than accidental duplicates.
- [ ] Author 100 original bilingual base queries using the annotation guide and the four V0.1 case types; Multi-valid starts in V1.0. Keep 50/50 language balance, attach source-independent IDs, and assign grouped 40/30/30 splits with the fixed seed.
- [ ] Have the primary annotator label relevant-tool sets, should-abstain, abstention reasons, label rationale, candidate-conflict notes, and Tool Corpus hash; run validation and export a dataset summary table in the Dataset Card.
- [ ] Freeze the V0.1 configuration, run both baselines locally on CPU, compare exploratory Max-score and Margin sweeps, and evaluate Test once. Report Brier, ECE, Risk-Coverage, and failure examples, but mark `policy.json`, empirical, and conservative policies `available=false` with reason `exploratory_v0_1`; do not advertise a stable 5% threshold.
- [ ] Write README sections for the problem, why abstention matters, architecture, installation, five-minute fixture demo, V0.1 results, artifact layout, external-router import, limitations, licenses, and roadmap. Include measured commands/runtime/hardware rather than projected performance claims.
- [ ] Run all tests plus `mrrl validate --profile v0.1` against the checked-in V0.1 validation profile; inspect both generated Reliability Cards and all error slices.
- [ ] Commit exactly these files and sanitized V0.1 report artifacts with `git commit -m "data: publish V0.1 reliability benchmark"`.

## Task 16: Expand, independently review, and release V1.0

**Files:**

- Modify: `data/tools.json`
- Modify: `data/queries.jsonl`
- Modify: `data/sources.csv`
- Modify: `data/annotations/primary.jsonl`
- Create: `data/annotations/secondary.jsonl`
- Create: `data/annotations/adjudicated.jsonl`
- Create: `data/annotations/disagreements.csv`
- Modify: `data/DATASET_CARD.md`
- Modify: `README.md`
- Create: `.github/workflows/ci.yml`
- Create: `docs/architecture.md`
- Create: `docs/demo-script.md`
- Create: `docs/resume-bullets.md`

**Acceptance contract:**

```text
40–60 tools
300 base queries: 100 clear, 60 hard-negative, 20 multi-valid, 60 clarify, 60 no-match
150 Chinese + 150 English base queries
Development / Calibration / Test = 100 / 100 / 100 base queries
90 variants from 30 Test bases: paraphrase + cross-language + noise
100 independently double-annotated samples with raw disagreements retained
```

- [ ] Expand the corpus to 40–60 licensed tools while preserving provenance. Expand to exactly 300 base queries with the agreed case and language totals, then create exactly three paired variants for each of 30 selected Test bases.
- [ ] Run production validation before routing. Verify no base-query group crosses splits, no normalized query duplicate exists, and the frozen Test set has not appeared in configuration or feature-selection inputs.
- [ ] Give a blinded, stratified 100-sample subset to a second human annotator. Store independent labels before adjudication, calculate route/abstain Cohen's Kappa and relevant-tool-set Jaccard, retain raw disagreement rows, and document adjudication changes. Independently re-check every Clarify, No-match, and Hard Negative record, including rationale, conflict notes, and corpus hash.
- [ ] Freeze corpus/query/config/source-commit hashes. If a Test label is corrected afterward, increment the dataset version and rerun every final experiment. Run BM25, multilingual embedding, and the external-import fixture; fit on Calibration, read Test once, calculate 1000-or-more grouped bootstrap replicates for every primary metric, and produce paired robustness results for all 90 variants.
- [ ] Add CI for Python 3.11 and 3.12 that installs core plus dev/report extras, runs Ruff, mypy, all unit/integration tests, and the tiny fixture benchmark without network, API keys, GPU, or full model downloads.
- [ ] Update the Dataset Card and README with V1.0 composition, category proportions, annotation agreement, measured results, reason-hint per-class metrics, confidence intervals, failure cases, licensing, ethical limitations, reproducibility commands, and warnings that probabilities/thresholds shift with class priors and that the benchmark tests queries over a known Tool Corpus rather than unseen-Tool cold start, and explicit statement that the policy is an empirical benchmark artifact rather than a formal guarantee. Put a Risk-Coverage figure, `mrrl benchmark` command, compact result card, external-import example, and competitive boundary on the README first screen.
- [ ] Write an architecture note, a two-minute demo script, and honest resume bullets tied only to measured results. Perform a clean-environment release rehearsal: install from the lock file, run the fixture offline, validate every manifest checksum, reproduce the full local run from recorded commands, and compare hashes for all deterministic artifacts.
- [ ] Tag only after the entire Definition of Done passes; commit exact release files with `git commit -m "release: complete MCP Route Reliability Lab V1.0"`.

## Suggested Schedule

| Timebox | Tasks | Deliverable |
|---|---|---|
| Week 1 | Learning checkpoint, 1–3 | Three-tool protocol understanding, contracts, validation, BM25 |
| Week 2 | 4–8 and V0.1 data start | Embedding/import, grouped data workflow, features, routing metrics |
| Week 3 | 9–14 and V0.1 data completion | Calibration, policies, runtime, robustness, artifacts, complete CLI |
| Week 4 | 15 | GitHub-ready V0.1 with 20 tools and 100 queries |
| Weeks 5–8 | 16 | Full dataset, independent review, frozen V1.0 release |

## Final Verification Gate

- [ ] Run `python -m pytest -q`, `python -m ruff check src tests`, and `python -m mypy src` in a clean Python 3.11 environment.
- [ ] Run `mrrl validate --tools data/tools.json --queries data/queries.jsonl --profile production` and verify all exact V1.0 counts.
- [ ] Run the offline tiny benchmark twice and compare deterministic artifact hashes.
- [ ] Verify a wrong Router ID/version, model version, corpus hash, feature version, or score contract fails before metrics or runtime decisions.
- [ ] Verify empirical/conservative unavailable states remain visible in JSON, Markdown, and CLI output.
- [ ] Verify `rg -n "relevant_tool_ids|case_type|should_abstain|abstain_reason" src/mcp_route_reliability/confidence src/mcp_route_reliability/runtime.py` finds no inference API access to gold labels.
- [ ] Verify every data record has provenance/license coverage and every manifest references the exact source commit. Scan tracked files for API keys, private data, and ambiguous licenses before release.
- [ ] Inspect the final Reliability Card, four PNG figure types, errors CSV, policies, agreement report, and paired robustness outputs before tagging.