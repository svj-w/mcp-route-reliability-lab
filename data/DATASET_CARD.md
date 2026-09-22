# MCP Route Reliability V0.1 Dataset Card

## Summary

V0.1 is an exploratory bilingual benchmark for MCP-style tool routing with explicit
abstention. It contains 20 normalized public Tool Schemas and 100 original base
queries. The corpus is intended to test whether a router can distinguish confusable
tools and lower confidence when a request requires clarification or has no matching
capability.

This release is suitable for development, demonstrations, and pipeline regression
tests. It is not a representative sample of production traffic and must not be used
to claim a production-safe error threshold.

## Files

| Path | Contents |
|---|---|
| `tools.json` | 20 schema `1.0` Tool records |
| `queries.jsonl` | 100 schema `1.0` labeled query cases |
| `annotations/primary.jsonl` | frozen primary annotation records |
| `sources.csv` | per-tool URL, revision, license, access date, transformation, redistribution status |

## Composition

### Split

| Split | Base queries |
|---|---:|
| Development | 40 |
| Calibration | 30 |
| Test | 30 |
| **Total** | **100** |

The split seed is `20260921`. Every V0.1 record is its own base-query group and no
group crosses a split. V0.1 contains no perturbation variants.

### Language

| Language | Count |
|---|---:|
| Chinese (`zh`) | 50 |
| English (`en`) | 50 |

### Case type

| Case type | Count | Label behavior |
|---|---:|---|
| Clear | 25 | one actionable tool |
| Hard Negative | 25 | one tool is correct despite a similar same-domain distractor |
| Clarify | 25 | capability exists, but intent or a required value is unresolved; abstain |
| No-match | 25 | the corpus lacks the requested capability; empty relevant set; abstain |

`multi_valid` is intentionally deferred to V1.0.

### Tool domains

| Domain | Tools |
|---|---:|
| Filesystem | 6 |
| Git | 5 |
| Memory / knowledge graph | 5 |
| Time | 2 |
| Web fetch | 1 |
| Sequential reasoning | 1 |
| **Total** | **20** |

Near-neighbor groups were retained deliberately: read-one/read-many,
write/line-edit, list/search, staged/unstaged diff, current-time/time-conversion, and
memory search/exact-open. Their normalized descriptions are distinct and were
manually reviewed so hard negatives are semantic contrasts rather than accidental
duplicate records.

## Provenance and transformations

All 20 tools come from the official
[`modelcontextprotocol/servers`](https://github.com/modelcontextprotocol/servers)
repository pinned to commit
`d73f99efbfd40c3aa1b61e88728b3d49fb52608f`, accessed 2026-09-22. V0.1 uses the
filesystem, git, memory, time, fetch, and sequential-thinking reference servers.

For every tool, `sources.csv` records the exact source file URL and original tool
identifier. Transformations were limited to:

- adding a domain namespace to the identifier;
- writing a concise normalized description;
- converting Zod/Pydantic input definitions to JSON Schema; and
- omitting server implementation code.

The pinned upstream repository documents a license transition: new contributions are
Apache-2.0, while earlier unrelicensed contributions remain MIT. V0.1 therefore
records `Apache-2.0 OR MIT` per transformed schema and retains attribution. Sources
without clear redistribution permission were excluded.

## Query authoring

Queries were authored for this benchmark in paired Chinese/English coverage, using
the frozen Tool Corpus rather than copying prompts from upstream. Templates were used
to balance case types and domain contrasts; scenario suffixes make each base query
text unique. The result is intentionally controlled and visibly synthetic.

Authoring followed `docs/annotation-guide.md`:

1. decide whether any corpus tool has the capability;
2. if so, decide whether required information or intent is unresolved;
3. distinguish one correct tool from same-domain hard negatives; and
4. record relevant tools, route/abstain, abstention reason, rationale, and conflict
   notes relative to the immutable Tool Corpus hash.

## Annotation

One primary annotator labeled all 100 records under annotator ID `primary-v0.1`.
Every annotation contains the query ID, relevant Tool set, `should_abstain`, reason,
rationale, conflict note, and exact Tool Corpus hash. Clarify and No-match cases must
abstain; No-match is the only case type with an empty relevant set.

There is no independent annotation or adjudication in V0.1. Agreement statistics are
therefore unavailable and must not be inferred. A blinded second annotation is a
hard requirement for V1.0.

## Validation and frozen identifiers

The checked-in data passes the V0.1 validation profile:

- 20 unique Tool IDs and 100 unique query IDs;
- no normalized duplicate query text;
- all relevant Tool IDs exist in the corpus;
- abstention semantics match the case type;
- exact 40/30/30 split, 50/50 language balance, and 25 cases per V0.1 type; and
- annotation and query records bind to the same Tool Corpus hash.

Frozen identifiers:

```text
Tool Corpus SHA-256:
sha256:a4cb1dc1d0ccfa9bb0b559dae7729c455c45afa15abe04219446e8b65a94eb50

Dataset SHA-256 used by V0.1 reports:
sha256:e1fca3ad3a871d3028c0e4989ad3c78bed515724f4f0f0f065dc8f173661c6eb
```

Reproduce structural validation with:

```powershell
mrrl validate --tools data/tools.json --queries data/queries.jsonl --profile v0.1
```

## Intended uses

- developing and testing router reliability metrics;
- comparing BM25, embedding, or externally imported routers on the same frozen data;
- exercising calibration/Test separation and fail-closed policy binding;
- inspecting bilingual and case-type failure slices; and
- teaching reproducible evaluation and artifact provenance.

## Out-of-scope uses

- certifying a router as safe for production;
- deriving a stable 5% selective-risk threshold;
- measuring general language understanding or end-to-end agent task success;
- evaluating tools absent from the known corpus; or
- representing real user or organization traffic.

## Limitations and risks

- Only 30 Test cases are frozen, so confidence intervals are wide.
- Queries are balanced and templated; natural production priors, wording, and noise
  will differ.
- Chinese and English counts are balanced, but translations are not a formally
  aligned parallel corpus and dialectal coverage was not measured.
- Tools come from one upstream repository and cover only six domains.
- V0.1 has no multi-valid labels, perturbation robustness set, or unseen-tool split.
- One author performed query creation and primary annotation, creating confirmation
  bias that independent review must address.
- Probabilities and thresholds are distribution-dependent; corpus or traffic changes
  require recalibration and a new evaluation.

## Licensing

Original queries, annotations, dataset documentation, and original normalized
descriptions are licensed CC BY 4.0 under `DATA_LICENSE`. Upstream schemas retain
their original licenses and attribution as recorded in `sources.csv`. The Python
software is separately licensed Apache-2.0 under the repository `LICENSE`.

## Maintenance

Any Tool Schema, label, or Test correction changes the dataset version and requires
regenerating all reports. V1.0 will expand the corpus, add grouped variants, preserve
raw independent annotations before adjudication, and publish agreement metrics.

