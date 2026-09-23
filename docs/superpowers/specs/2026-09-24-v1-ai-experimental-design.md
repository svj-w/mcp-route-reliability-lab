# V1 AI Experimental Dataset and Evaluation Design

## Purpose and release boundary

Produce a reproducible, portfolio-ready AI-annotated experiment from the existing
V1 candidate and the two completed AI annotation CSVs. The release is named
**V1 AI experimental**, not stable V1.0 or human-reviewed. It evaluates the known
40-tool corpus; it does not establish production routing reliability or a formal
safety guarantee. No GitHub push or external publication is part of this work.

The existing V0.1 dataset, reports, and `data/v1-candidate/` remain unchanged.
Derived data lives in `data/v1-ai-experimental/`; its reports live in
`reports/v1-ai-experimental/`. The original 180-row primary recheck CSV and
100-row secondary CSV are retained byte-for-byte under a raw-input directory in
the new dataset. The proposed rewrite and changelog CSVs are advisory only, not
annotation sources. Source file hashes and the transformation version are
recorded so the derived snapshot can be rebuilt.

## Data and annotation flow

1. Load the committed V1 candidate, 40-tool corpus, and both raw AI CSVs. Fail on
   missing or duplicate IDs, changed Tool Corpus hash, unknown Tool IDs, invalid
   label combinations, or unexpected text changes.
2. Treat the primary recheck CSV's 28 rewritten query texts as the preferred
   text candidates. Approve each against the Tool input schema and annotation
   guide before copying it to the derived query set. Rebuild the three variants
   of `v1_q_183` so they express the revised base intent. Never silently change
   V0.1 records or the original candidate.
3. For the 15 secondary-sample IDs whose raw texts differ from the preferred
   texts, create a separate, explicitly post-hoc AI revision record. Keep the
   original secondary row untouched. The revision must use the final query text
   and Tool schemas, explain its decision, and not be described as blinded or
   independent. The other 85 raw secondary rows remain identifiable as
   same-text comparisons.
4. Keep every primary/recheck/secondary disagreement visible in a decision log.
   Resolve `v01_q_039`, the eight disputed `multi_valid` cases, and any additional
   schema/intent conflicts by the annotation guide's precedence rules. Do not
   force a case type to preserve target quotas. Record the before/after text,
   labels, reason, reviewer type, and source IDs for each correction. Base labels
   not covered by either AI CSV retain their generator-provisional provenance.
5. Materialize one canonical query set and one final annotation per base query.
   Every variant inherits the final base label only after its wording is checked
   for semantic equivalence. The Tool Corpus hash remains the existing hash if
   Tool records do not change; the query-set hash is recomputed.

The raw 85 same-text comparisons may be summarized as source-comparison
diagnostics. Of those, 45 have a same-text AI primary recheck and 40 compare the
secondary AI row against a generator-provisional primary label; report those
subgroups separately. A 100-row statistic after revisions is post-hoc
concordance, not independent inter-annotator agreement. None substitutes for
the original human-review release gate.

## Split integrity and evaluation

Derive an auditable intent-family key by removing scenario-number suffixes and
grouping remaining repeated templates; manually inspect ambiguous families.
Assign all members of one family to one split. The derived release requires 100
base queries each in Development, Calibration, and Test without compromising
family isolation; fail the build if that allocation is impossible. Keep variants
with their base; if a currently selected variant base
cannot stay in Test, select a replacement Test base and regenerate that paired
variant set. Preserve 300 bases and 90 variants. Record actual language,
case-type, difficulty, and domain distributions rather than asserting the
original exact case quotas when semantic corrections change them.

The derived dataset has a dedicated experimental validation profile. It requires
40 tools, 300 bases, 90 variants, three 100-base splits, valid labels, complete
hashes, and zero cross-split intent-family collisions. It reports but does not
hide within-split repetition and strata deviations. If those requirements
cannot be met without inventing labels or changing semantics, do not publish a
passing report; document the remaining failures.

After freezing the final query set and labels, run the existing BM25 and
multilingual embedding pipelines on the same snapshot. Calibrate using only the
Calibration split; inspect Test once per frozen configuration. Regenerate
manifests, metrics, error slices, robustness figures, reliability cards, and
policies with the new query hash. Do not reuse V0.1 or candidate predictions,
thresholds, or performance claims. Report exact corpus/query hashes, sample
counts, AI-label provenance, template-family overlap audit, and empirical
uncertainty limitations.

## Components and failure handling

- A deterministic dataset builder owns raw-input validation, query/variant
  harmonization, family allocation, and derived files. It does not perform
  routing or scoring.
- A separate decision log owns label corrections and provenance. The builder
  applies those explicit decisions; it does not silently infer gold labels
  from a changed `case_type` or force two annotators to agree.
- Existing annotation import and evaluator code remain the source of schema,
  metric, and report behavior. Any new profile or helper is scoped to the AI
  experimental dataset, leaving the V1 candidate and V0.1 behavior intact.
- Validation failures name the query ID and violated invariant. The builder
  refuses to overwrite a frozen output unless an explicit rebuild workflow is
  used. Raw inputs are immutable within the derived dataset.

## Acceptance checks

- Raw copies match the user-supplied CSV hashes; all 180 and 100 source rows are
  accounted for, with 15 secondary rows flagged as post-hoc revisions.
- Every canonical base and variant has schema-valid, text-consistent labels;
  `v01_q_039`, all eight disputed `multi_valid` IDs, and all `v1_q_183`
  variants have explicit decisions.
- No normalized intent-family key crosses splits; each split has 100 bases,
  every variant shares its base split, and all 300 bases plus 90 variants load.
- Rebuilding from committed inputs yields byte-identical derived data and
  matching hashes; existing V0.1 and V1 candidate tests still pass.
- BM25 and embedding reports are generated from the frozen derived data and
  pass provenance checks. README and dataset/reliability cards say AI
  experimental, identify provisional labels, and never claim human review.
