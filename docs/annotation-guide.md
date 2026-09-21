# Annotation Guide

This guide defines labels relative to one immutable Tool Corpus hash. Annotators
must read the normalized Tool names, descriptions, and input schemas before
labeling. They label user intent—not lexical overlap—and record a short rationale.

## Required fields

Every annotation records the query ID, relevant Tool ID set, route/abstain choice,
abstention reason, rationale, candidate-conflict note, annotator ID, and exact Tool
Corpus hash. An empty relevant set is meaningful only for `no_match`.

## Case types

- `clear`: One Tool is the clearly appropriate capability. Example: “Search this
  repository for Python files containing FastAPI” maps to a code-search Tool.
- `hard_negative`: Similar Tools exist, but intent identifies a narrower correct
  Tool. Example: searching file contents must not be labeled as repository search.
- `multi_valid`: At least two Tools can satisfy the request without inventing
  missing information. Record every acceptable Tool, not merely a preferred one.
- `clarify`: The corpus contains relevant capability, but a required slot or
  disambiguating intent is missing. Use `insufficient_information` for a missing
  required value and `ambiguous` when multiple incompatible interpretations remain.
- `no_match`: No Tool in the hashed corpus can complete the request. Use an empty
  relevant set and reason `no_match`; do not infer capabilities absent from schemas.

## Decision precedence

1. If no corpus Tool has the capability, label `no_match`.
2. If capability exists but required information or intent is unresolved, label
   `clarify` with the appropriate reason.
3. If multiple Tools independently satisfy the complete request, label
   `multi_valid`.
4. If similar distractors exist but one Tool is correct, label `hard_negative`.
5. Otherwise label `clear`.

## Independent review protocol

The second annotator receives a blinded, stratified 100-query sample and this guide,
but never the primary labels. The sample covers language, case type, difficulty, and
Tool domain. Both raw files are frozen before adjudication. The project reports
Cohen's Kappa for route/abstain and per-sample plus mean Jaccard for relevant Tool
sets. Constant-label Kappa is reported as unavailable, not as perfect agreement.

All Clarify, No-match, and Hard Negative records receive a separate second self-check.
Disagreements preserve both original annotations; an adjudicator writes a third
record rather than overwriting either source. Any correction after Test freeze bumps
the dataset version and requires rerunning the final experiments.
