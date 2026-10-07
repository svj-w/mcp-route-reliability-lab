# V1 AI experimental Dataset Card

This is an AI experimental, synthetic tool-routing benchmark. It is not human-reviewed, not an independent inter-annotator study, and not a production safety guarantee.

- 40 Tool Corpus records; 300 base queries; 90 selected variants.
- 180 bases have an AI primary recheck; 120 do not.
- 100 raw AI secondary rows are retained unchanged. Fifteen require separately logged post-hoc relabeling because their raw text differs from the revised primary text.
- All decisions are in `decisions.jsonl`; source hashes and provenance counts are in `source-manifest.json`. Generator-provisional labels are identified there.
- Numbered scenario templates are grouped into one split. This prevents measured cross-split template leakage under the documented normalization, but cannot prove that every semantic paraphrase is isolated.
