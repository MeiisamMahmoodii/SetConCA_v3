# Concept Review Pages

## Goal

Make concept inspection human-readable.

The previous inspection produced CSV files, but raw CSV is hard to review manually. This step creates one Markdown page per top concept candidate, plus a review checklist where labels can be filled in.

## New Script

Added:

```text
scripts/build_concept_review_pages.py
```

The script reads:

- `concept_summary.csv`
- `concept_examples.csv`

and writes:

- `review_pages/concept_review_table.md`
- `review_pages/<rank>_<pair>_c<concept>.md`

Each concept page contains:

- manual review fields,
- source and target model metadata,
- target concept dimension,
- source concept dimension,
- inspection score,
- target top examples,
- mapped source top examples,
- a decision guide.

## Command Run

```powershell
python scripts\build_concept_review_pages.py `
  --inspection-dir results\concept_inspection_llama_qwen_e25 `
  --max-concepts 24
```

Output:

```text
Wrote 24 concept review pages to results\concept_inspection_llama_qwen_e25\review_pages
```

## Where To Review

Start here:

```text
results/concept_inspection_llama_qwen_e25/review_pages/concept_review_table.md
```

Then open each linked concept page and fill in:

```text
manual_label:
short_name:
confidence:
use_for_steering:
notes:
```

Suggested labels:

- `semantic`: clear human-interpretable concept,
- `broad_topic`: real but too broad,
- `style_artifact`: wording/source/format artifact,
- `unclear`: mixed or insufficient evidence.

## Early Example

`Concept 001` has a high bridge-inspection score, but its examples mix oil-market articles and sports articles.

That means it may be stable as a bridge dimension but not clean enough to call a semantic concept. This is exactly why the manual review gate is necessary before causal steering.

## Verification

The script compiled successfully and the test suite passed:

```text
14 passed
```
