# First-Pass Concept Labels

## Goal

Create a filled starting point for concept review so the user does not have to begin with empty labels.

## Output

Added first-pass labels under:

```text
results/concept_inspection_llama_qwen_e25/review_pages
```

Files:

- `first_pass_labels.md`
- `first_pass_labels.csv`

The review index now links to both files.

## Labeling Rule

The label asks whether the target examples and mapped-source examples share the same human idea.

Labels used:

- `clean_semantic`: both tables mostly show one clear idea,
- `broad_topic`: both tables share a broad category but not a precise concept,
- `mixed`: examples combine unrelated ideas,
- `artifact`: shared pattern is style/source/format rather than meaning,
- `unclear`: not enough evidence.

## First-Pass Keep List

The first candidates worth carrying forward are:

| Rank | Short name | Steering |
| ---: | --- | --- |
| 9 | oil and market uncertainty | yes |
| 10 | oil and market uncertainty | yes |
| 19 | finance investment and market uncertainty | yes |
| 21 | damage conflict and disaster | yes/maybe |

## Caveat

These are provisional labels. They should be treated as a review aid, not as final scientific claims.
