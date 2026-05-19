# Latent Dataset And Dictionary Plan

Date: 2026-05-14

Status: Succeeded for planning; FEVER foundation implemented.

Related notes:

- [[DATASET_PROJECT_PLAN]]
- [[TRUTHFULQA_SOURCE_ASSESSMENT]]
- [[2026-05-08_setconca_vs_pointwise_baseline_results]]
- [[2026-05-08_cross_family_pair_verification]]
- [[2026-05-08_proposal_alignment_audit]]

## Goal

Document the current dataset and benchmark direction before implementation starts.

The user wants to move beyond the current AG News constrained-rewrite dataset. The new dataset should support stronger claims about SetConCA learning shared latent variables and should later support thresholded concept dictionaries and cross-family dictionary alignment between Llama and Qwen.

## Current Decision

Dataset first. Dictionary alignment second. Intervention-tested steering third.

The project should not start by chasing steering or cross-family dictionary transfer until the dataset has clean source-certified latent sets. If the latent sets are noisy, cross-model dictionary alignment will only compare noisy labels.

## Dataset Plan

The new dataset should be a multi-view latent-set dataset:

```text
same source-certified latent variable
-> question
-> claim
-> evidence sentence
-> answer statement
-> entailment / contradiction view
```

This replaces the weaker structure:

```text
same sentence
-> constrained rewrites
```

The first release should focus on entity, fact, and event/scene latents.

Build now:

- FEVER / FEVEROUS for claim-evidence fact latents.
- HotpotQA for answer/evidence multi-hop latents.
- AllNLI or SNLI/MultiNLI for event/scene and entailment latents.
- PAWS as lexical-adversarial hard negatives.

Future work:

- Quora / AskUbuntu duplicate intent graphs.
- BBQ, RealToxicityPrompts, harmful-request refusal data.
- SycophancyEval / Anthropic evals.
- Algorithmic, code, or math latent sets.

Optional only:

- TruthfulQA as an eval-only truthfulness / misconception benchmark or data-generation template.

## TruthfulQA Decision

TruthfulQA is useful, but not as the core SetConCA training dataset.

It has 817 rows and strong truth-vs-misconception contrasts, but it does not provide rich structured multi-view evidence like FEVER or HotpotQA. It should be used as:

- eval-only intervention benchmark,
- template for generating our own source-grounded misconception data,
- possible held-out generalization test for FEVER-trained truth/evidence directions.

## Validation Plan

A set is accepted only if it passes:

- shared source key,
- positive graph connectivity,
- at least two view types,
- lexical diversity,
- semantic validation,
- hard-negative separation,
- global deduplication and split decontamination,
- calibrated `latent_set_confidence`.

The confidence score should be calibrated using a small manual annotation round:

1. Sample 100 candidate sets across FEVER, HotpotQA, and NLI.
2. Have 2-3 annotators rate valid/invalid latent-set membership.
3. Fit or choose confidence weights against that gold set.
4. Freeze the scoring config for the first pilot.

## Activation-Consistency Validator

The user raised an important representational validator: if many samples from the same latent set exist, they should trigger a stable shared activation signature.

The important point is not dimension counting. Even if a set has more samples than the model hidden dimension, that does not prove a real concept. Linear dependencies are guaranteed, but many can be meaningless.

The stronger test is:

```text
Does a direction/subspace/features estimated from one split of the set
generalize to held-out members
and beat hard negatives?
```

Proposed metrics:

- split-half centroid cosine,
- PCA subspace principal-angle similarity,
- thresholded feature support overlap,
- positive-vs-hard-negative activation margin,
- stability across view types.

This should be added after the first dataset schema/builders exist.

## Thresholded Concept Dictionary

The user proposed replacing fixed TopK feature selection with thresholded variable-size dictionary entries.

This is a good idea because concepts do not all have the same sparsity. Some concepts may need 2 trusted features, some 20, and some 0.

Feature acceptance should use calibrated metrics such as:

- positive mean activation,
- positive activation frequency,
- positive-vs-hard-negative margin,
- split-half stability,
- view-type stability.

Use normalized or calibrated thresholds rather than raw `0.8` thresholds, because activation scales differ across model families, layers, and training runs.

## Cross-Family Dictionary Alignment

The user asked whether we can train SetConCA separately on Llama and Qwen, then compare the learned dictionaries.

Decision: yes. This is likely one of the strongest downstream experiments.

Procedure:

1. Train SetConCA on Llama activations.
2. Train SetConCA separately on Qwen activations.
3. Build thresholded variable-size dictionaries for each model.
4. Learn a bridge between concept spaces:
   - Procrustes,
   - ridge,
   - CCA,
   - optional optimal transport,
   - optional MLP diagnostic.
5. Evaluate held-out concepts:
   - source concept retrieves target concept,
   - mapped source signature activates target positives above hard negatives,
   - shuffled labels and random bridges fail.

This should be framed as cross-family linear recoverability of learned concept dictionaries, not proof that the models use identical representations.

## Benchmark Plan

Representation benchmark first:

- set retrieval,
- clustering,
- hard-negative separation,
- set-size scaling,
- cross-model bridge transfer,
- old paraphrase-only dataset baseline,
- pointwise TopK baseline,
- shuffled-key control,
- lexical-overlap baseline.

Intervention benchmark later:

- truthful vs misconception,
- evidence use vs unsupported assertion,
- sycophancy resistance if needed later.

Use the wording "intervention-tested steering directions" rather than "causal steering directions" until the evidence is stronger.

## Where To Start

Start with the minimum viable dataset implementation:

1. Add `src/setconca_v2/latent_set_schema.py`.
2. Add tests for the latent set row shape and validation constraints.
3. Add `configs/dataset_sources.json`.
4. Add `src/setconca_v2/dataset_registry.py`.
5. Implement the FEVER builder first.

FEVER should be first because it has the cleanest schema sanity check: claims, labels, and evidence structure. HotpotQA is richer, but FEVER is simpler for proving the pipeline works.

## Result

Planning is documented in:

- `docs/DATASET_PROJECT_PLAN.md`
- `docs/TRUTHFULQA_SOURCE_ASSESSMENT.md`
- this progress note

## Implementation Started

Implemented the first FEVER foundation:

- `src/setconca_v2/latent_set_schema.py`
- `src/setconca_v2/dataset_registry.py`
- `src/setconca_v2/fever_builder.py`
- `configs/dataset_sources.json`
- `scripts/build_latent_sets.py`
- `tests/test_latent_set_schema_and_registry.py`
- `tests/test_fever_builder.py`

The FEVER builder supports two modes:

1. Evidence-text mode: pass a wiki-page JSONL/index so claim rows can attach real evidence sentences.
2. Reference-only mode: use `--include-reference-view` for smoke tests when the wiki sentence text is unavailable.

The reference-only mode is not enough for final dataset claims because it does not provide true evidence text. It exists so the pipeline can be tested without downloading or indexing the full FEVER wiki pages.

The CLI also supports a local official FEVER wiki archive:

```powershell
python scripts\build_latent_sets.py fever `
  --out-dir data\latent_sets\fever_evidence_pilot `
  --split labelled_dev `
  --candidate-limit 1000 `
  --limit 100 `
  --wiki-zip path\to\wiki-pages.zip
```

`--wiki-zip` selectively indexes only the Wikipedia page ids required by the candidate FEVER rows. This avoids prebuilding a full 7GB expanded wiki index for small pilots, though the official zip still has to exist locally.

The FEVER Hugging Face dataset still uses a legacy loading script that current `datasets` versions no longer execute. The CLI therefore falls back to FEVER's direct JSONL URLs:

- `https://fever.ai/download/fever/train.jsonl`
- `https://fever.ai/download/fever/shared_task_dev.jsonl`

The direct FEVER JSONL uses the original nested `evidence` field. `src/setconca_v2/fever_builder.py` now normalizes both that original shape and the flattened Hugging Face loader shape.

Example local smoke command:

```powershell
python scripts\build_latent_sets.py fever `
  --input path\to\fever_claims.jsonl `
  --wiki-jsonl path\to\wiki_pages.jsonl `
  --out-dir data\latent_sets\fever_pilot
```

Verification:

```powershell
python -m pytest tests\test_dataset_download.py tests\test_set_dataset_and_activation_extraction.py tests\test_latent_set_schema_and_registry.py tests\test_fever_builder.py -q --basetemp .test_tmp\pytest
```

Result:

```text
24 passed
```

First real FEVER reference-mode smoke artifact:

```powershell
python scripts\build_latent_sets.py fever `
  --out-dir data\latent_sets\fever_reference_smoke `
  --split labelled_dev `
  --candidate-limit 100 `
  --limit 5 `
  --include-reference-view
```

Result:

```text
{"accepted": 5, "rejected": 3}
```

Artifacts:

- `data/latent_sets/fever_reference_smoke/sets.jsonl`
- `data/latent_sets/fever_reference_smoke/rejected.jsonl`
- `data/latent_sets/fever_reference_smoke/rejection_report.md`
- `data/latent_sets/fever_reference_smoke/manifest.json`

These are real FEVER claim rows, but they still use `evidence_reference` placeholder text. The next scientific step is to attach real evidence sentence text from FEVER `wiki-pages.zip`.

First real FEVER evidence-text pilot:

```powershell
python scripts\build_latent_sets.py fever `
  --out-dir data\latent_sets\fever_evidence_pilot `
  --split labelled_dev `
  --candidate-limit 1000 `
  --limit 100 `
  --wiki-zip data\raw\wiki-pages.zip
```

Result:

```text
{"accepted": 100, "rejected": 32}
```

The official `wiki-pages.zip` contains macOS resource-fork files under `__MACOSX/`; the zip reader now skips those. FEVER wiki lines also include tab-separated hyperlink/entity tails after the sentence; `parse_fever_wiki_lines` now keeps only the sentence text before the first tab.

Evidence pilot artifacts:

- `data/latent_sets/fever_evidence_pilot/sets.jsonl`
- `data/latent_sets/fever_evidence_pilot/rejected.jsonl`
- `data/latent_sets/fever_evidence_pilot/rejection_report.md`
- `data/latent_sets/fever_evidence_pilot/manifest.json`

Pilot summary command:

```powershell
python scripts\summarize_latent_dataset.py `
  --sets data\latent_sets\fever_evidence_pilot\sets.jsonl `
  --rejected data\latent_sets\fever_evidence_pilot\rejected.jsonl `
  --out-dir data\latent_sets\fever_evidence_pilot `
  --examples 12
```

Summary artifacts:

- `data/latent_sets/fever_evidence_pilot/summary.json`
- `data/latent_sets/fever_evidence_pilot/summary_report.md`

Summary result:

| Metric | Value |
| --- | ---: |
| Sets | 100 |
| Evidence present views | 100 |
| Evidence missing/reference views | 0 |
| REFUTES | 58 |
| SUPPORTS | 42 |
| Mean claim/evidence Jaccard | 0.2068 |
| Mean claim tokens | 8.02 |
| Mean evidence tokens | 25.84 |
| Placeholder hard negatives | 100 |

Interpretation:

- The FEVER claim/evidence builder works on real evidence text.
- The first 100 accepted rows are structurally valid and have moderate lexical overlap, which is useful for avoiding pure paraphrase behavior.
- The main remaining blocker is hard negatives: all 100 rows still contain placeholder hard negatives. Before scaling to thousands, implement hard-negative mining.

Verification after summarizer:

```text
29 passed
```
