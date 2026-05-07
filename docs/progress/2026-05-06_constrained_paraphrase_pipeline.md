# Task: Constrained Paraphrase Pipeline

Tags: #progress #implementation #dataset-construction #semantic-sets

Related notes: [[README]] [[2026-05-06_v2_clean_restart]] [[2026-05-06_fresh_ag_news_dataset]]

## 1. Goal

Build the V2 paraphrase-set generation pipeline. The central requirement was to force rewrite models to preserve factual meaning while avoiding copied content words and meeting exact word-count bands.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| Main script | `scripts/generate_constrained_sets.py` |
| Config | `configs/rewrite_models.example.json` |
| Core modules | `text_constraints.py`, `rewrite_generation.py`, `semantic_validation.py`, `io_utils.py` |

## 3. Hypothesis Or Rationale

Set-ConCA V2 needs semantic sets where the shared concept is not trivially explained by surface lexical overlap. Banning likely copied content words and requiring several length bands should force more varied paraphrases. Recording both accepted and rejected attempts prevents silent selection bias.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Added length-band representation. | Needed fixed rewrite slots. | `LengthBand` and default bands `5-7`, `10-12`, `15-17`, `20-22` exist. | Succeeded |
| 2 | Added tokenization and banned-word extraction. | Needed to reduce lexical copying. | `extract_banned_words`, `contains_banned_word`, and `validate_rewrite` exist. | Succeeded |
| 3 | Added strong prompt template in config and code. | Needed models to follow exact constraints. | Prompt includes original sentence, forbidden words, required length, factual preservation, and final-rewrite-only instruction. | Succeeded |
| 4 | Added rewrite model specs and generator backends. | Needed to run many HF models one at a time and support dry runs. | `HFRewriteGenerator` and `DryRunRewriteGenerator` exist. | Succeeded |
| 5 | Added optional semantic validation. | Needed a future guard for meaning preservation. | Embedding cosine and optional NLI checks are implemented but disabled by default. | Succeeded |
| 6 | Added artifact writing. | Needed traceable outputs. | Attempts, accepted rewrites, grouped sets, review table, and manifest are saved. | Succeeded |
| 7 | Added unit tests. | Needed regression protection. | Tests cover constraints, cleaning, grouping, and disabled semantic validator behavior. | Succeeded |

## 5. Code And Pseudocode

```text
for model_spec in enabled_models:
    load one generator
    for original in originals:
        banned_words = original.banned_words or extract_banned_words(original.text)
        for band in length_bands:
            for attempt_idx in max_attempts_per_slot:
                prompt = build_prompt(original.text, banned_words, band)
                candidates = generator.generate(prompt)
                for candidate in candidates:
                    ok = validate_rewrite(candidate, banned_words, band)
                    if ok and semantic_validation_enabled:
                        ok = semantic_validator.validate(original.text, candidate)
                    write attempt row with reasons and metrics
                    if first accepted for slot:
                        write accepted row
                        stop retrying slot
    close generator
group accepted rewrites into semantic sets
write review table and run manifest
```

## 6. Results

### Output Artifacts

| Artifact | Meaning |
| --- | --- |
| `attempts.jsonl` | Every candidate with accepted/rejected status and reasons. |
| `accepted.jsonl` | First valid candidate per original/model/length slot. |
| `sets.jsonl` | Grouped semantic sets by original sentence. |
| `review_table.md` | Human-readable review table with original, banned words, model, length, and rewrite. |
| `run_manifest.json` | Config, device, counts, bands, timing, and output path. |

### Configured Rewrite Models

The example config lists Gemma, Llama, Qwen, Mistral, Phi, TinyLlama, and OLMo rewrite candidates. They are disabled by default so the pipeline can be configured for the local machine.

## 7. Interpretation

The pipeline directly addresses the main V2 risk: shallow lexical matching. It does not prove semantic preservation by itself, but it creates a better dataset-construction substrate and records failure reasons for audit.

## 8. Successes

The implementation is modular: constraints, generation, semantic validation, IO, and path handling can be tested or swapped independently.

## 9. Failures Or Limits

Semantic validation is disabled in the example config. Real model generation still needs to be run and manually audited. Banned-word matching is token-based and exact; near-copies and inflections are discouraged by prompt but not fully blocked by code yet.

## 10. External Works And Papers

| Work | Link | Core Objective | How We Used It |
| --- | --- | --- | --- |
| Sentence-BERT, Reimers and Gurevych 2019 | [arXiv 1908.10084](https://arxiv.org/abs/1908.10084) | Produce sentence embeddings that can be compared efficiently with cosine similarity. | Used as the basis for optional embedding-cosine semantic validation. |
| MultiNLI, Williams et al. 2018 | [arXiv 1704.05426](https://arxiv.org/abs/1704.05426) | Provide broad-coverage natural language inference data. | Used conceptually through NLI-style entailment/contradiction checks in optional semantic validation. |

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `configs/rewrite_models.example.json` | Added model list, generation config, prompt template, semantic-validation config. | Control the rewrite experiment without hardcoding values. |
| `src/setconca_v2/text_constraints.py` | Added tokenization, banned words, length bands, validation. | Enforce lexical and length constraints. |
| `src/setconca_v2/rewrite_generation.py` | Added prompt builder and generator classes. | Generate constrained rewrites. |
| `src/setconca_v2/semantic_validation.py` | Added optional embedding/NLI validator. | Guard meaning preservation. |
| `src/setconca_v2/io_utils.py` | Added JSONL writing, grouping, review table. | Preserve all outputs and make review easy. |
| `scripts/generate_constrained_sets.py` | Added end-to-end generation CLI. | Run the full constrained set pipeline. |
| `tests/test_text_constraints.py` | Added tests for constraints and grouping. | Protect core behavior. |

## 12. Follow-Up

- [ ] Add stronger morphological banned-word checks.
- [ ] Run a 10-sentence pilot with real enabled rewrite models.
- [ ] Add manual review outcomes to this note or a new pilot-result note.
