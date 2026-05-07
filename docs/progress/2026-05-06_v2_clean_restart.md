# Task: V2 Clean Restart And Project Scaffold

Tags: #progress #implementation #v2 #project-design

Related notes: [[README]] [[2026-05-06_constrained_paraphrase_pipeline]] [[2026-05-06_fresh_ag_news_dataset]]

## 1. Goal

Create a separate SetConCA V2 project so future training, testing, and paper work can restart from a cleaner base instead of relying on prior V1 results.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Main plan file | `V2_PLAN.md` |
| Main README | `README.md` |
| Main package | `src/setconca_v2` |

## 3. Hypothesis Or Rationale

The earlier work had accumulated tests, reports, and results whose validity needed to be rechecked. A clean V2 folder gives us a controlled base where each dataset, activation, model, result, and claim can be traced from scratch.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Created or organized a separate `SetConCA_V2` workspace. | Avoid mixing V2 with old V1 artifacts. | V2 exists as a self-contained project folder. | Succeeded |
| 2 | Added V2 planning document. | Needed a research roadmap. | `V2_PLAN.md` records dataset construction, activation extraction, training, tests, and paper phases. | Succeeded |
| 3 | Added V2 README. | Needed operational instructions. | `README.md` explains dataset download and constrained set generation. | Succeeded |
| 4 | Added core Python package under `src/setconca_v2`. | Needed reusable implementation modules. | Dataset, IO, paths, rewriting, semantic validation, and text constraints are separated. | Succeeded |
| 5 | Added modeling and training modules. | Needed downstream Set-ConCA V2 training path. | `model/setconca_v2.py`, `data/activation_sets.py`, and `training/losses.py` exist. | Succeeded |

## 5. Code And Pseudocode

The intended V2 lifecycle:

```text
download independent raw examples
for each original sentence:
    extract banned content words
    for each rewrite model:
        for each length band:
            prompt model for constrained paraphrase
            validate word count and banned-word avoidance
            optionally validate semantic similarity/NLI
save attempts, accepted rewrites, grouped sets, and manifests
extract activations later
train SetConCA V2 and baselines
evaluate transfer, steering, hard negatives, and set-size scaling
write only claims supported by tests
```

## 6. Results

### Core Files Present

| Area | Files |
| --- | --- |
| Planning | `README.md`, `V2_PLAN.md` |
| Dataset and generation | `src/setconca_v2/dataset_download.py`, `rewrite_generation.py`, `text_constraints.py`, `semantic_validation.py` |
| IO and paths | `src/setconca_v2/io_utils.py`, `paths.py` |
| Scripts | `scripts/download_news_dataset.py`, `scripts/generate_constrained_sets.py` |
| Modeling | `model/setconca_v2.py`, `training/losses.py`, `data/activation_sets.py` |
| Tests | `tests/test_dataset_download.py`, `tests/test_text_constraints.py` |

## 7. Interpretation

The V2 scaffold is a good reset because it makes the first project goal explicit: construct higher-quality semantic sets with reduced lexical copying. That is the right dependency to establish before activation extraction or model-transfer claims.

## 8. Successes

The clean restart succeeded because the new workspace has clear module boundaries and can run independently of V1 data.

## 9. Failures Or Limits

The scaffold is still early. Activation extraction, full training scripts, cross-model transfer tests, causal steering evaluations, and final paper-grade result generation remain future work.

## 10. External Works And Papers

| Work | Link | Core Objective | How We Used It |
| --- | --- | --- | --- |
| ConCA paper/project | Not yet pinned in this note | Concept alignment baseline and intellectual predecessor. | V2 is designed to stay faithful to ConCA while testing set-based semantic representations and cross-model concept transfer. |

## 11. Files Changed

This reconstruction note documents existing V2 files rather than making new implementation changes.

## 12. Follow-Up

- [ ] Add the exact ConCA citation and paper breakdown once the canonical source is pinned.
- [ ] Add a future task note for activation extraction design.
- [ ] Add a future task note for the full evaluation suite before running large experiments.
