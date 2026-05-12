# Bridged Concept Inspection

## Goal

Move from aggregate bridge scores to inspectable candidate concepts.

Previous runs showed that SetConCA has stronger shuffled-controlled linear bridgeability than the pointwise TopK baseline. This step asks a more concrete question:

> Which individual target concept dimensions are stable under a source-to-target linear bridge, and what text examples activate them?

This is the first gate before causal steering. It is still diagnostic, not a final proof of monosemanticity.

## New Script

Added:

```text
scripts/inspect_bridged_concepts.py
```

The script:

1. Reads saved SetConCA `codes.pt` files.
2. Reads the completed transfer run's `transfer_steering_results.csv`.
3. Selects strong candidate source/target pairs.
4. Fits the requested bridge on train concept codes.
5. Scores each target concept dimension on held-out test codes.
6. Attaches original dataset text and rewrite samples for manual inspection.
7. Writes CSVs, figures, selected-pair metadata, and a Markdown report.

It does not retrain SetConCA models and does not run the original LLMs.

## Command Run

```powershell
python scripts\inspect_bridged_concepts.py `
  --run-dir results\llama_qwen_set_vs_pointwise_linear_seed0 `
  --dataset data\generated\server_4gpu_2000\merged\sets_min16.jsonl `
  --out-dir results\concept_inspection_llama_qwen_e25 `
  --bridge ridge `
  --set-size 16 `
  --source-depth 60 `
  --target-depth 60 `
  --max-pairs 8 `
  --top-concepts 8
```

Output:

```text
Wrote concept inspection to results\concept_inspection_llama_qwen_e25
```

## Result Artifacts

Directory:

```text
results/concept_inspection_llama_qwen_e25
```

Files:

| File | Purpose |
| --- | --- |
| `concept_summary.csv` | Per-pair, per-concept diagnostic scores. |
| `concept_examples.csv` | Top activating examples for each selected concept. |
| `selected_pairs.json` | The source/target pairs selected from transfer results. |
| `REPORT.md` | Human-readable summary of the inspection run. |
| `figures/top_bridged_concepts.png` | Highest-scoring bridged concept candidates. |
| `figures/pair_concept_scores.png` | Average candidate quality by model pair. |

## Inspection Metric

The script reports:

```text
inspection_score =
  (alignment - shuffled_alignment)
  + (top_example_overlap - shuffled_top_example_overlap)
```

This is a ranking aid. It is intentionally controlled by shuffled examples, but it is not itself a semantic proof.

High score means:

- the bridged source dimension and target dimension align on held-out examples,
- their top activating examples overlap more than shuffled controls,
- the dimension is worth reading manually.

High score does not mean:

- the concept is cleanly monosemantic,
- the bridge is behaviorally causal,
- steering will work.

## Early Observation

The first run inspected 8 strong `60% -> 60%`, `S=16`, SetConCA ridge pairs and wrote:

```text
64 concept rows
512 example rows
```

The top candidates include both within-family and cross-family pairs:

- Llama 3 mid -> Llama 3 big,
- Qwen 3 big -> Llama 3 mid/big/small,
- Qwen 3 small -> Llama 3 small,
- Qwen 3 big -> Qwen 3 mid.

The first manual glance shows mixed evidence:

- some examples cluster around plausible themes such as market uncertainty, disasters, conflict, and technology/autonomy,
- some candidates are broad or mixed across news domains,
- therefore this is a candidate-discovery table, not yet a final concept list.

## Verification

The new script compiled successfully:

```text
python -m py_compile scripts\inspect_bridged_concepts.py
```

The test suite passed after the addition:

```text
14 passed
```

The pytest cache warning is a local permission warning and did not affect the tests.

## Next Step

Manually review `concept_examples.csv` and mark each candidate as:

- semantic concept,
- broad topic concept,
- syntax/style artifact,
- dataset/order artifact,
- unclear.

After that, select only a small number of clean candidates for real causal steering.
