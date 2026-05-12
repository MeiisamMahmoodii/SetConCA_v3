# SetConCA V2 Experiment Report

## Current Objective

SetConCA V2 is now moving from dataset construction to representation experiments.

The V1 project showed that Set-ConCA can learn set-level sparse concept codes with above-chance cross-model transfer and steering-style evidence, but it also found important limits: pointwise TopK can beat Set-ConCA on raw overlap, consistency loss was not the main driver in the verified TopK setup, and V1's layer/depth test was only a PCA-rank proxy.

The current V2 objective is therefore to test, more cleanly, when semantic set size improves learned concept representations, and whether that behavior changes across:

- model family,
- model size,
- model depth,
- and set size.

## Dataset Under Study

The controlled dataset for the first serious sweep is:

```text
data/generated/server_4gpu_2000/merged/sets_min16.jsonl
```

Why this file:

- It has 805 semantic sets.
- Every set has at least 16 accepted rewrites.
- It supports fair set-size comparisons from `S=2` through `S=16`.
- All set-size conditions can use the same original sentences.

## Model Grid

| Family | Small | Mid | Big | Current status |
| --- | --- | --- | --- | --- |
| Llama 3 | `meta-llama/Llama-3.2-1B` | `meta-llama/Llama-3.2-3B` | `meta-llama/Llama-3.1-8B` | Active |
| Qwen 3 | `Qwen/Qwen3-0.6B` | `Qwen/Qwen3-4B` | `Qwen/Qwen3-8B` | Active |
| Gemma 3 | `google/gemma-3-1b-pt` | `google/gemma-3-4b-pt` | `google/gemma-3-12b-pt` | Paused after weak shuffled-controlled bridge evidence |

These are nearest official family sizes. Some families do not have exact 1B/4B/7B releases.

## Layer Grid

For each model, extract hidden states from three depth points:

```text
20%, 60%, 90%
```

The script reads each model config and converts these fractions to actual layer indices.

## Activation Extraction Plan

For every model/layer pair:

1. Load `sets_min16.jsonl`.
2. Select 16 views per set.
3. Run the representation model.
4. Extract hidden states from the chosen layer.
5. Pool the last non-padding token by default.
6. Save an activation bank shaped:

```text
[805, 16, hidden_dim]
```

## Training Plan

After extraction, train SetConCA V2 with set sizes:

```text
S = 2, 4, 6, 8, 10, 12, 14, 16
```

This produces:

```text
27 activation banks x 8 set sizes = 216 training runs
```

## Why This Is Scientifically Clean

The activation extraction uses one fixed dataset source. Smaller set sizes are created by slicing the same activation bank, not by switching to a different dataset.

Therefore, changes across `S` are less confounded by changes in original sentence coverage.

This directly addresses a V1 limitation: V1 showed useful S-scaling and proxy depth trends, but V2 now has real model-family/size/layer activation banks and a controlled `S=2...16` sweep over the same semantic sets.

## Main Scripts

| Script | Purpose |
| --- | --- |
| `scripts/run_activation_grid.py` | Run all model-family/size/layer extraction jobs. |
| `scripts/extract_activation_bank.py` | Extract one activation bank. |
| `scripts/train_setconca_v2.py` | Train SetConCA V2 on one bank and one set size. |
| `scripts/run_transfer_steering_grid.py` | Train SetConCA V2 across selected activation banks/set sizes, evaluate bridge-based concept transfer, and run a concept-code steering proxy. |
| `scripts/summarize_and_filter_sets.py` | Create dataset stats and filtered set files. |

## Transfer And Steering Pipeline

The V2 transfer script keeps the current architecture unchanged and evaluates trained concept spaces with several bridge types:

```text
identity, procrustes, ridge, mlp
```

The steering output is currently a concept-code proxy. It tests whether bridged source concept directions move target concept codes more than random directions. It is not yet a full language-model behavioral steering intervention.

Smoke-tested command:

```bash
python scripts/run_transfer_steering_grid.py \
  --activation-root data/activations/model_grid_s16_min16_4A100 \
  --out-dir results/smoke_transfer_steering_qwen3_small \
  --only-family qwen3 \
  --only-size small \
  --set-sizes 2,4 \
  --max-banks 2 \
  --max-sets 24 \
  --epochs 1 \
  --batch-size 8 \
  --mlp-epochs 5 \
  --steering-alphas 0,1 \
  --bridges identity,procrustes,ridge,mlp \
  --include-self-pairs \
  --device cpu
```

Smoke status:

- 2 activation banks selected.
- 4 tiny SetConCA models trained.
- 64 pair/bridge/steering rows written.
- Figures and `REPORT.md` generated under `results/smoke_transfer_steering_qwen3_small`.

Seeded Qwen3-small pilot:

```bash
python scripts/run_transfer_steering_grid.py \
  --activation-root data/activations/model_grid_s16_min16_4A100 \
  --out-dir results/pilot_transfer_steering_qwen3_small_seed0 \
  --only-family qwen3 \
  --only-size small \
  --set-sizes 2,4,8,16 \
  --max-sets 100 \
  --epochs 3 \
  --batch-size 16 \
  --mlp-epochs 20 \
  --steering-alphas 0,1,2 \
  --bridges identity,procrustes,ridge,mlp \
  --device cpu \
  --seed 0
```

Pilot status:

- 3 Qwen3 0.6B layer banks selected.
- 12 SetConCA models trained.
- 288 transfer/bridge/steering rows written.
- Four figures generated: bridge summary, steering proxy, set-size bridge comparison, and training loss by layer/set size.
- This pilot is for pipeline validation only; it is not the final scientific result.

## Recommended Server Command

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/model_grid_s16_min16 \
  --gpus 4
```

## Recommended Pilot First

```bash
uv run python scripts/run_activation_grid.py \
  --config configs/activation_model_grid.json \
  --out-root data/activations/pilot_qwen3_small_s16 \
  --only-family qwen3 \
  --only-size small \
  --max-sets 25 \
  --gpus 1
```

## Current Status

- Large semantic-set dataset exists.
- `sets_min16.jsonl` is selected as the controlled sweep dataset.
- Activation extraction script exists.
- Training script exists.
- Model/layer grid config exists.
- Activation-grid launcher exists.
- Full copied 4A100 activation grid has passed local QA: 27/27 banks are present, all tensors load, all checked shapes are `[805, 16, hidden_dim]`, and no extraction logs contained failure patterns.
- Transfer/steering grid script exists and passed a tiny end-to-end smoke run.
- First within-family pilots are available for Gemma 3, Llama 3, and Qwen 3. Llama and Qwen show promising set-size scaling under linear bridges; Gemma shows very high bridge overlap but also very large reconstruction losses, so it needs a diagnostic audit before claims.
- First bridge/code diagnostic pass is complete. It showed that raw TopK bridge overlap can be inflated by high shuffled-anchor overlap, especially in the Gemma 3 and Qwen 3 small diagnostic slices. The transfer script now reports `real_minus_shuffled_topk` as the main first-pass controlled bridge score and adds adjusted-overlap figures.
- Larger family pilots with the updated reporting are complete at 25 epochs, batch size 128, max 300 sets. Llama 3 shows the strongest controlled linear bridge evidence, Qwen 3 shows moderate evidence, and Gemma 3 raw overlap is mostly explained by shuffled-anchor controls.
- All-family pair evaluation is complete and verified: it includes 15,552 cross-family alpha-0 rows and 19,008 cross-size alpha-0 rows. Within-family transfer is stronger than cross-family transfer. The strongest cross-family linear signals are Qwen 3 to Llama 3 and Llama 3 to Qwen 3; pairs involving Gemma 3 remain weak after shuffled-anchor correction.
- Gemma 3 is paused for the next phase. Future extraction config keeps the Gemma entries for provenance but disables them by default.
- The Llama/Qwen SetConCA-vs-pointwise baseline run is complete. Pointwise TopK has much higher raw overlap, but most of it remains under shuffled anchors. SetConCA has lower raw overlap but much stronger controlled transfer: Procrustes `0.1333` vs `0.0465`, ridge `0.1449` vs `0.0586`. This supports the value of set-level training for anchor-specific linear bridgeability.
- Automatic summary reporting is now part of `scripts/run_transfer_steering_grid.py`. Completed runs write grouped CSV/JSON summaries and summary figures under each result directory. The completed Llama/Qwen run was refreshed with `--resume`, so the report, summary tables, and figures were regenerated without retraining the 288 existing artifacts.
- The 25/50/100 epoch comparison is complete. Longer training improved the pointwise TopK baseline but did not improve SetConCA's controlled transfer score. SetConCA ridge decreased from `0.1452` at 25 epochs to `0.1416` at 50 and `0.1368` at 100. The 25-epoch SetConCA run remains the main candidate for concept extraction and causal-steering preparation.
- First bridged concept inspection is implemented and run for the strongest `S=16`, SetConCA, ridge, `60% -> 60%` candidates. It produced `results/concept_inspection_llama_qwen_e25` with concept summary tables, example tables, figures, and a report. Early examples show some plausible semantic/topic clusters, but also broad or mixed candidates, so manual concept labeling is required before causal steering claims.
- Human-readable concept review pages now exist under `results/concept_inspection_llama_qwen_e25/review_pages`. The next decision is manual: label each top candidate as semantic, broad topic, artifact, or unclear, then keep only clean semantic candidates for steering.
- A first-pass filled concept label table is available at `results/concept_inspection_llama_qwen_e25/review_pages/first_pass_labels.md`. The strongest first-pass steering candidates are concepts 9, 10, 19, and maybe 21.
- Dataset diversity audit found that the previous `--max-sets 300` slice was label-skewed: 194 science/technology, 45 business, 44 world, and only 17 sports. A balanced 300-set diverse slice now exists with 75 examples from each AG News label, plus matching filtered activation banks under `data/activations/model_grid_s16_min16_diverse300_4A100`.
- The balanced Llama/Qwen run is complete at 25 epochs. It trained 288 models and wrote 14,688 result rows under `results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0`. Controlled SetConCA transfer improved relative to the skewed first-300 run: ridge `real_minus_shuffled_topk` is `0.1827`, and Procrustes is `0.1674`. Pointwise TopK remains much lower after shuffled correction: ridge `0.0664`, Procrustes `0.0509`.
- Diverse concept inspection is complete under `results/concept_inspection_llama_qwen_diverse300_e25`. First-pass labels are available in `review_pages/first_pass_labels.md` and `review_pages/first_pass_labels.csv`. The strongest clean candidates are Google IPO / stock offering, Windows/software security updates, stock market/earnings/prices, and corporate earnings/company performance. Sports/Olympics candidates exist, but they are broad-topic controls rather than strong monosemantic claims.
- The first generation-time causal steering probe is now implemented without changing the SetConCA architecture. `scripts/build_steering_candidate_manifest.py` turns reviewed concept labels into a steering manifest, and `scripts/run_causal_steering_probe.py` uses the target SetConCA shared-decoder column as a hidden-state steering vector at the target model layer. This is the first real causal intervention path, but the automatic score is only keyword gain and must be manually audited.
- The first steering manifest is written under `results/causal_steering_candidates_diverse300`. It contains 15 reviewed candidates: 6 clean or clean-ish concepts and 9 broad sports/Olympics controls. A dry-run selection under `results/causal_steering_probe_diverse300_dryrun` confirms that the clean candidates are selected first.
- The first real causal steering smoke run completed under `results/causal_steering_probe_diverse300_clean_smoke` with 96 generation rows. The intervention changes many generations, so the hook is mechanically active, but concept-specific keyword gain is mostly flat. Only the Windows/software candidate showed a small positive gain at alpha `1`. This is a partial result: causal intervention works, but reliable concept steering is not yet demonstrated. A more news-like prompt file now exists at `configs/steering_probe_news_prompts.csv` for the next diagnostic run.
- The news-prompt causal steering run completed under `results/causal_steering_probe_diverse300_news_prompts` with 180 generation rows. This run is more promising: the duplicated Google IPO target direction shows stable gain `0.50`, corporate earnings reaches gain `0.50`, and stock-market/earnings reaches gain `0.17`. Windows/security is prompt-confounded and becomes negative at high alpha. The result is a partial success, not final proof: it justifies a focused positive-vs-negative direction test.
- Positive and absolute-negative steering runs are complete under `results/causal_steering_probe_diverse300_promising_pos` and `results/causal_steering_probe_diverse300_promising_neg`. Active sign is stronger than absolute-negative for Google IPO and corporate earnings, but this was not a clean opposite-sign control for every candidate because `--sign-mode negative` means absolute negative, not opposite of active sign. `scripts/run_causal_steering_probe.py` now supports `--sign-mode opposite_active`; the corrected control should be run before making a directional steering claim.
- The corrected opposite-active run is complete under `results/causal_steering_probe_diverse300_promising_opposite_active`, with comparison artifacts under `results/causal_steering_probe_diverse300_active_vs_opposite`. Google IPO and corporate earnings remain strongest: active gain `0.50` vs opposite-active `0.33`. Windows/software and software/IT show suppression under opposite-active but have weak or prompt-confounded active effects. Stock-market/earnings fails the directional keyword test because opposite-active reaches `0.33` vs active `0.17`. Current conclusion: early direction-sensitive behavioral evidence exists, but this is not yet a clean monosemantic steering claim.
- Next action is interpretability and real steering work on the strongest directions: Qwen 3 to Llama 3, Llama 3 to Qwen 3, and Llama 3 within-family.

## Automatic Reporting Artifacts

For the current Llama/Qwen result:

```text
results/llama_qwen_set_vs_pointwise_linear_seed0
```

Primary summary tables:

| File | Use |
| --- | --- |
| `summaries/method_relation_summary.csv` | Compare SetConCA and pointwise TopK across within-family and cross-family transfer. |
| `summaries/depth_pair_summary.csv` | Pick source/target layer-depth pairs for concept extraction. |
| `summaries/family_pair_summary.csv` | Pick source/target model families for cross-model transfer. |
| `summaries/set_size_summary.csv` | Inspect whether larger semantic sets improve controlled bridge signal. |

Primary figures:

| Figure | Use |
| --- | --- |
| `figures/summary_method_bridge_adjusted.png` | Best one-page method and bridge comparison. |
| `figures/summary_relation_bridge_adjusted.png` | Best one-page within-family vs cross-family comparison. |
| `figures/summary_depth_heatmap_setconca_ridge.png` | Best first depth-pair map for the linear SetConCA bridge. |
| `figures/summary_family_heatmap_setconca_ridge.png` | Best first family-pair map for the linear SetConCA bridge. |

The main score in these artifacts is still `real_minus_shuffled_topk`, because it subtracts the shuffled-anchor overlap from raw TopK overlap and is therefore a better first-pass estimate of anchor-specific bridgeability.

Current interpretation from the automatic tables:

- SetConCA remains stronger than pointwise TopK after shuffled-anchor correction.
- Ridge and Procrustes are the most useful bridge types for the current linear-bridge hypothesis.
- Cross-family Llama/Qwen transfer is weaker than within-family transfer but remains above the shuffled control.
- Target depth around `60%` is a better first candidate for real concept extraction than jumping directly to target `90%`.

This means the next stage should use SetConCA with linear bridges first, not an MLP bridge as the main claim. MLP can stay as a later diagnostic for nonlinear recoverability.
