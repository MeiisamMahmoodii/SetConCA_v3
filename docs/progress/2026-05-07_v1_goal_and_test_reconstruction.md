# 2026-05-07 V1 Goal and Test Reconstruction

## Goal

Read the original SetConCA project at:

```text
C:\Users\MPC\Documents\code\SetConCA
```

excluding the current V2 folder:

```text
C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2
```

The goal was to understand what V1 was actually trying to prove, what tests were run, what evidence supported the claims, what evidence weakened the claims, and how that should shape V2.

## V1 Research Goal

V1 Set-ConCA was a mechanistic interpretability and cross-model alignment project.

The central idea was:

```text
Concepts should be learned from sets of related hidden states, not from isolated single vectors.
```

Each input example was a tensor:

```text
X: [batch, set_size, hidden_dim]
```

where the `set_size` dimension could represent paraphrases, translations, trajectory steps, or local neighborhoods. The project hypothesis was that a concept code learned from multiple related views would suppress view-specific noise and preserve a more stable semantic core.

In V1 language, Set-ConCA extends ConCA from pointwise concept posterior estimation to set-level concept posterior estimation.

## V1 Model Design

The core implementation lived in:

```text
C:\Users\MPC\Documents\code\SetConCA\code\setconca\model
```

The model was:

```text
x [B, S, D]
  -> ElementEncoder
  -> SetAggregator
  -> optional TopK sparse concept code
  -> DualDecoder
  -> reconstructed hidden states [B, S, D]
```

### ElementEncoder

File:

```text
code/setconca/model/encoder.py
```

The encoder is a shared linear map applied independently to each view:

```text
u_i = W_e x_i + b_e
```

There is no activation function. V1 tests explicitly guarded against accidentally adding a ReLU or clipping behavior.

### SetAggregator

File:

```text
code/setconca/model/aggregator.py
```

The default aggregator is mean pooling over set elements:

```text
u_bar = mean_i(u_i)
z_hat = LayerNorm(u_bar)
```

This makes the set code permutation invariant. An attention aggregator also existed and was tested as an ablation.

### TopK Concept Code

File:

```text
code/setconca/model/setconca.py
```

In the main V1 experiments, sparsity was usually hard TopK:

```text
C = 128
k = 32
```

This means each set-level concept code kept exactly 32 active dimensions. Chance support overlap was often referenced as:

```text
k / C = 0.25
```

### DualDecoder

File:

```text
code/setconca/model/decoder.py
```

The decoder factorized reconstruction into:

```text
shared stream:   W_shared(z_hat)
residual stream: W_residual(u_i)
```

The intended meaning was:

- `z_hat` carries the set-invariant semantic core.
- `u_i` carries view-specific residual information needed to reconstruct each individual hidden state.

This design matters for V2 because it is the mechanism that tries to avoid forcing all syntax/detail into the concept code.

## V1 Loss

The training loss combined:

1. Shared reconstruction loss.
2. Full reconstruction loss.
3. Optional soft sparsity loss.
4. Subset consistency loss.

In TopK mode, the soft sigmoid sparsity term was disabled because the TopK mask already fixes the active count.

The consistency loss split each set into two random subsets, encoded both, and penalized squared distance between their concept codes. In V1, this was intended to make the learned code stable even when only part of the set was observed.

Important V1 finding:

```text
Consistency was not the dominant transfer driver in the verified TopK run.
```

EXP9 showed only about a 0.1 percentage point transfer difference with and without consistency.

## Main V1 Evaluation Suite

The main suite was:

```text
code/evaluation/run_evaluation_v2.py
```

Despite the filename, this belongs to the original V1 project history. It ran EXP1-16 with these core settings:

```text
N = 2048 anchors
S = 8 default set size
C = 128 concept dimensions
k = 32 TopK active units
epochs = 80
batch_size = 64
seeds = [42, 1337, 2024, 7, 314]
```

The primary result artifact was:

```text
code/outputs/results_v2.json
```

## EXP1-16: What V1 Tested

| Experiment | Question | Main V1 finding |
| --- | --- | --- |
| EXP1 | Set-ConCA vs pointwise training | Set-ConCA had higher reconstruction MSE than pointwise, which was framed as the cost of set-level invariance. Stability was similar. |
| EXP2 | Set size scaling | Reconstruction MSE improved monotonically as S increased from 1 to 32, with diminishing returns after about S=8. Stability was relatively flat. |
| EXP3 | Mean vs attention aggregation | Attention slightly improved MSE/stability in that run; mean remained simpler and deterministic. |
| EXP4 | Cross-family transfer | Gemma 4B -> Llama 8B achieved about 0.695 TopK overlap vs 0.25 chance. Reverse direction was about 0.596, showing asymmetry. |
| EXP5 | Intra-family transfer | Gemma-family transfer could be strong but was asymmetric; capacity and training recipe mismatch mattered. |
| EXP6 | SOTA-style baselines | Set-ConCA was competitive among sparse methods, but PCA remained a dense reconstruction reference and pointwise TopK later beat Set-ConCA on raw overlap. |
| EXP7 | Steering | Structured Set-ConCA directions improved target similarity with alpha; random directions collapsed. Weak-to-strong 1B -> 8B steering also improved similarity. |
| EXP8 | Convergence | Training loss converged steadily and stabilized before 80 epochs. |
| EXP9 | Consistency ablation | Removing consistency barely changed transfer in the TopK setup. |
| EXP10 | Corruption test | The tested corruption protocol did not collapse transfer; full corruption remained around 0.69. This weakened any claim that semantic grouping alone was the proven cause. |
| EXP11 | Layer/depth proxy | V1 did not have real multi-layer activations for this test; it used PCA rank as a proxy for information depth. |
| EXP12 | Linear vs nonlinear bridge | Linear bridge beat the tested MLP bridge, supporting approximate linear alignability. |
| EXP13 | Interpretability metrics | Set-ConCA had high proxy interpretability scores, but PCA/SAE comparisons meant this was relative evidence, not human-proven monosemanticity. |
| EXP14 | PCA-32 transfer | Aggressive PCA-32 input compression hurt transfer badly, around 0.314. |
| EXP15 | Soft sparsity consistency | Soft sigmoid-L1 mode performed near chance and was not a robust substitute for hard TopK in that setup. |
| EXP16 | Pointwise TopK vs Set-ConCA | Pointwise TopK SAE achieved higher raw transfer overlap, about 0.784 vs Set-ConCA about 0.695. This is the major negative result. |

## Supported V1 Claims

The evidence supports these claims:

1. Set-ConCA can learn sparse set-level concept codes from hidden-state sets.
2. These codes can transfer across model families far above the simple TopK chance reference.
3. Linear bridges are strong; nonlinear MLP bridging was not better in the verified run.
4. Structured Set-ConCA directions produced better steering-style similarity trajectories than random directions.
5. Increasing set size improved reconstruction MSE in V1.
6. The codebase had real unit tests, smoke tests, and claim-validation gates to prevent unsupported narrative claims.

## Claims V1 Had To Narrow

The evidence does not support stronger versions of the story:

1. Set-ConCA did not always beat pointwise TopK SAE. EXP16 showed pointwise TopK had higher raw overlap.
2. Semantic grouping was not isolated as the only cause of gains. The corruption/shuffled protocols suggested the set bottleneck itself may act as a strong structural regularizer.
3. Consistency loss was not proven to drive transfer in the TopK configuration.
4. PCA-32 compression did not preserve the useful transfer signal.
5. Soft sparsity was not an adequate replacement for hard TopK in the verified setting.
6. Layer conclusions in V1 were limited because the main layer/depth result was a PCA-rank proxy, not true multi-layer activation extraction.
7. Multilingual results were useful supporting evidence, not a universal dominance claim.

## Important Lesson For V2

V2 should be built as a cleaner scientific test of the open questions V1 could not fully settle.

The most important V2 difference is that we now have real activation banks across:

```text
families: Llama 3, Gemma 3, Qwen 3
sizes: small, mid, big
depths: 20%, 60%, 90%
views: 16 per semantic set
sets: 805 originals
```

That means V2 can replace V1's proxy layer test with real layer tests.

V2 can also test set size more cleanly:

```text
S = 2, 4, 6, 8, 10, 12, 14, 16
```

using the same 805 original semantic sets for every S. This avoids changing the underlying examples when changing set size.

## V2 Direction Based On V1

The right V2 goal is not:

```text
Prove Set-ConCA always beats every pointwise method.
```

The right V2 goal is:

```text
Measure when set-level semantic aggregation helps, when it does not, and how that depends on model family, size, depth, and set size.
```

V2 should keep the V1 discipline:

- Report negative results.
- Compare against pointwise TopK/SAE-style baselines.
- Keep set-size and layer effects separate.
- Treat steering and cross-model transfer as different kinds of evidence.
- Use numeric artifacts as truth, not narrative strings.

## Files Read

Key V1 files inspected:

```text
README.md
code/setconca/model/setconca.py
code/setconca/model/encoder.py
code/setconca/model/aggregator.py
code/setconca/model/decoder.py
code/setconca/losses/consistency.py
code/setconca/losses/sparsity.py
code/setconca/data/dataset.py
code/train.py
code/evaluation/run_evaluation_v2.py
code/evaluation/run_extended_alignment.py
code/scripts/run_full_pipeline.py
code/tests/test_setconca.py
code/tests/test_validation_gates.py
code/outputs/results_v2.json
code/outputs/extended_alignment_results.json
document/CODEBASE_WALKTHROUGH.md
document/full_project_diary/08_claims_vs_evidence.md
document/document/04_experiments.md
document/document/05_results.md
```

## Status

Status: `Succeeded`

No V2 code was changed. This note records V1's goal, test suite, result interpretation, limitations, and the resulting V2 research direction.
