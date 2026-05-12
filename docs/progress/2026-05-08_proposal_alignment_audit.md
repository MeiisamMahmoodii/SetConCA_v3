# 2026-05-08 Proposal Alignment Audit

## Goal

Check whether SetConCA V2 is still following the original Set-ConCA proposal and whether the current results support the intended high-level objective:

```text
learn set-level monosemantic concept representations that can be linearly bridged between models,
then use those bridges for concept transfer and eventually causal steering.
```

## Sources Checked

- Original Set-ConCA proposal text extracted in `C:\Users\MPC\Documents\code\SetConCA\set_conca_math.txt`
- Original ConCA paper text extracted in `C:\Users\MPC\Documents\code\SetConCA\conca_math.txt`
- V1 implementation under `C:\Users\MPC\Documents\code\SetConCA\code\setconca`
- V1 reconstruction note: `docs/progress/2026-05-07_v1_goal_and_test_reconstruction.md`
- V2 implementation:
  - `model/setconca_v2.py`
  - `training/losses.py`
  - `scripts/run_transfer_steering_grid.py`
- V2 result notes:
  - `docs/progress/2026-05-08_diagnostic_family_reruns_e25_b128.md`
  - `docs/progress/2026-05-08_cross_family_pair_verification.md`

The local PDF extractor failed on the PDF due a Windows/MiKTeX permission error, so this audit uses the existing extracted proposal text files and code.

## Original Proposal Requirements

The proposal defines Set-ConCA as:

```text
u_i = W_e f(x_i) + b_e
u_bar_X = P({u_1, ..., u_m})
z_hat_X = R(u_bar_X)
```

For the minimal implementation, the proposal explicitly lists:

- linear encoder,
- mean pooling,
- shared + residual decoder,
- smooth probability surrogate,
- L1 sparsity,
- subset consistency regularization.

The proposal's theoretical reason for mean pooling is that set-level posterior evidence can be treated as accumulated or normalized pointwise evidence:

```text
u_bar_X = (1/m) sum_j u_j
```

This is intended to approximate normalized set-level log-posterior evidence:

```text
z_hat_X approx [log p(z_i | X)]_i
```

## Does V2 Follow The Proposal?

Mostly yes, with two important deliberate deviations.

### Followed

V2 still uses:

- linear element encoder,
- mean pooling over views,
- permutation-invariant set code,
- LayerNorm after pooling,
- shared decoder,
- residual decoder,
- subset split consistency,
- linear bridge tests with Procrustes and ridge.

Current V2 model path:

```text
x [B,S,D]
  -> Linear encoder u [B,S,C]
  -> mean(u, dim=1)
  -> LayerNorm
  -> TopK sparse z [B,C]
  -> shared decoder W_s z
  -> residual decoder W_r u_i
```

This is structurally aligned with the original proposal.

### Deviated

V2 currently uses hard TopK sparsity instead of the proposal's minimal sigmoid/probability-domain L1.

Reason: V1 found soft sigmoid-L1 mode unreliable in the verified TopK setup. TopK was adopted as the more stable sparse support mechanism. This is a pragmatic change, not part of the original minimal proposal.

V2 also adds contrastive alignment and off-diagonal decorrelation losses. These are not in the original proposal. They were added to stabilize learned code geometry and make bridge testing more meaningful. They should be documented as V2 additions, not original Set-ConCA theory.

## What Happens If We Do Mean Pooling?

We already are doing mean pooling in V2:

```python
pooled = u.mean(dim=1)
```

Mean pooling is the most proposal-faithful aggregator because:

- it is permutation invariant,
- it has a direct normalized posterior-evidence interpretation,
- it avoids adding learned attention weights that could obscure the ConCA log-posterior story,
- it is simple enough that linear bridge results are easier to interpret.

V1 had an attention aggregator ablation. Attention slightly improved some V1 metrics, but it is less clean theoretically because learned attention changes the set posterior aggregation rule. For the current goal, mean pooling is the right default.

## Is The Goal Clear?

Yes. The goal should be written as:

```text
SetConCA aims to learn sparse set-level concept codes that approximate log-posterior concept evidence.
If those codes capture real monosemantic concepts, then corresponding concepts should be linearly bridgeable across models.
Successful bridges should allow concept transfer, and later causal steering, across model families and sizes.
```

## How Far Are We?

### Achieved

We have:

- a controlled paraphrase-set dataset,
- real hidden-state banks across 3 families, 3 sizes, and 3 depth points,
- set-size sweep from `S=2` to `S=16`,
- SetConCA training over all banks,
- within-family and cross-family bridge evaluation,
- linear bridges through Procrustes and ridge,
- shuffled-anchor controls,
- initial steering proxy in concept-code space.

### Current Evidence

The best evidence is for Llama 3 and Qwen 3:

- Llama 3 within-family linear bridge is strongest.
- Qwen 3 within-family bridge is moderate.
- Qwen 3 to Llama 3 and Llama 3 to Qwen 3 are the strongest cross-family directions.
- Larger set sizes help Llama clearly and Qwen moderately.

The current all-family run verified cross-family coverage:

- `15,552` cross-family alpha-0 rows,
- `19,008` cross-size alpha-0 rows.

Important linear bridge results:

| Relation | Procrustes adjusted | Ridge adjusted |
| --- | ---: | ---: |
| within-family | 0.1079 | 0.1119 |
| cross-family | 0.0424 | 0.0519 |
| qwen3 -> llama3 | 0.1504 | 0.1677 |
| llama3 -> qwen3 | 0.0921 | 0.1104 |

### Not Yet Proven

We have not yet proven monosemanticity.

Current bridge success shows alignment of sparse code supports above shuffled controls for some model pairs. That is evidence for shared structure, but not proof that each dimension is a human-interpretable monosemantic concept.

We also have not yet proven causal steering in the strong sense.

Current steering is a concept-code proxy. It does not yet inject directions into transformer activations during generation and measure output changes, specificity, or side effects.

Gemma remains a warning case. Raw overlap is high, but shuffled overlap is also high, so Gemma results should not be used as strong semantic bridge evidence yet.

## Honest Distance To The Main Objective

Current status:

```text
Dataset and representation pipeline:        strong
Proposal-faithful architecture:             mostly strong
Linear bridge evidence:                     moderate, strongest for Llama/Qwen
Cross-family concept transfer evidence:     early but real for Llama/Qwen
Monosemanticity proof:                      not yet
Behavioral causal steering proof:           not yet
Paper-level comparison to baselines:        incomplete
```

Overall distance:

```text
We are past "does the pipeline work?"
We are at "there is a real Llama/Qwen linear bridge signal."
We are not yet at "we proved monosemantic brain-to-brain concept transfer."
```

## Next Required Tests

1. Add pointwise TopK/ConCA baseline on the same V2 activation banks.
2. Add dense CKA/CCA/PCA bridge baselines on the same banks.
3. Add concept interpretability checks for top dimensions.
4. Add true activation steering for the strongest directions:
   - qwen3 -> llama3,
   - llama3 -> qwen3,
   - within-family llama3.
5. Add specificity controls:
   - shuffled concept direction,
   - off-target concept shift,
   - random same-norm direction,
   - unrelated semantic set.
6. Keep mean pooling as default until a better aggregator wins both empirically and theoretically.

## Status

Status: `Succeeded`

The architecture is close to the original proposal. The goal is clear. The current evidence is promising but still intermediate: it supports linearly bridgeable sparse set codes in Llama/Qwen more than it proves monosemantic causal concept transfer.
