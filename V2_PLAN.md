# SetConCA V2 Plan

## Phase 1: Dataset Construction

Build semantic sets that reduce lexical copying:

- Extract banned content words from each original sentence.
- Use multiple rewrite models.
- Ask each model for four length bands: 5-7, 10-12, 15-17, 20-22 words.
- Validate exact word count and banned-word avoidance.
- Save accepted and rejected generations.

Expected set size if 10 models are used:

- 10 models x 4 length bands = up to 40 rewrites per original.
- We can later sample S in {2, 4, 8, 16, 32, 40}.

## Phase 2: Activation Extraction

For each downstream representation model:

- Load one model at a time on the 3090.
- Extract residual-stream activations for original + rewrites.
- Record exact model ID, layer index, token position, dtype, and SHA256 hashes.

## Phase 3: Training

Train:

- Set-ConCA V2 with contrastive semantic loss.
- Pointwise ConCA.
- SAE-TopK.
- SAE-L1.
- Gated SAE.
- CrossCoder-style paired sparse baseline.

## Phase 4: Tests

Main tests:

- Semantic set vs shuffled set.
- Semantic set vs duplicated set.
- Same-label hard negatives.
- Wrong-translation hard negatives.
- S-sweep: 2, 4, 8, 16, 32, 40.
- Concept dimension sweep: 64, 128, 256, 512.
- In-family same-generation transfer.
- In-family cross-generation transfer.
- Out-family transfer.
- Multilingual transfer.
- Activation-level steering.
- Behavioral steering, if local model weights are available.

## Phase 5: Paper

Only claim what passes:

- If semantic sets beat shuffled/hard negatives, claim set semantics matter.
- If steering specificity is positive, claim activation-level steering.
- If generated-text behavior passes, claim behavioral steering evidence.

