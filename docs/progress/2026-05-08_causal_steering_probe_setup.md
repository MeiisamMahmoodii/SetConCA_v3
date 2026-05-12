# Causal Steering Probe Setup

## Goal

Move from concept-code transfer evidence to the first generation-time causal steering probe, while keeping the architecture unchanged.

## What Changed

Added a small steering utility module:

```text
src/setconca_v2/steering.py
```

Added a manifest builder:

```text
scripts/build_steering_candidate_manifest.py
```

Added a generation-time steering probe:

```text
scripts/run_causal_steering_probe.py
```

Added tests:

```text
tests/test_steering_manifest.py
```

## Method

For a reviewed concept candidate, the target SetConCA model provides a target concept dimension. The target SetConCA shared decoder maps that concept dimension back into the target model hidden-state space.

The steering direction is therefore:

```text
target_hidden_direction = target_setconca.shared_decoder.weight[:, target_concept_dim]
```

The probe normalizes this vector and adds it to the target language model hidden state at the extraction layer during generation.

This is a real causal intervention on the language model hidden state, but it is still a first probe. The output is scored with keyword hits and must be manually inspected before any strong claim.

## Why This Is Minimal

No SetConCA architecture was changed. The probe only reuses existing trained artifacts:

- reviewed concept candidates,
- target SetConCA checkpoint,
- target concept dimension,
- target model layer.

## Main Commands

Build the candidate manifest:

```bash
python scripts/build_steering_candidate_manifest.py \
  --concept-summary results/concept_inspection_llama_qwen_diverse300_e25/concept_summary.csv \
  --labels results/concept_inspection_llama_qwen_diverse300_e25/review_pages/first_pass_labels.csv \
  --out-csv results/causal_steering_candidates_diverse300/candidates.csv \
  --out-json results/causal_steering_candidates_diverse300/candidates.json \
  --include-use yes,yes_maybe,maybe_control
```

Dry-run the selected steering candidates:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_smoke \
  --max-candidates 2 \
  --dry-run
```

Run a small real probe on one GPU:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_smoke \
  --max-candidates 2 \
  --alphas 0,1,2,4 \
  --max-new-tokens 48 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

## Outputs

The steering probe writes:

- `run_manifest.json`,
- `selected_candidates.csv` for dry runs,
- `generations.csv`,
- `summary.csv`,
- `direction_meta.json`,
- `REPORT.md`.

Created candidate manifest:

```text
results/causal_steering_candidates_diverse300/candidates.csv
results/causal_steering_candidates_diverse300/candidates.json
```

The manifest contains 15 candidates:

- 6 clean or clean-ish candidates selected by `yes` / `yes_maybe`,
- 9 broad sports/Olympics controls selected by `maybe_control`.

Dry-run output:

```text
results/causal_steering_probe_diverse300_dryrun
```

The dry run confirmed that the default probe selection starts with the clean candidates:

1. Google IPO / stock offering
2. Windows / software security updates
3. software / IT products
4. stock market / earnings / prices
5. Google IPO / stock offering
6. corporate earnings / company performance

## Recommended First Real Run

For the first real causal probe, run only the clean candidates:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_clean_smoke \
  --max-candidates 6 \
  --only-use yes,yes_maybe \
  --alphas 0,1,2,4 \
  --max-new-tokens 48 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

For a later control run, include the broad sports controls:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_with_controls \
  --max-candidates 15 \
  --only-use yes,yes_maybe,maybe_control \
  --alphas 0,1,2,4 \
  --max-new-tokens 48 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

If the first run is too weak, rerun with stronger alpha values:

```text
--alphas 0,2,4,8
```

## Caveat

The current automatic score is keyword gain. This is intentionally weak and honest. The next scientific gate is manual reading of baseline-vs-steered generations, followed by a better evaluation set if the smoke probe shows a visible effect.

## Verification

Full local test suite:

```text
17 passed, 1 warning
```

The warning is from pytest cache creation permissions and does not affect the test result.

## Status

Succeeded.
