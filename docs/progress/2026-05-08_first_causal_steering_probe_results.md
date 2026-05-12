# First Causal Steering Probe Results

## Goal

Check whether reviewed SetConCA concept directions can causally steer real language-model generations.

## Run

Output directory:

```text
results/causal_steering_probe_diverse300_clean_smoke
```

Command shape:

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

The run completed and wrote:

```text
96 generation rows
```

## Result

Automatic keyword summary:

| Candidate | Best alpha | Best mean keyword gain |
| --- | ---: | ---: |
| Google IPO / stock offering | 0,1,2,4 | 0.00 |
| Windows / software security updates | 1 | 0.25 |
| software / IT products | 0,1,2,4 | 0.00 |
| stock market / earnings / prices | 0,1,2,4 | 0.00 |
| corporate earnings / company performance | 0,1,2,4 | 0.00 |

The hook clearly changed many generations, but the concept-specific keyword gain was mostly flat.

Text-change counts by candidate:

| Candidate | Changed rows | Total rows |
| --- | ---: | ---: |
| Google IPO / stock offering | 22 | 32 |
| Windows / software security updates | 12 | 16 |
| software / IT products | 11 | 16 |
| stock market / earnings / prices | 4 | 16 |
| corporate earnings / company performance | 9 | 16 |

## Honest Interpretation

This is a successful mechanical causal intervention test, but not yet strong concept steering evidence.

What we can say:

- The generation-time hook works.
- The SetConCA decoder directions are being injected at the intended target layers.
- The model outputs often change when the direction is injected.

What we cannot say yet:

- We cannot claim reliable Google IPO, stock-market, or corporate-earnings steering from this run.
- The Windows/software direction has one weak positive signal at alpha `1`, but it is too small and noisy to treat as a result.

## Likely Problem

The default prompts were too generic and sometimes produced odd base-LM continuations. That makes keyword-based concept detection weak even if a direction is doing something.

## Next Run

Created a more news-like prompt file:

```text
configs/steering_probe_news_prompts.csv
```

Recommended next command:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_news_prompts \
  --prompts configs/steering_probe_news_prompts.csv \
  --max-candidates 6 \
  --only-use yes,yes_maybe \
  --alphas 0,1,2,4,8 \
  --max-new-tokens 48 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

If this still stays flat, the next diagnostic is to try `--token-position all` and compare it to the default `last` token intervention.

## Status

Partial. The hook works, but concept steering is not yet demonstrated.
