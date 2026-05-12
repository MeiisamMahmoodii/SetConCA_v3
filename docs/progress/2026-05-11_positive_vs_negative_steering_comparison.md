# Positive Vs Negative Steering Comparison

## Goal

Compare the active-sign and negative-sign causal steering runs.

## Runs

Active-sign run:

```text
results/causal_steering_probe_diverse300_promising_pos
```

Absolute-negative run:

```text
results/causal_steering_probe_diverse300_promising_neg
```

Comparison output:

```text
results/causal_steering_probe_diverse300_pos_neg_comparison
```

## Important Correction

The completed negative run used:

```text
--sign-mode negative
```

That means absolute negative decoder direction, not necessarily the opposite of the active concept sign.

This is only a true opposite control for candidates whose active sign was positive:

- Google IPO / stock offering,
- corporate earnings / company performance.

For candidates whose active sign was already negative, the absolute-negative run reused the same sign as the active run. So Windows, software/IT, and stock-market candidates still need a corrected opposite-active run.

The script now supports the correct mode:

```text
--sign-mode opposite_active
```

## Results From Completed Runs

Best gains:

| Candidate | Active best gain | Absolute-negative best gain | Interpretation |
| --- | ---: | ---: | --- |
| Google IPO / stock offering | 0.50 | 0.33 | Active sign stronger, but negative still partially activates related terms. |
| Corporate earnings / company performance | 0.50 | 0.33 | Active sign stronger, but negative still partially activates related terms. |
| Stock market / earnings / prices | 0.17 | 0.17 | Not a valid sign comparison because active sign was negative. |
| Windows / software security updates | 0.17 | 0.17 | Not a valid sign comparison because active sign was negative. |
| Software / IT products | 0.17 | 0.17 | Not a valid sign comparison because active sign was negative. |

Figure:

![Active vs absolute negative](../../results/causal_steering_probe_diverse300_pos_neg_comparison/active_vs_negative_abs_clear_candidates.png)

## Honest Interpretation

The existing positive-vs-negative evidence is directionally encouraging for Google IPO and corporate earnings, but it is not yet a clean suppression test.

Why:

- active sign is stronger than absolute-negative for Google and corporate earnings,
- but absolute-negative still increases some related keywords,
- and three candidates were not actually inverted by `--sign-mode negative`.

## Correct Next Run

Run the true opposite-active control:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_promising_opposite_active \
  --prompts configs/steering_probe_news_prompts.csv \
  --only-use yes,yes_maybe \
  --alphas 0,0.5,1,2,4 \
  --sign-mode opposite_active \
  --max-new-tokens 64 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

Then compare:

- `promising_pos`,
- `promising_opposite_active`.

## Status

Partial. Useful comparison for two candidates; corrected control still needed for the full set.
