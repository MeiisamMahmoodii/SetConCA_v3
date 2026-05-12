# News-Prompt Causal Steering Results

## Goal

Repeat the first causal steering probe with more news-like prompts. The first smoke run used generic prompts and produced mostly flat keyword gains, even though the hook changed generations.

## Run

Output directory:

```text
results/causal_steering_probe_diverse300_news_prompts
```

Prompt file:

```text
configs/steering_probe_news_prompts.csv
```

The run completed and wrote:

```text
180 generation rows
```

This comes from:

- 6 clean or clean-ish candidates,
- 6 prompts,
- 5 alpha values: `0,1,2,4,8`.

## Figure

![Keyword gain by alpha](../../results/causal_steering_probe_diverse300_news_prompts/keyword_gain_by_alpha.png)

## Best Scores

| Rank | Candidate | Target model | Best alpha | Baseline keyword score | Steered keyword score | Best gain |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 2 | Google IPO / stock offering | `meta-llama/Llama-3.2-3B` | 1 | 0.17 | 0.67 | 0.50 |
| 9 | Windows / software security updates | `meta-llama/Llama-3.2-1B` | 0 | 0.83 | 0.83 | 0.00 |
| 16 | software / IT products | `meta-llama/Llama-3.2-3B` | 0 | 0.17 | 0.17 | 0.00 |
| 17 | stock market / earnings / prices | `Qwen/Qwen3-4B` | 2 | 0.33 | 0.50 | 0.17 |
| 19 | Google IPO / stock offering | `meta-llama/Llama-3.2-3B` | 1 | 0.17 | 0.67 | 0.50 |
| 30 | corporate earnings / company performance | `meta-llama/Llama-3.1-8B` | 2 | 1.00 | 1.50 | 0.50 |

## Interpretation

This is more promising than the first smoke run.

Evidence that looks encouraging:

- The two Google IPO candidates are actually the same target model/dimension from two source directions. They both show the same positive pattern: gain `0.50` from alpha `1` through `8`.
- The corporate earnings concept shows gain up to `0.50`, strongest around alpha `2` and `4`.
- The stock-market concept shows a smaller positive gain of `0.17`.
- The steering hook changes text in many rows, so the intervention is active.

Evidence that is not yet good:

- Windows/software security gets worse at higher alpha. The baseline prompts already contain Microsoft/Windows/software words, so this concept is prompt-confounded.
- Software / IT products stays flat.
- Keyword gain is still a crude metric. It can count shallow word changes and miss real semantic changes.

## Honest Claim

We now have early behavioral evidence for some causal steering directions, especially:

1. Google IPO / stock offering,
2. corporate earnings / company performance,
3. maybe stock market / earnings / prices.

This is not yet a finished scientific claim. It is a positive diagnostic result that justifies a stronger next evaluation.

## Next Step

Run a focused probe only on the promising candidates with:

- more prompts,
- repeated seeds or sampling,
- positive and negative direction signs,
- and a control direction.

Recommended next command:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_promising_pos \
  --prompts configs/steering_probe_news_prompts.csv \
  --only-use yes,yes_maybe \
  --alphas 0,0.5,1,2,4 \
  --max-new-tokens 64 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

Then run the opposite active sign:

```bash
python scripts/run_causal_steering_probe.py \
  --candidates results/causal_steering_candidates_diverse300/candidates.csv \
  --run-dir results/llama_qwen_diverse300_set_vs_pointwise_linear_seed0 \
  --out-dir results/causal_steering_probe_diverse300_promising_neg \
  --prompts configs/steering_probe_news_prompts.csv \
  --only-use yes,yes_maybe \
  --alphas 0,0.5,1,2,4 \
  --sign-mode opposite_active \
  --max-new-tokens 64 \
  --temperature 0 \
  --dtype float16 \
  --device cuda
```

The key test is whether the active sign increases concept evidence and the opposite active sign suppresses it.

## Status

Partial success. Better than smoke run, but not final proof.
