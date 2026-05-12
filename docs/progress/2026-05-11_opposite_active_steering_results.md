# Opposite-Active Steering Results

## Goal

Run the corrected directional control:

```text
--sign-mode opposite_active
```

This is the proper opposite of the concept's active sign, unlike the earlier absolute-negative run.

## Runs Compared

Active-sign run:

```text
results/causal_steering_probe_diverse300_promising_pos
```

Opposite-active run:

```text
results/causal_steering_probe_diverse300_promising_opposite_active
```

Comparison output:

```text
results/causal_steering_probe_diverse300_active_vs_opposite
```

## Figure

![Active vs opposite-active keyword gain](../../results/causal_steering_probe_diverse300_active_vs_opposite/active_vs_opposite_keyword_gain.png)

## Best-Gain Comparison

| Candidate | Active best gain | Opposite-active best gain | Interpretation |
| --- | ---: | ---: | --- |
| Google IPO / stock offering | 0.50 | 0.33 | Active is stronger, but opposite still activates related terms. Promising, not clean suppression. |
| Windows / software security updates | 0.17 | 0.00 | Directional suppression pattern is good, but baseline prompts are confounded by Windows/software words. |
| Software / IT products | 0.17 | 0.00 | Weak active effect and opposite suppresses it. Low-confidence positive. |
| Stock market / earnings / prices | 0.17 | 0.33 | Fails directional test; opposite-active is stronger on keyword score. |
| Corporate earnings / company performance | 0.50 | 0.33 | Active is stronger, but opposite still activates related business terms. Promising, not clean suppression. |

## Interpretation

This is the most honest current picture:

- Google IPO and corporate earnings remain the strongest behavioral candidates.
- Windows/software and software/IT show some directional suppression, but their positive effect is weak or prompt-confounded.
- Stock-market/earnings is ambiguous and should not be used as a main claim yet.
- Opposite-active does not reliably suppress all related words, probably because the concepts live in a broad business/news region and keyword scoring is coarse.

## What We Can Claim Now

We can claim:

```text
SetConCA decoder-derived directions can causally change generation, and some reviewed concept directions show early, direction-sensitive keyword effects.
```

We should not yet claim:

```text
SetConCA has proven clean monosemantic causal steering between models.
```

## Next Step

Move from keyword-only scoring to a targeted evaluation set:

1. Keep only the strongest candidates:
   - Google IPO / stock offering,
   - corporate earnings / company performance,
   - maybe Windows/software as a control.
2. Create neutral prompts that do not already contain the target keywords.
3. Add a random-direction control with the same norm.
4. Score generations with:
   - keyword gain,
   - manual examples,
   - and later an external classifier or LLM judge if available.

## Status

Partial success. Directional evidence exists, but it is not yet strong enough for a final monosemantic steering claim.
