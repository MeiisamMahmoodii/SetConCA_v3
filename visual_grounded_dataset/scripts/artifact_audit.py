from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean

from vg_common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a first-pass artifact audit for visual-grounded sets.")
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.sets)
    n_views = []
    model_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    lengths = []

    for row in rows:
        texts = row.get("texts", [])
        n_views.append(len(texts))
        topic_counts.update(row.get("metadata", {}).get("topic_tags", []))
        for text in texts:
            meta = text.get("metadata", {})
            model_counts[text.get("source_id", "")] += 1
            language_counts[meta.get("language_code", "")] += 1
            prompt_counts[meta.get("prompt_view", "")] += 1
            lengths.append(len(text.get("text", "")))

    lines = [
        "# Visual-Grounded Artifact Audit",
        "",
        "This is a structural audit. It does not replace SetConCA control runs.",
        "",
        "## Coverage",
        "",
        f"- Sets: {len(rows)}",
        f"- Total views: {sum(n_views)}",
        f"- Mean views per set: {mean(n_views):.2f}" if n_views else "- Mean views per set: 0",
        f"- Mean text length: {mean(lengths):.2f} chars" if lengths else "- Mean text length: 0 chars",
        "",
        "## Model Distribution",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in model_counts.most_common())
    lines.extend(["", "## Language Distribution", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in language_counts.most_common())
    lines.extend(["", "## Prompt View Distribution", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in prompt_counts.most_common())
    lines.extend(["", "## Topic Distribution", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in topic_counts.most_common())
    lines.extend(
        [
            "",
            "## Required Next Checks",
            "",
            "- Train/evaluate on real sets vs shuffled-image controls.",
            "- Train/evaluate on same-language, same-prompt, and same-model different-image controls.",
            "- Run language/model/prompt predictability probes on target LLM activations.",
            "- Compare multilingual/multi-prompt/multi-model sets against one-language, one-prompt, and one-model ablations.",
        ]
    )

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote artifact audit to {args.out}")


if __name__ == "__main__":
    main()

