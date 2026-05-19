from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from vg_common import read_jsonl, write_jsonl


STOPWORDS = set(
    """
    a an the and or but if then while with without of in on at to for from by as is are was were be been being it this that
    these those there here below above under over into onto within between against around near beside another other some several
    many one two three image shows show scene visible appears featuring features includes including has have had close likely suggests
    setting area foreground background front back top bottom left right
    un une le la les des de du et ou mais dans sur sous avec sans pour par comme est sont était étaient ce cette ces il elle ils
    elles au aux en l image montre visible semble plusieurs groupe personne personnes autre certains autour près devant derrière haut
    bas gauche droite
    """.split()
)


def latin_tokens(text: str, *, remove_stopwords: bool = True) -> set[str]:
    tokens = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", text.lower())
    cleaned = set()
    for token in tokens:
        token = token.strip("'")
        if len(token) < 3:
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        cleaned.add(token)
    return cleaned


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(len(a | b), 1)


def diversity_metrics(row: dict) -> dict:
    texts = list(row.get("texts", []))
    latin_by_language: dict[str, list[set[str]]] = {}
    for item in texts:
        code = item.get("metadata", {}).get("language_code")
        if code not in {"en", "fr", "de", "es", "pt", "id", "sw"}:
            continue
        tokens = latin_tokens(str(item.get("text", "")))
        if tokens:
            latin_by_language.setdefault(code, []).append(tokens)

    pair_scores = []
    for token_sets in latin_by_language.values():
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                pair_scores.append(jaccard(token_sets[i], token_sets[j]))

    all_latin_sets = [tokens for group in latin_by_language.values() for tokens in group]
    doc_freq = Counter()
    for tokens in all_latin_sets:
        doc_freq.update(tokens)
    repeated_frac = 0.0
    if all_latin_sets:
        repeated = sum(1 for count in doc_freq.values() if count / len(all_latin_sets) >= 0.5)
        repeated_frac = repeated / max(len(doc_freq), 1)

    return {
        "latin_view_count": len(all_latin_sets),
        "mean_same_language_content_jaccard": sum(pair_scores) / len(pair_scores) if pair_scores else 0.0,
        "max_same_language_content_jaccard": max(pair_scores) if pair_scores else 0.0,
        "repeated_content_word_fraction": repeated_frac,
    }


def should_keep(metrics: dict, args: argparse.Namespace) -> tuple[bool, list[str]]:
    reasons = []
    if metrics["mean_same_language_content_jaccard"] > args.max_mean_content_jaccard:
        reasons.append("mean_content_jaccard_too_high")
    if metrics["max_same_language_content_jaccard"] > args.max_pair_content_jaccard:
        reasons.append("max_pair_content_jaccard_too_high")
    if metrics["repeated_content_word_fraction"] > args.max_repeated_word_fraction:
        reasons.append("repeated_content_word_fraction_too_high")
    return not reasons, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Drop visual sets that are too lexically repetitive within language.")
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rejected-out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--max-mean-content-jaccard", type=float, default=0.40)
    parser.add_argument("--max-pair-content-jaccard", type=float, default=0.85)
    parser.add_argument("--max-repeated-word-fraction", type=float, default=0.12)
    args = parser.parse_args()

    source_rows = read_jsonl(args.sets)
    print("Set diversity filter run")
    print(f"- input: {args.sets}")
    print(f"- output: {args.out}")
    print(f"- rejected output: {args.rejected_out}")
    print(f"- report: {args.report}")
    print(f"- input sets: {len(source_rows)}")
    print(f"- max mean content Jaccard: {args.max_mean_content_jaccard}")
    print(f"- max pair content Jaccard: {args.max_pair_content_jaccard}")
    print(f"- max repeated word fraction: {args.max_repeated_word_fraction}")
    print("")

    kept = []
    rejected = []
    reason_counts: Counter[str] = Counter()
    metric_rows = []
    for row in source_rows:
        metrics = diversity_metrics(row)
        metric_rows.append(metrics)
        keep, reasons = should_keep(metrics, args)
        row_with_metrics = {
            **row,
            "validation": {
                **dict(row.get("validation", {})),
                "lexical_diversity": metrics,
            },
        }
        if keep:
            kept.append(row_with_metrics)
        else:
            rejected.append({**row_with_metrics, "diversity_rejection_reasons": reasons})
            reason_counts.update(reasons)

    write_jsonl(args.out, kept)
    write_jsonl(args.rejected_out, rejected)

    lines = [
        "# Set Diversity Filter Report",
        "",
        f"- Input sets: {len(kept) + len(rejected)}",
        f"- Kept sets: {len(kept)}",
        f"- Rejected sets: {len(rejected)}",
        f"- Max mean content Jaccard: {args.max_mean_content_jaccard}",
        f"- Max pair content Jaccard: {args.max_pair_content_jaccard}",
        f"- Max repeated word fraction: {args.max_repeated_word_fraction}",
        "",
        "## Rejection Reasons",
        "",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    lines.extend(["", "## Rejected Sets", ""])
    for row in rejected[:50]:
        metrics = row["validation"]["lexical_diversity"]
        lines.append(
            "- `{}` image={} mean_jaccard={:.3f} max_jaccard={:.3f} repeated_frac={:.3f} reasons={}".format(
                row.get("set_id"),
                row.get("metadata", {}).get("image_id"),
                metrics["mean_same_language_content_jaccard"],
                metrics["max_same_language_content_jaccard"],
                metrics["repeated_content_word_fraction"],
                ",".join(row["diversity_rejection_reasons"]),
            )
        )

    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Set diversity filter summary")
    print(f"- kept: {len(kept)} / {len(kept) + len(rejected)}")
    print(f"- rejected: {len(rejected)}")
    if metric_rows:
        mean_j = sum(item["mean_same_language_content_jaccard"] for item in metric_rows) / len(metric_rows)
        max_j = max(item["max_same_language_content_jaccard"] for item in metric_rows)
        mean_rep = sum(item["repeated_content_word_fraction"] for item in metric_rows) / len(metric_rows)
        print(f"- mean content Jaccard across sets: {mean_j:.3f}")
        print(f"- max pair content Jaccard across sets: {max_j:.3f}")
        print(f"- mean repeated word fraction: {mean_rep:.3f}")
    if reason_counts:
        print("- rejection reasons:")
        for reason, count in reason_counts.most_common():
            print(f"  - {reason}: {count}")
    print(f"- report: {args.report}")


if __name__ == "__main__":
    main()
