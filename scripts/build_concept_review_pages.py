from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_").lower()


def concept_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["source_key"],
        row["target_key"],
        row["bridge"],
        row["target_concept_dim"],
    )


def short_key(key: str) -> str:
    parts = key.split("__")
    if len(parts) < 7:
        return key
    family = parts[1]
    size = parts[2]
    layer = parts[-2]
    return f"{family} {size} {layer}"


def clean_cell(value: str) -> str:
    return " ".join(str(value).replace("\n", " ").replace("\r", " ").split())


def write_index(out_dir: Path, pages: list[dict[str, Any]]) -> None:
    lines = [
        "# Concept Review Table",
        "",
        "Use this file as the manual review checklist. Open each concept page, read the examples, then fill in the manual fields.",
        "",
        "Suggested labels:",
        "",
        "- `semantic`: clear human-interpretable concept",
        "- `broad_topic`: real but too broad",
        "- `style_artifact`: wording/source/format artifact",
        "- `unclear`: not enough evidence",
        "",
        "| Rank | Page | Source -> Target | Target concept | Score | Manual label | Use for steering? |",
        "| ---: | --- | --- | ---: | ---: | --- | --- |",
    ]
    for page in pages:
        lines.append(
            f"| {page['rank']} | [{page['title']}]({page['file'].name}) | "
            f"{page['source_short']} -> {page['target_short']} | {page['target_dim']} | "
            f"{page['score']:.4f} |  |  |"
        )
    (out_dir / "concept_review_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_page(path: Path, summary: dict[str, str], examples: list[dict[str, str]], rank: int) -> None:
    source_short = short_key(summary["source_key"])
    target_short = short_key(summary["target_key"])
    title = f"Concept {rank:03d}: {source_short} -> {target_short} | target c{summary['target_concept_dim']}"
    target_examples = [row for row in examples if row["example_kind"] == "target_top"]
    mapped_examples = [row for row in examples if row["example_kind"] == "mapped_source_top"]

    lines = [
        f"# {title}",
        "",
        "## Manual Review",
        "",
        "Fill these in after reading the evidence.",
        "",
        "```text",
        "manual_label:",
        "short_name:",
        "confidence:",
        "use_for_steering:",
        "notes:",
        "```",
        "",
        "## Candidate Metadata",
        "",
        f"- Source: `{source_short}`",
        f"- Target: `{target_short}`",
        f"- Bridge: `{summary['bridge']}`",
        f"- Target concept dim: `{summary['target_concept_dim']}`",
        f"- Source concept dim: `{summary['source_concept_dim']}`",
        f"- Inspection score: `{float(summary['inspection_score']):.4f}`",
        f"- Alignment delta: `{float(summary['alignment_delta']):.4f}`",
        f"- Example-overlap delta: `{float(summary['top_example_overlap_delta']):.4f}`",
        f"- Pair controlled TopK: `{float(summary['pair_controlled_topk']):.4f}`",
        "",
        "## Target Top Examples",
        "",
    ]
    lines.extend(example_table(target_examples))
    lines.extend(["", "## Mapped Source Top Examples", ""])
    lines.extend(example_table(mapped_examples))
    lines.extend(
        [
            "",
            "## Decision Guide",
            "",
            "- Mark `semantic` only if the examples share a clear human concept.",
            "- Mark `broad_topic` if it is meaningful but too broad for steering.",
            "- Mark `style_artifact` if it seems driven by source, wording, length, punctuation, or dataset structure.",
            "- Mark `unclear` if the evidence is mixed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def example_table(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return ["No examples."]
    lines = [
        "| Dataset index | Label | Values mapped/target/source | Original | Rewrite samples |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        values = (
            f"{float(row['mapped_source_value']):.3f} / "
            f"{float(row['target_value']):.3f} / "
            f"{float(row['source_value']):.3f}"
        )
        lines.append(
            f"| {row['dataset_index']} | {clean_cell(row['label'])} | `{values}` | "
            f"{clean_cell(row['original_text'])} | {clean_cell(row['rewrite_samples'])} |"
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Build human-readable Markdown pages for concept review.")
    parser.add_argument("--inspection-dir", default="results/concept_inspection_llama_qwen_e25")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-concepts", type=int, default=24)
    args = parser.parse_args()

    inspection_dir = Path(args.inspection_dir)
    out_dir = Path(args.out_dir) if args.out_dir else inspection_dir / "review_pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = read_rows(inspection_dir / "concept_summary.csv")
    examples = read_rows(inspection_dir / "concept_examples.csv")
    summaries.sort(key=lambda row: float(row["inspection_score"]), reverse=True)
    selected = summaries[: args.max_concepts]
    examples_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in examples:
        examples_by_key.setdefault(concept_key(row), []).append(row)

    pages = []
    for rank, summary in enumerate(selected, start=1):
        source_short = short_key(summary["source_key"])
        target_short = short_key(summary["target_key"])
        title = f"Concept {rank:03d}"
        filename = f"{rank:03d}_{safe_name(source_short)}_to_{safe_name(target_short)}_c{summary['target_concept_dim']}.md"
        path = out_dir / filename
        key = concept_key(summary)
        write_page(path, summary, examples_by_key.get(key, []), rank)
        pages.append(
            {
                "rank": rank,
                "title": title,
                "file": path,
                "source_short": source_short,
                "target_short": target_short,
                "target_dim": summary["target_concept_dim"],
                "score": float(summary["inspection_score"]),
            }
        )
    write_index(out_dir, pages)
    print(f"Wrote {len(pages)} concept review pages to {out_dir}")


if __name__ == "__main__":
    main()
