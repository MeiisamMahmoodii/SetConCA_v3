from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def load_existing_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in read_jsonl(path):
        keys.add((row["original_id"], row["model_name"], row["length_band"]))
    return keys


def group_accepted(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        item = grouped.setdefault(
            row["original_id"],
            {
                "original_id": row["original_id"],
                "original_text": row["original_text"],
                "label": row.get("label"),
                "source": row.get("source"),
                "banned_words": row["banned_words"],
                "rewrites": [],
            },
        )
        item["rewrites"].append(
            {
                "text": row["rewrite"],
                "model_name": row["model_name"],
                "model_id": row["model_id"],
                "length_band": row["length_band"],
                "word_count": row["word_count"],
                "semantic_metrics": row.get("semantic_metrics", {}),
            }
        )
    return list(grouped.values())


def write_review_table(path: Path, grouped_rows: List[Dict[str, Any]]) -> None:
    lines = [
        "# Constrained Paraphrase Set Review",
        "",
        "| Original ID | Original sentence | Banned words | Model | Length | Word count | Rewrite |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for item in grouped_rows:
        banned = ", ".join(item.get("banned_words", []))
        for rewrite in item.get("rewrites", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(item["original_id"]).replace("|", "\\|"),
                        str(item["original_text"]).replace("|", "\\|"),
                        banned.replace("|", "\\|"),
                        str(rewrite["model_name"]).replace("|", "\\|"),
                        str(rewrite["length_band"]),
                        str(rewrite["word_count"]),
                        str(rewrite["text"]).replace("|", "\\|"),
                    ]
                )
                + " |"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
