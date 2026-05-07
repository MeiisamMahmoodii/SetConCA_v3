from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io_utils import read_jsonl, write_json, write_jsonl


@dataclass
class SetDatasetStats:
    n_sets: int
    n_rewrites: int
    min_rewrites: int
    max_rewrites: int
    mean_rewrites: float
    rewrite_count_histogram: Dict[str, int]
    sets_at_least: Dict[str, int]
    label_counts: Dict[str, int]
    model_counts: Dict[str, int]
    length_band_counts: Dict[str, int]
    sha256: str


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sets(path: str | Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(Path(path)))


def compute_set_stats(path: str | Path, thresholds: Iterable[int] = (2, 4, 8, 16, 24, 32, 40)) -> SetDatasetStats:
    path = Path(path)
    rows = load_sets(path)
    rewrite_counts = [len(row.get("rewrites", [])) for row in rows]
    label_counts = Counter(str(row.get("label")) for row in rows)
    model_counts = Counter()
    length_band_counts = Counter()
    for row in rows:
        for rewrite in row.get("rewrites", []):
            model_counts[str(rewrite.get("model_name"))] += 1
            length_band_counts[str(rewrite.get("length_band"))] += 1

    n_sets = len(rows)
    n_rewrites = sum(rewrite_counts)
    min_rewrites = min(rewrite_counts) if rewrite_counts else 0
    max_rewrites = max(rewrite_counts) if rewrite_counts else 0
    mean_rewrites = n_rewrites / n_sets if n_sets else 0.0
    hist = Counter(rewrite_counts)
    return SetDatasetStats(
        n_sets=n_sets,
        n_rewrites=n_rewrites,
        min_rewrites=min_rewrites,
        max_rewrites=max_rewrites,
        mean_rewrites=mean_rewrites,
        rewrite_count_histogram={str(k): hist[k] for k in sorted(hist)},
        sets_at_least={str(k): sum(count >= k for count in rewrite_counts) for k in thresholds},
        label_counts=dict(sorted(label_counts.items())),
        model_counts=dict(model_counts.most_common()),
        length_band_counts=dict(sorted(length_band_counts.items())),
        sha256=file_sha256(path),
    )


def filter_sets_by_min_rewrites(rows: Iterable[Dict[str, Any]], min_rewrites: int) -> List[Dict[str, Any]]:
    return [row for row in rows if len(row.get("rewrites", [])) >= min_rewrites]


def write_stats_report(path: str | Path, stats: SetDatasetStats, source_path: str | Path) -> None:
    lines = [
        "# Set Dataset Statistics",
        "",
        f"Source: `{source_path}`",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Sets | {stats.n_sets} |",
        f"| Rewrites | {stats.n_rewrites} |",
        f"| Min rewrites per set | {stats.min_rewrites} |",
        f"| Max rewrites per set | {stats.max_rewrites} |",
        f"| Mean rewrites per set | {stats.mean_rewrites:.4f} |",
        f"| SHA256 | `{stats.sha256}` |",
        "",
        "## Sets At Least N Rewrites",
        "",
        "| Threshold | Sets |",
        "| ---: | ---: |",
    ]
    for threshold, count in stats.sets_at_least.items():
        lines.append(f"| {threshold} | {count} |")

    lines.extend(["", "## Rewrite Count Histogram", "", "| Rewrites | Sets |", "| ---: | ---: |"])
    for count, n_sets in stats.rewrite_count_histogram.items():
        lines.append(f"| {count} | {n_sets} |")

    lines.extend(["", "## Model Counts", "", "| Model | Accepted rewrites |", "| --- | ---: |"])
    for model, count in stats.model_counts.items():
        lines.append(f"| {model} | {count} |")

    lines.extend(["", "## Length Band Counts", "", "| Length band | Accepted rewrites |", "| --- | ---: |"])
    for band, count in stats.length_band_counts.items():
        lines.append(f"| {band} | {count} |")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stats_json(path: str | Path, stats: SetDatasetStats, source_path: str | Path) -> None:
    write_json(
        Path(path),
        {
            "source": str(source_path),
            **stats.__dict__,
        },
    )


def write_filtered_sets(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    write_jsonl(Path(path), rows)
