from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


@dataclass(frozen=True)
class SteeringCandidate:
    rank: int
    candidate: str
    label: str
    use_for_steering: str
    source_key: str
    target_key: str
    source_family: str
    target_family: str
    source_size: str
    target_size: str
    source_layer: str
    target_layer: str
    bridge: str
    pair_controlled_topk: float
    target_concept_dim: int
    source_concept_dim: int
    inspection_score: float
    keywords: list[str]
    notes: str = ""


DEFAULT_KEYWORDS = {
    "google ipo": ["google", "ipo", "offering", "shares", "stock", "sec", "bid"],
    "windows": ["microsoft", "windows", "xp", "security", "update", "software", "service pack"],
    "stock market": ["stock", "stocks", "market", "earnings", "prices", "shares", "profit"],
    "corporate earnings": ["earnings", "profit", "revenue", "shares", "stock", "company"],
    "sports": ["olympic", "sports", "soccer", "tennis", "medal", "athlete", "team"],
}


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def keyword_guess(candidate: str, label: str) -> list[str]:
    haystack = f"{candidate} {label}".lower()
    for marker, keywords in DEFAULT_KEYWORDS.items():
        if marker in haystack:
            return keywords
    words = [w.strip(" /_-").lower() for w in candidate.split() if len(w.strip(" /_-")) > 3]
    return sorted(set(words))[:8]


def parse_keywords(text: str) -> list[str]:
    return [item.strip().lower() for item in str(text).split(";") if item.strip()]


def keyword_score(text: str, keywords: Iterable[str]) -> tuple[int, list[str]]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        k = keyword.lower().strip()
        if k and k in lowered:
            hits.append(k)
    return len(hits), hits


def concept_sign_from_codes(codes_path: str | Path, dim: int) -> float:
    payload = torch.load(codes_path, map_location="cpu", weights_only=False)
    z = torch.cat([payload["train_z"], payload["test_z"]], dim=0).float()
    vals = z[:, dim]
    active = vals[vals.ne(0)]
    if len(active) == 0:
        return 1.0
    return 1.0 if float(active.mean()) >= 0 else -1.0


def model_dir_from_key(run_dir: str | Path, key: str) -> Path:
    parts = key.split("__")
    if len(parts) < 7:
        raise ValueError(f"Unexpected model key format: {key}")
    method = parts[0]
    family = parts[1]
    size = parts[2]
    model_slug = "__".join(parts[3:-2])
    layer_name = parts[-2]
    set_token = parts[-1]
    return Path(run_dir) / "models" / method / family / size / model_slug / layer_name / set_token


def candidate_rows_from_review(
    *,
    concept_summary_path: str | Path,
    label_path: str | Path,
    include_use: set[str] | None = None,
) -> list[dict[str, Any]]:
    concept_rows = read_csv_rows(concept_summary_path)
    label_rows = read_csv_rows(label_path)
    by_rank = {int(row["rank"]): row for row in label_rows}
    include_use = include_use or {"yes", "yes_maybe", "maybe_control"}
    out = []
    for idx, concept in enumerate(concept_rows, start=1):
        label = by_rank.get(idx)
        if not label:
            continue
        use_value = label.get("use_for_steering", "")
        if use_value not in include_use:
            continue
        keywords = parse_keywords(label.get("keywords", ""))
        if not keywords:
            keywords = keyword_guess(label.get("candidate", ""), label.get("label", ""))
        out.append(
            {
                "rank": idx,
                "candidate": label.get("candidate", ""),
                "label": label.get("label", ""),
                "use_for_steering": use_value,
                "source_key": concept["source_key"],
                "target_key": concept["target_key"],
                "source_family": concept["source_family"],
                "target_family": concept["target_family"],
                "source_size": concept["source_size"],
                "target_size": concept["target_size"],
                "source_layer": concept["source_layer"],
                "target_layer": concept["target_layer"],
                "bridge": concept["bridge"],
                "pair_controlled_topk": float(concept["pair_controlled_topk"]),
                "target_concept_dim": int(concept["target_concept_dim"]),
                "source_concept_dim": int(concept["source_concept_dim"]),
                "inspection_score": float(concept["inspection_score"]),
                "keywords": ";".join(keywords),
                "notes": label.get("notes", ""),
            }
        )
    return out


def load_steering_candidates(path: str | Path) -> list[SteeringCandidate]:
    candidates = []
    for row in read_csv_rows(path):
        candidates.append(
            SteeringCandidate(
                rank=int(row["rank"]),
                candidate=row["candidate"],
                label=row["label"],
                use_for_steering=row["use_for_steering"],
                source_key=row["source_key"],
                target_key=row["target_key"],
                source_family=row["source_family"],
                target_family=row["target_family"],
                source_size=row["source_size"],
                target_size=row["target_size"],
                source_layer=row["source_layer"],
                target_layer=row["target_layer"],
                bridge=row["bridge"],
                pair_controlled_topk=float(row["pair_controlled_topk"]),
                target_concept_dim=int(row["target_concept_dim"]),
                source_concept_dim=int(row["source_concept_dim"]),
                inspection_score=float(row["inspection_score"]),
                keywords=parse_keywords(row.get("keywords", "")),
                notes=row.get("notes", ""),
            )
        )
    return candidates


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

