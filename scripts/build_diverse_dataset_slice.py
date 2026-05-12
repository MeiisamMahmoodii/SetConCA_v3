from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_balanced_indices(
    rows: list[dict[str, Any]],
    *,
    total: int,
    labels: list[str] | None,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    by_label: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        label = str(row.get("label", ""))
        if labels is None or label in labels:
            by_label[label].append(idx)
    selected_labels = labels or sorted(by_label)
    if not selected_labels:
        raise ValueError("No labels available for selection.")
    base = total // len(selected_labels)
    remainder = total % len(selected_labels)
    selected: list[int] = []
    leftovers: list[int] = []
    for label_pos, label in enumerate(selected_labels):
        candidates = list(by_label.get(label, []))
        rng.shuffle(candidates)
        want = base + (1 if label_pos < remainder else 0)
        take = min(want, len(candidates))
        selected.extend(candidates[:take])
        leftovers.extend(candidates[take:])
    if len(selected) < total:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: total - len(selected)])
    selected = selected[:total]
    selected.sort()
    return selected


def copy_subset_activation_bank(src: Path, dst: Path, indices: list[int]) -> None:
    payload = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "hidden" not in payload:
        raise ValueError(f"Unexpected activation bank format: {src}")
    idx = torch.tensor(indices, dtype=torch.long)
    out = dict(payload)
    out["hidden"] = payload["hidden"].index_select(0, idx).contiguous()
    for key in ["texts", "view_texts", "set_ids", "labels", "sources", "rewrite_meta"]:
        value = payload.get(key)
        if isinstance(value, list) and len(value) >= max(indices) + 1:
            out[key] = [value[i] for i in indices]
    meta = dict(payload.get("meta") or {})
    meta["diverse_subset_source"] = str(src)
    meta["diverse_subset_n"] = len(indices)
    meta["diverse_subset_indices"] = indices
    out["meta"] = meta
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)


def build_activation_subset(src_root: Path, dst_root: Path, indices: list[int]) -> int:
    count = 0
    for src in src_root.rglob("activation_bank.pt"):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        copy_subset_activation_bank(src, dst, indices)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a balanced/diverse slice of a SetConCA dataset and activation root.")
    parser.add_argument("--dataset", default="data/generated/server_4gpu_2000/merged/sets_min16.jsonl")
    parser.add_argument("--activation-root", default="data/activations/model_grid_s16_min16_4A100")
    parser.add_argument("--out-dataset-dir", default="data/generated/server_4gpu_2000/diverse_s16_300")
    parser.add_argument("--out-activation-root", default="data/activations/model_grid_s16_min16_diverse300_4A100")
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--labels", default="business,sports,world,science_technology")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-activations", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    activation_root = Path(args.activation_root)
    out_dataset_dir = Path(args.out_dataset_dir)
    out_activation_root = Path(args.out_activation_root)
    labels = [x.strip() for x in args.labels.split(",") if x.strip()] if args.labels else None

    rows = load_jsonl(dataset_path)
    indices = select_balanced_indices(rows, total=args.total, labels=labels, seed=args.seed)
    subset = [rows[i] for i in indices]
    label_counts = Counter(row.get("label", "") for row in subset)
    out_dataset = out_dataset_dir / "sets_min16_diverse300.jsonl"
    write_jsonl(out_dataset, subset)
    index_rows = [
        {
            "new_index": new_idx,
            "old_index": old_idx,
            "label": rows[old_idx].get("label", ""),
            "original_id": rows[old_idx].get("original_id", ""),
            "original_text": rows[old_idx].get("original_text", ""),
        }
        for new_idx, old_idx in enumerate(indices)
    ]
    write_csv(out_dataset_dir / "selected_indices.csv", index_rows)
    (out_dataset_dir / "selected_indices.json").write_text(json.dumps(indices, indent=2), encoding="utf-8")
    stats = {
        "source_dataset": str(dataset_path),
        "source_activation_root": str(activation_root),
        "out_dataset": str(out_dataset),
        "out_activation_root": str(out_activation_root),
        "n_source_sets": len(rows),
        "n_selected_sets": len(subset),
        "seed": args.seed,
        "label_counts": dict(label_counts),
        "source_label_counts": dict(Counter(row.get("label", "") for row in rows)),
        "first_300_source_label_counts": dict(Counter(row.get("label", "") for row in rows[:300])),
    }
    if args.skip_activations:
        stats["activation_banks_written"] = 0
    else:
        if out_activation_root.exists():
            shutil.rmtree(out_activation_root)
        stats["activation_banks_written"] = build_activation_subset(activation_root, out_activation_root, indices)
    (out_dataset_dir / "diverse_slice_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    md = [
        "# Diverse S16 Dataset Slice",
        "",
        f"- Source dataset: `{dataset_path}`",
        f"- Output dataset: `{out_dataset}`",
        f"- Source activation root: `{activation_root}`",
        f"- Output activation root: `{out_activation_root}`",
        f"- Selected sets: `{len(subset)}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "| --- | ---: |",
    ]
    for label, count in sorted(label_counts.items()):
        md.append(f"| `{label}` | {count} |")
    md.extend(
        [
            "",
            "## Why This Exists",
            "",
            "The previous `--max-sets 300` run used the first 300 sets from `sets_min16.jsonl`, which was not label-balanced. This slice reorders the dataset and activation banks so existing training scripts can use a more diverse first 300 examples.",
        ]
    )
    (out_dataset_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
