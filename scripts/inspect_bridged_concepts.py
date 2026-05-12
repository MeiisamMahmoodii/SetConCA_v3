from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_transfer_steering_grid import apply_bridge, fit_bridge


@dataclass(frozen=True)
class CodeBank:
    key: str
    family: str
    size: str
    model_id: str
    layer_name: str
    layer: int
    set_size: int
    codes_path: Path
    train_z: torch.Tensor
    test_z: torch.Tensor


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def short_text(text: str, limit: int = 180) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def sample_rewrites(row: dict[str, Any], n: int = 3) -> str:
    rewrites = row.get("rewrites", [])[:n]
    return " | ".join(short_text(item.get("text", ""), 120) for item in rewrites)


def model_slug_from_key(key: str) -> str:
    parts = key.split("__")
    if len(parts) < 6:
        return key
    return "__".join(parts[3:-2])


def load_code_bank(run_dir: Path, key: str) -> CodeBank:
    parts = key.split("__")
    if len(parts) < 7:
        raise ValueError(f"Unexpected model key format: {key}")
    method, family, size, model_slug, layer_name, set_token = (
        parts[0],
        parts[1],
        parts[2],
        model_slug_from_key(key),
        parts[-2],
        parts[-1],
    )
    if method != "setconca":
        raise ValueError(f"Concept inspection currently expects SetConCA keys, got: {method}")
    codes_path = run_dir / "models" / method / family / size / model_slug / layer_name / set_token / "codes.pt"
    metrics_path = codes_path.with_name("metrics.json")
    if not codes_path.exists():
        raise FileNotFoundError(codes_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = metrics["manifest"]
    payload = torch.load(codes_path, map_location="cpu")
    return CodeBank(
        key=key,
        family=family,
        size=size,
        model_id=manifest["bank"]["model_id"],
        layer_name=layer_name,
        layer=int(manifest["bank"]["layer"]),
        set_size=int(manifest["set_size"]),
        codes_path=codes_path,
        train_z=payload["train_z"].float(),
        test_z=payload["test_z"].float(),
    )


def candidate_pairs(
    transfer_rows: list[dict[str, str]],
    *,
    bridge: str,
    set_size: int,
    source_depth: str | None,
    target_depth: str | None,
    max_pairs: int,
) -> list[dict[str, str]]:
    rows = []
    for row in transfer_rows:
        if row.get("method") != "setconca":
            continue
        if row.get("bridge") != bridge:
            continue
        if int(row["set_size"]) != set_size:
            continue
        if float(row["steering_alpha"]) != 0.0:
            continue
        if source_depth and str(row.get("source_depth_pct")) != source_depth:
            continue
        if target_depth and str(row.get("target_depth_pct")) != target_depth:
            continue
        rows.append(row)
    rows.sort(key=lambda r: float(r["real_minus_shuffled_topk"]), reverse=True)
    return rows[:max_pairs]


def abs_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if float(a.abs().sum()) == 0.0 or float(b.abs().sum()) == 0.0:
        return 0.0
    return float((F.normalize(a.float(), dim=0) * F.normalize(b.float(), dim=0)).sum().abs())


def top_index_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    k = min(k, len(a), len(b))
    if k <= 0:
        return 0.0
    ia = set(torch.topk(a.abs(), k).indices.tolist())
    ib = set(torch.topk(b.abs(), k).indices.tolist())
    return len(ia & ib) / k


def inspect_pair(
    row: dict[str, str],
    source: CodeBank,
    target: CodeBank,
    *,
    bridge_method: str,
    ridge_alpha: float,
    top_concepts: int,
    example_top_k: int,
    max_examples: int,
    dataset_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    n_train = min(len(source.train_z), len(target.train_z))
    n_test = min(len(source.test_z), len(target.test_z))
    bridge = fit_bridge(
        source.train_z[:n_train],
        target.train_z[:n_train],
        bridge_method,
        ridge_alpha=ridge_alpha,
        mlp_epochs=0,
    )
    mapped = apply_bridge(source.test_z[:n_test], bridge)
    target_z = target.test_z[:n_test]
    source_z = source.test_z[:n_test]
    matrix = bridge.get("matrix")
    train_count = len(source.train_z)

    concept_rows = []
    example_rows = []
    shuffled = target_z[torch.randperm(n_test)]
    for target_dim in range(target_z.shape[1]):
        mapped_dim = mapped[:, target_dim]
        target_dim_values = target_z[:, target_dim]
        real_alignment = abs_cosine(mapped_dim, target_dim_values)
        shuffled_alignment = abs_cosine(mapped_dim, shuffled[:, target_dim])
        overlap = top_index_overlap(mapped_dim, target_dim_values, example_top_k)
        shuffled_overlap = top_index_overlap(mapped_dim, shuffled[:, target_dim], example_top_k)
        source_dim = -1
        bridge_weight = math.nan
        if matrix is not None:
            source_dim = int(torch.argmax(matrix[:, target_dim].abs()).item())
            bridge_weight = float(matrix[source_dim, target_dim])
        source_freq = float(source_z[:, source_dim].ne(0).float().mean()) if source_dim >= 0 else math.nan
        target_freq = float(target_dim_values.ne(0).float().mean())
        score = (real_alignment - shuffled_alignment) + (overlap - shuffled_overlap)
        concept_rows.append(
            {
                "source_key": source.key,
                "target_key": target.key,
                "source_family": source.family,
                "target_family": target.family,
                "source_size": source.size,
                "target_size": target.size,
                "source_layer": source.layer_name,
                "target_layer": target.layer_name,
                "bridge": bridge_method,
                "pair_controlled_topk": float(row["real_minus_shuffled_topk"]),
                "target_concept_dim": target_dim,
                "source_concept_dim": source_dim,
                "bridge_weight": bridge_weight,
                "target_active_freq": target_freq,
                "source_active_freq": source_freq,
                "alignment": real_alignment,
                "shuffled_alignment": shuffled_alignment,
                "alignment_delta": real_alignment - shuffled_alignment,
                "top_example_overlap": overlap,
                "shuffled_top_example_overlap": shuffled_overlap,
                "top_example_overlap_delta": overlap - shuffled_overlap,
                "inspection_score": score,
            }
        )

    concept_rows.sort(key=lambda r: r["inspection_score"], reverse=True)
    selected = concept_rows[:top_concepts]
    for rank, concept in enumerate(selected, start=1):
        concept["concept_rank_in_pair"] = rank
        target_dim = int(concept["target_concept_dim"])
        source_dim = int(concept["source_concept_dim"])
        mapped_dim = mapped[:, target_dim]
        target_dim_values = target_z[:, target_dim]
        example_ids = []
        for kind, values in [("target_top", target_dim_values), ("mapped_source_top", mapped_dim)]:
            for local_idx in torch.topk(values.abs(), min(max_examples, n_test)).indices.tolist():
                global_idx = train_count + local_idx
                if global_idx >= len(dataset_rows):
                    continue
                data = dataset_rows[global_idx]
                example_ids.append((kind, local_idx, global_idx, data))
        seen = set()
        for kind, local_idx, global_idx, data in example_ids:
            key = (kind, global_idx)
            if key in seen:
                continue
            seen.add(key)
            example_rows.append(
                {
                    "source_key": source.key,
                    "target_key": target.key,
                    "bridge": bridge_method,
                    "concept_rank_in_pair": rank,
                    "target_concept_dim": target_dim,
                    "source_concept_dim": source_dim,
                    "example_kind": kind,
                    "local_test_index": local_idx,
                    "dataset_index": global_idx,
                    "original_id": data.get("original_id", ""),
                    "label": data.get("label", ""),
                    "mapped_source_value": float(mapped_dim[local_idx]),
                    "target_value": float(target_dim_values[local_idx]),
                    "source_value": float(source_z[local_idx, source_dim]) if source_dim >= 0 else math.nan,
                    "original_text": short_text(data.get("original_text", ""), 260),
                    "rewrite_samples": sample_rewrites(data),
                }
            )
    return selected, example_rows


def plot_summary(out_dir: Path, concept_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    top_rows = sorted(concept_rows, key=lambda r: r["inspection_score"], reverse=True)[:14]
    labels = [
        f"{r['source_family']} {r['source_size']} -> {r['target_family']} {r['target_size']} | c{r['target_concept_dim']}"
        for r in top_rows
    ]
    values = [float(r["inspection_score"]) for r in top_rows]
    y = list(range(len(values)))
    fig, ax = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    bars = ax.barh(y, values, color="#1b9e77")
    ax.set_yticks(y, labels, fontsize=9)
    ax.invert_yaxis()
    for bar in bars:
        width = float(bar.get_width())
        ax.text(width + 0.015, bar.get_y() + bar.get_height() / 2, f"{width:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Concept inspection score")
    ax.set_title("Top Bridged Concept Candidates", fontsize=15, pad=12)
    ax.grid(axis="x", color="#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "top_bridged_concepts.png", dpi=220)
    plt.close(fig)

    pair_scores: dict[str, list[float]] = {}
    for row in concept_rows:
        label = f"{row['source_family']} {row['source_size']} -> {row['target_family']} {row['target_size']}"
        pair_scores.setdefault(label, []).append(float(row["inspection_score"]))
    labels = sorted(pair_scores)
    means = [sum(pair_scores[label]) / len(pair_scores[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    ax.barh(range(len(labels)), means, color="#4c78a8")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Mean top-concept inspection score")
    ax.set_title("Concept Candidate Quality By Model Pair", fontsize=15, pad=12)
    ax.grid(axis="x", color="#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.savefig(fig_dir / "pair_concept_scores.png", dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, concept_rows: list[dict[str, Any]], example_rows: list[dict[str, Any]], pairs: list[dict[str, str]]) -> None:
    top = sorted(concept_rows, key=lambda r: r["inspection_score"], reverse=True)[:12]
    lines = [
        "# Bridged Concept Inspection",
        "",
        "## Scope",
        "",
        "This report inspects saved SetConCA concept codes from an already-completed transfer run. It does not retrain models and does not run the original LLMs.",
        "",
        f"- Candidate pairs inspected: {len(pairs)}",
        f"- Concept rows written: {len(concept_rows)}",
        f"- Example rows written: {len(example_rows)}",
        "",
        "## How To Read The Scores",
        "",
        "`inspection_score` combines held-out per-dimension alignment and top-example overlap after subtracting shuffled controls. It is a ranking aid for manual inspection, not yet proof of a human semantic concept.",
        "",
        "## Top Candidates",
        "",
        "| Rank | Source -> Target | Target concept | Source concept | Score | Alignment delta | Example-overlap delta |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(top, start=1):
        pair = f"{row['source_family']} {row['source_size']} -> {row['target_family']} {row['target_size']}"
        lines.append(
            f"| {idx} | {pair} | {row['target_concept_dim']} | {row['source_concept_dim']} | "
            f"{row['inspection_score']:.4f} | {row['alignment_delta']:.4f} | {row['top_example_overlap_delta']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `concept_summary.csv`",
            "- `concept_examples.csv`",
            "- `figures/top_bridged_concepts.png`",
            "- `figures/pair_concept_scores.png`",
            "",
            "## Caveat",
            "",
            "The next step is manual reading of `concept_examples.csv`. A high score means a bridged dimension is stable under this diagnostic; it does not guarantee that the dimension is a clean semantic concept.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect bridged SetConCA concept dimensions.")
    parser.add_argument("--run-dir", default="results/llama_qwen_set_vs_pointwise_linear_seed0")
    parser.add_argument("--dataset", default="data/generated/server_4gpu_2000/merged/sets_min16.jsonl")
    parser.add_argument("--out-dir", default="results/concept_inspection_llama_qwen_e25")
    parser.add_argument("--bridge", default="ridge", choices=["ridge", "procrustes"])
    parser.add_argument("--set-size", type=int, default=16)
    parser.add_argument("--source-depth", default="60")
    parser.add_argument("--target-depth", default="60")
    parser.add_argument("--max-pairs", type=int, default=8)
    parser.add_argument("--top-concepts", type=int, default=8)
    parser.add_argument("--example-top-k", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    max_sets = manifest["args"].get("max_sets")
    dataset_rows = load_jsonl(Path(args.dataset), limit=max_sets)
    transfer_rows = read_rows(run_dir / "transfer_steering_results.csv")
    pairs = candidate_pairs(
        transfer_rows,
        bridge=args.bridge,
        set_size=args.set_size,
        source_depth=args.source_depth,
        target_depth=args.target_depth,
        max_pairs=args.max_pairs,
    )
    if not pairs:
        raise RuntimeError("No candidate pairs matched the requested filters.")

    cache: dict[str, CodeBank] = {}
    all_concepts: list[dict[str, Any]] = []
    all_examples: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        src_key = pair["source_key"]
        tgt_key = pair["target_key"]
        cache.setdefault(src_key, load_code_bank(run_dir, src_key))
        cache.setdefault(tgt_key, load_code_bank(run_dir, tgt_key))
        concepts, examples = inspect_pair(
            pair,
            cache[src_key],
            cache[tgt_key],
            bridge_method=args.bridge,
            ridge_alpha=args.ridge_alpha,
            top_concepts=args.top_concepts,
            example_top_k=args.example_top_k,
            max_examples=args.max_examples,
            dataset_rows=dataset_rows,
        )
        print(f"[{idx}/{len(pairs)}] inspected {src_key} -> {tgt_key}: {len(concepts)} concepts")
        all_concepts.extend(concepts)
        all_examples.extend(examples)

    all_concepts.sort(key=lambda r: r["inspection_score"], reverse=True)
    write_rows(out_dir / "concept_summary.csv", all_concepts)
    write_rows(out_dir / "concept_examples.csv", all_examples)
    (out_dir / "selected_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    plot_summary(out_dir, all_concepts)
    write_report(out_dir, all_concepts, all_examples, pairs)
    print(f"Wrote concept inspection to {out_dir}")


if __name__ == "__main__":
    main()
