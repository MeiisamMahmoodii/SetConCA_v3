from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Runs must use LABEL=PATH format.")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("Run label cannot be empty.")
    return label, Path(path)


def read_csv(path: Path) -> list[dict[str, str]]:
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


def pretty_method(method: str) -> str:
    return {"setconca": "SetConCA", "pointwise_topk": "Pointwise TopK"}.get(method, method)


def pretty_bridge(bridge: str) -> str:
    return {"procrustes": "Procrustes", "ridge": "Ridge", "identity": "Identity"}.get(bridge, bridge)


def load_run(label: str, run_dir: Path) -> dict[str, Any]:
    summary_dir = run_dir / "summaries"
    if not summary_dir.exists():
        raise FileNotFoundError(f"Missing summaries directory: {summary_dir}")
    run_summary_path = run_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.exists() else {}
    return {
        "label": label,
        "path": run_dir,
        "run_summary": run_summary,
        "bridge": read_csv(summary_dir / "bridge_method_summary.csv"),
        "set_size": read_csv(summary_dir / "set_size_summary.csv"),
        "relation": read_csv(summary_dir / "method_relation_summary.csv"),
    }


def collect_bridge_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for row in run["bridge"]:
            rows.append(
                {
                    "run": run["label"],
                    "method": row["method"],
                    "bridge": row["bridge"],
                    "n": int(row["n"]),
                    "raw_topk": float(row["raw_topk"]),
                    "shuffled_topk": float(row["shuffled_topk"]),
                    "real_minus_shuffled_topk": float(row["real_minus_shuffled_topk"]),
                    "train_test_topk_gap": float(row["train_test_topk_gap"]),
                    "transfer_mse": float(row["transfer_mse"]),
                    "cosine": float(row["cosine"]),
                }
            )
    return rows


def collect_set_size_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        for row in run["set_size"]:
            rows.append(
                {
                    "run": run["label"],
                    "method": row["method"],
                    "set_size": int(row["set_size"]),
                    "bridge": row["bridge"],
                    "real_minus_shuffled_topk": float(row["real_minus_shuffled_topk"]),
                }
            )
    return rows


def plot_epoch_bridge(out_dir: Path, rows: list[dict[str, Any]], run_order: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    filtered = [row for row in rows if row["bridge"] in {"procrustes", "ridge"}]
    groups = [
        ("setconca", "procrustes"),
        ("setconca", "ridge"),
        ("pointwise_topk", "procrustes"),
        ("pointwise_topk", "ridge"),
    ]
    colors = {
        ("setconca", "procrustes"): "#66c2a5",
        ("setconca", "ridge"): "#1b9e77",
        ("pointwise_topk", "procrustes"): "#8da0cb",
        ("pointwise_topk", "ridge"): "#4c78a8",
    }
    markers = {"setconca": "o", "pointwise_topk": "s"}
    linestyles = {"setconca": "-", "pointwise_topk": "--"}
    lookup = {(row["run"], row["method"], row["bridge"]): row["real_minus_shuffled_topk"] for row in filtered}
    xs = list(range(len(run_order)))
    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    for method, bridge in groups:
        ys = [lookup.get((run, method, bridge), float("nan")) for run in run_order]
        ax.plot(
            xs,
            ys,
            marker=markers[method],
            linestyle=linestyles[method],
            linewidth=2.4,
            color=colors[(method, bridge)],
            label=f"{pretty_method(method)} + {pretty_bridge(bridge)}",
        )
    ax.set_xticks(xs, run_order)
    ax.set_xlabel("Training run")
    ax.set_ylabel("Controlled TopK overlap\n(real minus shuffled)")
    ax.set_title("Epoch Sweep: Controlled Bridge Signal", fontsize=15, pad=12)
    ax.grid(axis="y", color="#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.savefig(fig_dir / "epoch_bridge_controlled.png", dpi=220)
    plt.close(fig)


def plot_setconca_set_size(out_dir: Path, rows: list[dict[str, Any]], run_order: list[str]) -> None:
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    filtered = [row for row in rows if row["method"] == "setconca" and row["bridge"] == "ridge"]
    lookup = {(row["run"], row["set_size"]): row["real_minus_shuffled_topk"] for row in filtered}
    set_sizes = sorted({row["set_size"] for row in filtered})
    colors = ["#1b9e77", "#4c78a8", "#984ea3", "#ff7f00"]
    fig, ax = plt.subplots(figsize=(9.5, 5.3), constrained_layout=True)
    for idx, run in enumerate(run_order):
        ys = [lookup.get((run, set_size), float("nan")) for set_size in set_sizes]
        ax.plot(set_sizes, ys, marker="o", linewidth=2.4, color=colors[idx % len(colors)], label=run)
    ax.set_xlabel("Number of paraphrases per semantic set")
    ax.set_ylabel("Controlled TopK overlap\n(real minus shuffled)")
    ax.set_title("SetConCA + Ridge: Set-Size Trend Across Epochs", fontsize=15, pad=12)
    ax.grid(axis="y", color="#dddddd")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.savefig(fig_dir / "epoch_setconca_ridge_set_size.png", dpi=220)
    plt.close(fig)


def write_report(out_dir: Path, runs: list[dict[str, Any]], bridge_rows: list[dict[str, Any]]) -> None:
    def score(run: str, method: str, bridge: str) -> float:
        for row in bridge_rows:
            if row["run"] == run and row["method"] == method and row["bridge"] == bridge:
                return float(row["real_minus_shuffled_topk"])
        return float("nan")

    run_order = [run["label"] for run in runs]
    lines = [
        "# Transfer Run Epoch Comparison",
        "",
        "## Scope",
        "",
        "Compared completed transfer runs using their existing summary tables. No model training was performed by this comparison script.",
        "",
        "| Run | Path | Trained models | Result rows |",
        "| --- | --- | ---: | ---: |",
    ]
    for run in runs:
        summary = run["run_summary"]
        lines.append(
            f"| `{run['label']}` | `{run['path']}` | {summary.get('n_trained', '')} | {summary.get('n_result_rows', '')} |"
        )
    lines.extend(
        [
            "",
            "## Main Scores",
            "",
            "| Run | SetConCA + Procrustes | SetConCA + Ridge | Pointwise TopK + Procrustes | Pointwise TopK + Ridge |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in run_order:
        lines.append(
            f"| `{run}` | {score(run, 'setconca', 'procrustes'):.4f} | {score(run, 'setconca', 'ridge'):.4f} | "
            f"{score(run, 'pointwise_topk', 'procrustes'):.4f} | {score(run, 'pointwise_topk', 'ridge'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher epochs improved the pointwise TopK baseline on the controlled bridge metric.",
            "- Higher epochs did not improve SetConCA in this sweep; the 25-epoch run remains strongest for SetConCA ridge/procrustes.",
            "- This suggests we should not assume lower training loss means better transferable concepts.",
            "- For the next concept-extraction step, keep the 25-epoch SetConCA run as the main candidate unless a new hyperparameter sweep reverses this result.",
            "",
            "## Artifacts",
            "",
            "- `bridge_epoch_comparison.csv`",
            "- `set_size_epoch_comparison.csv`",
            "- `figures/epoch_bridge_controlled.png`",
            "- `figures/epoch_setconca_ridge_set_size.png`",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed transfer/steering summary runs.")
    parser.add_argument("--run", action="append", type=parse_run, required=True, help="Run in LABEL=PATH format.")
    parser.add_argument("--out-dir", required=True, help="Directory for comparison CSVs, figures, and report.")
    args = parser.parse_args()

    runs = [load_run(label, path) for label, path in args.run]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_rows = collect_bridge_rows(runs)
    set_size_rows = collect_set_size_rows(runs)
    write_rows(out_dir / "bridge_epoch_comparison.csv", bridge_rows)
    write_rows(out_dir / "set_size_epoch_comparison.csv", set_size_rows)
    run_order = [label for label, _ in args.run]
    plot_epoch_bridge(out_dir, bridge_rows, run_order)
    plot_setconca_set_size(out_dir, set_size_rows, run_order)
    write_report(out_dir, runs, bridge_rows)
    print(f"Wrote epoch comparison to {out_dir}")


if __name__ == "__main__":
    main()
