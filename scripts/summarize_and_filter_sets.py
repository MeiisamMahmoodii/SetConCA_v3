from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.paths import resolve_project_path
from setconca_v2.set_dataset import (
    compute_set_stats,
    filter_sets_by_min_rewrites,
    load_sets,
    write_filtered_sets,
    write_stats_json,
    write_stats_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and optionally filter grouped semantic-set JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-rewrites", type=int, default=None)
    parser.add_argument("--filtered-name", default=None)
    args = parser.parse_args()

    input_path = resolve_project_path(args.input)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_set_stats(input_path)
    write_stats_json(out_dir / "set_stats.json", stats, input_path)
    write_stats_report(out_dir / "set_stats.md", stats, input_path)
    print(
        f"Stats saved to {out_dir} | sets={stats.n_sets} | rewrites={stats.n_rewrites} | "
        f"mean_rewrites={stats.mean_rewrites:.2f}"
    )

    if args.min_rewrites is not None:
        rows = load_sets(input_path)
        filtered = filter_sets_by_min_rewrites(rows, args.min_rewrites)
        name = args.filtered_name or f"sets_min{args.min_rewrites}.jsonl"
        filtered_path = out_dir / name
        write_filtered_sets(filtered_path, filtered)
        filtered_stats = compute_set_stats(filtered_path)
        write_stats_json(out_dir / f"{filtered_path.stem}_stats.json", filtered_stats, filtered_path)
        write_stats_report(out_dir / f"{filtered_path.stem}_stats.md", filtered_stats, filtered_path)
        print(
            f"Filtered sets saved to {filtered_path} | min_rewrites={args.min_rewrites} | "
            f"sets={filtered_stats.n_sets} | rewrites={filtered_stats.n_rewrites}"
        )


if __name__ == "__main__":
    main()
