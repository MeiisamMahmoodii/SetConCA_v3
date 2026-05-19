from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.io_utils import read_jsonl, write_json
from setconca_v2.latent_dataset_summary import rejection_reason_counts, summarize_latent_rows, write_summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a latent-set dataset.")
    parser.add_argument("--sets", required=True, help="Input latent-set JSONL.")
    parser.add_argument("--rejected", help="Optional rejected rows JSONL.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--examples", type=int, default=10, help="Number of review examples to include.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sets_path = Path(args.sets)
    out_dir = Path(args.out_dir)
    rows = list(read_jsonl(sets_path))
    rejected = list(read_jsonl(Path(args.rejected))) if args.rejected else []

    summary = summarize_latent_rows(rows)
    rejection_counts = rejection_reason_counts(rejected)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        out_dir / "summary.json",
        {
            "sets": str(sets_path),
            "rejected": str(args.rejected) if args.rejected else None,
            "summary": summary.to_dict(),
            "rejection_reason_counts": rejection_counts,
        },
    )
    report = write_summary_report(
        summary=summary,
        rejection_counts=rejection_counts,
        examples=rows[: max(0, args.examples)],
    )
    (out_dir / "summary_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()

