from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.paths import resolve_project_path
from setconca_v2.steering import candidate_rows_from_review, write_csv_rows, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reviewed concept manifest for causal steering probes.")
    parser.add_argument("--concept-summary", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--include-use", default="yes,yes_maybe,maybe_control")
    args = parser.parse_args()

    include_use = {item.strip() for item in args.include_use.split(",") if item.strip()}
    rows = candidate_rows_from_review(
        concept_summary_path=resolve_project_path(args.concept_summary),
        label_path=resolve_project_path(args.labels),
        include_use=include_use,
    )
    out_csv = resolve_project_path(args.out_csv)
    write_csv_rows(out_csv, rows)
    if args.out_json:
        write_json(resolve_project_path(args.out_json), rows)
    print(f"Wrote {len(rows)} steering candidate(s) to {out_csv}")


if __name__ == "__main__":
    main()

