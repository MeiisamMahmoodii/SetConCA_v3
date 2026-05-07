from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.dataset_download import format_ag_news_rows, rows_to_jsonl_dicts
from setconca_v2.io_utils import write_json, write_jsonl
from setconca_v2.paths import resolve_project_path


def download_ag_news(split: str, limit: int | None):
    from datasets import load_dataset

    ds = load_dataset("ag_news", split=split)
    return format_ag_news_rows(ds, split=split, limit=limit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a fresh V2 news dataset independent of V1.")
    parser.add_argument("--dataset", default="ag_news", choices=["ag_news"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", default=str(ROOT / "data" / "raw" / "ag_news_train.jsonl"))
    args = parser.parse_args()

    out_path = resolve_project_path(args.out)
    if args.dataset == "ag_news":
        rows = download_ag_news(args.split, args.limit)
    else:
        raise ValueError(args.dataset)

    records = rows_to_jsonl_dicts(rows)
    write_jsonl(out_path, records)
    manifest = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit,
        "n_rows": len(records),
        "out": str(out_path),
        "format": {"id": "string", "text": "string", "source": "string", "label": "string|null"},
    }
    write_json(out_path.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
