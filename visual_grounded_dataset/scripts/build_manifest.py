from __future__ import annotations

import argparse
from pathlib import Path

from vg_common import normalize_tags, read_csv_rows, write_jsonl


def build_manifest_rows(csv_path: Path) -> list[dict[str, object]]:
    rows = []
    for idx, row in enumerate(read_csv_rows(csv_path)):
        image_id = (row.get("image_id") or f"image_{idx:08d}").strip()
        image_path = (row.get("image_path") or row.get("url") or "").strip()
        if not image_path:
            raise ValueError(f"row {idx} is missing image_path or url")
        rows.append(
            {
                "image_id": image_id,
                "image_path": image_path,
                "source_dataset": (row.get("source_dataset") or "unknown").strip(),
                "topic_tags": normalize_tags(row.get("topic_tags")),
                "metadata": {key: value for key, value in row.items() if key not in {"image_id", "image_path", "url", "source_dataset", "topic_tags"}},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual-grounded image manifest from a CSV file.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows = build_manifest_rows(Path(args.input_csv))
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} image manifest rows to {args.out}")


if __name__ == "__main__":
    main()

