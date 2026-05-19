from __future__ import annotations

import argparse
from pathlib import Path

from vg_common import stable_hash, write_jsonl


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_image_id(path: Path) -> str:
    stem = path.stem.strip()
    return stem or stable_hash(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an image manifest by scanning a local image folder.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-dataset", default="open_images_v7_train_partial")
    parser.add_argument("--topic-tags", default="open_images_train_partial")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    iterator = image_dir.rglob("*") if args.recursive else image_dir.iterdir()
    rows = []
    tags = [tag.strip() for tag in args.topic_tags.split("|") if tag.strip()]
    for path in sorted(iterator):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rows.append(
            {
                "image_id": infer_image_id(path),
                "image_path": str(path.resolve()),
                "source_dataset": args.source_dataset,
                "topic_tags": tags,
                "metadata": {
                    "filename": path.name,
                    "scan_root": str(image_dir.resolve()),
                },
            }
        )
        if args.limit is not None and len(rows) >= args.limit:
            break

    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} image manifest rows to {args.out}")


if __name__ == "__main__":
    main()

