from __future__ import annotations

import argparse
from collections import defaultdict

from vg_common import read_jsonl, stable_hash, write_jsonl


def response_to_view(row: dict) -> dict:
    view_type = f"{row['model_source_id']}__{row['language_code']}__{row['prompt_view']}"
    return {
        "id": row["response_id"],
        "text": row["output_text"],
        "view_type": view_type,
        "source_id": row["model_source_id"],
        "generated": True,
        "metadata": {
            "model_id": row["model_id"],
            "language_code": row["language_code"],
            "language_name": row["language_name"],
            "prompt_view": row["prompt_view"],
            "prompt_hash": row["prompt_hash"],
            "image_path": row["image_path"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Group filtered responses into SetConCA-compatible visual-scene sets.")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-views", type=int, default=12)
    args = parser.parse_args()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.responses):
        grouped[str(row["image_id"])].append(row)

    sets = []
    for image_id, rows in sorted(grouped.items()):
        if len(rows) < args.min_views:
            continue
        first = rows[0]
        source_dataset = first.get("source_dataset", "unknown")
        topic_tags = first.get("topic_tags", [])
        sets.append(
            {
                "set_id": f"visual_scene_{stable_hash({'image_id': image_id})}",
                "latent_type": "visual_scene",
                "latent_key": f"{source_dataset}:{image_id}",
                "source_dataset": source_dataset,
                "source_ids": [image_id],
                "domain": "vision_language",
                "texts": [response_to_view(row) for row in rows],
                "usage": "train",
                "confidence": 0.8,
                "validation": {
                    "builder": "visual_grounded_dataset",
                    "n_views": len(rows),
                    "n_models": len({row["model_source_id"] for row in rows}),
                    "n_languages": len({row["language_code"] for row in rows}),
                    "n_prompt_views": len({row["prompt_view"] for row in rows}),
                },
                "metadata": {
                    "image_id": image_id,
                    "image_path": first.get("image_path", ""),
                    "topic_tags": topic_tags,
                    "anchor_type": "image",
                    "caption_source_used_for_generation": False,
                },
            }
        )

    write_jsonl(args.out, sets)
    print(f"wrote {len(sets)} visual-scene sets to {args.out}")


if __name__ == "__main__":
    main()

