from __future__ import annotations

import argparse
from pathlib import Path

from vg_common import read_jsonl, write_jsonl


def convert_row(row: dict, *, mode: str) -> dict:
    texts = list(row.get("texts", []))
    if not texts:
        raise ValueError(f"set {row.get('set_id')} has no texts")

    if mode == "all_views_as_rewrites":
        original_text = texts[0]["text"]
        rewrite_views = texts
    elif mode == "first_view_as_original":
        original_text = texts[0]["text"]
        rewrite_views = texts[1:]
    else:
        raise ValueError(f"unknown conversion mode: {mode}")

    rewrites = []
    for idx, view in enumerate(rewrite_views):
        meta = dict(view.get("metadata", {}))
        rewrites.append(
            {
                "id": view.get("id", f"view_{idx:03d}"),
                "text": view["text"],
                "model_name": view.get("source_id") or meta.get("model_id") or "unknown_vlm",
                "length_band": meta.get("prompt_view", view.get("view_type", "visual_view")),
                "view_type": view.get("view_type", ""),
                "metadata": meta,
            }
        )

    return {
        "original_id": row.get("set_id") or row.get("latent_key"),
        "original_text": original_text,
        "label": row.get("latent_type", "visual_scene"),
        "source": row.get("source_dataset", "visual_grounded_dataset"),
        "rewrites": rewrites,
        "metadata": {
            **dict(row.get("metadata", {})),
            "latent_key": row.get("latent_key"),
            "source_ids": row.get("source_ids", []),
            "activation_conversion_mode": mode,
            "activation_note": "Use --no-original with all_views_as_rewrites to sample only generated visual views.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert visual latent-set rows to the legacy activation extractor shape.")
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        choices=["all_views_as_rewrites", "first_view_as_original"],
        default="all_views_as_rewrites",
        help="Use all visual views as sampleable rewrites, or reserve the first as original_text.",
    )
    args = parser.parse_args()

    rows = [convert_row(row, mode=args.mode) for row in read_jsonl(args.sets)]
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} activation-ready rows to {args.out}")


if __name__ == "__main__":
    main()

