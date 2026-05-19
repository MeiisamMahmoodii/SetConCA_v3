from __future__ import annotations

import argparse
import random
from pathlib import Path

from vg_common import read_jsonl, stable_hash, write_jsonl


def clone_with_control_id(row: dict, control_type: str, texts: list[dict]) -> dict:
    return {
        **row,
        "set_id": f"{control_type}_{stable_hash({'base': row['set_id'], 'texts': [text['id'] for text in texts]})}",
        "latent_type": f"control_{control_type}",
        "texts": texts,
        "risk_tag": "control_not_train",
        "usage": "eval",
        "validation": {**row.get("validation", {}), "control_type": control_type},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched control sets from visual-scene sets.")
    parser.add_argument("--sets", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sets = read_jsonl(args.sets)
    all_texts = [text for row in sets for text in row.get("texts", [])]
    by_language: dict[str, list[dict]] = {}
    by_prompt: dict[str, list[dict]] = {}
    by_model: dict[str, list[dict]] = {}
    for text in all_texts:
        meta = text.get("metadata", {})
        by_language.setdefault(meta.get("language_code", ""), []).append(text)
        by_prompt.setdefault(meta.get("prompt_view", ""), []).append(text)
        by_model.setdefault(text.get("source_id", ""), []).append(text)

    shuffled = []
    same_language = []
    same_prompt = []
    same_model = []
    for row in sets:
        n = len(row.get("texts", []))
        original_ids = {text["id"] for text in row.get("texts", [])}
        pool = [text for text in all_texts if text["id"] not in original_ids]
        if len(pool) >= n:
            shuffled.append(clone_with_control_id(row, "shuffled_image", rng.sample(pool, n)))

        language_matched = []
        prompt_matched = []
        model_matched = []
        for text in row.get("texts", []):
            meta = text.get("metadata", {})
            for target, buckets, key in [
                (language_matched, by_language, meta.get("language_code", "")),
                (prompt_matched, by_prompt, meta.get("prompt_view", "")),
                (model_matched, by_model, text.get("source_id", "")),
            ]:
                candidates = [candidate for candidate in buckets.get(key, []) if candidate["id"] not in original_ids]
                if candidates:
                    target.append(rng.choice(candidates))
        if len(language_matched) == n:
            same_language.append(clone_with_control_id(row, "same_language_different_image", language_matched))
        if len(prompt_matched) == n:
            same_prompt.append(clone_with_control_id(row, "same_prompt_different_image", prompt_matched))
        if len(model_matched) == n:
            same_model.append(clone_with_control_id(row, "same_model_different_image", model_matched))

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "controls_shuffled_image.jsonl", shuffled)
    write_jsonl(out_dir / "controls_same_language_different_image.jsonl", same_language)
    write_jsonl(out_dir / "controls_same_prompt_different_image.jsonl", same_prompt)
    write_jsonl(out_dir / "controls_same_model_different_image.jsonl", same_model)
    print(f"wrote controls to {out_dir}")


if __name__ == "__main__":
    main()

