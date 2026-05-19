from __future__ import annotations

import argparse
from itertools import islice

from vg_common import load_enabled_models, load_languages, load_views, read_jsonl, render_prompt, stable_hash, write_jsonl


def limited(items: list[dict], limit: int | None) -> list[dict]:
    return list(islice(items, limit)) if limit is not None else items


def main() -> None:
    parser = argparse.ArgumentParser(description="Render VLM generation jobs from image/model/language/view configs.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit-images", type=int)
    parser.add_argument("--limit-models", type=int)
    parser.add_argument("--limit-languages", type=int)
    parser.add_argument("--limit-views", type=int)
    args = parser.parse_args()

    images = limited(read_jsonl(args.manifest), args.limit_images)
    models = limited(load_enabled_models(), args.limit_models)
    languages = limited(load_languages(), args.limit_languages)
    views, template = load_views()

    jobs = []
    for image in images:
        for model in models:
            for language in languages:
                for view in limited(views, args.limit_views):
                    prompt = render_prompt(language, view, template)
                    job = {
                        "job_id": stable_hash(
                            {
                                "image_id": image["image_id"],
                                "model": model["source_id"],
                                "language": language["code"],
                                "view": view["view_id"],
                            }
                        ),
                        "image": image,
                        "model": model,
                        "language": language,
                        "view": view,
                        "prompt": prompt,
                        "prompt_hash": stable_hash(prompt),
                    }
                    jobs.append(job)

    write_jsonl(args.out, jobs)
    print(f"wrote {len(jobs)} generation jobs to {args.out}")


if __name__ == "__main__":
    main()

