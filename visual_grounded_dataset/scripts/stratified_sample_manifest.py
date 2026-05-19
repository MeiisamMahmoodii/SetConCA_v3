from __future__ import annotations

import argparse
import random
from collections import defaultdict

from vg_common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample a topic-balanced subset from an image manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-topic", type=int, default=500)
    parser.add_argument("--max-total", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_jsonl(args.manifest)
    by_topic: dict[str, list[dict]] = defaultdict(list)
    untagged = []
    for row in rows:
        tags = row.get("topic_tags", [])
        if not tags:
            untagged.append(row)
        for tag in tags:
            by_topic[str(tag)].append(row)

    selected_by_id: dict[str, dict] = {}
    topic_report = {}
    for topic, topic_rows in sorted(by_topic.items()):
        sampled = topic_rows[:]
        rng.shuffle(sampled)
        sampled = sampled[: args.per_topic]
        topic_report[topic] = len(sampled)
        for row in sampled:
            selected_by_id[str(row["image_id"])] = row

    selected = list(selected_by_id.values())
    rng.shuffle(selected)
    if args.max_total is not None:
        selected = selected[: args.max_total]

    write_jsonl(args.out, selected)
    print(f"wrote {len(selected)} stratified manifest rows to {args.out}")
    for topic, count in topic_report.items():
        print(f"{topic}: {count}")


if __name__ == "__main__":
    main()

