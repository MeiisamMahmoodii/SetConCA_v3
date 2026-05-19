from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from vg_common import compact_text, leaks_metadata, read_jsonl, refusal_like, script_matches, write_jsonl


def rejection_reasons(row: dict, min_chars: int, max_chars: int) -> list[str]:
    text = compact_text(str(row.get("output_text", "")))
    reasons = []
    if row.get("generation_status") == "error":
        reasons.append("generation_error")
    if len(text) < min_chars:
        reasons.append("too_short")
    if len(text) > max_chars:
        reasons.append("too_long")
    if refusal_like(text):
        reasons.append("refusal_like")
    if leaks_metadata(text):
        reasons.append("metadata_leak")
    if not script_matches(text, str(row.get("language_script_hint", ""))):
        reasons.append("script_mismatch")
    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter generated visual-grounded responses.")
    parser.add_argument("--responses", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()

    rows = read_jsonl(args.responses)
    seen_by_image: dict[str, set[str]] = defaultdict(set)
    accepted = []
    rejected = []
    reason_counts: Counter[str] = Counter()

    for row in rows:
        text = compact_text(str(row.get("output_text", "")))
        reasons = rejection_reasons(row, args.min_chars, args.max_chars)
        duplicate_key = text.casefold()
        image_id = str(row.get("image_id", ""))
        if duplicate_key in seen_by_image[image_id]:
            reasons.append("duplicate_within_image")
        if reasons:
            rejected.append({**row, "output_text": text, "validation_status": "rejected", "rejection_reasons": reasons})
            reason_counts.update(reasons)
        else:
            seen_by_image[image_id].add(duplicate_key)
            accepted.append({**row, "output_text": text, "validation_status": "accepted", "rejection_reasons": []})

    write_jsonl(args.out, accepted)
    lines = [
        "# Filter Report",
        "",
        f"- Input responses: {len(rows)}",
        f"- Accepted: {len(accepted)}",
        f"- Rejected: {len(rejected)}",
        "",
        "## Rejection Reasons",
        "",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"- `{reason}`: {count}")
    from pathlib import Path

    target = Path(args.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"accepted {len(accepted)} / {len(rows)} responses; report written to {args.report}")


if __name__ == "__main__":
    main()
