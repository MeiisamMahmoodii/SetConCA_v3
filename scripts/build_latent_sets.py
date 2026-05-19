from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.dataset_registry import load_dataset_registry
from setconca_v2.fever_builder import build_fever_latent_sets, build_wiki_sentence_index, parse_fever_wiki_lines, required_fever_pages
from setconca_v2.io_utils import read_jsonl, write_json, write_jsonl


FEVER_DIRECT_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "labelled_dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "unlabelled_dev": "https://fever.ai/download/fever/shared_task_dev_public.jsonl",
    "unlabelled_test": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "paper_dev": "https://fever.ai/download/fever/paper_dev.jsonl",
    "paper_test": "https://fever.ai/download/fever/paper_test.jsonl",
}


def load_records_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    return list(read_jsonl(path))


def load_jsonl_from_url(url: str, *, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with urllib.request.urlopen(url) as response:
        for raw_line in response:
            if limit is not None and len(rows) >= limit:
                break
            line = raw_line.decode("utf-8").strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_fever_direct(split: str, *, limit: int | None = None) -> List[Dict[str, Any]]:
    if split not in FEVER_DIRECT_URLS:
        raise ValueError(f"no direct FEVER URL configured for split: {split}")
    return load_jsonl_from_url(FEVER_DIRECT_URLS[split], limit=limit)


def build_wiki_sentence_index_from_zip(zip_path: Path, required_pages: set[str]) -> Dict[str, Dict[int, str]]:
    index: Dict[str, Dict[int, str]] = {}
    if not required_pages:
        return index
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if len(index) >= len(required_pages):
                break
            path_parts = Path(name).parts
            member_name = Path(name).name
            if "__MACOSX" in path_parts or member_name.startswith("._") or not name.endswith(".jsonl"):
                continue
            with zf.open(name) as f:
                for raw_line in f:
                    if len(index) >= len(required_pages):
                        break
                    row = json.loads(raw_line.decode("utf-8"))
                    page_id = str(row.get("id", "")).strip()
                    if page_id in required_pages:
                        index[page_id] = parse_fever_wiki_lines(str(row.get("lines", "")))
    return index


def load_records_from_hf(source_id: str, *, split: str, limit: int | None = None) -> List[Dict[str, Any]]:
    sources = {source.source_id: source for source in load_dataset_registry()}
    if source_id not in sources:
        raise ValueError(f"unknown source_id in registry: {source_id}")
    source = sources[source_id]
    if source_id == "fever":
        return load_fever_direct(split, limit=limit)
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Hugging Face loading") from exc

    kwargs: Dict[str, Any] = {"split": split, "trust_remote_code": True}
    if source.config:
        dataset = load_dataset(source.hf_dataset, source.config, **kwargs)
    else:
        dataset = load_dataset(source.hf_dataset, **kwargs)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return [dict(row) for row in dataset]


def write_rejection_report(path: Path, rejected: Iterable[Dict[str, Any]]) -> None:
    lines = [
        "# Latent Set Rejection Report",
        "",
        "| Source index | Source id | Label | Reasons | Claim |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rejected:
        reasons = ", ".join(str(item) for item in row.get("reasons", []))
        claim = str(row.get("claim", "")).replace("|", "\\|")
        lines.append(
            f"| {row.get('source_index')} | {row.get('source_id')} | {row.get('label')} | "
            f"{reasons.replace('|', '\\|')} | {claim} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_fever(args: argparse.Namespace) -> None:
    if args.input:
        records = load_records_from_jsonl(Path(args.input))
        source_mode = "jsonl"
    else:
        records = load_records_from_hf("fever", split=args.split, limit=args.candidate_limit)
        source_mode = "hf"

    wiki_sentence_index = None
    if args.wiki_jsonl:
        wiki_rows = load_records_from_jsonl(Path(args.wiki_jsonl))
        wiki_sentence_index = build_wiki_sentence_index(wiki_rows)
    elif args.wiki_zip:
        wiki_sentence_index = build_wiki_sentence_index_from_zip(Path(args.wiki_zip), required_fever_pages(records))

    result = build_fever_latent_sets(
        records,
        split=args.split,
        wiki_sentence_index=wiki_sentence_index,
        include_reference_view=args.include_reference_view,
        limit=args.limit,
    )

    out_dir = Path(args.out_dir)
    rows = [row.to_dict() for row in result.accepted]
    write_jsonl(out_dir / "sets.jsonl", rows)
    write_jsonl(out_dir / "rejected.jsonl", result.rejected)
    write_rejection_report(out_dir / "rejection_report.md", result.rejected)
    write_json(
        out_dir / "manifest.json",
        {
            "builder": "fever_claim_evidence",
            "source_mode": source_mode,
            "split": args.split,
            "input": str(args.input) if args.input else None,
            "wiki_jsonl": str(args.wiki_jsonl) if args.wiki_jsonl else None,
            "wiki_zip": str(args.wiki_zip) if args.wiki_zip else None,
            "include_reference_view": bool(args.include_reference_view),
            "candidate_rows": len(records),
            "accepted_sets": len(result.accepted),
            "rejected_rows": len(result.rejected),
            "outputs": {
                "sets": "sets.jsonl",
                "rejected": "rejected.jsonl",
                "rejection_report": "rejection_report.md",
            },
        },
    )
    print(json.dumps({"accepted": len(result.accepted), "rejected": len(result.rejected)}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build latent-set datasets from configured sources.")
    sub = parser.add_subparsers(dest="source", required=True)

    fever = sub.add_parser("fever", help="Build FEVER claim/evidence latent sets.")
    fever.add_argument("--input", help="Optional local FEVER claim JSONL. If omitted, loads from Hugging Face.")
    fever.add_argument("--wiki-jsonl", help="Optional FEVER wiki_pages JSONL with id/lines fields for evidence text.")
    fever.add_argument("--wiki-zip", help="Optional official FEVER wiki-pages.zip. Only required evidence pages are indexed.")
    fever.add_argument("--out-dir", required=True, help="Output directory.")
    fever.add_argument("--split", default="train", help="FEVER split name.")
    fever.add_argument("--limit", type=int, help="Maximum accepted sets to write.")
    fever.add_argument("--candidate-limit", type=int, help="Maximum source rows to load from Hugging Face before filtering.")
    fever.add_argument(
        "--include-reference-view",
        action="store_true",
        help="Allow evidence-reference text when wiki evidence text is unavailable. Good for smoke tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "fever":
        build_fever(args)
    else:
        raise ValueError(f"unsupported source: {args.source}")


if __name__ == "__main__":
    main()
