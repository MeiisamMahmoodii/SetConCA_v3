from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.io_utils import group_accepted, read_json, read_jsonl, write_json, write_jsonl, write_review_table
from setconca_v2.paths import resolve_project_path


def read_manifest(path: Path) -> Dict[str, Any]:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.exists():
        return {"out_dir": str(path), "missing_manifest": True}
    return read_json(manifest_path)


def read_rows(path: Path, name: str) -> List[Dict[str, Any]]:
    file_path = path / name
    if not file_path.exists():
        return []
    return list(read_jsonl(file_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge generated dataset shard artifact folders.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("shard_dirs", nargs="+")
    args = parser.parse_args()

    t0 = time.time()
    out_dir = resolve_project_path(args.out_dir)
    shard_dirs = [resolve_project_path(path) for path in args.shard_dirs]
    out_dir.mkdir(parents=True, exist_ok=True)

    attempts: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    manifests = []
    for shard_dir in shard_dirs:
        manifests.append(read_manifest(shard_dir))
        attempts.extend(read_rows(shard_dir, "attempts.jsonl"))
        accepted.extend(read_rows(shard_dir, "accepted.jsonl"))

    attempts.sort(
        key=lambda row: (
            str(row.get("model_name", "")),
            str(row.get("original_id", "")),
            str(row.get("length_band", "")),
            int(row.get("attempt_idx", -1)),
            int(row.get("candidate_idx", -1)),
        )
    )
    accepted.sort(
        key=lambda row: (
            str(row.get("original_id", "")),
            str(row.get("model_name", "")),
            str(row.get("length_band", "")),
        )
    )
    grouped = group_accepted(accepted)

    write_jsonl(out_dir / "attempts.jsonl", attempts)
    write_jsonl(out_dir / "accepted.jsonl", accepted)
    write_jsonl(out_dir / "sets.jsonl", grouped)
    write_review_table(out_dir / "review_table.md", grouped)
    write_json(
        out_dir / "run_manifest.json",
        {
            "kind": "merged_shards",
            "out_dir": str(out_dir),
            "shard_dirs": [str(path) for path in shard_dirs],
            "shard_manifests": manifests,
            "n_shards": len(shard_dirs),
            "n_attempts": len(attempts),
            "n_accepted": len(accepted),
            "n_sets": len(grouped),
            "elapsed_s": time.time() - t0,
        },
    )
    print(
        f"Merged {len(shard_dirs)} shard(s) into {out_dir} | "
        f"attempts={len(attempts)} | accepted={len(accepted)} | sets={len(grouped)}"
    )


if __name__ == "__main__":
    main()
