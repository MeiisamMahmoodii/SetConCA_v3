from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.activation_extraction import build_activation_bank
from setconca_v2.paths import resolve_project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract representation activations for grouped semantic sets.")
    parser.add_argument("--sets", required=True, help="Grouped semantic-set JSONL, e.g. sets_min8.jsonl")
    parser.add_argument("--out", required=True, help="Output .pt activation bank")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--token-position", default="last", choices=["last", "mean"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-sets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--no-original", action="store_true", help="Use only rewrites as views.")
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic fake activations for pipeline tests.")
    parser.add_argument("--fake-hidden-dim", type=int, default=64)
    args = parser.parse_args()

    meta = build_activation_bank(
        resolve_project_path(args.sets),
        out_path=resolve_project_path(args.out),
        model_id=args.model_id,
        layer=args.layer,
        views=args.views,
        token_position=args.token_position,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_sets=args.max_sets,
        include_original=not args.no_original,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        dry_run=args.dry_run,
        fake_hidden_dim=args.fake_hidden_dim,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
