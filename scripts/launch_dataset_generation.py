from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]


def build_generation_cmd(args: argparse.Namespace, out_dir: Path, shard: str | None = None) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "generate_constrained_sets.py"),
        "--models-config",
        args.models_config,
        "--input",
        args.input,
        "--out-dir",
        str(out_dir),
        "--backend",
        args.backend,
    ]
    if args.length_bands:
        cmd.extend(["--length-bands", args.length_bands])
    if args.max_originals is not None:
        cmd.extend(["--max-originals", str(args.max_originals)])
    if args.dry_run:
        cmd.append("--dry-run")
    if args.include_disabled:
        cmd.append("--include-disabled")
    if shard:
        cmd.extend(["--model-shard", shard])
    return cmd


def run_single(args: argparse.Namespace) -> int:
    out_dir = ROOT / args.out_dir
    cmd = build_generation_cmd(args, out_dir)
    print("Running single-process generation:")
    print(" ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def run_multi(args: argparse.Namespace) -> int:
    out_dir = ROOT / args.out_dir
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    for gpu_idx in range(args.gpus):
        shard_dir = out_dir / f"shard_{gpu_idx}"
        shard = f"{gpu_idx}/{args.gpus}"
        cmd = build_generation_cmd(args, shard_dir, shard=shard)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        log_path = logs_dir / f"shard_{gpu_idx}.log"
        log_file = log_path.open("w", encoding="utf-8")
        print(f"Launching shard {shard} on GPU {gpu_idx}: {log_path}", flush=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((gpu_idx, proc, log_file))

    exit_code = 0
    for gpu_idx, proc, log_file in processes:
        code = proc.wait()
        log_file.close()
        print(f"Shard {gpu_idx}/{args.gpus} finished with exit code {code}", flush=True)
        if code != 0:
            exit_code = code

    if exit_code != 0:
        print("At least one shard failed. Not merging outputs.", flush=True)
        return exit_code

    merge_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "merge_generated_shards.py"),
        "--out-dir",
        str(out_dir / "merged"),
    ]
    merge_cmd.extend(str(out_dir / f"shard_{gpu_idx}") for gpu_idx in range(args.gpus))
    print("Merging shard outputs:", flush=True)
    print(" ".join(merge_cmd), flush=True)
    return subprocess.call(merge_cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch SetConCA V2 dataset generation locally or across GPUs.")
    parser.add_argument("--models-config", default="configs/rewrite_models.example.json")
    parser.add_argument("--input", default="data/raw/ag_news_train_full.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPU shard processes to launch.")
    parser.add_argument("--length-bands", default=None)
    parser.add_argument("--max-originals", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    args = parser.parse_args()

    if args.gpus < 1:
        raise SystemExit("--gpus must be at least 1")
    if args.gpus > 1 and args.backend != "vllm":
        raise SystemExit("Multi-GPU launch is intended for --backend vllm. Use --gpus 1 for hf.")

    code = run_single(args) if args.gpus == 1 else run_multi(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
