from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from setconca_v2.io_utils import read_json, write_json
from setconca_v2.paths import resolve_project_path


@dataclass
class ExtractionJob:
    family: str
    size: str
    size_label: str
    model_id: str
    layer_fraction: float
    layer_index: int
    n_layers: int
    out_dir: Path
    out_file: Path
    log_file: Path


def slugify(text: str) -> str:
    text = text.lower().replace("/", "__")
    text = re.sub(r"[^a-z0-9_.-]+", "_", text)
    return text.strip("_")


def pct_label(value: float) -> str:
    return f"{int(round(value * 100)):02d}pct"


def resolve_num_layers(model_id: str, trust_remote_code: bool) -> int:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    for attr in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(cfg, attr, None)
        if value is not None:
            return int(value)
    for nested_attr in ("text_config", "language_config", "llm_config"):
        nested = getattr(cfg, nested_attr, None)
        if nested is None:
            continue
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(nested, attr, None)
            if value is not None:
                return int(value)
    raise ValueError(f"Could not determine number of hidden layers for {model_id}")


def layer_from_fraction(n_layers: int, fraction: float) -> int:
    if not 0 < fraction <= 1:
        raise ValueError(f"Layer fraction must be in (0, 1], got {fraction}")
    return max(1, min(n_layers, round(n_layers * fraction)))


def build_jobs(
    cfg: Dict[str, Any],
    out_root: Path,
    only_family: str | None,
    only_size: str | None,
    *,
    use_config_layers: bool,
) -> List[ExtractionJob]:
    extraction = cfg.get("extraction", {})
    trust_remote_code = bool(extraction.get("trust_remote_code", False))
    fractions = [float(x) for x in extraction.get("layer_fractions", [0.2, 0.6, 0.9])]
    jobs: List[ExtractionJob] = []

    layer_cache: Dict[str, int] = {}
    for model in cfg.get("models", []):
        if not bool(model.get("enabled", True)):
            continue
        family = str(model["family"])
        size = str(model["size"])
        if only_family and family != only_family:
            continue
        if only_size and size != only_size:
            continue
        model_id = str(model["model_id"])
        if use_config_layers and model.get("num_hidden_layers") is not None:
            n_layers = int(model["num_hidden_layers"])
        else:
            try:
                n_layers = layer_cache.setdefault(model_id, resolve_num_layers(model_id, trust_remote_code))
            except ValueError:
                if model.get("num_hidden_layers") is None:
                    raise
                n_layers = int(model["num_hidden_layers"])
                print(
                    f"Warning: using configured num_hidden_layers={n_layers} for {model_id} "
                    "because live config resolution did not expose a standard layer field.",
                    flush=True,
                )
        for fraction in fractions:
            layer_idx = layer_from_fraction(n_layers, fraction)
            family_dir = out_root / family / f"{size}_{model.get('size_label', size)}" / slugify(model_id)
            layer_dir = family_dir / f"layer_{layer_idx:02d}_{pct_label(fraction)}"
            jobs.append(
                ExtractionJob(
                    family=family,
                    size=size,
                    size_label=str(model.get("size_label", size)),
                    model_id=model_id,
                    layer_fraction=fraction,
                    layer_index=layer_idx,
                    n_layers=n_layers,
                    out_dir=layer_dir,
                    out_file=layer_dir / "activation_bank.pt",
                    log_file=layer_dir / "extract.log",
                )
            )
    return jobs


def command_for_job(args: argparse.Namespace, cfg: Dict[str, Any], job: ExtractionJob) -> List[str]:
    dataset = cfg.get("dataset", {})
    extraction = cfg.get("extraction", {})
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "extract_activation_bank.py"),
        "--sets",
        args.sets or dataset.get("sets", "data/generated/server_4gpu_2000/merged/sets_min16.jsonl"),
        "--out",
        str(job.out_file),
        "--model-id",
        job.model_id,
        "--layer",
        str(job.layer_index),
        "--views",
        str(args.views or dataset.get("views", 16)),
        "--token-position",
        str(args.token_position or extraction.get("token_position", "last")),
        "--batch-size",
        str(args.batch_size or extraction.get("batch_size", 8)),
        "--max-length",
        str(args.max_length or extraction.get("max_length", 256)),
        "--dtype",
        str(args.dtype or extraction.get("dtype", "auto")),
        "--seed",
        str(args.seed),
    ]
    if args.max_sets is not None:
        cmd.extend(["--max-sets", str(args.max_sets)])
    if args.dry_run:
        cmd.append("--dry-run")
        cmd.extend(["--fake-hidden-dim", str(args.fake_hidden_dim)])
    if bool(extraction.get("trust_remote_code", False)):
        cmd.append("--trust-remote-code")
    return cmd


def write_grid_manifest(out_root: Path, cfg: Dict[str, Any], jobs: List[ExtractionJob], args: argparse.Namespace) -> None:
    write_json(
        out_root / "activation_grid_manifest.json",
        {
            "config": str(resolve_project_path(args.config)),
            "out_root": str(out_root),
            "sets": args.sets or cfg.get("dataset", {}).get("sets"),
            "views": args.views or cfg.get("dataset", {}).get("views"),
            "layer_fractions": cfg.get("extraction", {}).get("layer_fractions"),
            "set_size_sweep": cfg.get("dataset", {}).get("set_size_sweep"),
            "dry_run": args.dry_run,
            "gpus": args.gpus,
            "n_jobs": len(jobs),
            "jobs": [
                {
                    "family": job.family,
                    "size": job.size,
                    "size_label": job.size_label,
                    "model_id": job.model_id,
                    "n_layers": job.n_layers,
                    "layer_fraction": job.layer_fraction,
                    "layer_index": job.layer_index,
                    "out_file": str(job.out_file),
                    "log_file": str(job.log_file),
                }
                for job in jobs
            ],
        },
    )


def run_jobs(args: argparse.Namespace, cfg: Dict[str, Any], jobs: List[ExtractionJob]) -> int:
    if args.print_only:
        for job in jobs:
            print(" ".join(command_for_job(args, cfg, job)))
        return 0

    if args.gpus <= 1:
        for idx, job in enumerate(jobs, start=1):
            job.out_dir.mkdir(parents=True, exist_ok=True)
            cmd = command_for_job(args, cfg, job)
            print(f"[{idx}/{len(jobs)}] running {job.family}/{job.size_label} layer {job.layer_index}: {job.model_id}")
            with job.log_file.open("w", encoding="utf-8") as f:
                code = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
            if code != 0:
                print(f"Job failed with exit code {code}. See {job.log_file}")
                return code
        return 0

    queue = deque(jobs)
    running: List[tuple[int, ExtractionJob, subprocess.Popen[Any], Any]] = []
    exit_code = 0
    while queue or running:
        while queue and len(running) < args.gpus:
            gpu_idx = len(running)
            job = queue.popleft()
            job.out_dir.mkdir(parents=True, exist_ok=True)
            cmd = command_for_job(args, cfg, job)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            log_handle = job.log_file.open("w", encoding="utf-8")
            print(
                f"launch gpu={gpu_idx} {job.family}/{job.size_label} "
                f"layer={job.layer_index}/{job.n_layers} model={job.model_id}"
            )
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
            running.append((gpu_idx, job, proc, log_handle))

        still_running = []
        for gpu_idx, job, proc, log_handle in running:
            code = proc.poll()
            if code is None:
                still_running.append((gpu_idx, job, proc, log_handle))
                continue
            log_handle.close()
            print(f"finish gpu={gpu_idx} code={code} {job.family}/{job.size_label} layer={job.layer_index}")
            if code != 0 and exit_code == 0:
                exit_code = code
        running = still_running
        if running:
            time.sleep(args.poll_seconds)
        if exit_code != 0 and args.stop_on_failure:
            for _, _, proc, log_handle in running:
                proc.terminate()
                log_handle.close()
            return exit_code
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SetConCA activation extraction model/layer grid.")
    parser.add_argument("--config", default="configs/activation_model_grid.json")
    parser.add_argument("--out-root", default="data/activations/model_grid_s16_min16")
    parser.add_argument("--sets", default=None)
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--token-position", default=None)
    parser.add_argument("--max-sets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--only-family", default=None, choices=["llama3", "gemma3", "qwen3"])
    parser.add_argument("--only-size", default=None, choices=["small", "mid", "big"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fake-hidden-dim", type=int, default=32)
    parser.add_argument(
        "--use-config-layers",
        action="store_true",
        help="Use num_hidden_layers from the grid config instead of resolving model configs from Hugging Face.",
    )
    parser.add_argument("--print-only", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()

    cfg = read_json(resolve_project_path(args.config))
    out_root = resolve_project_path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(
        cfg,
        out_root,
        args.only_family,
        args.only_size,
        use_config_layers=args.use_config_layers or args.dry_run or args.print_only,
    )
    write_grid_manifest(out_root, cfg, jobs, args)
    print(f"Prepared {len(jobs)} extraction job(s). Manifest: {out_root / 'activation_grid_manifest.json'}")
    raise SystemExit(run_jobs(args, cfg, jobs))


if __name__ == "__main__":
    main()
