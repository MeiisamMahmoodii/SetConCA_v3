from __future__ import annotations

import argparse
from collections import Counter
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from setconca_v2.io_utils import group_accepted, read_json, read_jsonl, write_json, write_jsonl, write_review_table
from setconca_v2.rewrite_generation import (
    DryRunRewriteGenerator,
    HFRewriteGenerator,
    RewriteModelSpec,
    VLLMRewriteGenerator,
    build_prompt,
)
from setconca_v2.text_constraints import (
    DEFAULT_LENGTH_BANDS,
    LengthBand,
    count_words,
    extract_banned_words,
    validate_rewrite,
)
from setconca_v2.semantic_validation import SemanticValidator
from setconca_v2.paths import resolve_project_path


def format_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def summarize_rejection_reasons(attempts: List[Dict[str, Any]], max_items: int = 4) -> str:
    reasons = Counter()
    for row in attempts:
        if row.get("status") == "rejected":
            for reason in row.get("reasons", []):
                reasons[reason.split("=", 1)[0]] += 1
        elif row.get("status") == "generation_error":
            reasons["generation_error"] += 1
    if not reasons:
        return "none"
    return ", ".join(f"{reason}:{count}" for reason, count in reasons.most_common(max_items))


def parse_bands(raw: str | None) -> List[LengthBand]:
    if not raw:
        return DEFAULT_LENGTH_BANDS
    bands = []
    for part in raw.split(","):
        lo, hi = part.strip().split("-")
        bands.append(LengthBand(part.strip(), int(lo), int(hi)))
    return bands


def model_specs(config: Dict[str, Any], include_disabled: bool = False) -> List[RewriteModelSpec]:
    specs = []
    for row in config.get("models", []):
        enabled = bool(row.get("enabled", True))
        if not enabled and not include_disabled:
            continue
        specs.append(
            RewriteModelSpec(
                name=row["name"],
                model_id=row["model_id"],
                enabled=enabled,
                revision=row.get("revision"),
                torch_dtype=row.get("torch_dtype", "auto"),
                trust_remote_code=bool(row.get("trust_remote_code", False)),
            )
        )
    return specs


def parse_shard(raw: str | None) -> tuple[int, int] | None:
    if not raw:
        return None
    try:
        idx_raw, total_raw = raw.split("/", 1)
        idx = int(idx_raw)
        total = int(total_raw)
    except ValueError as exc:
        raise ValueError("--model-shard must use INDEX/TOTAL, e.g. 0/4") from exc
    if total <= 0 or idx < 0 or idx >= total:
        raise ValueError("--model-shard INDEX must satisfy 0 <= INDEX < TOTAL")
    return idx, total


def apply_model_shard(specs: List[RewriteModelSpec], shard: tuple[int, int] | None) -> List[RewriteModelSpec]:
    if shard is None:
        return specs
    idx, total = shard
    return [spec for pos, spec in enumerate(specs) if pos % total == idx]


def validate_candidate_row(
    candidate: str,
    candidate_idx: int,
    original: Dict[str, Any],
    original_id: str,
    original_text: str,
    banned_words: List[str],
    band: LengthBand,
    spec: RewriteModelSpec,
    attempt_idx: int,
    semantic_validator: SemanticValidator,
) -> tuple[Dict[str, Any], bool]:
    ok, reasons = validate_rewrite(candidate, banned_words, band)
    semantic = semantic_validator.validate(original_text, candidate) if ok else None
    semantic_metrics = semantic.metrics if semantic is not None else {}
    if semantic is not None and not semantic.passed:
        ok = False
        reasons = reasons + semantic.reasons
    row = {
        "status": "accepted" if ok else "rejected",
        "reasons": reasons,
        "original_id": original_id,
        "original_text": original_text,
        "label": original.get("label"),
        "source": original.get("source"),
        "model_name": spec.name,
        "model_id": spec.model_id,
        "length_band": band.label,
        "attempt_idx": attempt_idx,
        "candidate_idx": candidate_idx,
        "rewrite": candidate,
        "word_count": count_words(candidate),
        "banned_words": banned_words,
        "semantic_metrics": semantic_metrics,
    }
    return row, ok


def generate_for_model_vllm_batched(
    spec: RewriteModelSpec,
    model_idx: int,
    n_models: int,
    originals: List[Dict[str, Any]],
    bands: List[LengthBand],
    generation_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
    prompt_template: str | None,
    semantic_validator: SemanticValidator,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model_t0 = time.time()
    print(f"[{model_idx}/{n_models}] loading {spec.name}: {spec.model_id} via vllm-batched", flush=True)
    generator = VLLMRewriteGenerator(spec, vllm_cfg)
    print(f"[{model_idx}/{n_models}] loaded {spec.name} in {format_elapsed(time.time() - model_t0)}", flush=True)

    attempts: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    max_attempts = int(generation_cfg.get("max_attempts_per_slot", 3))
    active_slots: List[Dict[str, Any]] = []

    for original_idx, original in enumerate(originals, start=1):
        original_id = original.get("id") or original.get("original_id")
        original_text = original["text"]
        banned_words = original.get("banned_words") or extract_banned_words(original_text)
        for band in bands:
            active_slots.append(
                {
                    "original": original,
                    "original_idx": original_idx,
                    "original_id": original_id,
                    "original_text": original_text,
                    "banned_words": banned_words,
                    "band": band,
                }
            )

    batch_size = int(vllm_cfg.get("batch_size", len(active_slots) or 1))
    print(
        f"[{model_idx}/{n_models}] prepared {len(active_slots)} slot prompt(s) | "
        f"batch_size={batch_size} | max_attempts_per_slot={max_attempts}",
        flush=True,
    )

    try:
        for attempt_idx in range(max_attempts):
            if not active_slots:
                break
            round_t0 = time.time()
            prompts = [
                build_prompt(slot["original_text"], slot["banned_words"], slot["band"], prompt_template)
                for slot in active_slots
            ]
            print(
                f"[{model_idx}/{n_models}] attempt round {attempt_idx + 1}/{max_attempts}: "
                f"generating {len(prompts)} prompt(s)",
                flush=True,
            )
            try:
                results = generator.generate_many(prompts, generation_cfg)
            except Exception as exc:
                for slot in active_slots:
                    attempts.append(
                        {
                            "status": "generation_error",
                            "reason": repr(exc),
                            "original_id": slot["original_id"],
                            "original_text": slot["original_text"],
                            "model_name": spec.name,
                            "model_id": spec.model_id,
                            "length_band": slot["band"].label,
                            "attempt_idx": attempt_idx,
                            "banned_words": slot["banned_words"],
                        }
                    )
                print(
                    f"[{model_idx}/{n_models}] attempt round {attempt_idx + 1}/{max_attempts}: "
                    f"generation error for {len(active_slots)} slot(s): {exc!r}",
                    flush=True,
                )
                break

            next_slots: List[Dict[str, Any]] = []
            accepted_this_round = 0
            for slot, candidates in zip(active_slots, results):
                slot_accepted = False
                for candidate_idx, candidate in enumerate(candidates):
                    row, ok = validate_candidate_row(
                        candidate,
                        candidate_idx,
                        slot["original"],
                        slot["original_id"],
                        slot["original_text"],
                        slot["banned_words"],
                        slot["band"],
                        spec,
                        attempt_idx,
                        semantic_validator,
                    )
                    attempts.append(row)
                    if ok and not slot_accepted:
                        accepted.append(row)
                        slot_accepted = True
                        accepted_this_round += 1
                if not slot_accepted:
                    next_slots.append(slot)

            print(
                f"[{model_idx}/{n_models}] attempt round {attempt_idx + 1}/{max_attempts}: "
                f"accepted={accepted_this_round} | remaining={len(next_slots)} | "
                f"round_attempts={sum(len(items) for items in results)} | "
                f"reasons={summarize_rejection_reasons(attempts)} | "
                f"round_time={format_elapsed(time.time() - round_t0)}",
                flush=True,
            )
            active_slots = next_slots
    finally:
        generator.close()
        print(
            f"[{model_idx}/{n_models}] closed {spec.name} | "
            f"model_attempts={len(attempts)} | model_accepted={len(accepted)} | "
            f"model_time={format_elapsed(time.time() - model_t0)}",
            flush=True,
        )

    return attempts, accepted


def generate_for_model(
    spec: RewriteModelSpec,
    model_idx: int,
    n_models: int,
    backend: str,
    originals: List[Dict[str, Any]],
    bands: List[LengthBand],
    generation_cfg: Dict[str, Any],
    vllm_cfg: Dict[str, Any],
    prompt_template: str | None,
    semantic_validator: SemanticValidator,
    dry_run: bool,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model_t0 = time.time()
    print(f"[{model_idx}/{n_models}] loading {spec.name}: {spec.model_id} via {backend}", flush=True)
    if dry_run:
        generator = DryRunRewriteGenerator(spec)
    elif backend == "hf":
        generator = HFRewriteGenerator(spec)
    elif backend == "vllm":
        generator = VLLMRewriteGenerator(spec, vllm_cfg)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    print(f"[{model_idx}/{n_models}] loaded {spec.name} in {format_elapsed(time.time() - model_t0)}", flush=True)
    attempts: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    max_attempts = int(generation_cfg.get("max_attempts_per_slot", 3))
    n_slots = len(originals) * len(bands)
    completed_slots = 0

    try:
        for original_idx, original in enumerate(originals, start=1):
            original_id = original.get("id") or original.get("original_id")
            original_text = original["text"]
            banned_words = original.get("banned_words") or extract_banned_words(original_text)
            preview = original_text[:90] + ("..." if len(original_text) > 90 else "")
            print(
                f"[{model_idx}/{n_models}] original {original_idx}/{len(originals)} "
                f"{original_id} | banned={len(banned_words)} | {preview}",
                flush=True,
            )
            for band in bands:
                slot_t0 = time.time()
                slot_accepted = 0
                slot_attempts_start = len(attempts)
                print(
                    f"[{model_idx}/{n_models}]   band {band.label}: generating "
                    f"up to {max_attempts} attempt(s)",
                    flush=True,
                )
                for attempt_idx in range(max_attempts):
                    prompt = build_prompt(original_text, banned_words, band, prompt_template)
                    try:
                        candidates = generator.generate(prompt, generation_cfg)
                    except Exception as exc:
                        attempts.append(
                            {
                                "status": "generation_error",
                                "reason": repr(exc),
                                "original_id": original_id,
                                "original_text": original_text,
                                "model_name": spec.name,
                                "model_id": spec.model_id,
                                "length_band": band.label,
                                "attempt_idx": attempt_idx,
                                "banned_words": banned_words,
                            }
                        )
                        print(
                            f"[{model_idx}/{n_models}]   band {band.label}: generation error "
                            f"on attempt {attempt_idx + 1}/{max_attempts}: {exc!r}",
                            flush=True,
                        )
                        break

                    accepted_this_attempt = 0
                    for candidate_idx, candidate in enumerate(candidates):
                        row, ok = validate_candidate_row(
                            candidate,
                            candidate_idx,
                            original,
                            original_id,
                            original_text,
                            banned_words,
                            band,
                            spec,
                            attempt_idx,
                            semantic_validator,
                        )
                        attempts.append(row)
                        if ok and slot_accepted == 0:
                            accepted.append(row)
                            slot_accepted += 1
                            accepted_this_attempt += 1
                    print(
                        f"[{model_idx}/{n_models}]   band {band.label}: attempt "
                        f"{attempt_idx + 1}/{max_attempts} produced {len(candidates)} candidate(s), "
                        f"accepted_this_attempt={accepted_this_attempt}",
                        flush=True,
                    )
                    if slot_accepted:
                        break
                completed_slots += 1
                slot_attempts = attempts[slot_attempts_start:]
                slot_status = "accepted" if slot_accepted else "no accepted rewrite"
                print(
                    f"[{model_idx}/{n_models}]   band {band.label}: {slot_status} | "
                    f"slot_attempts={len(slot_attempts)} | "
                    f"reasons={summarize_rejection_reasons(slot_attempts)} | "
                    f"slot_time={format_elapsed(time.time() - slot_t0)} | "
                    f"slots={completed_slots}/{n_slots}",
                    flush=True,
                )
    finally:
        generator.close()
        print(
            f"[{model_idx}/{n_models}] closed {spec.name} | "
            f"model_attempts={len(attempts)} | model_accepted={len(accepted)} | "
            f"model_time={format_elapsed(time.time() - model_t0)}",
            flush=True,
        )

    return attempts, accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--length-bands", default=None, help="Comma separated bands, e.g. 5-7,10-12")
    parser.add_argument("--max-originals", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--model-shard", default=None, help="Run only model shard INDEX/TOTAL, e.g. 0/4")
    parser.add_argument(
        "--backend",
        default="hf",
        choices=["hf", "vllm"],
        help="Generation backend. Use vllm from Linux/WSL2 with a CUDA-capable GPU.",
    )
    args = parser.parse_args()

    t0 = time.time()
    config_path = resolve_project_path(args.models_config)
    input_path = resolve_project_path(args.input)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = read_json(config_path)
    shard = parse_shard(args.model_shard)
    specs_all = model_specs(cfg, include_disabled=args.include_disabled)
    specs = apply_model_shard(specs_all, shard)
    if not specs:
        raise SystemExit("No enabled models found. Enable models in config or pass --include-disabled for dry runs.")

    originals = list(read_jsonl(input_path))
    if args.max_originals is not None:
        originals = originals[: args.max_originals]
    bands = parse_bands(args.length_bands)
    generation_cfg = cfg.get("generation", {})
    vllm_cfg = cfg.get("vllm", {})
    prompt_template = cfg.get("prompting", {}).get("template")
    semantic_validator = SemanticValidator(cfg.get("semantic_validation", {}))

    all_attempts: List[Dict[str, Any]] = []
    all_accepted: List[Dict[str, Any]] = []

    print("SetConCA V2 constrained-set generation", flush=True)
    print(f"  input: {input_path}", flush=True)
    print(f"  out_dir: {out_dir}", flush=True)
    print(f"  backend: {args.backend}", flush=True)
    print(f"  models: {len(specs)} of {len(specs_all)}", flush=True)
    print(f"  model_shard: {args.model_shard or 'none'}", flush=True)
    print(f"  originals: {len(originals)}", flush=True)
    print(f"  bands: {', '.join(band.label for band in bands)}", flush=True)
    print(f"  max_attempts_per_slot: {generation_cfg.get('max_attempts_per_slot', 3)}", flush=True)
    print(f"  num_return_sequences: {generation_cfg.get('num_return_sequences', 1)}", flush=True)
    print(f"  semantic_validation: {semantic_validator.enabled}", flush=True)
    print(f"  device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

    for idx, spec in enumerate(specs, start=1):
        if args.backend == "vllm" and not args.dry_run:
            attempts, accepted = generate_for_model_vllm_batched(
                spec,
                idx,
                len(specs),
                originals,
                bands,
                generation_cfg,
                vllm_cfg,
                prompt_template,
                semantic_validator,
            )
        else:
            attempts, accepted = generate_for_model(
                spec,
                idx,
                len(specs),
                args.backend,
                originals,
                bands,
                generation_cfg,
                vllm_cfg,
                prompt_template,
                semantic_validator,
                args.dry_run,
            )
        all_attempts.extend(attempts)
        all_accepted.extend(accepted)
        write_jsonl(out_dir / "attempts.jsonl", all_attempts)
        write_jsonl(out_dir / "accepted.jsonl", all_accepted)
        grouped = group_accepted(all_accepted)
        write_jsonl(out_dir / "sets.jsonl", grouped)
        write_review_table(out_dir / "review_table.md", grouped)
        print(
            f"[{idx}/{len(specs)}] saved artifacts | "
            f"total_attempts={len(all_attempts)} | total_accepted={len(all_accepted)} | "
            f"sets={len(grouped)} | elapsed={format_elapsed(time.time() - t0)}",
            flush=True,
        )

    manifest = {
        "models_config": str(config_path),
        "input": str(input_path),
        "out_dir": str(out_dir),
        "dry_run": args.dry_run,
        "backend": args.backend,
        "model_shard": args.model_shard,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "n_originals": len(originals),
        "length_bands": [band.__dict__ for band in bands],
        "n_models": len(specs),
        "n_models_total": len(specs_all),
        "n_attempts": len(all_attempts),
        "n_accepted": len(all_accepted),
        "elapsed_s": time.time() - t0,
    }
    write_json(out_dir / "run_manifest.json", manifest)
    semantic_validator.close()
    print(
        f"Done. Saved dataset artifacts to {out_dir} | "
        f"attempts={len(all_attempts)} | accepted={len(all_accepted)} | "
        f"elapsed={format_elapsed(time.time() - t0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
