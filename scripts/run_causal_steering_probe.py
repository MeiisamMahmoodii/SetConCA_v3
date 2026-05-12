from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from model.setconca_v2 import SetConCAV2
from setconca_v2.paths import resolve_project_path
from setconca_v2.steering import (
    SteeringCandidate,
    concept_sign_from_codes,
    keyword_score,
    load_steering_candidates,
    model_dir_from_key,
    write_csv_rows,
    write_json,
)


DEFAULT_PROMPTS = [
    "Write one short news headline about a current event:",
    "Complete this news sentence in one sentence: Officials announced",
    "Write a neutral one-sentence news brief:",
    "In one sentence, describe a recent development:",
]


def dtype_from_name(name: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def read_prompts(path: str | Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    path = Path(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [str(item) for item in obj]
        return [str(item["prompt"]) for item in obj["prompts"]]
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["prompt"] for row in rows]


def load_setconca_model(model_dir: Path) -> tuple[SetConCAV2, dict[str, Any]]:
    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = payload["manifest"]
    model = SetConCAV2(
        hidden_dim=int(manifest["bank"]["hidden_dim"]),
        concept_dim=int(manifest["concept_dim"]),
        topk=int(manifest["topk"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, manifest


def decoder_direction(
    run_dir: Path,
    candidate: SteeringCandidate,
    *,
    sign_mode: str,
    device: torch.device,
    scale_to: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    target_dir = model_dir_from_key(run_dir, candidate.target_key)
    setconca, manifest = load_setconca_model(target_dir)
    dim = candidate.target_concept_dim
    direction = setconca.shared_decoder.weight[:, dim].detach().float()
    sign = 1.0
    if sign_mode == "active_mean":
        sign = concept_sign_from_codes(target_dir / "codes.pt", dim)
    elif sign_mode == "opposite_active":
        sign = -concept_sign_from_codes(target_dir / "codes.pt", dim)
    elif sign_mode == "negative":
        sign = -1.0
    direction = direction * sign
    norm = float(direction.norm())
    if norm > 0:
        direction = direction / norm * float(scale_to)
    meta = {
        "target_model_dir": str(target_dir),
        "target_model_id": manifest["bank"]["model_id"],
        "target_layer": int(manifest["bank"]["layer"]),
        "target_concept_dim": dim,
        "direction_norm_before_scale": norm,
        "sign": sign,
        "scale_to": scale_to,
    }
    return direction.to(device), meta


def get_decoder_layers(model: torch.nn.Module) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise ValueError("Could not find a supported decoder layer stack on this model.")


def add_to_layer_output(output: Any, delta: torch.Tensor, token_position: str) -> Any:
    if isinstance(output, tuple):
        hidden = output[0]
        rest = output[1:]
    else:
        hidden = output
        rest = None
    if token_position == "last":
        hidden = hidden.clone()
        hidden[:, -1:, :] = hidden[:, -1:, :] + delta.view(1, 1, -1).to(hidden.dtype)
    elif token_position == "all":
        hidden = hidden + delta.view(1, 1, -1).to(hidden.dtype)
    else:
        raise ValueError(f"Unsupported token_position: {token_position}")
    return (hidden, *rest) if rest is not None else hidden


def generate_once(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> str:
    torch.manual_seed(seed)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generate_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    with torch.no_grad():
        out = model.generate(**generate_kwargs)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :].strip() if text.startswith(prompt) else text.strip()


def run_candidate(
    candidate: SteeringCandidate,
    *,
    run_dir: Path,
    prompts: list[str],
    alphas: list[float],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    direction, direction_meta = decoder_direction(
        run_dir,
        candidate,
        sign_mode=args.sign_mode,
        device=device,
        scale_to=args.direction_norm,
    )
    tokenizer = AutoTokenizer.from_pretrained(direction_meta["target_model_id"], trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        direction_meta["target_model_id"],
        torch_dtype=dtype_from_name(args.dtype),
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()
    layers = get_decoder_layers(model)
    target_layer = int(direction_meta["target_layer"])
    rows = []
    for prompt_idx, prompt in enumerate(prompts):
        baseline = generate_once(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed + prompt_idx,
        )
        base_score, base_hits = keyword_score(baseline, candidate.keywords)
        for alpha in alphas:
            if float(alpha) == 0.0:
                steered = baseline
            else:
                handle = layers[target_layer].register_forward_hook(
                    lambda _module, _inputs, output, a=float(alpha): add_to_layer_output(
                        output,
                        direction * a,
                        args.token_position,
                    )
                )
                try:
                    steered = generate_once(
                        model,
                        tokenizer,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        seed=args.seed + prompt_idx,
                    )
                finally:
                    handle.remove()
            steer_score, steer_hits = keyword_score(steered, candidate.keywords)
            rows.append(
                {
                    "rank": candidate.rank,
                    "candidate": candidate.candidate,
                    "label": candidate.label,
                    "use_for_steering": candidate.use_for_steering,
                    "target_model_id": direction_meta["target_model_id"],
                    "target_layer": direction_meta["target_layer"],
                    "target_concept_dim": candidate.target_concept_dim,
                    "alpha": alpha,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "baseline": baseline,
                    "steered": steered,
                    "baseline_keyword_score": base_score,
                    "steered_keyword_score": steer_score,
                    "keyword_gain": steer_score - base_score,
                    "baseline_keyword_hits": ";".join(base_hits),
                    "steered_keyword_hits": ";".join(steer_hits),
                    "keywords": ";".join(candidate.keywords),
                }
            )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, direction_meta


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["rank"], row["candidate"], row["target_model_id"], row["alpha"])
        grouped.setdefault(key, []).append(row)
    out = []
    for key, vals in sorted(grouped.items()):
        out.append(
            {
                "rank": key[0],
                "candidate": key[1],
                "target_model_id": key[2],
                "alpha": key[3],
                "n": len(vals),
                "mean_baseline_keyword_score": sum(float(v["baseline_keyword_score"]) for v in vals) / len(vals),
                "mean_steered_keyword_score": sum(float(v["steered_keyword_score"]) for v in vals) / len(vals),
                "mean_keyword_gain": sum(float(v["keyword_gain"]) for v in vals) / len(vals),
            }
        )
    return out


def write_report(out_dir: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Causal Steering Probe Report",
        "",
        "This is a first behavioral probe, not a final steering claim.",
        "",
        "The intervention adds a normalized SetConCA target-decoder concept direction to the target model hidden state during generation.",
        "",
        "## Manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
        "## Summary",
        "",
        "| Rank | Candidate | Target model | Alpha | Mean keyword gain |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['rank']} | {row['candidate']} | {row['target_model_id']} | "
            f"{float(row['alpha']):.2f} | {float(row['mean_keyword_gain']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
            "- Keyword gain is only a weak first proxy for behavioral steering.",
            "- Manual reading of generations is required before claiming causal concept control.",
            "- Negative or zero gain is still useful evidence and should be kept.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-pass generation-time causal steering probes.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--prompts", default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--only-use", default="yes,yes_maybe")
    parser.add_argument("--alphas", default="0,1,2,4")
    parser.add_argument("--direction-norm", type=float, default=1.0)
    parser.add_argument("--sign-mode", choices=["active_mean", "opposite_active", "positive", "negative"], default="active_mean")
    parser.add_argument("--token-position", choices=["last", "all"], default="last")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    candidates = load_steering_candidates(resolve_project_path(args.candidates))
    allowed = {item.strip() for item in args.only_use.split(",") if item.strip()}
    candidates = [item for item in candidates if item.use_for_steering in allowed]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]
    prompts = read_prompts(resolve_project_path(args.prompts) if args.prompts else None)
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]
    run_dir = resolve_project_path(args.run_dir)
    out_dir = resolve_project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "candidates": str(resolve_project_path(args.candidates)),
        "run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "n_candidates": len(candidates),
        "n_prompts": len(prompts),
        "alphas": alphas,
        "args": vars(args),
    }
    write_json(out_dir / "run_manifest.json", manifest)
    if args.dry_run:
        write_csv_rows(out_dir / "selected_candidates.csv", [c.__dict__ | {"keywords": ";".join(c.keywords)} for c in candidates])
        print(json.dumps(manifest, indent=2))
        return

    all_rows = []
    direction_meta = []
    for idx, candidate in enumerate(candidates, start=1):
        print(f"[{idx}/{len(candidates)}] steering {candidate.candidate} on {candidate.target_key}", flush=True)
        rows, meta = run_candidate(candidate, run_dir=run_dir, prompts=prompts, alphas=alphas, args=args)
        all_rows.extend(rows)
        direction_meta.append({"rank": candidate.rank, "candidate": candidate.candidate, **meta})
        write_csv_rows(out_dir / "generations.csv", all_rows)
    summary = summarize(all_rows)
    write_csv_rows(out_dir / "summary.csv", summary)
    write_json(out_dir / "direction_meta.json", direction_meta)
    write_report(out_dir, all_rows, summary, manifest)
    print(f"Wrote {len(all_rows)} generation rows to {out_dir}")


if __name__ == "__main__":
    main()
