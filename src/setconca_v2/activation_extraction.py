from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from .io_utils import read_jsonl
from .set_dataset import file_sha256


@dataclass
class SemanticSetView:
    original_id: str
    label: str | None
    source: str | None
    texts: List[str]
    rewrite_meta: List[Dict[str, Any]]


def stable_seed(*parts: str, base_seed: int = 0) -> int:
    h = hashlib.sha256()
    h.update(str(base_seed).encode("utf-8"))
    for part in parts:
        h.update(str(part).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big")


def load_semantic_views(
    path: str | Path,
    *,
    views: int,
    include_original: bool = True,
    max_sets: int | None = None,
    seed: int = 0,
) -> List[SemanticSetView]:
    rows = []
    for idx, row in enumerate(read_jsonl(Path(path))):
        if max_sets is not None and len(rows) >= max_sets:
            break
        rewrites = list(row.get("rewrites", []))
        n_rewrites_needed = views - 1 if include_original else views
        if len(rewrites) < n_rewrites_needed:
            continue
        rng = random.Random(stable_seed(str(row.get("original_id")), base_seed=seed))
        sampled = rewrites[:]
        rng.shuffle(sampled)
        sampled = sampled[:n_rewrites_needed]
        texts = [row["original_text"]] if include_original else []
        texts.extend(item["text"] for item in sampled)
        rows.append(
            SemanticSetView(
                original_id=str(row.get("original_id")),
                label=row.get("label"),
                source=row.get("source"),
                texts=texts,
                rewrite_meta=sampled,
            )
        )
    return rows


def extract_position(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "last":
        idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[batch_idx, idx]
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    raise ValueError(f"Unsupported token position mode: {mode}")


def make_fake_hidden(n_sets: int, views: int, hidden_dim: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return torch.randn(n_sets, views, hidden_dim, generator=gen)


def extract_hf_activations(
    texts: List[str],
    *,
    model_id: str,
    layer: int,
    token_position: str,
    batch_size: int,
    max_length: int,
    device: str | None,
    dtype: str,
    trust_remote_code: bool,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = {
        "auto": torch.float16 if resolved_device == "cuda" else torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    ).to(resolved_device)
    model.eval()

    chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(resolved_device)
            out = model(**encoded, output_hidden_states=True, use_cache=False)
            layer_hidden = out.hidden_states[layer]
            pooled = extract_position(layer_hidden, encoded["attention_mask"], token_position)
            chunks.append(pooled.detach().cpu().float())

    meta = {
        "model_id": model_id,
        "layer": layer,
        "token_position": token_position,
        "device": resolved_device,
        "dtype": dtype,
        "max_length": max_length,
        "hidden_dim": int(chunks[0].shape[-1]) if chunks else 0,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0), meta


def build_activation_bank(
    sets_path: str | Path,
    *,
    out_path: str | Path,
    model_id: str,
    layer: int,
    views: int,
    token_position: str = "last",
    batch_size: int = 8,
    max_length: int = 256,
    max_sets: int | None = None,
    include_original: bool = True,
    seed: int = 0,
    device: str | None = None,
    dtype: str = "auto",
    trust_remote_code: bool = False,
    dry_run: bool = False,
    fake_hidden_dim: int = 64,
) -> Dict[str, Any]:
    t0 = time.time()
    sets_path = Path(sets_path)
    out_path = Path(out_path)
    semantic_views = load_semantic_views(
        sets_path,
        views=views,
        include_original=include_original,
        max_sets=max_sets,
        seed=seed,
    )
    if not semantic_views:
        raise ValueError("No semantic sets had enough views for the requested extraction.")

    flat_texts = [text for item in semantic_views for text in item.texts]
    if dry_run:
        hidden_dim = fake_hidden_dim
        flat_hidden = make_fake_hidden(len(flat_texts), 1, hidden_dim, seed=seed).squeeze(1)
        extraction_meta = {
            "model_id": model_id,
            "layer": layer,
            "token_position": token_position,
            "device": "dry-run",
            "dtype": "float32",
            "max_length": max_length,
            "hidden_dim": hidden_dim,
        }
    else:
        flat_hidden, extraction_meta = extract_hf_activations(
            flat_texts,
            model_id=model_id,
            layer=layer,
            token_position=token_position,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

    hidden = flat_hidden.reshape(len(semantic_views), views, -1).contiguous()
    meta = {
        **extraction_meta,
        "sets_path": str(sets_path),
        "sets_sha256": file_sha256(sets_path),
        "out_path": str(out_path),
        "n_sets": len(semantic_views),
        "views": views,
        "include_original": include_original,
        "seed": seed,
        "dry_run": dry_run,
        "elapsed_s": time.time() - t0,
    }
    payload = {
        "hidden": hidden,
        "texts": [item.texts[0] for item in semantic_views],
        "view_texts": [item.texts for item in semantic_views],
        "set_ids": [item.original_id for item in semantic_views],
        "labels": [item.label for item in semantic_views],
        "sources": [item.source for item in semantic_views],
        "rewrite_meta": [item.rewrite_meta for item in semantic_views],
        "meta": meta,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return meta
