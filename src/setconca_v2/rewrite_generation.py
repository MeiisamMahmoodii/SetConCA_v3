from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from .text_constraints import LengthBand, clean_model_output


@dataclass
class RewriteModelSpec:
    name: str
    model_id: str
    enabled: bool = True
    revision: str | None = None
    torch_dtype: str = "auto"
    trust_remote_code: bool = False


def resolve_dtype(dtype_name: str) -> Any:
    if dtype_name == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


DEFAULT_PROMPT_TEMPLATE = (
    "You are creating a controlled paraphrase dataset for a scientific experiment. "
    "Rewrite the sentence so that factual meaning is preserved, but wording is as different as possible.\n\n"
    "Original sentence:\n{original}\n\n"
    "Forbidden words. Do not use any of these exact words:\n{banned_words}\n\n"
    "Required length: {min_words}-{max_words} words.\n\n"
    "Rules:\n"
    "1. Return exactly one sentence.\n"
    "2. Preserve factual meaning, entities, polarity, time, and event relation.\n"
    "3. Do not add new facts.\n"
    "4. Do not use any forbidden word.\n"
    "5. Change syntax and vocabulary substantially.\n\n"
    "Final rewrite only:"
)


def build_prompt(
    original: str,
    banned_words: Sequence[str],
    band: LengthBand,
    template: str | None = None,
) -> str:
    banned = ", ".join(banned_words) if banned_words else "(none)"
    prompt_template = template or DEFAULT_PROMPT_TEMPLATE
    return prompt_template.format(
        original=original,
        banned_words=banned,
        min_words=band.min_words,
        max_words=band.max_words,
        length_band=band.label,
    )


class HFRewriteGenerator:
    def __init__(self, spec: RewriteModelSpec, device: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = resolve_dtype(spec.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            revision=spec.revision,
            torch_dtype=dtype,
            trust_remote_code=spec.trust_remote_code,
            local_files_only=False,
        ).to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, generation_cfg: Dict[str, Any]) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        num_return_sequences = int(generation_cfg.get("num_return_sequences", 1))
        gen_kwargs = {
            "max_new_tokens": int(generation_cfg.get("max_new_tokens", 80)),
            "temperature": float(generation_cfg.get("temperature", 0.8)),
            "top_p": float(generation_cfg.get("top_p", 0.9)),
            "do_sample": bool(generation_cfg.get("do_sample", True)),
            "num_return_sequences": num_return_sequences,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        texts = []
        prompt_len = inputs["input_ids"].shape[-1]
        for ids in output_ids:
            generated = ids[prompt_len:]
            texts.append(clean_model_output(self.tokenizer.decode(generated, skip_special_tokens=True)))
        return texts

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class VLLMRewriteGenerator:
    """vLLM offline generator for high-throughput CUDA/WSL/Linux runs."""

    def __init__(self, spec: RewriteModelSpec, vllm_cfg: Dict[str, Any] | None = None):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError(
                "vLLM is not installed in this Python environment. Install it in Linux/WSL2 "
                "or use the default --backend hf."
            ) from exc

        self.spec = spec
        self.vllm_cfg = vllm_cfg or {}
        self.SamplingParams = SamplingParams
        llm_kwargs: Dict[str, Any] = {
            "model": spec.model_id,
            "trust_remote_code": spec.trust_remote_code,
            "dtype": spec.torch_dtype,
        }
        if spec.revision:
            llm_kwargs["revision"] = spec.revision
        for key in (
            "tensor_parallel_size",
            "gpu_memory_utilization",
            "max_model_len",
            "download_dir",
            "enforce_eager",
            "quantization",
        ):
            if key in self.vllm_cfg:
                llm_kwargs[key] = self.vllm_cfg[key]
        self.llm = LLM(**llm_kwargs)

    def generate(self, prompt: str, generation_cfg: Dict[str, Any]) -> List[str]:
        do_sample = bool(generation_cfg.get("do_sample", True))
        temperature = float(generation_cfg.get("temperature", 0.8)) if do_sample else 0.0
        sampling_params = self.SamplingParams(
            n=int(generation_cfg.get("num_return_sequences", 1)),
            max_tokens=int(generation_cfg.get("max_new_tokens", 80)),
            temperature=temperature,
            top_p=float(generation_cfg.get("top_p", 0.9)),
        )
        outputs = self.llm.generate([prompt], sampling_params)
        if not outputs:
            return []
        return [clean_model_output(item.text) for item in outputs[0].outputs]

    def close(self) -> None:
        del self.llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class DryRunRewriteGenerator:
    """Deterministic fallback for tests and pipeline dry-runs."""

    def __init__(self, spec: RewriteModelSpec):
        self.spec = spec

    def generate(self, prompt: str, generation_cfg: Dict[str, Any]) -> List[str]:
        # The caller validates these, so include several lengths.
        return [
            "Markets climbed as overseas chip demand improved",
            "Researchers found seas warmed unusually fast this summer",
            "Officials discussed peace terms during regional talks",
        ]

    def close(self) -> None:
        return None
