from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from vg_common import compact_text, read_json, read_jsonl, stable_hash, write_jsonl


def quiet_common_warnings() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=r".*cache-system uses symlinks.*")
    warnings.filterwarnings("ignore", message=r".*`torch_dtype` is deprecated.*")
    warnings.filterwarnings("ignore", message=r".*Passing `generation_config` together with generation-related arguments.*")
    warnings.filterwarnings("ignore", message=r".*Both `max_new_tokens`.*")
    warnings.filterwarnings("ignore", message=r".*Keyword argument `do_sample` is not a valid argument.*")
    warnings.filterwarnings("ignore", message=r".*Keyword argument `temperature` is not a valid argument.*")
    warnings.filterwarnings("ignore", message=r".*Keyword argument `top_p` is not a valid argument.*")
    warnings.filterwarnings("ignore", message=r".*Kwargs passed to `processor.__call__`.*")
    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass


class VLMAdapter(Protocol):
    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        ...

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any]) -> list[str]:
        ...


def mock_generate(job: dict) -> str:
    image_id = job["image"]["image_id"]
    language = job["language"]["name"]
    script_hint = job["language"].get("script_hint", "latin")
    model_source = job["model"].get("source_id", "mock_model")
    view = job["view"]["view_id"].replace("_", " ")
    tags = ", ".join(job["image"].get("topic_tags", [])) or "general visual content"
    if script_hint == "arabic":
        return f"مشهد {view}: يصف نموذج {model_source} الصورة كمحتوى بصري عن {tags} مع عناصر واضحة مرتبطة بالصورة {image_id}."
    if script_hint == "devanagari":
        return f"दृश्य {view}: मॉडल {model_source} के अनुसार यह छवि {tags} से जुड़ी दृश्य सामग्री दिखाती है और विवरण चित्र {image_id} पर आधारित है."
    if script_hint == "cjk" and job["language"].get("code") == "zh":
        return f"图像 {view}: 模型 {model_source} 认为这张图像展示了与 {tags} 有关的视觉场景，描述基于图像 {image_id}。"
    if script_hint == "cjk":
        return f"画像 {view}: モデル {model_source} は、この画像が {tags} に関する視覚的な場面を示すと説明し、画像 {image_id} に基づいています。"
    if script_hint == "hangul":
        return f"장면 {view}: 모델 {model_source}는 이 이미지가 {tags} 관련 시각적 내용을 보여 준다고 설명하며 이미지 {image_id}에 근거합니다."
    return f"{language} {view}: Model {model_source} says this image appears to show {tags}. The description is grounded in image anchor {image_id}."


def image_ref(path: str, mode: str) -> Any:
    resolved = Path(path).resolve()
    if mode == "path":
        return str(resolved)
    if mode == "file_uri":
        return resolved.as_uri()
    if mode == "pil":
        return Image.open(resolved).convert("RGB")
    raise ValueError(f"unknown image ref mode: {mode}")


def decode_pipeline_output(output: Any) -> str:
    if isinstance(output, str):
        return compact_text(output)
    if isinstance(output, list) and output:
        return decode_pipeline_output(output[0])
    if isinstance(output, dict):
        for key in ["generated_text", "text", "answer"]:
            if key in output:
                value = output[key]
                if isinstance(value, list) and value and isinstance(value[-1], dict):
                    content = value[-1].get("content", "")
                    if isinstance(content, list):
                        return compact_text(" ".join(str(item.get("text", "")) for item in content if isinstance(item, dict)))
                    return compact_text(str(content))
                return compact_text(str(value))
    return compact_text(str(output))


def resolve_load_dtype(dtype: str) -> str:
    if dtype not in {"auto", ""}:
        return dtype
    import torch

    if torch.cuda.is_available():
        return "bfloat16"
    return "float32"


def configure_cuda_runtime() -> None:
    import torch

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True


def resolve_device_map(device_map: str) -> str | dict[str, int]:
    """Pin the full model to the sole visible GPU (avoids device_map=auto quirks in workers)."""
    if device_map != "auto":
        return device_map
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and "," not in visible:
        return {"": 0}
    return "auto"


def resolve_batch_size(generation_config: dict[str, Any], cli_batch_size: int | None) -> int:
    if cli_batch_size is not None:
        return max(1, int(cli_batch_size))
    return max(1, int(generation_config.get("vlm_batch_size", 4)))


def build_model_kwargs(*, device_map: str, dtype: str, trust_remote_code: bool = False) -> dict[str, Any]:
    resolved_dtype = resolve_load_dtype(dtype)
    kwargs: dict[str, Any] = {"device_map": device_map}
    if resolved_dtype != "none":
        kwargs["dtype"] = resolved_dtype
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    kwargs["attn_implementation"] = "sdpa"
    configure_cuda_runtime()
    return kwargs


def generation_kwargs_from_config(generation_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 120)),
        "do_sample": float(generation_config.get("temperature", 0.0)) > 0,
        "temperature": float(generation_config.get("temperature", 0.2)),
        "top_p": float(generation_config.get("top_p", 0.9)),
    }


def qwen_vl_model_class(model_id: str) -> str:
    if "qwen3" in model_id.lower():
        return "Qwen3VLForConditionalGeneration"
    return "Qwen2_5_VLForConditionalGeneration"


def qwen_job_messages(job: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(Path(job["image"]["image_path"]).resolve())},
                {"type": "text", "text": job["prompt"]},
            ],
        }
    ]


def batch_input_lengths(inputs: Any) -> list[int]:
    if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
        return [int(length) for length in inputs.attention_mask.sum(dim=1).tolist()]
    seq_len = inputs.input_ids.shape[-1]
    batch_size = inputs.input_ids.shape[0]
    return [seq_len] * batch_size


def decode_generated_batch(
    processor: Any,
    outputs: Any,
    input_lengths: list[int],
    *,
    skip_special_tokens: bool = True,
) -> list[str]:
    texts: list[str] = []
    for index, in_len in enumerate(input_lengths):
        token_ids = outputs[index, int(in_len) :]
        text = processor.decode(token_ids, skip_special_tokens=skip_special_tokens)
        if hasattr(processor, "parse_response"):
            parsed = processor.parse_response(text)
            if isinstance(parsed, dict):
                text = parsed.get("answer") or parsed.get("response") or parsed.get("content") or text
            elif parsed is not None:
                text = str(parsed)
        texts.append(compact_text(text))
    return texts


class GenericPipelineAdapter:
    def __init__(self, model_id: str, *, device_map: str, dtype: str, image_ref_mode: str) -> None:
        from transformers import pipeline

        resolved_dtype = resolve_load_dtype(dtype)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "task": "image-text-to-text",
            "device_map": device_map,
            "model_kwargs": {"attn_implementation": "sdpa"},
        }
        if resolved_dtype != "none":
            kwargs["dtype"] = resolved_dtype
        configure_cuda_runtime()
        self.pipe = pipeline(**kwargs)
        self.image_ref_mode = image_ref_mode

    def _messages_for_job(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        image = image_ref(job["image"]["image_path"], self.image_ref_mode)
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image},
                    {"type": "text", "text": job["prompt"]},
                ],
            }
        ]

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        return self.generate_batch([job], generation_config)[0]

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any]) -> list[str]:
        if not jobs:
            return []
        messages_list = [self._messages_for_job(job) for job in jobs]
        pipe_kwargs = {
            **generation_kwargs_from_config(generation_config),
            "return_full_text": False,
        }
        try:
            if len(jobs) == 1:
                output = self.pipe(text=messages_list[0], **pipe_kwargs)
                return [decode_pipeline_output(output)]
            output = self.pipe(text=messages_list, **pipe_kwargs)
            if isinstance(output, list):
                return [decode_pipeline_output(item) for item in output]
            return [decode_pipeline_output(output)]
        except Exception:
            return [self._generate_single(job, generation_config) for job in jobs]

    def _generate_single(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        output = self.pipe(text=self._messages_for_job(job), **{**generation_kwargs_from_config(generation_config), "return_full_text": False})
        return decode_pipeline_output(output)


class AutoImageTextAdapter:
    def __init__(
        self,
        model_id: str,
        *,
        model_class: str,
        device_map: str,
        dtype: str,
        image_ref_mode: str,
        trust_remote_code: bool = False,
    ) -> None:
        import torch
        from transformers import AutoProcessor

        transformers = __import__("transformers", fromlist=[model_class])
        model_cls = getattr(transformers, model_class)
        model_kwargs = build_model_kwargs(device_map=device_map, dtype=dtype, trust_remote_code=trust_remote_code)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        try:
            self.model = model_cls.from_pretrained(model_id, **model_kwargs).eval()
        except TypeError:
            model_kwargs.pop("attn_implementation", None)
            self.model = model_cls.from_pretrained(model_id, **model_kwargs).eval()
        self.torch = torch
        self.image_ref_mode = image_ref_mode

    def _messages_for_job(self, job: dict[str, Any]) -> list[dict[str, Any]]:
        image = image_ref(job["image"]["image_path"], self.image_ref_mode)
        image_item = {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image}
        return [{"role": "user", "content": [image_item, {"type": "text", "text": job["prompt"]}]}]

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        return self.generate_batch([job], generation_config)[0]

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any]) -> list[str]:
        if not jobs:
            return []
        messages_list = [self._messages_for_job(job) for job in jobs]
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
        images: list[Any] = []
        for messages in messages_list:
            image = messages[0]["content"][0]
            images.append(image.get("image") or image.get("url"))
        inputs = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_lengths = batch_input_lengths(inputs)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs_from_config(generation_config))
        return decode_generated_batch(self.processor, outputs, input_lengths, skip_special_tokens=False)


class QwenVLAdapter(AutoImageTextAdapter):
    def __init__(self, model_id: str, *, device_map: str, dtype: str, image_ref_mode: str) -> None:
        try:
            import qwen_vl_utils  # noqa: F401
        except Exception:
            print(
                "qwen-vl-utils is not installed; falling back to the generic image-text-to-text pipeline for Qwen.",
                file=sys.stderr,
            )
            self._fallback = GenericPipelineAdapter(model_id, device_map=device_map, dtype=dtype, image_ref_mode=image_ref_mode)
            return

        self._fallback = None
        super().__init__(
            model_id,
            model_class=qwen_vl_model_class(model_id),
            device_map=device_map,
            dtype=dtype,
            image_ref_mode="path",
        )

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        return self.generate_batch([job], generation_config)[0]

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any]) -> list[str]:
        if self._fallback is not None:
            return self._fallback.generate_batch(jobs, generation_config)
        if not jobs:
            return []

        from qwen_vl_utils import process_vision_info

        messages_list = [qwen_job_messages(job) for job in jobs]
        texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_list
        ]
        image_inputs, video_inputs = process_vision_info(messages_list)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_lengths = batch_input_lengths(inputs)
        with self.torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs_from_config(generation_config))
        return decode_generated_batch(self.processor, outputs, input_lengths, skip_special_tokens=True)


class InternVLAdapter:
    def __init__(self, model_id: str, *, device_map: str, dtype: str, max_tiles: int) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        torch_dtype = torch.bfloat16 if dtype in {"auto", "bfloat16", "bf16"} else torch.float16
        kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if device_map != "none":
            kwargs["device_map"] = device_map
        self.model = AutoModel.from_pretrained(model_id, **kwargs).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.dtype = torch_dtype
        self.max_tiles = max_tiles

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any]) -> list[str]:
        return [self.generate(job, generation_config) for job in jobs]

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        pixel_values = load_internvl_image(job["image"]["image_path"], max_num=self.max_tiles).to(self.dtype)
        if hasattr(self.model, "device"):
            pixel_values = pixel_values.to(self.model.device)
        elif self.torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        prompt = "<image>\n" + job["prompt"]
        config = {
            "max_new_tokens": int(generation_config.get("max_new_tokens", 120)),
            "do_sample": float(generation_config.get("temperature", 0.0)) > 0,
            "temperature": float(generation_config.get("temperature", 0.2)),
            "top_p": float(generation_config.get("top_p", 0.9)),
        }
        with self.torch.inference_mode():
            response = self.model.chat(self.tokenizer, pixel_values, prompt, config)
        return compact_text(response)


def load_internvl_image(image_file: str, *, input_size: int = 448, max_num: int = 12) -> Any:
    import numpy as np
    import torch

    image = Image.open(image_file).convert("RGB")
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    tensors = []
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    for tile in tiles:
        tile = tile.resize((input_size, input_size), Image.Resampling.BICUBIC)
        array = np.asarray(tile).astype(np.float32) / 255.0
        array = (array - mean) / std
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        tensors.append(tensor)
    return torch.stack(tensors)


def dynamic_preprocess(image: Image.Image, *, min_num: int = 1, max_num: int = 12, image_size: int = 448, use_thumbnail: bool = False) -> list[Image.Image]:
    width, height = image.size
    aspect_ratio = width / height
    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda item: item[0] * item[1],
    )
    best_ratio = min(
        target_ratios,
        key=lambda ratio: (
            abs(aspect_ratio - (ratio[0] / ratio[1])),
            -width * height if width * height > 0.5 * image_size * image_size * ratio[0] * ratio[1] else 0,
        ),
    )
    target_width = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for block in range(best_ratio[0] * best_ratio[1]):
        x = (block % (target_width // image_size)) * image_size
        y = (block // (target_width // image_size)) * image_size
        tiles.append(resized.crop((x, y, x + image_size, y + image_size)))
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def make_adapter(model: dict[str, Any], args: argparse.Namespace) -> VLMAdapter:
    adapter = model.get("adapter", "generic_pipeline")
    model_id = model["model_id"]
    image_ref_mode = model.get("image_ref_mode") or args.image_ref_mode
    device_map = resolve_device_map(args.device_map)
    if adapter == "qwen_vl":
        return QwenVLAdapter(model_id, device_map=device_map, dtype=args.dtype, image_ref_mode=image_ref_mode)
    if adapter == "internvl_chat":
        return InternVLAdapter(model_id, device_map=args.device_map, dtype=args.dtype, max_tiles=args.internvl_max_tiles)
    if adapter == "gemma4_multimodal":
        return AutoImageTextAdapter(
            model_id,
            model_class="AutoModelForMultimodalLM",
            device_map=args.device_map,
            dtype=args.dtype,
            image_ref_mode=image_ref_mode,
        )
    if adapter == "auto_image_text":
        return AutoImageTextAdapter(
            model_id,
            model_class="AutoModelForImageTextToText",
            device_map=args.device_map,
            dtype=args.dtype,
            image_ref_mode=image_ref_mode,
        )
    if adapter == "generic_pipeline":
        return GenericPipelineAdapter(model_id, device_map=args.device_map, dtype=args.dtype, image_ref_mode=image_ref_mode)
    raise ValueError(f"unknown adapter {adapter!r} for {model_id}")


def existing_response_ids(path: str | None) -> set[str]:
    if not path or not Path(path).exists():
        return set()
    return {str(row.get("response_id", "")) for row in read_jsonl(path)}


def main() -> None:
    quiet_common_warnings()
    parser = argparse.ArgumentParser(description="Generate image descriptions for rendered jobs.")
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=["mock", "transformers"], default="mock")
    parser.add_argument("--generation-config", default="visual_grounded_dataset/configs/generation.json")
    parser.add_argument("--only-model-source", help="Run only jobs for one model source_id, e.g. qwen3_vl_4b.")
    parser.add_argument("--limit", type=int, help="Maximum number of jobs to run from the selected job file.")
    parser.add_argument("--start", type=int, default=0, help="Skip this many selected jobs before generation.")
    parser.add_argument("--resume", action="store_true", help="Append to --out and skip existing response IDs.")
    parser.add_argument("--continue-on-error", action="store_true", help="Write error rows instead of stopping on generation errors.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N generated jobs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Jobs per forward pass (overrides generation.json vlm_batch_size when set).",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map; use 'none' for adapters that should not pass it.")
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype (auto -> bfloat16 on CUDA). Use 'none' to omit dtype.",
    )
    parser.add_argument("--image-ref-mode", choices=["file_uri", "path", "pil"], default="file_uri")
    parser.add_argument("--internvl-max-tiles", type=int, default=6)
    args = parser.parse_args()

    generation_config = read_json(args.generation_config)
    if args.batch_size is not None:
        batch_size = max(1, args.batch_size)
    else:
        batch_size = max(1, int(generation_config.get("vlm_batch_size", 4)))
    load_dtype = resolve_load_dtype(args.dtype)
    all_jobs = read_jsonl(args.jobs)
    jobs = all_jobs
    if args.only_model_source:
        jobs = [job for job in jobs if job["model"]["source_id"] == args.only_model_source]
    jobs = jobs[args.start :]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if args.backend == "transformers":
        jobs = sorted(jobs, key=lambda job: job["model"]["source_id"])

    skip_ids = existing_response_ids(args.out) if args.resume else set()
    selected_before_skip = len(jobs)
    jobs = [job for job in jobs if stable_hash({"job_id": job["job_id"], "backend": args.backend}) not in skip_ids]
    model_counts = Counter(job["model"]["source_id"] for job in jobs)
    language_counts = Counter(job["language"]["code"] for job in jobs)
    view_counts = Counter(job["view"]["view_id"] for job in jobs)
    print("Generation run")
    print(f"- jobs file: {args.jobs}")
    print(f"- output: {args.out}")
    print(f"- backend: {args.backend}")
    print(f"- total jobs in file: {len(all_jobs)}")
    print(f"- selected jobs before resume skip: {selected_before_skip}")
    print(f"- skipped existing responses: {selected_before_skip - len(jobs)}")
    print(f"- jobs to generate: {len(jobs)}")
    print(f"- models: {dict(model_counts)}")
    print(f"- languages: {dict(language_counts)}")
    print(f"- views: {dict(view_counts)}")
    if args.backend == "transformers":
        print(f"- batch size: {batch_size}")
        print(f"- load dtype: {load_dtype}")
        print(f"- attn: sdpa (when supported)")
    print("", flush=True)

    rows = []
    adapter: VLMAdapter | None = None
    loaded_source_id: str | None = None
    generated = 0
    errors = 0
    started = time.time()

    def append_row(job: dict[str, Any], output_text: str, status: str, error: str | None) -> None:
        rows.append(
            {
                "response_id": stable_hash({"job_id": job["job_id"], "backend": args.backend}),
                "job_id": job["job_id"],
                "image_id": job["image"]["image_id"],
                "image_path": job["image"]["image_path"],
                "source_dataset": job["image"].get("source_dataset", "unknown"),
                "topic_tags": job["image"].get("topic_tags", []),
                "model_source_id": job["model"]["source_id"],
                "model_id": job["model"]["model_id"],
                "model_adapter": job["model"].get("adapter", "generic_pipeline"),
                "language_code": job["language"]["code"],
                "language_name": job["language"]["name"],
                "language_script_hint": job["language"].get("script_hint", ""),
                "prompt_view": job["view"]["view_id"],
                "prompt": job["prompt"],
                "prompt_hash": job["prompt_hash"],
                "generation_config": generation_config,
                "output_text": output_text,
                "generation_status": status,
                **({"generation_error": error} if error else {}),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "backend": args.backend,
            }
        )

    def report_progress() -> None:
        elapsed = time.time() - started
        rate = generated / elapsed if elapsed > 0 else 0.0
        remaining = (len(jobs) - generated) / rate if rate > 0 else 0.0
        print(
            f"progress {generated}/{len(jobs)} | ok={generated - errors} error={errors} "
            f"| {rate:.2f} jobs/s | elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m",
            flush=True,
        )

    def ensure_adapter(job: dict[str, Any]) -> VLMAdapter:
        nonlocal adapter, loaded_source_id
        source_id = job["model"]["source_id"]
        if adapter is None or source_id != loaded_source_id:
            print(f"Loading model adapter: {source_id} ({job['model']['model_id']})", flush=True)
            load_started = time.time()
            adapter = make_adapter(job["model"], args)
            loaded_source_id = source_id
            print(f"Loaded {source_id} in {time.time() - load_started:.1f}s", flush=True)
        return adapter

    job_index = 0
    while job_index < len(jobs):
        chunk = jobs[job_index : job_index + batch_size]
        job_index += len(chunk)

        if args.backend == "mock":
            for job in chunk:
                append_row(job, mock_generate(job), "ok", None)
                generated += 1
            if generated == 1 or generated % args.progress_every == 0 or generated == len(jobs):
                report_progress()
            continue

        try:
            active_adapter = ensure_adapter(chunk[0])
            if batch_size > 1 and len(chunk) > 1:
                outputs = active_adapter.generate_batch(chunk, generation_config)
            else:
                outputs = [active_adapter.generate(chunk[0], generation_config)]
            if len(outputs) != len(chunk):
                raise RuntimeError(f"batch returned {len(outputs)} outputs for {len(chunk)} jobs")
            for job, output_text in zip(chunk, outputs, strict=True):
                append_row(job, output_text, "ok", None)
                generated += 1
        except Exception as exc:
            if not args.continue_on_error:
                raise
            if batch_size > 1 and len(chunk) > 1:
                for job in chunk:
                    try:
                        output_text = ensure_adapter(job).generate(job, generation_config)
                        append_row(job, output_text, "ok", None)
                    except Exception as inner_exc:
                        append_row(job, "", "error", repr(inner_exc))
                        errors += 1
                    generated += 1
            else:
                for job in chunk:
                    append_row(job, "", "error", repr(exc))
                    errors += 1
                    generated += 1

        if generated == 1 or generated % args.progress_every == 0 or generated == len(jobs):
            report_progress()

    if args.resume:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open(mode, encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        write_jsonl(args.out, rows)
    elapsed = time.time() - started
    print("")
    print("Generation summary")
    print(f"- wrote rows: {len(rows)}")
    print(f"- ok: {len(rows) - errors}")
    print(f"- errors: {errors}")
    print(f"- elapsed: {elapsed/60:.2f} min")
    print(f"- output: {args.out}")


if __name__ == "__main__":
    main()
