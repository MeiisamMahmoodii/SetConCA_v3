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


def generate_batch_with_adapter(
    adapter: VLMAdapter,
    jobs: list[dict[str, Any]],
    generation_config: dict[str, Any],
    batch_size: int,
) -> list[str]:
    batch_method = getattr(adapter, "generate_batch", None)
    if callable(batch_method) and len(jobs) > 1:
        return list(batch_method(jobs, generation_config, batch_size))
    return [adapter.generate(job, generation_config) for job in jobs]


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


class GenericPipelineAdapter:
    def __init__(self, model_id: str, *, device_map: str, dtype: str, image_ref_mode: str) -> None:
        from transformers import pipeline

        kwargs: dict[str, Any] = {"model": model_id, "task": "image-text-to-text", "device_map": device_map}
        if dtype != "none":
            kwargs["dtype"] = dtype
        self.pipe = pipeline(**kwargs)
        self.image_ref_mode = image_ref_mode

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        return self.generate_batch([job], generation_config, batch_size=1)[0]

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any], batch_size: int) -> list[str]:
        if len(jobs) == 1:
            return [self.generate_one_by_one(jobs[0], generation_config)]
        messages_batch = []
        for job in jobs:
            image = image_ref(job["image"]["image_path"], self.image_ref_mode)
            prompt = job["prompt"]
            messages_batch.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            )
        outputs = self.pipe(
            text=messages_batch,
            batch_size=batch_size,
            max_new_tokens=int(generation_config.get("max_new_tokens", 120)),
            do_sample=float(generation_config.get("temperature", 0.0)) > 0,
            temperature=float(generation_config.get("temperature", 0.2)),
            top_p=float(generation_config.get("top_p", 0.9)),
            return_full_text=False,
        )
        if len(jobs) == 1:
            return [decode_pipeline_output(outputs)]
        return [decode_pipeline_output(output) for output in outputs]

    def generate_one_by_one(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        image = image_ref(job["image"]["image_path"], self.image_ref_mode)
        prompt = job["prompt"]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        output = self.pipe(
            text=messages,
            max_new_tokens=int(generation_config.get("max_new_tokens", 120)),
            do_sample=float(generation_config.get("temperature", 0.0)) > 0,
            temperature=float(generation_config.get("temperature", 0.2)),
            top_p=float(generation_config.get("top_p", 0.9)),
            return_full_text=False,
        )
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
        model_kwargs: dict[str, Any] = {"device_map": device_map}
        if dtype != "none":
            model_kwargs["dtype"] = dtype
        if trust_remote_code:
            model_kwargs["trust_remote_code"] = True
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = model_cls.from_pretrained(model_id, **model_kwargs).eval()
        self.torch = torch
        self.image_ref_mode = image_ref_mode

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        image = image_ref(job["image"]["image_path"], self.image_ref_mode)
        image_item = {"type": "image", "url": image} if isinstance(image, str) else {"type": "image", "image": image}
        messages = [{"role": "user", "content": [image_item, {"type": "text", "text": job["prompt"]}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(generation_config.get("max_new_tokens", 120)),
                do_sample=float(generation_config.get("temperature", 0.0)) > 0,
                temperature=float(generation_config.get("temperature", 0.2)),
                top_p=float(generation_config.get("top_p", 0.9)),
            )
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        if hasattr(self.processor, "parse_response"):
            parsed = self.processor.parse_response(response)
            if isinstance(parsed, dict):
                response = parsed.get("answer") or parsed.get("response") or parsed.get("content") or response
            elif parsed is not None:
                response = str(parsed)
        return compact_text(response)


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
            model_class="Qwen2_5_VLForConditionalGeneration",
            device_map=device_map,
            dtype=dtype,
            image_ref_mode="path",
        )

    def generate(self, job: dict[str, Any], generation_config: dict[str, Any]) -> str:
        return self.generate_batch([job], generation_config, batch_size=1)[0]

    def generate_batch(self, jobs: list[dict[str, Any]], generation_config: dict[str, Any], batch_size: int) -> list[str]:
        if self._fallback is not None:
            return self._fallback.generate_batch(jobs, generation_config, batch_size)

        from qwen_vl_utils import process_vision_info

        message_batches = []
        for job in jobs:
            message_batches.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(Path(job["image"]["image_path"]).resolve())},
                            {"type": "text", "text": job["prompt"]},
                        ],
                    }
                ]
            )
        text = [self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in message_batches]
        image_inputs, video_inputs = process_vision_info(message_batches)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=int(generation_config.get("max_new_tokens", 120)),
                do_sample=float(generation_config.get("temperature", 0.0)) > 0,
                temperature=float(generation_config.get("temperature", 0.2)),
                top_p=float(generation_config.get("top_p", 0.9)),
            )
        decoded = self.processor.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        return [compact_text(text) for text in decoded]


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
    if adapter == "qwen_vl":
        return QwenVLAdapter(model_id, device_map=args.device_map, dtype=args.dtype, image_ref_mode=image_ref_mode)
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
    parser.add_argument("--batch-size", type=int, default=1, help="Number of jobs to generate per VLM forward pass when the adapter supports batching.")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map; use 'none' for adapters that should not pass it.")
    parser.add_argument("--dtype", default="auto", help="Transformers dtype. Use 'none' to omit dtype.")
    parser.add_argument("--image-ref-mode", choices=["file_uri", "path", "pil"], default="file_uri")
    parser.add_argument("--internvl-max-tiles", type=int, default=6)
    args = parser.parse_args()

    generation_config = read_json(args.generation_config)
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
    print(f"- batch size: {args.batch_size}")
    print(f"- models: {dict(model_counts)}")
    print(f"- languages: {dict(language_counts)}")
    print(f"- views: {dict(view_counts)}")
    print("", flush=True)

    mode = "a" if args.resume else "w"
    rows = []
    adapter: VLMAdapter | None = None
    loaded_source_id: str | None = None
    generated = 0
    errors = 0
    started = time.time()
    batch_size = max(1, int(args.batch_size))
    index = 0
    while index < len(jobs):
        job = jobs[index]
        source_id = job["model"]["source_id"]
        next_index = index + 1
        while (
            next_index < len(jobs)
            and next_index - index < batch_size
            and jobs[next_index]["model"]["source_id"] == source_id
        ):
            next_index += 1
        batch_jobs = jobs[index:next_index]
        batch_outputs: list[str] = []
        batch_statuses = ["ok"] * len(batch_jobs)
        batch_errors: list[str | None] = [None] * len(batch_jobs)

        try:
            if args.backend == "mock":
                batch_outputs = [mock_generate(batch_job) for batch_job in batch_jobs]
            else:
                if adapter is None or source_id != loaded_source_id:
                    print(f"Loading model adapter: {source_id} ({job['model']['model_id']})", flush=True)
                    load_started = time.time()
                    adapter = make_adapter(job["model"], args)
                    loaded_source_id = source_id
                    print(f"Loaded {source_id} in {time.time() - load_started:.1f}s", flush=True)
                assert adapter is not None
                batch_outputs = generate_batch_with_adapter(adapter, batch_jobs, generation_config, batch_size)
                if len(batch_outputs) != len(batch_jobs):
                    raise RuntimeError(f"adapter returned {len(batch_outputs)} outputs for {len(batch_jobs)} jobs")
        except Exception as exc:
            if not args.continue_on_error:
                raise
            batch_outputs = []
            for batch_job in batch_jobs:
                try:
                    if args.backend == "mock":
                        batch_outputs.append(mock_generate(batch_job))
                    else:
                        assert adapter is not None
                        batch_outputs.append(adapter.generate(batch_job, generation_config))
                except Exception as item_exc:
                    batch_outputs.append("")
                    item_index = len(batch_outputs) - 1
                    batch_statuses[item_index] = "error"
                    batch_errors[item_index] = repr(item_exc)
                    errors += 1

        for batch_job, output_text, status, error in zip(batch_jobs, batch_outputs, batch_statuses, batch_errors):
            response_id_base = stable_hash({"job_id": batch_job["job_id"], "backend": args.backend})
            rows.append(
                {
                    "response_id": response_id_base,
                    "job_id": batch_job["job_id"],
                    "image_id": batch_job["image"]["image_id"],
                    "image_path": batch_job["image"]["image_path"],
                    "source_dataset": batch_job["image"].get("source_dataset", "unknown"),
                    "topic_tags": batch_job["image"].get("topic_tags", []),
                    "model_source_id": batch_job["model"]["source_id"],
                    "model_id": batch_job["model"]["model_id"],
                    "model_adapter": batch_job["model"].get("adapter", "generic_pipeline"),
                    "language_code": batch_job["language"]["code"],
                    "language_name": batch_job["language"]["name"],
                    "language_script_hint": batch_job["language"].get("script_hint", ""),
                    "prompt_view": batch_job["view"]["view_id"],
                    "prompt": batch_job["prompt"],
                    "prompt_hash": batch_job["prompt_hash"],
                    "generation_config": generation_config,
                    "output_text": output_text,
                    "generation_status": status,
                    **({"generation_error": error} if error else {}),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "backend": args.backend,
                }
            )
        generated += len(batch_jobs)
        index = next_index
        if generated == 1 or generated % args.progress_every == 0 or generated == len(jobs):
            elapsed = time.time() - started
            rate = generated / elapsed if elapsed > 0 else 0.0
            remaining = (len(jobs) - generated) / rate if rate > 0 else 0.0
            print(
                f"progress {generated}/{len(jobs)} | ok={generated - errors} error={errors} "
                f"| {rate:.2f} jobs/s | elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m",
                flush=True,
            )

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
