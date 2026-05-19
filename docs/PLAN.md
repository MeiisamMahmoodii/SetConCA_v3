# XM3600 Visual-Grounded Set Dataset Plan

## Summary
Build a new, isolated dataset builder under `visual_grounded_xm3600/` without changing the main SetConCA project. Each latent set is anchored by the **same image**, not by an existing caption. Multiple VLMs will independently describe the image across languages and prompt perspectives, producing text views that can later be fed into the existing SetConCA activation pipeline.

Use [XM3600](https://google.github.io/crossmodal-3600/) as the image source: 3,600 geographically diverse images with multilingual human captions available for optional validation only, not as generation input. The dataset’s core causal claim will come from randomized model/language/prompt variation plus surface-matched controls.

## Key Changes
Create one self-contained folder:

`visual_grounded_xm3600/`

Inside it:
- `configs/`: model list, language list, prompt-view list, generation settings.
- `scripts/`: download/prepare XM3600, run VLM generation, validate/filter outputs, build latent-set JSONL, build controls, run artifact audits.
- `data/`: local manifests, generated responses, filtered sets, control sets, reports.
- `README.md`: exact commands, dataset logic, assumptions, and audit interpretation.

No imports from the main project unless optional read-only schema compatibility helpers are useful. Output should match the existing latent-set JSONL shape so later SetConCA scripts can consume it.

## Dataset Design
Use XM3600 images as:

```text
latent_type = visual_scene
latent_key = xm3600:{image_id}
set_id = xm3600_visual_scene_{image_id}
```

Do not show VLMs the XM3600 captions. Captions are used only for optional post-hoc semantic validation.

Recommended pilot:
- 500 images first.
- Then full 3,600 images if filters/audits look good.

Recommended languages:
- Core 12: English, Arabic, Chinese, French, German, Hindi, Indonesian, Japanese, Korean, Portuguese, Spanish, Swahili.
- Reason: script diversity, language-family diversity, high VLM support, enough non-English pressure.
- Optional expansion: all XM3600 languages after pilot.

Recommended VLMs:
- `Qwen/Qwen2.5-VL-7B-Instruct` as the main reliable local baseline.
- `OpenGVLab/InternVL3-8B` for independent model-family diversity.
- `CohereLabs/aya-vision-8b` for multilingual multimodal strength.
- `google/paligemma2-3b-mix-448` as a smaller captioning-style model.
- Optional high-quality API comparator: GPT-4o / Gemini / Claude vision, stored as separate `source_id` so it can be excluded from local-only experiments.

Recommended prompt views per image/language/model:
- `neutral_scene`: describe visible objects, people, actions, setting.
- `relations`: focus on spatial/object/person relationships.
- `action_event`: focus on what is happening.
- `mood_emotion`: describe mood without inventing private thoughts.
- `positive_frame`: describe with a positive framing.
- `negative_frame`: describe with a cautious/critical framing.
- `news_anchor`: describe like a factual news anchor.
- `child_explanation`: explain simply for a child.

Use deterministic generation by default:
```text
temperature = 0.2
top_p = 0.9
max_new_tokens = 120
one output per model/language/prompt/image
```

Prompt template:
```text
You are given an image. Write the answer in {language_name} only.

Task: {view_instruction}

Rules:
- Describe only what is visible or strongly implied by the image.
- Do not mention that you are an AI model.
- Do not mention uncertainty unless the image is genuinely unclear.
- Do not translate from another caption; inspect the image directly.
- Use 1 to 3 concise sentences.
```

## Filtering And Validation
For every generated answer, store raw output plus metadata:
```text
image_id, image_path, model_id, language, prompt_view, prompt_hash,
generation_params, output_text, timestamp, validation_status
```

Filter out:
- Empty or too-short answers.
- Wrong-language outputs using language ID.
- Refusals or meta answers like “I cannot see the image.”
- Outputs that mention prompt text, dataset names, or model limitations.
- Near-duplicates within the same image/model/language.
- Captions with excessive hallucination risk, checked by optional CLIP/text-image similarity or VLM self-check.

Use XM3600 human captions only as an optional validator:
- Compare generated output embedding to same-image human captions.
- Compare against nearest captions from other images.
- Keep examples where same-image similarity is meaningfully higher.
- Do not train on the human captions in this dataset version.

Build these outputs:
- `responses_raw.jsonl`
- `responses_filtered.jsonl`
- `sets_visual_scene.jsonl`
- `controls_shuffled_image.jsonl`
- `controls_same_prompt_different_image.jsonl`
- `controls_same_language_different_image.jsonl`
- `artifact_audit_report.md`

## Proof/Audit Strategy
The dataset cannot prove causality absolutely from data alone, but it can support a strong causal identification argument.

Define variables:
```text
I = image / visual scene
M = VLM model
L = language
P = prompt perspective
T = generated text
H = target LLM activation
Z = SetConCA concept code
```

Construction goal:
```text
I -> T -> H -> Z
M, L, P also affect T, but vary within each image set.
```

Core assumption:
If model, language, and prompt vary inside a positive set while the image stays fixed, and controls match model/language/prompt distributions while changing the image, then a concept that survives controls is more likely tied to visual scene content than surface artifacts.

Required audits:
- **Real vs shuffled:** same-image sets should train/evaluate better than shuffled-image sets.
- **Surface-matched negatives:** same language/model/prompt but different image should not cluster as strongly.
- **Prompt artifact probe:** train a classifier to predict prompt view from embeddings; report whether SetConCA concepts are dominated by prompt style.
- **Language artifact probe:** train a classifier to predict language; require image-set signal to remain after language-balanced controls.
- **Model artifact probe:** train a classifier to predict VLM source; require image-set signal to remain after model-balanced controls.
- **Ablation:** compare multilingual+multi-prompt+multi-model vs English-only, one-prompt, and one-model subsets.
- **Human-caption validation:** generated views should be closer to same-image XM3600 captions than to random-image captions.

Acceptance criteria:
- Filter keeps at least 70% of attempted generations.
- Each retained image has at least 3 models, 8 languages, and 6 prompt views after filtering.
- Real image sets outperform shuffled/surface-matched controls on SetConCA reconstruction/contrastive metrics.
- Image identity signal remains positive after removing or balancing language/model/prompt predictors.
- Audit report identifies top residual artifacts before using the dataset for paper claims.

## Test Plan
Unit tests:
- Config loading for models/languages/views.
- Prompt rendering is deterministic and language-specific.
- Response schema validation.
- Filter behavior for wrong language, refusal, empty output, duplicates.
- Latent-set JSONL compatibility with existing SetConCA row shape.

Integration tests:
- Build a 5-image dry-run using mocked VLM outputs.
- Build a 10-image real pilot with one VLM, two languages, two prompt views.
- Generate control sets and verify they preserve language/model/prompt distributions.
- Run artifact audit on the pilot and write a Markdown report.

## Assumptions
- XM3600 is sufficient for v1; no COCO/WIT mixing in the first build.
- The main project is not modified.
- All dataset code, configs, generated manifests, and reports live under `visual_grounded_xm3600/`.
- Open-weight local VLMs are preferred; API VLMs are optional comparators.
- Existing XM3600 captions are validation references only, not source captions.
- The first scientific claim should be cautious: “visual-grounded semantic sets reduce surface artifact reliance under matched controls,” not “the dataset is perfectly causal.”
