# Goal And Build Plan

## Scientific Goal

Build a topic-diverse, visual-grounded SetConCA dataset where each semantic set
is anchored by an image and not by an existing caption. The generated text views
should vary across model, language, and perspective so the common factor is the
underlying visual scene.

The dataset should let us test whether SetConCA finds concept-level invariants
instead of shallow invariants such as language, prompt style, model style, or
caption length.

## Image Sources

Use Open Images V7 as the main source. Sample by topic buckets, not randomly:

- people and social scenes
- animals
- vehicles and transportation
- food and cooking
- sports and actions
- indoor rooms
- outdoor natural scenes
- urban scenes
- tools and objects
- workplaces
- text, signs, and document-like images
- art, posters, and illustrations
- multi-object cluttered scenes
- relationship-heavy scenes
- rare-object scenes

Add Visual Genome for relationship-heavy images and COCO as a small familiar
baseline. Keep XM3600 only for optional multilingual evaluation.

## Generation Design

For every image, generate independent descriptions with multiple VLMs, multiple
languages, and multiple prompt perspectives. Do not show original captions.

Recommended local/open VLMs:

- Qwen2.5-VL-7B-Instruct
- InternVL3-8B
- Aya Vision 8B
- PaliGemma 2 mix

Recommended core languages:

- English
- Arabic
- Chinese
- French
- German
- Hindi
- Indonesian
- Japanese
- Korean
- Portuguese
- Spanish
- Swahili

Recommended prompt views:

- neutral scene
- object and attribute inventory
- spatial/object/person relations
- action/event description
- mood and emotion
- positive framing
- cautious/negative framing
- news-anchor style
- child-friendly explanation
- concise one-sentence summary

## Filtering

Reject outputs that are empty, too short, wrong-language, refusal-like, metadata
leaking, duplicate, or likely hallucinated. Keep raw outputs and rejection
reasons so the filtering process is auditable.

## Proof Strategy

The causal argument is not absolute. It is a controlled-design argument.

Positive sets hold the image fixed and vary model/language/prompt. Control sets
hold model/language/prompt distributions fixed and change the image. If SetConCA
learns stronger shared codes for positive sets than matched controls, and the
effect remains after language/model/prompt probes, the dataset is less likely to
be surface-level.

Acceptance criteria:

- At least 70% generation retention after filtering.
- Each retained image has at least 3 models, 8 languages, and 6 prompt views in
  the full run.
- Real sets beat shuffled and surface-matched controls.
- Artifact probes show language/model/prompt are not sufficient to explain the
  concept signal.

