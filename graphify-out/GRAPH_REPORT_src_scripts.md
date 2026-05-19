# Graph Report - src + scripts  (2026-05-12)

## Corpus Check
- 25 files · ~0 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 241 nodes · 406 edges · 23 communities detected
- Extraction: 99% EXTRACTED · 1% INFERRED · 0% AMBIGUOUS · INFERRED: 6 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_run_transfer_steering_grid.py  main()  evaluate_pairs()|run_transfer_steering_grid.py / main() / evaluate_pairs()]]
- [[_COMMUNITY_text_constraints.py  VLLMRewriteGenerator  rewrite_generation.py|text_constraints.py / VLLMRewriteGenerator / rewrite_generation.py]]
- [[_COMMUNITY_io_utils.py  set_dataset.py  activation_extraction.py|io_utils.py / set_dataset.py / activation_extraction.py]]
- [[_COMMUNITY_inspect_bridged_concepts.py  main()  inspect_pair()|inspect_bridged_concepts.py / main() / inspect_pair()]]
- [[_COMMUNITY_compare_transfer_runs.py  main()  plot_epoch_bridge()|compare_transfer_runs.py / main() / plot_epoch_bridge()]]
- [[_COMMUNITY_run_causal_steering_probe.py  run_candidate()  main()|run_causal_steering_probe.py / run_candidate() / main()]]
- [[_COMMUNITY_steering.py  candidate_rows_from_review()  load_steering_candidates()|steering.py / candidate_rows_from_review() / load_steering_candidates()]]
- [[_COMMUNITY_generate_constrained_sets.py  main()  generate_for_model()|generate_constrained_sets.py / main() / generate_for_model()]]
- [[_COMMUNITY_run_activation_grid.py  build_jobs()  main()|run_activation_grid.py / build_jobs() / main()]]
- [[_COMMUNITY_build_concept_review_pages.py  main()  write_page()|build_concept_review_pages.py / main() / write_page()]]
- [[_COMMUNITY_build_diverse_dataset_slice.py  main()  build_activation_subset()|build_diverse_dataset_slice.py / main() / build_activation_subset()]]
- [[_COMMUNITY_PointwiseTopKModel  encode_set()  extract_pointwise_codes()|PointwiseTopKModel / encode_set() / extract_pointwise_codes()]]
- [[_COMMUNITY_train_setconca_v2.py  train()  evaluate()|train_setconca_v2.py / train() / evaluate()]]
- [[_COMMUNITY_SemanticValidator  SemanticValidationResult  .validate()|SemanticValidator / SemanticValidationResult / .validate()]]
- [[_COMMUNITY_launch_dataset_generation.py  build_generation_cmd()  main()|launch_dataset_generation.py / build_generation_cmd() / main()]]
- [[_COMMUNITY_dataset_download.py  format_ag_news_rows()  DatasetRow|dataset_download.py / format_ag_news_rows() / DatasetRow]]
- [[_COMMUNITY_main()  merge_generated_shards.py  read_manifest()|main() / merge_generated_shards.py / read_manifest()]]
- [[_COMMUNITY_resolve_project_path()  project_root()  paths.py|resolve_project_path() / project_root() / paths.py]]
- [[_COMMUNITY_download_ag_news()  main()  download_news_dataset.py|download_ag_news() / main() / download_news_dataset.py]]
- [[_COMMUNITY_main()  build_steering_candidate_manifest.py|main() / build_steering_candidate_manifest.py]]
- [[_COMMUNITY_main()  extract_activation_bank.py|main() / extract_activation_bank.py]]
- [[_COMMUNITY_summarize_and_filter_sets.py  main()|summarize_and_filter_sets.py / main()]]
- [[_COMMUNITY_SetConCA V2 utilities.  __init__.py|SetConCA V2 utilities. / __init__.py]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `main()` - 9 edges
3. `evaluate_pairs()` - 9 edges
4. `plot_summary_artifacts()` - 9 edges
5. `main()` - 8 edges
6. `main()` - 8 edges
7. `main()` - 7 edges
8. `build_jobs()` - 7 edges
9. `run_candidate()` - 7 edges
10. `PointwiseTopKModel` - 7 edges

## Surprising Connections (you probably didn't know these)
- `RewriteModelSpec` --uses--> `LengthBand`  [INFERRED]
  src\setconca_v2\rewrite_generation.py → src\setconca_v2\text_constraints.py
- `vLLM offline generator for high-throughput CUDA/WSL/Linux runs.` --uses--> `LengthBand`  [INFERRED]
  src\setconca_v2\rewrite_generation.py → src\setconca_v2\text_constraints.py
- `Deterministic fallback for tests and pipeline dry-runs.` --uses--> `LengthBand`  [INFERRED]
  src\setconca_v2\rewrite_generation.py → src\setconca_v2\text_constraints.py
- `HFRewriteGenerator` --uses--> `LengthBand`  [INFERRED]
  src\setconca_v2\rewrite_generation.py → src\setconca_v2\text_constraints.py
- `VLLMRewriteGenerator` --uses--> `LengthBand`  [INFERRED]
  src\setconca_v2\rewrite_generation.py → src\setconca_v2\text_constraints.py

## Communities

### Community 0 - "run_transfer_steering_grid.py / main() / evaluate_pairs()"
Cohesion: 0.1
Nodes (43): add_bar_labels(), apply_bridge(), average(), BankSpec, batch_indices(), code_diagnostics_for_item(), controlled_label(), cosine_mean() (+35 more)

### Community 1 - "text_constraints.py / VLLMRewriteGenerator / rewrite_generation.py"
Cohesion: 0.12
Nodes (14): DryRunRewriteGenerator, HFRewriteGenerator, vLLM offline generator for high-throughput CUDA/WSL/Linux runs., Deterministic fallback for tests and pipeline dry-runs., resolve_dtype(), RewriteModelSpec, VLLMRewriteGenerator, contains_banned_word() (+6 more)

### Community 2 - "io_utils.py / set_dataset.py / activation_extraction.py"
Cohesion: 0.11
Nodes (13): build_activation_bank(), extract_hf_activations(), extract_position(), load_semantic_views(), make_fake_hidden(), SemanticSetView, stable_seed(), load_existing_keys() (+5 more)

### Community 3 - "inspect_bridged_concepts.py / main() / inspect_pair()"
Cohesion: 0.25
Nodes (15): abs_cosine(), candidate_pairs(), CodeBank, inspect_pair(), load_code_bank(), load_jsonl(), main(), model_slug_from_key() (+7 more)

### Community 4 - "compare_transfer_runs.py / main() / plot_epoch_bridge()"
Cohesion: 0.28
Nodes (11): collect_bridge_rows(), collect_set_size_rows(), load_run(), main(), plot_epoch_bridge(), plot_setconca_set_size(), pretty_bridge(), pretty_method() (+3 more)

### Community 5 - "run_causal_steering_probe.py / run_candidate() / main()"
Cohesion: 0.32
Nodes (11): add_to_layer_output(), decoder_direction(), dtype_from_name(), generate_once(), get_decoder_layers(), load_setconca_model(), main(), read_prompts() (+3 more)

### Community 6 - "steering.py / candidate_rows_from_review() / load_steering_candidates()"
Cohesion: 0.26
Nodes (6): candidate_rows_from_review(), keyword_guess(), load_steering_candidates(), parse_keywords(), read_csv_rows(), SteeringCandidate

### Community 7 - "generate_constrained_sets.py / main() / generate_for_model()"
Cohesion: 0.42
Nodes (10): apply_model_shard(), format_elapsed(), generate_for_model(), generate_for_model_vllm_batched(), main(), model_specs(), parse_bands(), parse_shard() (+2 more)

### Community 8 - "run_activation_grid.py / build_jobs() / main()"
Cohesion: 0.35
Nodes (10): build_jobs(), command_for_job(), ExtractionJob, layer_from_fraction(), main(), pct_label(), resolve_num_layers(), run_jobs() (+2 more)

### Community 9 - "build_concept_review_pages.py / main() / write_page()"
Cohesion: 0.4
Nodes (9): clean_cell(), concept_key(), example_table(), main(), read_rows(), safe_name(), short_key(), write_index() (+1 more)

### Community 10 - "build_diverse_dataset_slice.py / main() / build_activation_subset()"
Cohesion: 0.46
Nodes (7): build_activation_subset(), copy_subset_activation_bank(), load_jsonl(), main(), select_balanced_indices(), write_csv(), write_jsonl()

### Community 11 - "PointwiseTopKModel / encode_set() / extract_pointwise_codes()"
Cohesion: 0.36
Nodes (4): encode_set(), extract_pointwise_codes(), PointwiseTopKModel, Pointwise sparse baseline: train on individual views, pool codes only at evaluat

### Community 12 - "train_setconca_v2.py / train() / evaluate()"
Cohesion: 0.73
Nodes (5): average_metrics(), batch_indices(), evaluate(), main(), train()

### Community 13 - "SemanticValidator / SemanticValidationResult / .validate()"
Cohesion: 0.4
Nodes (2): SemanticValidationResult, SemanticValidator

### Community 14 - "launch_dataset_generation.py / build_generation_cmd() / main()"
Cohesion: 0.8
Nodes (4): build_generation_cmd(), main(), run_multi(), run_single()

### Community 15 - "dataset_download.py / format_ag_news_rows() / DatasetRow"
Cohesion: 0.6
Nodes (3): DatasetRow, format_ag_news_rows(), normalize_news_text()

### Community 16 - "main() / merge_generated_shards.py / read_manifest()"
Cohesion: 0.83
Nodes (3): main(), read_manifest(), read_rows()

### Community 17 - "resolve_project_path() / project_root() / paths.py"
Cohesion: 0.67
Nodes (3): project_root(), Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.      If a, resolve_project_path()

### Community 18 - "download_ag_news() / main() / download_news_dataset.py"
Cohesion: 1.0
Nodes (2): download_ag_news(), main()

### Community 19 - "main() / build_steering_candidate_manifest.py"
Cohesion: 1.0
Nodes (0): 

### Community 20 - "main() / extract_activation_bank.py"
Cohesion: 1.0
Nodes (0): 

### Community 21 - "summarize_and_filter_sets.py / main()"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "SetConCA V2 utilities. / __init__.py"
Cohesion: 1.0
Nodes (1): SetConCA V2 utilities.

## Knowledge Gaps
- **3 isolated node(s):** `Pointwise sparse baseline: train on individual views, pool codes only at evaluat`, `SetConCA V2 utilities.`, `Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.      If a`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `main() / build_steering_candidate_manifest.py`** (2 nodes): `main()`, `build_steering_candidate_manifest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `main() / extract_activation_bank.py`** (2 nodes): `main()`, `extract_activation_bank.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `summarize_and_filter_sets.py / main()`** (2 nodes): `summarize_and_filter_sets.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SetConCA V2 utilities. / __init__.py`** (2 nodes): `SetConCA V2 utilities.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PointwiseTopKModel` connect `PointwiseTopKModel / encode_set() / extract_pointwise_codes()` to `run_transfer_steering_grid.py / main() / evaluate_pairs()`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **What connects `Pointwise sparse baseline: train on individual views, pool codes only at evaluat`, `SetConCA V2 utilities.`, `Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.      If a` to the rest of the system?**
  _3 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run_transfer_steering_grid.py / main() / evaluate_pairs()` be split into smaller, more focused modules?**
  _Cohesion score 0.1 - nodes in this community are weakly interconnected._
- **Should `text_constraints.py / VLLMRewriteGenerator / rewrite_generation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._
- **Should `io_utils.py / set_dataset.py / activation_extraction.py` be split into smaller, more focused modules?**
  _Cohesion score 0.11 - nodes in this community are weakly interconnected._