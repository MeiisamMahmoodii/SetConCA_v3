# Graph Report - docs + existing src/scripts graph  (2026-05-12)

## Corpus Check
- 62 files · ~65,642 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 453 nodes · 694 edges · 30 communities detected
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_run transfer steering grid.py  main()  evaluate pairs()|run transfer steering grid.py / main() / evaluate pairs()]]
- [[_COMMUNITY_Ridge Bridge  real minus shuffled topk Controlled Bridge Metric  LlamaQwen SetConCA vs Pointwise Linear Run|Ridge Bridge / real minus shuffled topk Controlled Bridge Metric / Llama/Qwen SetConCA vs Pointwise Linear Run]]
- [[_COMMUNITY_Latent-Set Dataset  SetConCA V2 Project Map  Constrained Paraphrase Pipeline|Latent-Set Dataset / SetConCA V2 Project Map / Constrained Paraphrase Pipeline]]
- [[_COMMUNITY_News-Prompt Causal Steering Probe  Opposite-Active Steering Control  SetConCA Vs Pointwise TopK Baseline Results|News-Prompt Causal Steering Probe / Opposite-Active Steering Control / SetConCA Vs Pointwise TopK Baseline Results]]
- [[_COMMUNITY_Real-Minus-Shuffled TopK Metric  Sparse Set Code z  Semantic Sets|Real-Minus-Shuffled TopK Metric / Sparse Set Code z / Semantic Sets]]
- [[_COMMUNITY_text constraints.py  VLLMRewriteGenerator  rewrite generation.py|text constraints.py / VLLMRewriteGenerator / rewrite generation.py]]
- [[_COMMUNITY_io utils.py  set dataset.py  activation extraction.py|io utils.py / set dataset.py / activation extraction.py]]
- [[_COMMUNITY_27 Activation Banks  Model Family and Layer Grid Plan  Seeded Qwen3 Small Pilot|27 Activation Banks / Model Family and Layer Grid Plan / Seeded Qwen3 Small Pilot]]
- [[_COMMUNITY_Activation Extraction Pipeline  Multi-GPU Server Usage Guide  sets min8 Semantic Dataset|Activation Extraction Pipeline / Multi-GPU Server Usage Guide / sets min8 Semantic Dataset]]
- [[_COMMUNITY_inspect bridged concepts.py  main()  inspect pair()|inspect bridged concepts.py / main() / inspect pair()]]
- [[_COMMUNITY_compare transfer runs.py  main()  plot epoch bridge()|compare transfer runs.py / main() / plot epoch bridge()]]
- [[_COMMUNITY_run causal steering probe.py  run candidate()  main()|run causal steering probe.py / run candidate() / main()]]
- [[_COMMUNITY_steering.py  candidate rows from review()  load steering candidates()|steering.py / candidate rows from review() / load steering candidates()]]
- [[_COMMUNITY_generate constrained sets.py  main()  generate for model()|generate constrained sets.py / main() / generate for model()]]
- [[_COMMUNITY_run activation grid.py  build jobs()  main()|run activation grid.py / build jobs() / main()]]
- [[_COMMUNITY_build concept review pages.py  main()  write page()|build concept review pages.py / main() / write page()]]
- [[_COMMUNITY_Multi-GPU Server Usage Guide  vLLM Backend  AG News Source Dataset|Multi-GPU Server Usage Guide / vLLM Backend / AG News Source Dataset]]
- [[_COMMUNITY_build diverse dataset slice.py  main()  build activation subset()|build diverse dataset slice.py / main() / build activation subset()]]
- [[_COMMUNITY_train setconca v2.py  train()  evaluate()|train setconca v2.py / train() / evaluate()]]
- [[_COMMUNITY_SemanticValidator  SemanticValidationResult  .validate()|SemanticValidator / SemanticValidationResult / .validate()]]
- [[_COMMUNITY_launch dataset generation.py  build generation cmd()  main()|launch dataset generation.py / build generation cmd() / main()]]
- [[_COMMUNITY_dataset download.py  format ag news rows()  DatasetRow|dataset download.py / format ag news rows() / DatasetRow]]
- [[_COMMUNITY_main()  merge generated shards.py  read manifest()|main() / merge generated shards.py / read manifest()]]
- [[_COMMUNITY_resolve project path()  project root()  paths.py|resolve project path() / project root() / paths.py]]
- [[_COMMUNITY_download ag news()  main()  download news dataset.py|download ag news() / main() / download news dataset.py]]
- [[_COMMUNITY_main()  build steering candidate manifest.py|main() / build steering candidate manifest.py]]
- [[_COMMUNITY_main()  extract activation bank.py|main() / extract activation bank.py]]
- [[_COMMUNITY_summarize and filter sets.py  main()|summarize and filter sets.py / main()]]
- [[_COMMUNITY_SetConCA V2 utilities.    init  .py|SetConCA V2 utilities. /   init  .py]]
- [[_COMMUNITY_activation extraction Module  extract activation bank CLI|activation extraction Module / extract activation bank CLI]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 17 edges
2. `main()` - 9 edges
3. `evaluate_pairs()` - 9 edges
4. `plot_summary_artifacts()` - 9 edges
5. `Latent-Set Dataset` - 9 edges
6. `main()` - 8 edges
7. `main()` - 8 edges
8. `SetConCA V2 Project Map` - 8 edges
9. `main()` - 7 edges
10. `build_jobs()` - 7 edges
11. `run_candidate()` - 7 edges
12. `PointwiseTopKModel` - 7 edges

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
- `DryRunRewriteGenerator` --uses--> `LengthBand`  [INFERRED]
  src\setconca_v2\rewrite_generation.py → src\setconca_v2\text_constraints.py
- `SetConCA PDF Report` --semantically_similar_to--> `SetConCA Temporary Project Report`  [EXTRACTED] [semantically similar]
  docs/temp_setconca_report/SetConCA_Report.pdf → docs/temp_setconca_report/SetConCA_Temporary_Report.md
- `SetConCA Mathematical Summary` --conceptually_related_to--> `Semantic Sets`  [EXTRACTED]
  docs/temp_setconca_report/figures/setconca_math_summary.png → docs/temp_setconca_report/SetConCA_Temporary_Report.md

## Hyperedges (group relationships)
- **Constrained Generation Artifact Flow** — chunk1_ag_news, chunk1_constrained_rewrites, chunk1_attempts_jsonl, chunk1_sets_jsonl, chunk1_semantic_sets [EXTRACTED 1.00]
- **Controlled Activation And Transfer Sweep** — chunk1_sets_min16_dataset, chunk1_activation_banks, chunk1_model_family_grid, chunk1_layer_grid, chunk1_set_size_sweep, chunk1_transfer_steering_grid, chunk1_real_minus_shuffled_topk [EXTRACTED 1.00]
- **Latent Dataset Expansion Plan** — chunk1_source_registry, chunk1_latent_set_dataset, chunk1_latent_variable_criterion, chunk1_hard_negative_separation, chunk1_latent_set_builders, chunk1_steering_benchmark_dataset, chunk1_pilot_latent_dataset_milestone [EXTRACTED 1.00]
- **Semantic Sets to Activations to Training** — chunk2_sets_min8_dataset, chunk2_activation_extraction_pipeline, chunk2_activation_bank_format, chunk2_train_setconca_v2_entrypoint [EXTRACTED 1.00]
- **Full Family Size Layer Set-Size Grid** — chunk2_model_families_llama_gemma_qwen, chunk2_layer_fraction_policy, chunk2_27_activation_banks, chunk2_216_training_runs [EXTRACTED 1.00]
- **Transfer and Steering Evaluation Stack** — chunk2_run_transfer_steering_grid, chunk2_bridge_methods, chunk2_steering_proxy_metric, chunk2_seeded_qwen3_small_pilot [EXTRACTED 1.00]
- **Linear Bridge Evidence Flow** — chunk3_setconca, chunk3_controlled_bridge_metric, chunk3_ridge_bridge, chunk3_procrustes_bridge, chunk3_llama_qwen_linear_run [EXTRACTED 1.00]
- **Concept Inspection To Causal Steering Pipeline** — chunk3_inspect_bridged_concepts_script, chunk3_build_concept_review_pages, chunk3_build_steering_candidate_manifest, chunk3_run_causal_steering_probe [EXTRACTED 1.00]
- **Family Diagnostic Interpretation** — chunk3_gemma3_caution, chunk3_llama3_controlled_signal, chunk3_qwen3_controlled_signal, chunk3_code_diagnostics, chunk3_normalized_reconstruction [EXTRACTED 1.00]
- **Steering Candidate Evaluation** — chunk4_news_prompt_probe, chunk4_google_ipo_candidate, chunk4_corporate_earnings_candidate, chunk4_stock_market_candidate, chunk4_opposite_active_control [EXTRACTED 1.00]
- **Bridge Baseline Evidence** — chunk4_pointwise_topk_baseline, chunk4_setconca_vs_pointwise_results, chunk4_real_minus_shuffled_metric, chunk4_set_size_pattern [EXTRACTED 1.00]
- **Project Documentation System** — chunk4_progress_log_system, chunk4_task_log_template, chunk4_progress_assets_guidance, chunk4_project_graph_refresh, chunk4_graphify_code_graph [EXTRACTED 1.00]
- **Dataset Generation Pipeline** — chunk5_ag_news_source, chunk5_raw_jsonl_dataset_artifacts, chunk5_constrained_paraphrase_generation, chunk5_semantic_sets, chunk5_dataset_audit_manifest [EXTRACTED 1.00]
- **SetConCA Shared And Residual Architecture** — chunk5_shared_encoder, chunk5_sparse_set_code, chunk5_shared_decoder, chunk5_residual_decoder, chunk5_training_loss_objective [EXTRACTED 1.00]
- **Controlled Bridge Evaluation Pattern** — chunk5_topk_sparse_overlap, chunk5_raw_topk_reinterpretation, chunk5_real_minus_shuffled_metric, chunk5_bridge_methods, chunk5_setconca_vs_pointwise_result [EXTRACTED 1.00]

## Communities

### Community 0 - "run transfer steering grid.py / main() / evaluate pairs()"
Cohesion: 0.08
Nodes (47): add_bar_labels(), apply_bridge(), average(), BankSpec, batch_indices(), code_diagnostics_for_item(), controlled_label(), cosine_mean() (+39 more)

### Community 1 - "Ridge Bridge / real minus shuffled topk Controlled Bridge Metric / Llama/Qwen SetConCA vs Pointwise Linear Run"
Cohesion: 0.06
Nodes (44): All-Family Cross-Family Diagnostic Run, scripts/build_concept_review_pages.py, scripts/build_diverse_dataset_slice.py, scripts/build_steering_candidate_manifest.py, Clean Business and Technology Steering Candidates, Code Diagnostics, scripts/compare_transfer_runs.py, concept_examples.csv (+36 more)

### Community 2 - "Latent-Set Dataset / SetConCA V2 Project Map / Constrained Paraphrase Pipeline"
Cohesion: 0.07
Nodes (43): Activation Banks, AG News, attempts.jsonl Audit Trail, Generation-Time Causal Steering Probe, Bridged Concept Inspection, Constrained Paraphrase Pipeline, Constrained Rewrites, Dataset Project Plan (+35 more)

### Community 3 - "News-Prompt Causal Steering Probe / Opposite-Active Steering Control / SetConCA Vs Pointwise TopK Baseline Results"
Cohesion: 0.08
Nodes (38): First Causal Steering Smoke Probe, First-Pass Concept Labels, Contrastive Alignment And Off-Diagonal Decorrelation Additions, Corporate Earnings / Company Performance Candidate, 60% to 60% SetConCA Depth Pair, Gemma 3 Disabled For Future Extraction, Generation-Time Steering Hook, Generic Prompt Keyword Problem (+30 more)

### Community 4 - "Real-Minus-Shuffled TopK Metric / Sparse Set Code z / Semantic Sets"
Cohesion: 0.08
Nodes (36): Activation Banks, Active Direction Versus Opposite-Active Control, Banned-Word Surface-Form Pressure, Bridge Methods, Causal Steering Probe, Concept Inspection, Constrained Paraphrase Generation, Dataset Audit Manifest (+28 more)

### Community 5 - "text constraints.py / VLLMRewriteGenerator / rewrite generation.py"
Cohesion: 0.12
Nodes (14): DryRunRewriteGenerator, HFRewriteGenerator, vLLM offline generator for high-throughput CUDA/WSL/Linux runs., Deterministic fallback for tests and pipeline dry-runs., resolve_dtype(), RewriteModelSpec, VLLMRewriteGenerator, contains_banned_word() (+6 more)

### Community 6 - "io utils.py / set dataset.py / activation extraction.py"
Cohesion: 0.11
Nodes (13): build_activation_bank(), extract_hf_activations(), extract_position(), load_semantic_views(), make_fake_hidden(), SemanticSetView, stable_seed(), load_existing_keys() (+5 more)

### Community 7 - "27 Activation Banks / Model Family and Layer Grid Plan / Seeded Qwen3 Small Pilot"
Cohesion: 0.13
Nodes (22): 216 SetConCA Training Runs, 27 Activation Banks, Activation Grid 4A100 QA, Identity, Procrustes, Ridge, and MLP Bridges, Fair Set-Size Sweep Rationale, Late-Layer Loss Scale Caveat, Layer Fraction Policy, Llama 3, Gemma 3, Qwen 3 Families (+14 more)

### Community 8 - "Activation Extraction Pipeline / Multi-GPU Server Usage Guide / sets min8 Semantic Dataset"
Cohesion: 0.15
Nodes (18): Activation Bank Format, Activation Extraction Pipeline, Fake Activation Smoke Bank, min_rewrites=8 Decision, Model Sharding, Multi-GPU Server Usage Guide, 50-Original CUDA Pilot Baseline, server_4gpu_2000 Dataset (+10 more)

### Community 9 - "inspect bridged concepts.py / main() / inspect pair()"
Cohesion: 0.25
Nodes (15): abs_cosine(), candidate_pairs(), CodeBank, inspect_pair(), load_code_bank(), load_jsonl(), main(), model_slug_from_key() (+7 more)

### Community 10 - "compare transfer runs.py / main() / plot epoch bridge()"
Cohesion: 0.28
Nodes (11): collect_bridge_rows(), collect_set_size_rows(), load_run(), main(), plot_epoch_bridge(), plot_setconca_set_size(), pretty_bridge(), pretty_method() (+3 more)

### Community 11 - "run causal steering probe.py / run candidate() / main()"
Cohesion: 0.32
Nodes (11): add_to_layer_output(), decoder_direction(), dtype_from_name(), generate_once(), get_decoder_layers(), load_setconca_model(), main(), read_prompts() (+3 more)

### Community 12 - "steering.py / candidate rows from review() / load steering candidates()"
Cohesion: 0.26
Nodes (6): candidate_rows_from_review(), keyword_guess(), load_steering_candidates(), parse_keywords(), read_csv_rows(), SteeringCandidate

### Community 13 - "generate constrained sets.py / main() / generate for model()"
Cohesion: 0.42
Nodes (10): apply_model_shard(), format_elapsed(), generate_for_model(), generate_for_model_vllm_batched(), main(), model_specs(), parse_bands(), parse_shard() (+2 more)

### Community 14 - "run activation grid.py / build jobs() / main()"
Cohesion: 0.35
Nodes (10): build_jobs(), command_for_job(), ExtractionJob, layer_from_fraction(), main(), pct_label(), resolve_num_layers(), run_jobs() (+2 more)

### Community 15 - "build concept review pages.py / main() / write page()"
Cohesion: 0.4
Nodes (9): clean_cell(), concept_key(), example_table(), main(), read_rows(), safe_name(), short_key(), write_index() (+1 more)

### Community 16 - "Multi-GPU Server Usage Guide / vLLM Backend / AG News Source Dataset"
Cohesion: 0.25
Nodes (9): AG News Source Dataset, Model Sharding Across GPUs, Multi-GPU Server Usage Guide, Progress Guides Index, Raw JSON/JSONL To SetConCA V2 Dataset Guide, Server Guides Index, Single-GPU Pilot Baseline, vLLM Tensor Parallelism (+1 more)

### Community 17 - "build diverse dataset slice.py / main() / build activation subset()"
Cohesion: 0.46
Nodes (7): build_activation_subset(), copy_subset_activation_bank(), load_jsonl(), main(), select_balanced_indices(), write_csv(), write_jsonl()

### Community 18 - "train setconca v2.py / train() / evaluate()"
Cohesion: 0.73
Nodes (5): average_metrics(), batch_indices(), evaluate(), main(), train()

### Community 19 - "SemanticValidator / SemanticValidationResult / .validate()"
Cohesion: 0.4
Nodes (2): SemanticValidationResult, SemanticValidator

### Community 20 - "launch dataset generation.py / build generation cmd() / main()"
Cohesion: 0.8
Nodes (4): build_generation_cmd(), main(), run_multi(), run_single()

### Community 21 - "dataset download.py / format ag news rows() / DatasetRow"
Cohesion: 0.6
Nodes (3): DatasetRow, format_ag_news_rows(), normalize_news_text()

### Community 22 - "main() / merge generated shards.py / read manifest()"
Cohesion: 0.83
Nodes (3): main(), read_manifest(), read_rows()

### Community 23 - "resolve project path() / project root() / paths.py"
Cohesion: 0.67
Nodes (3): project_root(), Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.      If a, resolve_project_path()

### Community 24 - "download ag news() / main() / download news dataset.py"
Cohesion: 1.0
Nodes (2): download_ag_news(), main()

### Community 25 - "main() / build steering candidate manifest.py"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "main() / extract activation bank.py"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "summarize and filter sets.py / main()"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "SetConCA V2 utilities. /   init  .py"
Cohesion: 1.0
Nodes (1): SetConCA V2 utilities.

### Community 29 - "activation extraction Module / extract activation bank CLI"
Cohesion: 1.0
Nodes (2): activation_extraction Module, extract_activation_bank CLI

## Knowledge Gaps
- **51 isolated node(s):** `Pointwise sparse baseline: train on individual views, pool codes only at evaluat`, `SetConCA V2 utilities.`, `Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.      If a`, `Progress Notebook`, `Project Graph Documentation` (+46 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `main() / build steering candidate manifest.py`** (2 nodes): `main()`, `build_steering_candidate_manifest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `main() / extract activation bank.py`** (2 nodes): `main()`, `extract_activation_bank.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `summarize and filter sets.py / main()`** (2 nodes): `summarize_and_filter_sets.py`, `main()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `SetConCA V2 utilities. /   init  .py`** (2 nodes): `SetConCA V2 utilities.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `activation extraction Module / extract activation bank CLI`** (2 nodes): `activation_extraction Module`, `extract_activation_bank CLI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Raw JSONL Dataset Artifacts` connect `Real-Minus-Shuffled TopK Metric / Sparse Set Code z / Semantic Sets` to `Multi-GPU Server Usage Guide / vLLM Backend / AG News Source Dataset`?**
  _High betweenness centrality (0.004) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `Latent-Set Dataset` (e.g. with `Constrained Rewrites` and `sets_min16.jsonl Controlled Dataset`) actually correct?**
  _`Latent-Set Dataset` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Pointwise sparse baseline: train on individual views, pool codes only at evaluat`, `SetConCA V2 utilities.`, `Resolve CLI paths so scripts work from repo root or SetConCA_V2 root.      If a` to the rest of the system?**
  _51 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `run transfer steering grid.py / main() / evaluate pairs()` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Ridge Bridge / real minus shuffled topk Controlled Bridge Metric / Llama/Qwen SetConCA vs Pointwise Linear Run` be split into smaller, more focused modules?**
  _Cohesion score 0.06 - nodes in this community are weakly interconnected._
- **Should `Latent-Set Dataset / SetConCA V2 Project Map / Constrained Paraphrase Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.07 - nodes in this community are weakly interconnected._
- **Should `News-Prompt Causal Steering Probe / Opposite-Active Steering Control / SetConCA Vs Pointwise TopK Baseline Results` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._
- **Should `Real-Minus-Shuffled TopK Metric / Sparse Set Code z / Semantic Sets` be split into smaller, more focused modules?**
  _Cohesion score 0.08 - nodes in this community are weakly interconnected._