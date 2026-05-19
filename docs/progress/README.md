# Progress Log

This folder is the project lab notebook. Each task gets one Markdown file that records what we tried, why we tried it, what changed, what evidence we collected, and whether the action succeeded or failed.

Use Obsidian links to connect notes:

- [[2026-05-06_progress_system_setup]]
- [[2026-05-06_progress_update_protocol]]
- [[2026-05-06_project_graph_refresh]]
- [[raw_json_to_dataset_guide]]
- [[multi_gpu_server_usage_guide]]
- [[TEMPLATE_task_log]]

## Logging Rules

1. One task, one file.
2. Record the exact goal before changing code.
3. Record context, assumptions, files touched, commands, outputs, images, tables, figures, and result artifacts.
4. Explain the reasoning behind each action.
5. Mark each action as `Succeeded`, `Failed`, `Partial`, or `Deferred`.
6. If a method, algorithm, tool, dataset, or claim comes from another work, add a source link and a short paper breakdown.
7. Include code snippets or pseudocode when they clarify how a change works.
8. Link images and result files with relative Markdown links so the note remains portable.
9. Keep failed attempts. Failed evidence is still evidence.

## Task Index

| Date | Task | Status | Note |
| --- | --- | --- | --- |
| 2026-05-06 | Progress logging system setup | Succeeded | [[2026-05-06_progress_system_setup]] |
| 2026-05-06 | Session reconstruction from Codex chat ID | Partial | [[2026-05-06_session_reconstruction]] |
| 2026-05-06 | V2 clean restart and project scaffold | Succeeded | [[2026-05-06_v2_clean_restart]] |
| 2026-05-06 | Constrained paraphrase pipeline | Succeeded | [[2026-05-06_constrained_paraphrase_pipeline]] |
| 2026-05-06 | Fresh AG News dataset for V2 | Succeeded | [[2026-05-06_fresh_ag_news_dataset]] |
| 2026-05-06 | Run-from-V2 path fix | Succeeded | [[2026-05-06_run_from_v2_path_fix]] |
| 2026-05-06 | Project graph documentation | Succeeded | [[2026-05-06_project_graph_documentation]] |
| 2026-05-06 | Raw JSON to dataset guide | Succeeded | [[2026-05-06_raw_json_to_dataset_guide]] |
| 2026-05-06 | Progress update protocol | Succeeded | [[2026-05-06_progress_update_protocol]] |
| 2026-05-06 | Project graph simplification | Succeeded | [[2026-05-06_project_graph_simplification]] |
| 2026-05-06 | Project graph refresh | Succeeded | [[2026-05-06_project_graph_refresh]] |
| 2026-05-06 | Progress sync after graph refresh | Succeeded | [[2026-05-06_progress_sync_after_graph_refresh]] |
| 2026-05-07 | Progress sync from update command | Succeeded | [[2026-05-07_progress_sync_update_command]] |
| 2026-05-07 | Multi-GPU server usage guide | Succeeded | [[2026-05-07_multi_gpu_server_usage_guide]] |
| 2026-05-07 | Server 4-GPU 2000-original dataset QA | Succeeded | [[2026-05-07_server_4gpu_2000_dataset_qa]] |
| 2026-05-07 | Activation extraction pipeline | Succeeded | [[2026-05-07_activation_extraction_pipeline]] |
| 2026-05-07 | SetConCA V2 training entrypoint | Succeeded | [[2026-05-07_setconca_v2_training_entrypoint]] |
| 2026-05-07 | Model family and layer grid plan | Succeeded | [[2026-05-07_model_family_layer_grid_plan]] |
| 2026-05-07 | 4A100 activation grid QA | Succeeded | [[2026-05-07_activation_grid_4A100_qa]] |
| 2026-05-07 | V1 goal and test reconstruction | Succeeded | [[2026-05-07_v1_goal_and_test_reconstruction]] |
| 2026-05-07 | V2 transfer and steering pipeline | Succeeded | [[2026-05-07_v2_transfer_steering_pipeline]] |
| 2026-05-08 | Family pilot results review | Succeeded | [[2026-05-08_family_pilot_results_review]] |
| 2026-05-08 | Bridge diagnostics and shuffled controls | Succeeded | [[2026-05-08_bridge_diagnostics]] |
| 2026-05-08 | Diagnostic family reruns, 25 epochs batch 128 | Succeeded | [[2026-05-08_diagnostic_family_reruns_e25_b128]] |
| 2026-05-08 | Cross-family pair verification | Succeeded | [[2026-05-08_cross_family_pair_verification]] |
| 2026-05-08 | Proposal alignment audit | Succeeded | [[2026-05-08_proposal_alignment_audit]] |
| 2026-05-08 | Llama/Qwen focus and pointwise baseline | Ready | [[2026-05-08_llama_qwen_focus_and_pointwise_baseline]] |
| 2026-05-08 | SetConCA vs pointwise baseline results | Succeeded | [[2026-05-08_setconca_vs_pointwise_baseline_results]] |
| 2026-05-08 | Layer-pair breakdown | Succeeded | [[2026-05-08_layer_pair_breakdown]] |
| 2026-05-08 | Automatic summary reporting for Llama/Qwen bridge results | Succeeded | [[2026-05-08_automatic_summary_reporting]] |
| 2026-05-08 | Epoch sweep comparison, 25 vs 50 vs 100 | Succeeded | [[2026-05-08_epoch_sweep_comparison]] |
| 2026-05-08 | Bridged concept inspection | Succeeded | [[2026-05-08_bridged_concept_inspection]] |
| 2026-05-08 | Concept review pages | Succeeded | [[2026-05-08_concept_review_pages]] |
| 2026-05-08 | First-pass concept labels | Succeeded | [[2026-05-08_first_pass_concept_labels]] |
| 2026-05-08 | Diverse steering slice | Succeeded | [[2026-05-08_diverse_steering_slice]] |
| 2026-05-08 | Diverse run and concept inspection | Succeeded | [[2026-05-08_diverse_run_and_concept_inspection]] |
| 2026-05-08 | Causal steering probe setup | Succeeded | [[2026-05-08_causal_steering_probe_setup]] |
| 2026-05-08 | First causal steering probe results | Partial | [[2026-05-08_first_causal_steering_probe_results]] |
| 2026-05-08 | News-prompt causal steering results | Partial success | [[2026-05-08_news_prompt_causal_steering_results]] |
| 2026-05-11 | Positive vs negative steering comparison | Partial | [[2026-05-11_positive_vs_negative_steering_comparison]] |
| 2026-05-11 | Opposite-active steering results | Partial success | [[2026-05-11_opposite_active_steering_results]] |
| 2026-05-12 | Project graph refresh | Succeeded | [[2026-05-12_project_graph_refresh]] |
| 2026-05-12 | Graphify code graph for src and scripts | Succeeded | [[2026-05-12_graphify_src_scripts_code_graph]] |
| 2026-05-12 | Graphify docs semantic graph | Succeeded | [[2026-05-12_graphify_docs_semantic_graph]] |
| 2026-05-13 | Zotero project references | Succeeded | [[2026-05-13_zotero_project_references]] |
| 2026-05-13 | Zotero idea-origin references | Succeeded | [[2026-05-13_zotero_idea_origin_references]] |
| 2026-05-14 | Latent dataset and dictionary plan | Succeeded | [[2026-05-14_latent_dataset_and_dictionary_plan]] |

## Guide Index

| Topic | Guide |
| --- | --- |
| Raw JSON/JSONL to SetConCA V2 dataset | [[raw_json_to_dataset_guide]] |
| Multi-GPU server dataset generation | [[multi_gpu_server_usage_guide]] |

## Workflow Notes

Session reference:

```text
019dfb85-1c81-74f1-9bfe-ffa587d400ad
```

When the user says `update`, update the relevant progress documentation from the current conversation and workspace state. See [[2026-05-06_progress_update_protocol]].

## Suggested Tags

`#progress` `#experiment` `#implementation` `#evaluation` `#paper` `#failure-analysis` `#results`
