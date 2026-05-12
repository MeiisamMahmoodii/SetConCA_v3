# Task: Project Graph Refresh

Tags: #progress #documentation #graph #transfer #steering

Related notes: [[README]] [[PROJECT_GRAPH]] [[2026-05-07_v2_transfer_steering_pipeline]] [[2026-05-08_concept_review_pages]] [[2026-05-11_opposite_active_steering_results]]

## 1. Goal

The user ran `/graphify .` on 2026-05-12. The goal was to refresh the project map so it reflects the current V2 workspace, including transfer, concept review, and causal steering work added after the original May 6 graph.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-12 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Main map | `docs/PROJECT_GRAPH.md` |
| Progress index | `docs/progress/README.md` |

## 3. Hypothesis Or Rationale

The previous map already covered dataset generation, activation extraction, and training, but the repo now also contains a focused transfer-and-steering branch. The graph needed a new section showing how activation banks lead to SetConCA/pointwise training, bridge evaluation, concept inspection, steering candidate manifests, behavioral probes, and active/opposite-active comparisons.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Scanned top-level project folders. | Needed to confirm current structure. | Found generated data, activation/results folders, expanded scripts/tests, and docs. | Succeeded |
| 2 | Read current `docs/PROJECT_GRAPH.md`. | Needed to update the existing map instead of replacing it. | Confirmed it already included server generation and activation/training sections. | Succeeded |
| 3 | Listed scripts, package modules, tests, configs, and progress notes. | Needed to identify new graph nodes. | Found transfer, concept-review, candidate-manifest, and causal-steering scripts. | Succeeded |
| 4 | Read steering-related notes and reports. | Needed honest result wording. | Confirmed current steering evidence is partial success, not final monosemantic proof. | Succeeded |
| 5 | Updated the project graph. | Needed the map to match current code and results. | Added transfer/steering flow, expanded code map, updated tests/results/risks, and linked current steering notes. | Succeeded |
| 6 | Updated the progress index. | Needed this graph refresh to be discoverable. | Added this note to `docs/progress/README.md`. | Succeeded |

## 5. Code And Pseudocode

```text
semantic sets
    -> activation banks
    -> train SetConCA and pointwise baselines
    -> fit bridges between source and target concept spaces
    -> evaluate controlled real-minus-shuffled transfer
    -> inspect strong bridged concepts
    -> label candidates
    -> build steering manifest
    -> run active/opposite-active behavioral probes
    -> compare keyword gains and manual evidence
```

## 6. Results

| File | Update |
| --- | --- |
| `docs/PROJECT_GRAPH.md` | Added transfer-and-steering flow, new scripts, new tests, steering results, and updated risks. |
| `docs/progress/README.md` | Added this task note to the task index. |
| `docs/progress/2026-05-12_project_graph_refresh.md` | Added this progress note. |

## 7. Interpretation

The project map now better represents the actual research path: V2 is no longer only a dataset-construction project. It now includes controlled bridge-transfer experiments, concept inspection, and early behavioral steering probes.

The current honest steering conclusion remains conservative: there is directional evidence for some reviewed concept directions, but not yet a clean monosemantic causal-steering claim.

## 8. Successes

The refresh succeeded because it incorporated the newer steering branch without discarding the readable project-map style.

## 9. Failures Or Limits

No tests were run, and Mermaid rendering was not visually checked. This was a documentation update based on repository inspection.

## 10. External Works And Papers

No new external papers, datasets, algorithms, or technologies were introduced in this graph refresh.

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `docs/PROJECT_GRAPH.md` | Expanded graph and map to cover transfer and steering. | Keep `/graphify .` current with the workspace. |
| `docs/progress/README.md` | Added this note to the task index. | Keep notebook navigation current. |
| `docs/progress/2026-05-12_project_graph_refresh.md` | Added task note. | Record this graph refresh. |

## 12. Follow-Up

- [ ] Add a separate evaluation-plan note for neutral prompts, random-direction controls, and non-keyword steering scores.
- [ ] Re-render Mermaid diagrams in Obsidian or a Markdown preview before using them in a report.
