# Task: Project Graph Simplification

Tags: #progress #documentation #project-map

Related notes: [[README]] [[PROJECT_GRAPH]] [[raw_json_to_dataset_guide]] [[2026-05-06_project_graph_documentation]] [[2026-05-06_progress_update_protocol]]

## 1. Goal

The user said `docs/PROJECT_GRAPH.md` was too complicated and did not look good. The goal was to make it easier to understand and to establish that it should be updated as the project changes.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-06 |
| File updated | `docs/PROJECT_GRAPH.md` |
| Reason | Simplify project structure, diagrams, and explanations. |

## 3. Hypothesis Or Rationale

A project graph should help orientation first. The previous version was technically dense and had many nodes, which made it harder to read. A better map should use fewer diagrams, plain English summaries, small tables, and links to deeper notes.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Replaced the dense graph with a simpler project map. | Needed a more understandable first read. | `docs/PROJECT_GRAPH.md` now starts with a big-picture pipeline. | Succeeded |
| 2 | Added folder-role table. | Needed to explain where things live. | Main folders now have simple descriptions. | Succeeded |
| 3 | Added a smaller data-pipeline diagram. | Needed visual clarity without overwhelming detail. | Raw data to generated artifacts is now shown in one diagram. | Succeeded |
| 4 | Added dataset row shapes. | Needed concrete examples of the data contracts. | Raw, accepted rewrite, and grouped set examples are included. | Succeeded |
| 5 | Added core code map and commands. | Needed practical navigation. | Main scripts/modules and run commands are listed. | Succeeded |
| 6 | Added "Keep This Updated" section. | Needed to satisfy the user's request to update the graph going forward. | The file now states what sections to update when project pieces change. | Succeeded |
| 7 | Recorded this task in progress docs. | Needed the lab notebook to track the documentation change. | This note was created and the progress index was updated. | Succeeded |

## 5. Code And Pseudocode

Documentation update logic:

```text
if project structure changes:
    update PROJECT_GRAPH folder roles and code map
if data pipeline changes:
    update PROJECT_GRAPH data pipeline and row shapes
if outputs or tests change:
    update PROJECT_GRAPH artifacts and tests
always:
    add or update matching docs/progress note
```

## 6. Results

| Output | Result |
| --- | --- |
| `docs/PROJECT_GRAPH.md` | Simplified and rewritten as a readable project map. |
| `docs/progress/2026-05-06_project_graph_simplification.md` | Added task note. |
| `docs/progress/README.md` | Added index entry for this task. |

## 7. Interpretation

The project graph is now meant to be a navigation page, not a full technical report. Detailed explanations belong in linked progress notes and guides.

## 8. Successes

The update succeeded because the document now separates orientation, data flow, data shapes, code roles, commands, risks, and update rules.

## 9. Failures Or Limits

No diagrams were rendered in this task. The Mermaid blocks are valid-looking Markdown, but visual rendering should be checked in Obsidian or another Mermaid-capable viewer.

## 10. External Works And Papers

No external papers, algorithms, datasets, or new technologies were introduced in this documentation cleanup.

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `docs/PROJECT_GRAPH.md` | Rewritten as a simpler project map. | Improve readability and maintainability. |
| `docs/progress/2026-05-06_project_graph_simplification.md` | Added task log. | Record this documentation action. |
| `docs/progress/README.md` | Added task index entry. | Keep progress notebook connected. |

## 12. Follow-Up

- [ ] Check Mermaid rendering in Obsidian.
- [ ] Update this map whenever new modules, commands, outputs, tests, or dataset steps are added.

