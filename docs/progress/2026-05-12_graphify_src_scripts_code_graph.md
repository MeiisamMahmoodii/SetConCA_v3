# Task: Graphify Code Graph For Src And Scripts

Tags: #progress #documentation #graphify #code-graph

Related notes: [[README]] [[PROJECT_GRAPH]] [[2026-05-12_project_graph_refresh]]

## 1. Goal

The user invoked the local graphify skill and selected `src + scripts` after the full workspace was too large for safe semantic extraction.

The goal was to generate a persistent code knowledge graph for the implementation slice of SetConCA V2.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-12 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Requested input | `src + scripts` |
| Output directory | `graphify-out` |
| Extraction mode | Code-only AST extraction |

## 3. Hypothesis Or Rationale

The full workspace contained hundreds of docs and result files, so running semantic graph extraction over `.` would be expensive and noisy. The code slice is small enough to graph deterministically and gives a clean architecture graph for the implementation modules and runnable scripts.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Read `C:\Users\MPC\.agents\skills\graphify\SKILL.md`. | Needed to follow the graphify workflow. | Confirmed output requirements: HTML, JSON, and report. | Succeeded |
| 2 | Checked graphify installation. | Needed to avoid installing unnecessarily. | `graphify` was already installed under `C:\Python314\python.exe`. | Succeeded |
| 3 | Ran detector on the full workspace. | Needed to obey corpus-size safety rules. | Full workspace had 326 supported files and about 4.6M words, so a narrower target was required. | Succeeded |
| 4 | User selected `src + scripts`. | Needed a practical code-focused scope. | Proceeded with 25 code files. | Succeeded |
| 5 | Ran graphify library pipeline on combined `src` and `scripts`. | Needed a single combined graph. | Generated graph with 241 nodes, 406 edges, and 23 communities. | Succeeded |
| 6 | Read report sections. | Needed to summarize useful graph output. | Extracted God Nodes, Surprising Connections, and Suggested Questions. | Succeeded |

## 5. Code And Pseudocode

```text
files = collect_files(src) + collect_files(scripts)
extraction = graphify.extract(files)
G = graphify.build_from_json(extraction)
communities = graphify.cluster(G)
cohesion = graphify.score_all(G, communities)
report = graphify.report.generate(...)
export graph.json
export graph.html
write GRAPH_REPORT.md
```

## 6. Results

| Metric | Value |
| --- | ---: |
| Files graphed | 25 |
| Nodes | 241 |
| Edges | 406 |
| Communities | 23 |

### Output Files

| File | Purpose |
| --- | --- |
| `graphify-out/graph.html` | Interactive graph visualization. |
| `graphify-out/graph.json` | GraphRAG-ready graph data. |
| `graphify-out/GRAPH_REPORT.md` | Plain-language graph audit report. |
| `graphify-out/run_summary_src_scripts.json` | Compact summary of this run. |

## 7. Interpretation

The graph identifies `run_transfer_steering_grid.py` as the largest current hub. That matches the codebase shape: it owns training, pointwise baseline comparison, bridge fitting, transfer scoring, steering proxy scoring, summaries, and plots.

The suggested questions point toward possible refactors, especially separating responsibilities currently concentrated in `run_transfer_steering_grid.py`.

## 8. Successes

The graph was generated without running LLM semantic extraction because the selected corpus was code-only. This made the run cheap and deterministic.

## 9. Failures Or Limits

Because semantic extraction was skipped, rationale and cross-document research intent are not deeply represented in this graph. The graph is strongest for code structure, functions, classes, and inferred implementation relationships.

## 10. External Works And Papers

No new papers were used. The graphify skill/tool was used as local project tooling.

## 11. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `graphify-out/graph.html` | Generated interactive graph. | Visual code navigation. |
| `graphify-out/graph.json` | Generated graph data. | Persistent graph and future queries. |
| `graphify-out/GRAPH_REPORT.md` | Generated audit report. | Human-readable graph summary. |
| `graphify-out/run_summary_src_scripts.json` | Generated run summary. | Preserve counts and top findings. |
| `docs/progress/README.md` | Added this task to the index. | Keep notebook current. |
| `docs/progress/2026-05-12_graphify_src_scripts_code_graph.md` | Added this note. | Record graphify run. |

## 12. Follow-Up

- [ ] Use `graphify query` to trace the `PointwiseTopKModel` bridge question.
- [ ] Consider a future semantic graph over `docs/progress` only, not the whole `results` folder.
