# Task: Graphify Docs Semantic Graph

Tags: #progress #documentation #graphify #semantic-graph

Related notes: [[README]] [[2026-05-12_graphify_src_scripts_code_graph]] [[2026-05-12_project_graph_refresh]]

## 1. Goal

The user invoked the local graphify skill for `docs` and asked to use parallel extraction agents while keeping the information that already existed in graphify.

The goal was to add a semantic documentation graph to the existing `src + scripts` code graph without overwriting the preserved code-only outputs.

## 2. Context

| Field | Value |
| --- | --- |
| Date | 2026-05-12 |
| Workspace | `C:\Users\MPC\Documents\code\SetConCA\SetConCA_V2` |
| Requested input | `docs` |
| Output directory | `graphify-out` |
| Extraction mode | Parallel semantic extraction plus existing code graph merge |

## 3. Rationale

The earlier graphify run produced a deterministic implementation graph for `src + scripts`. The docs contain the research narrative, decisions, rationale, results, limitations, figures, and guides that are not visible from AST extraction alone.

The useful graph is therefore a combined one: keep the implementation graph intact, add semantic documentation nodes and relationships, then recluster the full graph.

## 4. Actions

| Step | Action | Why | Result | Status |
| --- | --- | --- | --- | --- |
| 1 | Preserved the prior code graph outputs. | Needed to keep the existing graphify information. | Kept `graph_src_scripts.json`, `graph_src_scripts.html`, and `GRAPH_REPORT_src_scripts.md`. | Succeeded |
| 2 | Detected supported files under `docs`. | Needed to scope semantic extraction. | Found 62 supported files, about 65,642 words: 54 docs, 1 paper, 7 images; 1 sensitive file skipped. | Succeeded |
| 3 | Split semantic extraction across five parallel agents. | Needed faster extraction while keeping chunk outputs auditable. | Wrote `chunk_01_docs.json` through `chunk_05_docs.json`. | Succeeded |
| 4 | Validated chunk JSON files. | Needed to ensure merge inputs were parseable. | Extracted 212 semantic nodes, 294 semantic edges, and 15 hyperedges. | Succeeded |
| 5 | Merged semantic docs graph with the preserved `src + scripts` graph. | Needed one combined graphify view. | Generated 453 nodes, 694 edges, and 30 communities. | Succeeded |
| 6 | Re-exported graphify outputs. | Needed current default graph outputs plus explicit combined copies. | Updated `graph.html`, `graph.json`, and `GRAPH_REPORT.md`; also wrote `graph_docs_combined.*`. | Succeeded |

## 5. Results

| Metric | Value |
| --- | ---: |
| Docs files detected | 62 |
| Estimated words | 65,642 |
| Semantic extraction chunks | 5 |
| Semantic nodes | 212 |
| Semantic edges | 294 |
| Semantic hyperedges | 15 |
| Combined nodes | 453 |
| Combined edges | 694 |
| Combined communities | 30 |

### Output Files

| File | Purpose |
| --- | --- |
| `graphify-out/graph.html` | Current combined interactive graph visualization. |
| `graphify-out/graph.json` | Current combined GraphRAG-ready graph data. |
| `graphify-out/GRAPH_REPORT.md` | Current combined graph audit report. |
| `graphify-out/graph_docs_combined.html` | Named copy of the combined docs/code visualization. |
| `graphify-out/graph_docs_combined.json` | Named copy of the combined docs/code graph data. |
| `graphify-out/GRAPH_REPORT_docs_combined.md` | Named copy of the combined docs/code report. |
| `graphify-out/.graphify_semantic_docs.json` | Merged semantic extraction payload from the five doc chunks. |
| `graphify-out/run_summary_docs_combined.json` | Compact run summary and counts. |
| `graphify-out/graph_src_scripts.*` | Preserved previous code-only graph outputs. |

## 6. Interpretation

The combined graph now joins implementation hubs such as `run_transfer_steering_grid.py`, training scripts, activation extraction, and steering utilities with the research documentation around latent sets, controlled bridge metrics, SetConCA versus pointwise baselines, causal steering probes, and dataset quality limits.

The main benefit is that the graph can answer both code-navigation questions and research-history questions in one structure.

## 7. Limits

The semantic extraction is an agent-generated abstraction over docs, paper, and image assets. It is useful for navigation and connection-finding, but individual inferred edges should still be checked against the source notes before being treated as evidence.

The token counters in the chunk JSON are zero because the extraction agents returned structured outputs directly rather than reporting token usage.

## 8. Files Changed

| File | Change | Reason |
| --- | --- | --- |
| `graphify-out/graph.html` | Updated combined graph visualization. | Current graphify default output. |
| `graphify-out/graph.json` | Updated combined graph data. | Current persistent graph. |
| `graphify-out/GRAPH_REPORT.md` | Updated combined graph report. | Human-readable graph audit. |
| `graphify-out/graph_docs_combined.*` | Added named combined graph copies. | Preserve this docs merge run explicitly. |
| `graphify-out/GRAPH_REPORT_docs_combined.md` | Added named combined report copy. | Preserve this docs merge run explicitly. |
| `graphify-out/chunk_01_docs.json` through `chunk_05_docs.json` | Added extraction chunks. | Audit trail for parallel semantic extraction. |
| `graphify-out/.graphify_semantic_docs.json` | Added merged semantic extraction payload. | Reproducible merge input. |
| `graphify-out/run_summary_docs_combined.json` | Added run summary. | Preserve counts and output inventory. |
| `docs/progress/README.md` | Added this task to the index. | Keep notebook current. |
| `docs/progress/2026-05-12_graphify_docs_semantic_graph.md` | Added this note. | Record graphify docs run. |

## 9. Follow-Up

- [ ] Use the combined graph to trace how dataset design choices connect to causal steering probe limitations.
- [ ] Inspect inferred edges in `GRAPH_REPORT.md` before citing them as evidence.
